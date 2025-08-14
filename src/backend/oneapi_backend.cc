#include "backend/oneapi_backend.h"
#include "closure_hydrator.h"
#include "async_event.h"
#include "thread_queue.h"
#include "system_info.h"
#include "dense_map.h"
#include <level_zero/ze_api.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>

namespace celerity::detail {

class ze_event_impl;
async_event make_event_from_ze(ze_event_handle_t start, ze_event_handle_t end, ze_event_pool_handle_t pool);
async_event make_event_from_ze(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_pool);

// Helper to check Level-Zero return codes
static void ze_check(ze_result_t result, const char* message) {
    if(result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("Level-Zero error: ") + message +
                                 " (" + std::to_string(static_cast<int>(result)) + ")");
    }
}

std::unique_ptr<backend> make_oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                    const oneapi_backend::configuration& lvl0_cfg) {
  return std::make_unique<oneapi_backend>(devices, lvl0_cfg);
}

//------------------------------------------------------------------------------
// 1) Constructor / Destructor
//------------------------------------------------------------------------------

oneapi_backend::oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                               const configuration& config)
    : m_config(config),
      m_host(nullptr, config.profiling) {

    // Initialize Level Zero
    ze_check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");

    // (1) Find a Level-Zero driver that covers the given devices
    uint32_t driverCount = 0;
    ze_check(zeDriverGet(&driverCount, nullptr), "zeDriverGet (count)");
    if(driverCount == 0) throw std::runtime_error("No Level-Zero drivers found");
    
    std::vector<ze_driver_handle_t> driverHandles(driverCount);
    ze_check(zeDriverGet(&driverCount, driverHandles.data()), "zeDriverGet (handles)");

    // Find driver that supports all devices
    for(auto drv : driverHandles) {
        uint32_t devCount = 0;
        ze_check(zeDeviceGet(drv, &devCount, nullptr), "zeDeviceGet (count)");
        if(devCount == 0) continue;
        
        std::vector<ze_device_handle_t> drvDevices(devCount);
        ze_check(zeDeviceGet(drv, &devCount, drvDevices.data()), "zeDeviceGet (list)");

        bool allFound = true;
        for(auto userDev : devices) {
            if(std::find(drvDevices.begin(), drvDevices.end(), userDev) == drvDevices.end()) {
                allFound = false;
                break;
            }
        }
        
        if(allFound) {
            m_driver = drv;
            break;
        }
    }
    if(!m_driver) throw std::runtime_error("No driver supports all devices");

    // (2) Create context
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_check(zeContextCreate(m_driver, &contextDesc, &m_context), "zeContextCreate");
    
    // (3) Initialize devices
    size_t max_id = 0;
    for(const auto& dev : devices) {
        device_id did = reinterpret_cast<uintptr_t>(dev);
        max_id = std::max(max_id, static_cast<size_t>(did));
    }
    m_devices.resize(max_id + 1);
    
    for(const auto& dev : devices) {
        device_id did = reinterpret_cast<uintptr_t>(dev);
        device_state st;
        st.device = dev;
        st.context = m_context;

        // Create command queue
        ze_command_queue_desc_t queueDesc = {};
        queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        queueDesc.ordinal = 0;
        queueDesc.index = 0;
        queueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        queueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        queueDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
        
        ze_command_queue_handle_t cmdQueue = nullptr;
        ze_check(zeCommandQueueCreate(m_context, dev, &queueDesc, &cmdQueue),
                 "zeCommandQueueCreate");
        st.queue = cmdQueue;

        // Spawn submission thread if requested
        if(m_config.per_device_submission_threads) {
            st.submit_thread = std::make_optional<thread_queue>(named_threads::task_type_device_submitter(0));
        }

        m_devices[did] = std::move(st);
    }

    // (4) Populate system info
    m_system.num_devices = devices.size();
    for(const auto& dev : devices) {
        ze_device_properties_t props{};
        ze_check(zeDeviceGetProperties(dev, &props), "zeDeviceGetProperties");

        ze_device_compute_properties_t computeProps{};
        ze_check(zeDeviceGetComputeProperties(dev, &computeProps), "zeDeviceGetComputeProperties");
        
        // Use maxTotalGroupSize for work group size
        m_system.max_work_group_size = computeProps.maxTotalGroupSize;
        
        // Intel ARC specific properties
        if(props.vendorId == 0x8086) {
            m_system.max_compute_units = props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice;
        }
    }
    
    // (5) Initialize host state
    m_host = host_state(m_context, m_config.profiling);
}

oneapi_backend::~oneapi_backend() {
    for(auto& kv : m_devices) {
        device_state& st = kv;
        st.submit_thread.reset();
        if(st.queue) zeCommandQueueDestroy(st.queue);
    }
    if(m_context) zeContextDestroy(m_context);
}

//------------------------------------------------------------------------------
// 2) System Info
//------------------------------------------------------------------------------

const system_info& oneapi_backend::get_system_info() const { return m_system; }
system_info& oneapi_backend::get_system_info() { return m_system; }

//------------------------------------------------------------------------------
// 3) Initialization
//------------------------------------------------------------------------------

void oneapi_backend::init() {
    for(auto& kv : m_devices) {
        device_state& st = kv;
        ze_result_t res = zeCommandQueueSynchronize(st.queue, 0);
        if(res != ZE_RESULT_SUCCESS && res != ZE_RESULT_NOT_READY) {
            throw std::runtime_error("Level-Zero queue sync failed in init()");
        }
    }
}

//------------------------------------------------------------------------------
// 4) Memory Management
//------------------------------------------------------------------------------

void* oneapi_backend::debug_alloc(size_t size) {
    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* ptr = nullptr;
    ze_check(zeMemAllocHost(m_context, &hostDesc, size, 1, &ptr),
            "zeMemAllocHost in debug_alloc");
    return ptr;
}

void oneapi_backend::debug_free(void* ptr) {
    if(ptr) ze_check(zeMemFree(m_context, ptr), "zeMemFree in debug_free");
}

async_event oneapi_backend::enqueue_host_alloc(size_t size, size_t alignment) {
    return m_host.alloc_queue.submit([this, size, alignment]() -> void* {
        ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
        void* ptr = nullptr;
        ze_check(zeMemAllocHost(m_context, &hostDesc, size, alignment, &ptr),
                "zeMemAllocHost in enqueue_host_alloc");
        return ptr;
    });
}

async_event oneapi_backend::enqueue_device_alloc(device_id did, size_t size, size_t alignment) {
    return enqueue_work(did, 0, [this, size, alignment](device_state& st) -> async_event {
        ze_device_mem_alloc_desc_t devDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
        void* ptr = nullptr;
        ze_check(zeMemAllocDevice(st.context, &devDesc, size, alignment, st.device, &ptr),
                "zeMemAllocDevice in enqueue_device_alloc");
        return make_complete_event(ptr);
    });
}

async_event oneapi_backend::enqueue_host_free(void* ptr) {
    return m_host.alloc_queue.submit([this, ptr]() -> void* {
        if(ptr) ze_check(zeMemFree(m_context, ptr), "zeMemFree in enqueue_host_free");
        return nullptr;
    });
}

async_event oneapi_backend::enqueue_device_free(device_id did, void* ptr) {
    return enqueue_work(did, 0, [this, ptr](device_state& st) -> async_event {
        if(ptr) ze_check(zeMemFree(st.context, ptr), "zeMemFree in enqueue_device_free");
        return make_complete_event();
    });
}

//------------------------------------------------------------------------------
// 5) Task Execution
//------------------------------------------------------------------------------

async_event oneapi_backend::enqueue_host_task(size_t host_lane,
                                              const host_task_launcher& launcher,
                                              std::vector<closure_hydrator::accessor_info> infos,
                                              const range<3>& global_range,
                                              const box<3>& execution_range,
                                              const communicator* comm) {
    thread_queue& queue = m_host.get_queue(host_lane);
    return queue.submit(
        [launcher, infos = std::move(infos), global_range, execution_range, comm]() -> void* {
            std::vector<void*> ptrs;
            ptrs.reserve(infos.size());
            for(const auto& info : infos) ptrs.push_back(info.ptr);
            launcher(global_range, execution_range, comm);
            return nullptr;
        }
    );
}

async_event oneapi_backend::enqueue_device_kernel(device_id did, size_t lane,
    const device_kernel_launcher& launch,
    std::vector<closure_hydrator::accessor_info> infos,
    const box<3>& execution_range,
    const std::vector<void*>& reduction_ptrs) {

  return enqueue_work(did, lane,
    [this, launch, infos = std::move(infos), execution_range, reduction_ptrs](device_state& st) -> async_event {

    // 1) Build a sycl::device from the native handle:
    sycl::device dev =
      sycl::make_device<sycl::backend::ext_oneapi_level_zero>(st.device);
        
    // 2) Wrap the native context + device list:
    using ctx_input_t = sycl::backend_input_t<
      sycl::backend::ext_oneapi_level_zero, sycl::context>;
    ctx_input_t ctxInput{
      st.context,                        // ze_context_handle_t
      std::vector<sycl::device>{dev},    // must list at least one device
      sycl::ext::oneapi::level_zero::ownership::keep
    };
    
    // 3) Create the SYCL context:
    sycl::context ctx =
      sycl::make_context<sycl::backend::ext_oneapi_level_zero>(ctxInput);
    
    // 4) Wrap the native queue handle + device:
    using queue_input_t = sycl::backend_input_t<
      sycl::backend::ext_oneapi_level_zero, sycl::queue>;
    queue_input_t qInput{
      st.queue,  // ze_command_queue_handle_t
      dev,       // the device it runs on
      sycl::ext::oneapi::level_zero::ownership::keep,
      /* property_list */ {}
    };
    
    // 5) Create the SYCL queue:
    sycl::queue queue =
      sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(qInput, ctx);

    // 6) Submit the kernel:
    sycl::event ev = queue.submit([&](sycl::handler& cgh) {
        launch(cgh, execution_range, reduction_ptrs);
      });

    // 7) Extract the native L0 event and wrap it:
    ze_event_handle_t ze_ev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ev);
      return make_event_from_ze(ze_ev, nullptr, false);
  });
}

//------------------------------------------------------------------------------
// 6) Data Transfers
//------------------------------------------------------------------------------

async_event oneapi_backend::enqueue_host_copy(size_t host_lane,
                                              const void* source,
                                              void* dest,
                                              const region_layout& src_layout,
                                              const region_layout& dst_layout,
                                              const region<3>& copy_region,
                                              size_t elem_size) {
    thread_queue& queue = m_host.get_queue(host_lane);
    return queue.submit([=]() -> void* {
        const auto& cr = copy_region;
        const auto& s  = std::get<strided_layout>(src_layout).allocation.get_range();
        const auto& d  = std::get<strided_layout>(dst_layout).allocation.get_range();
        const auto& sr = cr.get_boxes()[0].get_range();

        const char* src_base = static_cast<const char*>(source);
        char* dst_base = static_cast<char*>(dest);

        for(size_t z = 0; z < sr[2]; ++z) {
            for(size_t y = 0; y < sr[1]; ++y) {
                const char* src_ptr = src_base + (z * s[2] + y * s[1] + 0) * elem_size;
                char* dst_ptr = dst_base + (z * d[2] + y * d[1] + 0) * elem_size;
                size_t bytes = sr[0] * elem_size;
                std::memcpy(dst_ptr, src_ptr, bytes);
            }
        }
        return nullptr;
    });
}

async_event oneapi_backend::enqueue_device_copy(device_id did, size_t lane, 
    const void* source, void* dest,
    const region_layout& src_layout, const region_layout& dst_layout, 
    const region<3>& copy_region, size_t elem_size) {
    
    return enqueue_work(did, lane, [this, source, dest, src_layout, dst_layout, 
                         copy_region, elem_size](device_state& st) -> async_event {
        // 1. Create command list
        ze_command_list_handle_t cmdlist = nullptr;
        ze_command_list_desc_t listDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
        ze_check(zeCommandListCreate(st.context, st.device, &listDesc, &cmdlist),
                 "zeCommandListCreate");

        // 2. Create events for profiling
        ze_event_pool_handle_t evt_pool = nullptr;
        ze_event_handle_t evt = nullptr;
        
        if(is_profiling_enabled()) {
            ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
            poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
            poolDesc.count = 1;
            ze_check(zeEventPoolCreate(st.context, &poolDesc, 1, &st.device, &evt_pool),
                     "zeEventPoolCreate");
            
            ze_event_desc_t evtDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
            evtDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
            ze_check(zeEventCreate(evt_pool, &evtDesc, &evt), "zeEventCreate");
        }

        // 3. Configure 3D copy parameters
        const auto& cr = copy_region;
        const auto& s  = std::get<strided_layout>(src_layout).allocation.get_range();
        const auto& d  = std::get<strided_layout>(dst_layout).allocation.get_range();
        const auto& sr = cr.get_boxes()[0].get_range();

        ze_copy_region_t src_region = {
            0, 0, 0,
            static_cast<uint32_t>(sr[0]),
            static_cast<uint32_t>(sr[1]),
            static_cast<uint32_t>(sr[2])
        };

        ze_copy_region_t dst_region = {
            0, 0, 0,
            static_cast<uint32_t>(sr[0]),
            static_cast<uint32_t>(sr[1]),
            static_cast<uint32_t>(sr[2])
        };

        // 4. Append 3D memory copy
        zeCommandListAppendMemoryCopyRegion(
            cmdlist, 
            dest, &dst_region, 
            static_cast<uint32_t>(d[0] * elem_size),  // dstPitch
            static_cast<uint32_t>(d[1] * d[0] * elem_size),  // dstSlicePitch
            source, &src_region,
            static_cast<uint32_t>(s[0] * elem_size),  // srcPitch
            static_cast<uint32_t>(s[1] * s[0] * elem_size),  // srcSlicePitch
            (is_profiling_enabled() ? evt : nullptr), 
            0, nullptr
        );

        // 5. Close and execute command list
        ze_check(zeCommandListClose(cmdlist), "zeCommandListClose");
        ze_check(zeCommandQueueExecuteCommandLists(st.queue, 1, &cmdlist, nullptr),
                 "zeCommandQueueExecuteCommandLists");

        // 6. Cleanup
        zeCommandListDestroy(cmdlist);

        // 7. Return async event
        if(is_profiling_enabled()) {
            return make_event_from_ze(evt, evt_pool, true);
        }
        return make_complete_event();
    });
}

//------------------------------------------------------------------------------
// 7) Error Checking
//------------------------------------------------------------------------------

void oneapi_backend::check_async_errors() {
    for (auto& kv : m_devices) {
        device_state& st = kv;
        ze_result_t res = zeCommandQueueSynchronize(st.queue, UINT64_MAX);
        if (res != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Level-Zero asynchronous error detected");
        }
    }
}

//------------------------------------------------------------------------------
// 8) Helper Implementations
//------------------------------------------------------------------------------

// Level-Zero event implementation
class ze_event_impl final : public async_event_impl {
public:
    ze_event_impl(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_pool)
        : m_event(evt), m_pool(pool), m_owns_pool(owns_pool) {}

    ~ze_event_impl() override {
        if(m_event) {
            // Ensure event is complete before destroying
            zeEventHostSynchronize(m_event, UINT64_MAX);
            zeEventDestroy(m_event);
        }
        if(m_owns_pool && m_pool) zeEventPoolDestroy(m_pool);
    }

    bool is_complete() override {
        ze_result_t status = zeEventQueryStatus(m_event);
        return status == ZE_RESULT_SUCCESS;
    }

private:
    ze_event_handle_t m_event;
    ze_event_pool_handle_t m_pool;
    bool m_owns_pool;
};

// Create async_event from Level-Zero event
async_event make_event_from_ze(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_pool) {
  return make_async_event<ze_event_impl>(evt, pool, owns_pool);
}

// Create async_event with start and end events for profiling
async_event make_event_from_ze(ze_event_handle_t start, ze_event_handle_t end, ze_event_pool_handle_t pool) {
    // This is simplified - actual implementation would need to handle both events
    return make_async_event<ze_event_impl>(end, pool, true);
}

celerity::detail::oneapi_backend::host_state::host_state(ze_context_handle_t ctx, bool profiling)
    : context(ctx), alloc_queue(named_threads::thread_type::alloc, profiling) {}

celerity::detail::thread_queue& celerity::detail::oneapi_backend::host_state::get_queue(size_t lane) {
    while (lane >= host_queues.size()) {
        host_queues.emplace_back(named_threads::task_type_host_queue(host_queues.size()));
    }
    return host_queues[lane];
}

bool celerity::detail::oneapi_backend::is_profiling_enabled() const {
    return m_config.profiling;
}

async_event oneapi_backend::enqueue_work(device_id did, size_t lane,
                                         std::function<async_event(device_state&)> work) {
    // 1) Validate the device_id against our dense_map
    if(did >= m_devices.size() || !m_devices[did].device) {
        throw std::runtime_error("enqueue_work: invalid device_id");
    }
    device_state& st = m_devices[did];
    return work(st);
}

} // namespace celerity::detail
