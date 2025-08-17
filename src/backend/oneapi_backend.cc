#include "backend/oneapi_backend.h"
#include "backend/sycl_backend.h"
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
#include <algorithm>
#include <variant>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include "log.h"
#include <spdlog/spdlog.h>

namespace celerity::detail {

// Forward declarations
class ze_event_impl;
void* allocate_host_memory(size_t size, size_t alignment);

// Helper functions for event management
async_event make_event_from_ze_owned(ze_event_handle_t evt, ze_event_pool_handle_t pool);

// Helper to check Level-Zero return codes
static void ze_check(ze_result_t result, const char* message) {
    if(result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("Level-Zero error: ") + message +
                                 " (" + std::to_string(static_cast<int>(result)) + ")");
    }
}

std::unique_ptr<backend> make_oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                    const oneapi_backend::configuration& cfg) {
  return std::make_unique<oneapi_backend>(devices, cfg);
}

//------------------------------------------------------------------------------
// 1) Constructor / Destructor
//------------------------------------------------------------------------------

oneapi_backend::oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                               const configuration& config)
    : m_config(config) {

    // Initialize Level Zero
    ze_check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");

    spdlog::info("Celerity backend ACTIVE: oneAPI/Level-Zero specialization.");
    spdlog::info("oneAPI backend: initializing Level-Zero for {} device(s).", devices.size());

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

    spdlog::info("oneAPI backend: selected Level-Zero driver handle {}", (void*)m_driver);

    // (2) Create context
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_check(zeContextCreate(m_driver, &contextDesc, &m_context), "zeContextCreate");

    spdlog::info("oneAPI backend: created Level-Zero context {}", (void*)m_context);

    // (3) Initialize devices
    m_devices.resize(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        device_state st{};
        st.device  = devices[i];
        st.context = m_context;

        ze_command_queue_desc_t queueDesc{};
        queueDesc.stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        queueDesc.ordinal  = 0;
        queueDesc.index    = 0;
        queueDesc.mode     = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        queueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        // Consider dropping EXPLICIT_ONLY unless you truly need it:
        // queueDesc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;

        ze_check(zeCommandQueueCreate(m_context, st.device, &queueDesc, &st.queue),
                 "zeCommandQueueCreate");

        // Initialize SYCL side once
        st.sycl_dev = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(st.device);
        
        using ctx_in_t = sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context>;
        ctx_in_t ctxIn{ m_context,
                        std::vector<sycl::device>{ st.sycl_dev },
                        sycl::ext::oneapi::level_zero::ownership::keep };
        st.sycl_ctx = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(ctxIn);

        if (m_config.per_device_submission_threads) {
          st.submit_thread = std::make_optional<thread_queue>(
              named_threads::task_type_device_submitter(i));
        }
        m_devices[i] = std::move(st);
    }

    // (4) Populate system info
    m_system.num_devices = devices.size();
    m_system.max_work_group_size = 0; // Initialize to 0, will be updated with max across all devices
    
    for(const auto& dev : devices) {
        ze_device_properties_t props{};
        ze_check(zeDeviceGetProperties(dev, &props), "zeDeviceGetProperties");

        ze_device_compute_properties_t computeProps{};
        ze_check(zeDeviceGetComputeProperties(dev, &computeProps), "zeDeviceGetComputeProperties");
        
        // Track the maximum work group size across all devices
        m_system.max_work_group_size = std::max(m_system.max_work_group_size, 
                                               static_cast<size_t>(computeProps.maxTotalGroupSize));
        
        // Intel ARC specific properties
        if(props.vendorId == 0x8086) {
            m_system.max_compute_units = props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice;
        }
    }
    
    // (5) Initialize host state (create it exactly once)
    m_host = std::make_unique<host_state>(m_context, m_config.profiling);
    spdlog::info("Celerity backend ACTIVE: oneAPI specialization (Level-Zero). Devices initialized: {}. Max WG size: {}.", m_system.num_devices, m_system.max_work_group_size);
}

oneapi_backend::~oneapi_backend() {
    for (auto& st : m_devices) {
        st.submit_thread.reset();
        if (st.queue) zeCommandQueueSynchronize(st.queue, UINT64_MAX);
        if (st.queue) zeCommandQueueDestroy(st.queue);
    }
    m_host.reset();
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
    ze_device_mem_alloc_desc_t devDesc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t  hostDesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* ptr = nullptr;
    ze_device_handle_t dev = m_devices.empty() ? nullptr : m_devices[0].device;
    ze_check(zeMemAllocShared(m_context, &devDesc, &hostDesc, size, /*alignment*/0, dev, &ptr),
             "zeMemAllocShared in debug_alloc");
    return ptr;
}

void oneapi_backend::debug_free(void* ptr) {
    if(ptr) ze_check(zeMemFree(m_context, ptr), "zeMemFree in debug_free");
}

async_event oneapi_backend::enqueue_host_alloc(size_t size, size_t alignment) {
    return m_host->alloc_queue.submit([this, size, alignment]() -> void* {
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
    return m_host->alloc_queue.submit([this, ptr]() -> void* {
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

async_event oneapi_backend::enqueue_host_task(
    size_t host_lane, const host_task_launcher& launcher,
    std::vector<closure_hydrator::accessor_info> accessor_infos,
    const range<3>& global_range, const box<3>& execution_range,
    const communicator* collective_comm)
{
    // 1) Arm & hydrate on the calling thread
    auto& hydrator = closure_hydrator::get_instance();
    hydrator.arm(target::host_task, std::move(accessor_infos));
    auto launch_hydrated = hydrator.hydrate<target::host_task>(launcher);

    // 2) Submit the already-hydrated closure to your host queue
    thread_queue& q = m_host->get_queue(host_lane);
    return q.submit([=, launch_hydrated = std::move(launch_hydrated)]() {
        launch_hydrated(global_range, execution_range, collective_comm);
        return nullptr;
    });
}

async_event oneapi_backend::enqueue_device_kernel(
    device_id did, size_t lane, const device_kernel_launcher& launch,
    std::vector<closure_hydrator::accessor_info> accessor_infos,
    const box<3>& execution_range, const std::vector<void*>& reduction_ptrs)
{
    return enqueue_work(did, lane,
        [=, this, acc_infos = std::move(accessor_infos)](device_state& st) mutable -> async_event {
            // Use the pre-initialized per-lane SYCL queue
            auto& queue = st.sycl_queue_for_lane(lane, m_config.profiling);

            // Submit with hydrator inside the handler
            sycl::event ev = queue.submit([&, acc_infos = std::move(acc_infos)](sycl::handler& cgh) mutable {
                auto& hydrator = closure_hydrator::get_instance();
                hydrator.arm(target::device, std::move(acc_infos));
                auto launch_h = hydrator.hydrate<target::device>(cgh, launch);
                launch_h(cgh, execution_range, reduction_ptrs);
            });

            celerity::detail::sycl_backend_detail::flush(queue);
            using sy_ev = celerity::detail::sycl_backend_detail::sycl_event;
            return make_async_event<sy_ev>(std::move(ev), m_config.profiling);
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
    thread_queue& tq = m_host->get_queue(host_lane);
    return tq.submit([=]() -> void* {
        const auto box  = copy_region.get_boxes()[0];
        const auto off  = box.get_min();      // {ox, oy, oz}
        const auto ext  = box.get_range();    // {nx, ny, nz}
        const auto s    = std::get<strided_layout>(src_layout).allocation.get_range();
        const auto d    = std::get<strided_layout>(dst_layout).allocation.get_range();

        const auto src = static_cast<const char*>(source);
        auto       dst = static_cast<char*>(dest);
        const size_t sx = s[0], sy = s[1];
        const size_t dx = d[0], dy = d[1];

        for (size_t z = 0; z < ext[2]; ++z) {
            for (size_t y = 0; y < ext[1]; ++y) {
                size_t src_off = ((off[2]+z) * sy + (off[1]+y)) * sx + off[0];
                size_t dst_off = ((off[2]+z) * dy + (off[1]+y)) * dx + off[0];
                std::memcpy(dst + dst_off*elem_size, src + src_off*elem_size, ext[0]*elem_size);
            }
        }
        return nullptr;
    });
}

async_event oneapi_backend::enqueue_device_copy(device_id did, size_t lane, 
    const void* source, void* dest,
    const region_layout& src_layout, const region_layout& dst_layout, 
    const region<3>& copy_region, size_t elem_size) {
        
    return enqueue_work(did, lane, [=, this](device_state& st) -> async_event {
        // Fast-path: use SYCL memcpy for linearized layouts when profiling is enabled
        if (m_config.profiling &&
            std::holds_alternative<linearized_layout>(src_layout) &&
            std::holds_alternative<linearized_layout>(dst_layout)) {

            auto& q = st.sycl_queue_for_lane(lane, /*profiling*/ true);

            // unit_box means "1 * elem_size" bytes in these tests
            const size_t bytes = copy_region.get_boxes()[0].get_area() * elem_size;

            sycl::event ev = q.memcpy(dest, source, bytes);
            celerity::detail::sycl_backend_detail::flush(q);

            using sy_ev = celerity::detail::sycl_backend_detail::sycl_event;
            return make_async_event<sy_ev>(std::move(ev), /*profiling*/ true);
        }

        // Fall back to L0 3D-region implementation
        // 1. Create command list
        ze_command_list_handle_t cmdlist = nullptr;
        ze_command_list_desc_t listDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
        ze_check(zeCommandListCreate(st.context, st.device, &listDesc, &cmdlist),
                 "zeCommandListCreate");

        // 2. Create a host-visible event we always return (profiling or not)
        ze_event_pool_handle_t evt_pool = nullptr;
        ze_event_handle_t evt = nullptr;

        ze_event_pool_desc_t poolDesc{};
        poolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        poolDesc.count = 1;
        ze_check(zeEventPoolCreate(st.context, &poolDesc, 1, &st.device, &evt_pool),
                 "zeEventPoolCreate");

        ze_event_desc_t evtDesc{};
        evtDesc.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
        evtDesc.index  = 0; // single-event pool
        evtDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        evtDesc.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
        ze_check(zeEventCreate(evt_pool, &evtDesc, &evt), "zeEventCreate");

        // 3. Configure 3D region using min corner as origin
        const auto box = copy_region.get_boxes()[0];
        ze_copy_region_t src_region{ (uint32_t)box.get_min()[0], (uint32_t)box.get_min()[1], (uint32_t)box.get_min()[2],
                                     (uint32_t)box.get_range()[0], (uint32_t)box.get_range()[1], (uint32_t)box.get_range()[2] };
        ze_copy_region_t dst_region = src_region;
        
        const auto s = std::get<strided_layout>(src_layout).allocation.get_range();
        const auto d = std::get<strided_layout>(dst_layout).allocation.get_range();
        
        ze_check(zeCommandListAppendMemoryCopyRegion(
            cmdlist,
            dest,   &dst_region, (uint32_t)(d[0] * elem_size), (uint32_t)(d[0] * d[1] * elem_size),
            source, &src_region, (uint32_t)(s[0] * elem_size), (uint32_t)(s[0] * s[1] * elem_size),
            evt, 0, nullptr),
            "zeCommandListAppendMemoryCopyRegion");
        
        // 5. Close and execute command list
        ze_check(zeCommandListClose(cmdlist), "zeCommandListClose");
        ze_check(zeCommandQueueExecuteCommandLists(st.queue, 1, &cmdlist, nullptr),
                 "zeCommandQueueExecuteCommandLists");

        // 6. Cleanup
        zeCommandListDestroy(cmdlist);

        // 7. Return async event
        return make_event_from_ze_owned(evt, evt_pool);
    });
}

//------------------------------------------------------------------------------
// 7) Error Checking
//------------------------------------------------------------------------------

void oneapi_backend::check_async_errors() {
    for (auto& st : m_devices) {
        // Wait on SYCL queues first
        for (auto& qp : st.sycl_lanes) if (qp) qp->wait_and_throw();
        
        // Then check L0 queue
        ze_result_t res = zeCommandQueueSynchronize(st.queue, 0);
        if (res != ZE_RESULT_SUCCESS && res != ZE_RESULT_NOT_READY) {
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
    ze_event_impl(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_evt, bool owns_pool)
        : m_event(evt), m_pool(pool), m_owns_evt(owns_evt), m_owns_pool(owns_pool) {}

    ~ze_event_impl() override {
        if(m_owns_evt && m_event) {
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
    bool m_owns_evt;
    bool m_owns_pool;
};

// Create async_event from Level-Zero event that you own
async_event make_event_from_ze_owned(ze_event_handle_t evt, ze_event_pool_handle_t pool) {
    return make_async_event<ze_event_impl>(evt, pool, /*owns_evt*/true, /*owns_pool*/true);
}

// Legacy function for backward compatibility - use the specific ones above instead
async_event make_event_from_ze(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_pool) {
    return make_async_event<ze_event_impl>(evt, pool, /*owns_evt*/true, owns_pool);
}

// Create async_event with start and end events for profiling
async_event make_event_from_ze(ze_event_handle_t start, ze_event_handle_t end, ze_event_pool_handle_t pool) {
    // This is simplified - actual implementation would need to handle both events
    return make_async_event<ze_event_impl>(end, pool, /*owns_evt*/true, /*owns_pool*/true);
}

celerity::detail::oneapi_backend::host_state::host_state(ze_context_handle_t ctx, bool profiling)
    : context(ctx), profiling(profiling), alloc_queue(named_threads::thread_type::alloc, profiling) {}

celerity::detail::thread_queue& celerity::detail::oneapi_backend::host_state::get_queue(size_t lane) {
    while (lane >= host_queues.size()) {
        host_queues.emplace_back(named_threads::task_type_host_queue(host_queues.size()), profiling);
    }
    return host_queues[lane];
}

bool celerity::detail::oneapi_backend::is_profiling_enabled() const {
    return m_config.profiling;
}

async_event oneapi_backend::enqueue_work(device_id did, size_t lane,
                                         std::function<async_event(device_state&)> work) {
  const size_t idx = static_cast<size_t>(did);
  if (idx >= m_devices.size()) throw std::runtime_error("enqueue_work: invalid device_id");
  device_state& st = m_devices[idx];
  return work(st);
}

} // namespace celerity::detail
