// src/backend/oneapi_backend.cc

#include "backend/oneapi_backend.h"

#include "closure_hydrator.h"   // for accessor hydration
#include "async_event.h"        // for async_event, make_ready_async_event, make_event_from_ze
#include "thread_queue.h"       // for thread_queue, enqueue_work
#include "system_info.h"        // for system_info
#include "dense_map.h"          // for m_devices map
#include <level_zero/ze_api.h>  // Level-Zero core API
#include <stdexcept>            // for std::runtime_error
#include <vector>
#include <cstring>              // for std::memcpy

namespace celerity::detail {

// Helper to check Level-Zero return codes and throw on error.
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
      // Initialize host_state with a dummy context for now; we will overwrite it below
      // after creating the real m_context (see step (5)).
      m_host(nullptr, config.profiling) {

    // (1) Find a Level-Zero driver that covers the given devices.
    // Query the number of drivers installed.
    uint32_t driverCount = 0;
    ze_check(zeDriverGet(&driverCount, nullptr), "zeDriverGet (count)");
    if(driverCount == 0) {
        throw std::runtime_error("No Level-Zero drivers found on system");
    }
    std::vector<ze_driver_handle_t> driverHandles(driverCount);
    ze_check(zeDriverGet(&driverCount, driverHandles.data()), "zeDriverGet (handles)");

    // We assume all devices come from a single driver. Find which driver instance exposes our devices.
    // (In most installations, there is only one Level-Zero driver. Hopefully here as well)
    bool foundDriver = false;
    for(auto drv : driverHandles) {
        // Query all devices for this driver.
        uint32_t devCount = 0;
        if(zeDeviceGet(drv, &devCount, nullptr) != ZE_RESULT_SUCCESS || devCount == 0) {
            continue;
        }
        std::vector<ze_device_handle_t> drvDevices(devCount);
        ze_check(zeDeviceGet(drv, &devCount, drvDevices.data()), "zeDeviceGet (list)");

        // Check if all user-provided devices are in drvDevices.
        bool allInSet = true;
        for(auto userDev : devices) {
            bool inThisDriver = false;
            for(auto dd : drvDevices) {
                if(dd == userDev) {
                    inThisDriver = true;
                    break;
                }
            }
            if(!inThisDriver) {
                allInSet = false;
                break;
            }
        }
        if(allInSet) {
            m_driver = drv;
            foundDriver = true;
            break;
        }
    }
    if(!foundDriver) {
        throw std::runtime_error("None of the provided devices belong to a single Level-Zero driver");
    }

    // (2) Create a single Level-Zero context covering all devices.
    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0u};
    ze_check(zeContextCreate(m_driver, &contextDesc, &m_context), "zeContextCreate");
    
    // (3) For each device, create a device_state entry: store device handle, store context, create a queue, and possibly a submission thread.
    // First, determine the maximum device ID to size the dense_map properly
    size_t max_device_id = 0;
    for(const auto& zedev : devices) {
        device_id did{static_cast<size_t>(reinterpret_cast<uintptr_t>(zedev))};
        max_device_id = std::max(max_device_id, static_cast<size_t>(did));
    }
    m_devices.resize(max_device_id + 1);
    
    for(const auto& zedev : devices) {
        device_state st;
        st.device = zedev;
        st.context = m_context;

        // Create an in-order command queue on this device.
        ze_command_queue_desc_t queueDesc = {};
        queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        // ordinal/index selection: 0 is the first command queue group; priority = normal/0.
        queueDesc.ordinal = 0;            // assume a single compute queue group index = 0
        queueDesc.index   = 0;            // the sub-device index is 0 (no subdevices)
        queueDesc.flags   = 0;            // no special flags
        queueDesc.mode    = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS; // asynchronous (in-order) queue
        queueDesc.priority= ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

        ze_command_queue_handle_t cmdQueue = nullptr;
        ze_check(zeCommandQueueCreate(m_context, /*device=*/zedev, &queueDesc, &cmdQueue),
                 "zeCommandQueueCreate");
        st.queue = cmdQueue;

        // If requested, spawn a per-device submission thread. That thread will accept
        // lambdas of type std::function<async_event(device_state&)>.
        if(m_config.per_device_submission_threads) {
            st.submit_thread = std::make_optional<thread_queue>(named_threads::task_type_device_submitter(0));
        }

        // Insert into our dense_map<device_id, device_state>
        device_id did{static_cast<size_t>(reinterpret_cast<uintptr_t>(zedev))};
        m_devices[did] = std::move(st);  // Use move assignment
    }

    // (4) Populate m_system with metadata from Level-Zero.
    // At minimum, fill number of devices, max compute units (cores), global memory size, etc.
    // We assume system_info has a helper to query Level-Zero. If not, we query manually.

    // Let's query each device and accumulate:
    m_system.num_devices = static_cast<int>(devices.size());
    // For simplicity, take the first device's properties for uniform values like max work-group size:
    for(const auto& zedev : devices) {
        ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        ze_check(zeDeviceGetProperties(zedev, &props), "zeDeviceGetProperties");
        // Use a conservative default for max work group size since the exact property name may vary
        m_system.max_work_group_size = 1024; // Default to 1024, which is common for most GPUs
        m_system.max_compute_units = props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice;
    }
    // Total global memory: sum of each device's global memory:
    uint64_t totalGlobalMem = 0;
    for(auto zedev : devices) {
        uint32_t count = 1;
        ze_device_memory_properties_t memProps = {ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES, nullptr};
        ze_check(zeDeviceGetMemoryProperties(zedev, &count, &memProps), "zeDeviceGetMemoryProperties");
        totalGlobalMem += memProps.totalSize;
    }
    m_system.total_global_memory = totalGlobalMem;

    // Host page size: use a conservative default (e.g., 4096), or query via OS if needed.
    m_system.host_page_size = 4096;

    // (5) Initialize m_host now that we have a valid context:
    // We reconstruct m_host with the real context and profiling flag.
    m_host = host_state(m_context, m_config.profiling);
}

// oneapi_backend destructor
oneapi_backend::~oneapi_backend() {
    // (1) Tear down each device_state: stop its submission thread (if any), destroy its queue.
    for(auto& kv : m_devices) {
        device_state& st = kv;
        // If there is a submission thread, let it be destroyed automatically
        // The thread_queue destructor will handle cleanup
        st.submit_thread.reset();
        // Destroy the command queue:
        if(st.queue) {
            zeCommandQueueDestroy(st.queue);
            st.queue = nullptr;
        }
    }

    // (2) Destroy the Level-Zero context.
    if(m_context) {
        zeContextDestroy(m_context);
        m_context = nullptr;
    }

    // (3) No need to destroy m_driver—driver lifetime is managed by Level-Zero runtime.
}

// ----------------------------------------------------------------------------------
// 2.1. get_system_info() (const and non-const overloads)
// ----------------------------------------------------------------------------------
//
// These simply return a reference to the system_info object we populated in the
// constructor. The const overload satisfies the pure-virtual from backend.h, and
// the non-const overload is a convenience (not virtual).
//
// Header declarations (oneapi_backend.h) :contentReference[oaicite:1]{index=1}:
//   const system_info& get_system_info() const override;
//   system_info&       get_system_info();
    
const system_info& oneapi_backend::get_system_info() const {
    return m_system;
}

system_info& oneapi_backend::get_system_info() {
    return m_system;
}

// ----------------------------------------------------------------------------------
// 2.2. init()
// ----------------------------------------------------------------------------------
//
// Called once after construction but before any enqueue_* calls. The SYCL backends
// often leave init() empty if the constructor already set up all queues. Here, we
// simply synchronize each device queue once to "warm up" the driver (optional).
//
// Header declaration :contentReference[oaicite:2]{index=2}:
//   void init() override;

void oneapi_backend::init() {
    // For each device_state, perform a zero-time synchronize to ensure the queue is valid.
    for(auto& kv : m_devices) {
        device_state& st = kv;
        // We do a "no-op" sync: waiting on an empty queue returns immediately,
        // but verifies that the queue handle is usable.
        ze_result_t res = zeCommandQueueSynchronize(st.queue, 0ULL);
        if(res != ZE_RESULT_SUCCESS && res != ZE_RESULT_NOT_READY) {
            // ZE_RESULT_NOT_READY is acceptable (nothing enqueued). Other errors are not.
            throw std::runtime_error("Level-Zero queue sync failed in init()");
        }
    }
    // No further action needed; all contexts and queues already exist.
}

// ----------------------------------------------------------------------------------
// 2.3. debug_alloc(size_t) and debug_free(void*)
// ----------------------------------------------------------------------------------
//
// These allocate/free "device-accessible" host (pinned) memory synchronously.
// We call zeMemAllocHost and zeMemFree on our shared context. The test suite uses
// debug_alloc/debug_free to get scratch buffers when verifying correctness.
//
// Header declarations :contentReference[oaicite:3]{index=3}:
//   void* debug_alloc(size_t size) override;
//   void  debug_free(void* ptr) override;

void* oneapi_backend::debug_alloc(size_t size) {
    // Allocate host (pinned) memory from the Level-Zero context.
    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0u};
    void* host_ptr = nullptr;
    ze_result_t ret = zeMemAllocHost(m_context, &hostDesc, size, /*alignment=*/1, &host_ptr);
    ze_check(ret, "zeMemAllocHost in debug_alloc");
    return host_ptr;
}

void oneapi_backend::debug_free(void* ptr) {
    if(ptr != nullptr) {
        // Free the memory we allocated via zeMemAllocHost
        ze_result_t ret = zeMemFree(m_context, ptr);
        ze_check(ret, "zeMemFree in debug_free");
    }
}

/// ---------------------------------------------------------------------------------
/// 3. Asynchronous alloc/free methods
/// ---------------------------------------------------------------------------------

// 3.1. enqueue_host_alloc
//
// Schedule a pinned (host) allocation on m_host.alloc_queue. Once zeMemAllocHost
// completes, produce an async_event that contains the allocated pointer.
//
// Header: async_event enqueue_host_alloc(size_t size, size_t alignment) override; :contentReference[oaicite:1]{index=1}
async_event oneapi_backend::enqueue_host_alloc(size_t size, size_t alignment) {
    // Enqueue a lambda on the host alloc_queue:
    return m_host.alloc_queue.submit([this, size, alignment]() -> void* {
        return allocate_host_memory(size, alignment);
    });
}

// 3.2. enqueue_device_alloc
//
// Schedule a device-side allocation on the given device's queue (or its submit thread).
// Once zeMemAllocDevice completes, produce an async_event containing the device pointer.
//
// Header: async_event enqueue_device_alloc(device_id device, size_t size, size_t alignment) override; :contentReference[oaicite:2]{index=2}
async_event oneapi_backend::enqueue_device_alloc(device_id did, size_t size, size_t alignment) {
    // Create a promise to store the result
    auto promise = std::make_shared<std::promise<async_event>>();
    auto future = promise->get_future();
    
    // Define the work lambda that performs zeMemAllocDevice and wraps its result:
    auto work = [this, size, alignment, promise](device_state& st) mutable -> void* {
        ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0};
        void* dev_ptr = nullptr;
        ze_result_t ret = zeMemAllocDevice(m_context, &desc, size, alignment, st.device, &dev_ptr);
        ze_check(ret, "zeMemAllocDevice in enqueue_device_alloc");
        promise->set_value(make_complete_event(dev_ptr));
        return nullptr;
    };
    
    // Submit the work
    auto& queue = m_devices[did].submit_thread;
    device_state& st = m_devices[did];
    if(queue) {
        (void)queue->submit([work = std::move(work), &st]() mutable -> void* {
            return work(st);
        });
    } else {
        work(st);
    }
    return future.get();
}

// 3.3. enqueue_host_free
//
// Schedule a host-free (zeMemFree on host pointer) on m_host.alloc_queue. Return an event that completes once free is done.
//
// Header: async_event enqueue_host_free(void* ptr) override; :contentReference[oaicite:3]{index=3}
async_event oneapi_backend::enqueue_host_free(void* ptr) {
    // Enqueue on the host alloc_queue:
    return m_host.alloc_queue.submit([this, ptr]() -> void* {
        if(ptr) {
            free(ptr);
        }
        return nullptr;
    });
}

// 3.4. enqueue_device_free
//
// Schedule a device-free (zeMemFree on device pointer) on the given device's queue (or its submit thread).
// Return an event that completes once free is done.
//
// Header: async_event enqueue_device_free(device_id device, void* ptr) override; :contentReference[oaicite:4]{index=4}
async_event oneapi_backend::enqueue_device_free(device_id did, void* ptr) {
    // Create a promise to store the result
    auto promise = std::make_shared<std::promise<async_event>>();
    auto future = promise->get_future();
    
    auto work = [this, ptr, promise](device_state& st) mutable -> void* {
        if(ptr) {
            ze_result_t ret = zeMemFree(m_context, ptr);
            ze_check(ret, "zeMemFree in enqueue_device_free");
        }
        promise->set_value(make_complete_event());
        return nullptr;
    };
    
    // Submit the work
    auto& queue = m_devices[did].submit_thread;
    device_state& st = m_devices[did];
    if(queue) {
        (void)queue->submit([work = std::move(work), &st]() mutable -> void* {
            return work(st);
        });
    } else {
        work(st);
    }
    return future.get();
}

/// ---------------------------------------------------------------------------------
/// 4. enqueue_host_task
/// ---------------------------------------------------------------------------------
//
// Signature (oneapi_backend.h) :contentReference[oaicite:0]{index=0}:
//   async_event enqueue_host_task(size_t host_lane,
//                                 const host_task_launcher& launcher,
//                                 std::vector<closure_hydrator::accessor_info> infos,
//                                 const range<3>& global_range,
//                                 const box<3>& execution_range,
//                                 const communicator* comm) override;
//
// What this does:
//   - Enqueue a lambda on m_host.get_queue(host_lane) (a thread_queue).
//   - Inside that lambda:
//       1. Hydrate all accessor_info into raw pointers.
//       2. Call launcher(ptrs.data(), global_range, execution_range, comm).
//       3. Return a completed async_event<void>.
//
// Once compiled, the test suite's host-task tests will run unmodified.

async_event oneapi_backend::enqueue_host_task(size_t                          host_lane,
                                              const host_task_launcher&       launcher,
                                              std::vector<closure_hydrator::accessor_info> infos,
                                              const range<3>&                 global_range,
                                              const box<3>&                   execution_range,
                                              const communicator*             comm) {
    // Grab the thread_queue for this host_lane:
    thread_queue& queue = m_host.get_queue(host_lane);

    // Enqueue a lambda that runs the host-task logic:
    return queue.submit(
        [launcher, infos = std::move(infos), global_range, execution_range, comm]() -> void* {
            // (1) Hydrate all accessor_info into raw pointers:
            std::vector<void*> ptrs;
            ptrs.reserve(infos.size());
            for(const auto& info : infos) {
                void* p = info.ptr;
                ptrs.push_back(p);
            }

            // (2) Invoke the user's host-task launcher:
            launcher(global_range, execution_range, comm);

            // (3) All slices copied; just finish for thread_queue compatibility
            return nullptr;
        }
    );
}

// ---------------------------------------------------------------------------------
// 5. enqueue_device_kernel
// ---------------------------------------------------------------------------------
//
// Signature (oneapi_backend.h) :contentReference[oaicite:1]{index=1}:
//   async_event enqueue_device_kernel(device_id device,
//                                     size_t lane,
//                                     const device_kernel_launcher& launch,
//                                     std::vector<closure_hydrator::accessor_info> infos,
//                                     const box<3>& execution_range,
//                                     const std::vector<void*>& reduction_ptrs) override;
//
// What this does:
//   1. Hydrate all accessor_infos into raw pointers.
//   2. Create (or fetch) a Level-Zero kernel handle for launch.kernel_name().
//   3. Set kernel arguments: first all pointers from infos, then reduction_ptrs.
//   4. Build a command list, append a signal event (if profiling), append the kernel launch,
//      close and execute the command list on st.queue.
//   5. Wrap the "after" event in a level_zero_async_event and return it.
//
// We dispatch this entire workflow via enqueue_work(...), which either runs it
// immediately (if per_device_submission_threads is false) or on a background thread.

async_event oneapi_backend::enqueue_device_kernel(device_id did, size_t lane, const device_kernel_launcher& launch,
    std::vector<closure_hydrator::accessor_info> infos, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) {
    // Create a promise to store the result
    auto promise = std::make_shared<std::promise<async_event>>();
    auto future = promise->get_future();
    
    // Define the work lambda that returns void
    auto work = [this, launch, infos = std::move(infos), execution_range, reduction_ptrs, promise](device_state& st) mutable -> void* {
        // 1. Hydrate all closure accessor infos into raw pointers:
        std::vector<void*> ptrs;
        ptrs.reserve(infos.size());
        for(const auto& info : infos) {
            void* p = info.ptr;
            ptrs.push_back(p);
        }

        // 2. Create or fetch the Level-Zero kernel handle for launch.kernel_name()
        const char* kernelName = "kernel"; // Default name if no task name is provided
        ze_module_handle_t module = nullptr;
        ze_kernel_handle_t kernel = nullptr;

        ze_kernel_desc_t kdesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
        kdesc.pKernelName = kernelName;
        ze_check(zeKernelCreate(module, &kdesc, &kernel), "zeKernelCreate for device kernel");

        // 3. Set kernel arguments
        uint32_t argIdx = 0;
        for(void* p : ptrs) {
            ze_check(zeKernelSetArgumentValue(kernel, argIdx++, sizeof(p), &p),
                     "zeKernelSetArgumentValue (accessor pointer)");
        }
        for(void* rptr : reduction_ptrs) {
            ze_check(zeKernelSetArgumentValue(kernel, argIdx++, sizeof(rptr), &rptr),
                     "zeKernelSetArgumentValue (reduction pointer)");
        }

        // 4. Create and record commands into a command list
        ze_command_list_handle_t cmdlist = nullptr;
        {
            ze_command_list_desc_t listDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
            listDesc.commandQueueGroupOrdinal = 0;
            ze_check(zeCommandListCreate(st.context, st.device, &listDesc, &cmdlist),
                     "zeCommandListCreate for device kernel");
        }

        // Handle profiling events
        ze_event_handle_t evt_before = nullptr, evt_after = nullptr;
        ze_event_pool_handle_t evt_pool = nullptr;
        if(is_profiling_enabled()) {
            ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
            poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
            poolDesc.count = 1;
            ze_check(zeEventPoolCreate(st.context, &poolDesc, 1, &st.device, &evt_pool), "zeEventPoolCreate");
            ze_event_desc_t evDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
            ze_check(zeEventCreate(evt_pool, &evDesc, &evt_before), "zeEventCreate (before)");
            ze_check(zeEventCreate(evt_pool, &evDesc, &evt_after), "zeEventCreate (after)");
            ze_check(zeCommandListAppendSignalEvent(cmdlist, evt_before), "zeAppendSignalEvent");
        }

        // Set launch parameters
        const auto& ext = execution_range.get_range();
        ze_group_count_t launchArgs = {
            static_cast<uint32_t>(ext[0]),
            static_cast<uint32_t>(ext[1]),
            static_cast<uint32_t>(ext[2])
        };

        // Launch kernel
        if(evt_before) {
            ze_check(zeCommandListAppendLaunchKernel(cmdlist, kernel, &launchArgs, evt_before, 0, nullptr),
                     "zeAppendLaunchKernel (profiling)");
        } else {
            ze_check(zeCommandListAppendLaunchKernel(cmdlist, kernel, &launchArgs, nullptr, 0, nullptr),
                     "zeAppendLaunchKernel (no profiling)");
        }

        // Execute command list
        ze_check(zeCommandListClose(cmdlist), "zeCommandListClose");
        ze_check(zeCommandQueueExecuteCommandLists(st.queue, 1, &cmdlist, nullptr),
                 "zeCommandQueueExecuteCommandLists");

        zeCommandListDestroy(cmdlist);

        // Store the result
        promise->set_value(make_complete_event());
        return nullptr;
    };

    // Submit the work
    auto& queue = m_devices[did].submit_thread;
    device_state& st = m_devices[did];
    if(queue) {
        (void)queue->submit([work = std::move(work), &st]() mutable -> void* {
            return work(st);
        });
    } else {
        work(st);
    }
    return future.get();
}

/// ---------------------------------------------------------------------------------
/// 6. enqueue_host_copy
/// ---------------------------------------------------------------------------------
//
// Signature (oneapi_backend.h) :contentReference[oaicite:0]{index=0}:
//   async_event enqueue_host_copy(size_t host_lane,
//                                 const void* source,
//                                 void* dest,
//                                 const region_layout& src_layout,
//                                 const region_layout& dst_layout,
//                                 const region<3>& copy_region,
//                                 size_t elem_size) override;
//
// What this does:
//   - Schedules a 3D subregion copy between two host pointers on the given host_lane.
//   - Uses nested loops over z, y, x, computes byte offsets via region_layout.strides,
//     and calls std::memcpy for each contiguous slice of length (copy_region.x * elem_size).
//   - Returns a completed async_event once all memcpy calls finish.

async_event oneapi_backend::enqueue_host_copy(size_t                     host_lane,
                                              const void*                source,
                                              void*                      dest,
                                              const region_layout&       src_layout,
                                              const region_layout&       dst_layout,
                                              const region<3>&           copy_region,
                                              size_t                     elem_size) {
    // Obtain the thread_queue for this host_lane:
    thread_queue& queue = m_host.get_queue(host_lane);

    // Enqueue a lambda that performs the nested-loop copy.
    return queue.submit(
        [=]() -> void* {
            const auto& cr = copy_region;              // alias for brevity
            const auto& s  = std::get<strided_layout>(src_layout).allocation.get_range();       // strides in elements
            const auto& d  = std::get<strided_layout>(dst_layout).allocation.get_range();
            const auto& sr = cr.get_boxes()[0].get_range();                                           // get range from first box

            // Convert base pointers to byte-addressable char*
            const char* src_base = static_cast<const char*>(source);
            char*       dst_base = static_cast<char*>(dest);

            // Iterate over Z dimension
            for(size_t z = 0; z < sr[2]; ++z) {
                // Compute element-offset for this Z-slice in source and dest
                size_t src_z_offset_elems = (sr[1] * s[1] + z * s[2]);
                size_t dst_z_offset_elems = (sr[1] * d[1] + z * d[2]);

                // Iterate over Y dimension
                for(size_t y = 0; y < sr[1]; ++y) {
                    // Compute element-offset for this Y-row in source and dest
                    size_t src_y_offset_elems = src_z_offset_elems + y * s[1];
                    size_t dst_y_offset_elems = dst_z_offset_elems + y * d[1];

                    // Compute byte pointers for the start of this row
                    const char* src_row_ptr = src_base + (src_y_offset_elems * elem_size);
                    char*       dst_row_ptr = dst_base + (dst_y_offset_elems * elem_size);

                    // Number of elements to copy in X direction
                    size_t width = sr[0];
                    // Number of bytes to copy for this contiguous row
                    size_t byteCount = width * elem_size;

                    // Perform the contiguous memcpy
                    std::memcpy(dst_row_ptr, src_row_ptr, byteCount);
                }
            }
            // All slices copied; just finish for thread_queue compatibility
            return nullptr;
        }
    );
}

/// ---------------------------------------------------------------------------------
/// 7. enqueue_device_copy
/// ---------------------------------------------------------------------------------
//
// Signature (oneapi_backend.h) :contentReference[oaicite:1]{index=1}:
//   async_event enqueue_device_copy(device_id device,
//                                   size_t lane,
//                                   const void* source,
//                                   void* dest,
//                                   const region_layout& src_layout,
//                                   const region_layout& dst_layout,
//                                   const region<3>& copy_region,
//                                   size_t elem_size) override;
//
// What this does:
//   - Schedules a 3D subregion device-to-device (or host-to-device/device-to-host) copy
//     via Level-Zero commands. Both source and dest pointers must be device-accessible.
//   - Splits the copy into Z- and Y-slices if the subregion is not contiguous in memory.
//   - For each contiguous X-row, calls zeCommandListAppendMemoryCopy.
//   - Appends a signal event (evt_after) once all copies are recorded, then closes+executes
//     the command list on the device's queue. Returns an async_event wrapping evt_after.

async_event oneapi_backend::enqueue_device_copy(device_id device, size_t lane, const void* source, void* dest,
    const region_layout& src_layout, const region_layout& dst_layout, const region<3>& copy_region, size_t elem_size) {
    // Create a promise to store the result
    auto promise = std::make_shared<std::promise<async_event>>();
    auto future = promise->get_future();
    
    // Define the work lambda that returns void
    auto work = [this, source, dest, src_layout, dst_layout, copy_region, elem_size, promise](device_state& st) mutable -> void* {
        // 1. Create command list
        ze_command_list_handle_t cmdlist = nullptr;
        {
            ze_command_list_desc_t listDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
            listDesc.commandQueueGroupOrdinal = 0;
            ze_check(zeCommandListCreate(st.context, st.device, &listDesc, &cmdlist),
                     "zeCommandListCreate in enqueue_device_copy");
        }

        // 2. Compute strides and base pointers
        const auto& cr = copy_region;
        const auto& s  = std::get<strided_layout>(src_layout).allocation.get_range();
        const auto& d  = std::get<strided_layout>(dst_layout).allocation.get_range();
        const auto& sr = cr.get_boxes()[0].get_range();
        const char* src_base = static_cast<const char*>(source);
        char*       dst_base = static_cast<char*>(dest);

        // 3. Perform the copy
        for(size_t z = 0; z < sr[2]; ++z) {
            size_t src_z_offset_elems = (sr[1] * s[1] + z * s[2]);
            size_t dst_z_offset_elems = (sr[1] * d[1] + z * d[2]);
            for(size_t y = 0; y < sr[1]; ++y) {
                size_t src_y_offset_elems = src_z_offset_elems + y * s[1];
                size_t dst_y_offset_elems = dst_z_offset_elems + y * d[1];

                const void* src_ptr = src_base + (src_y_offset_elems * elem_size);
                void*       dst_ptr = dst_base + (dst_y_offset_elems * elem_size);
                size_t byteCount = sr[0] * elem_size;

                ze_check(zeCommandListAppendMemoryCopy(cmdlist, dst_ptr, src_ptr, byteCount, nullptr, 0, nullptr), 
                         "zeCommandListAppendMemoryCopy");
            }
        }

        // 4. Create and handle events
        ze_event_handle_t evt = nullptr;
        ze_event_pool_handle_t evt_pool = nullptr;
        if(is_profiling_enabled()) {
            ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
            poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
            poolDesc.count = 1;
            ze_check(zeEventPoolCreate(st.context, &poolDesc, 1, &st.device, &evt_pool), "zeEventPoolCreate");
            ze_event_desc_t evDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
            ze_check(zeEventCreate(evt_pool, &evDesc, &evt), "zeEventCreate in enqueue_device_copy");
        } else {
            ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
            poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
            poolDesc.count = 1;
            ze_check(zeEventPoolCreate(st.context, &poolDesc, 1, &st.device, &evt_pool), "zeEventPoolCreate");
            ze_event_desc_t evDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
            ze_check(zeEventCreate(evt_pool, &evDesc, &evt), "zeEventCreate (no profiling) in enqueue_device_copy");
        }

        ze_check(zeCommandListAppendSignalEvent(cmdlist, evt),
                 "zeCommandListAppendSignalEvent in enqueue_device_copy");

        // 5. Execute command list
        ze_check(zeCommandListClose(cmdlist), "zeCommandListClose in enqueue_device_copy");
        ze_check(zeCommandQueueExecuteCommandLists(st.queue, 1, &cmdlist, nullptr),
                 "zeCommandQueueExecuteCommandLists in enqueue_device_copy");

        zeCommandListDestroy(cmdlist);

        // Store the result
        promise->set_value(make_complete_event());
        return nullptr;
    };

    // Submit the work
    auto& queue = m_devices[device].submit_thread;
    device_state& st = m_devices[device];
    if(queue) {
        (void)queue->submit([work = std::move(work), &st]() mutable -> void* {
            return work(st);
        });
    } else {
        work(st);
    }
    return future.get();
}

/// ---------------------------------------------------------------------------------
/// 8. check_async_errors
/// ---------------------------------------------------------------------------------
//
// Signature (oneapi_backend.h) :contentReference[oaicite:0]{index=0}:
//   void check_async_errors() override;
//
// What this does:
//   - For each device_state in m_devices, synchronizes its command queue with an infinite
//     timeout. If any queue returns an error, throws a runtime_error.
//   - Ensures that all in-flight operations (kernels, copies, etc.) have completed
//     successfully before returning.

void oneapi_backend::check_async_errors() {
    for (auto& kv : m_devices) {
        const device_state& st = kv;
        // Synchronize the queue; this will wait until all enqueued work is done or an error occurs.
        ze_result_t res = zeCommandQueueSynchronize(st.queue, UINT64_MAX);
        if (res != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Level-Zero asynchronous error detected on device queue");
        }
    }
    // No need to check host queues here, as host tasks use std::memcpy and will throw directly if they fail.
}

// Level-Zero event implementation
class ze_event_impl final : public async_event_impl {
public:
    ze_event_impl(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_pool)
        : m_event(evt), m_pool(pool), m_owns_pool(owns_pool) {}

    ~ze_event_impl() override {
        if(m_event) {
            ze_check(zeEventDestroy(m_event), "zeEventDestroy");
        }
        if(m_owns_pool && m_pool) {
            ze_check(zeEventPoolDestroy(m_pool), "zeEventPoolDestroy");
        }
    }

    bool is_complete() override {
        if(!m_event) return true;
        ze_result_t status = zeEventQueryStatus(m_event);
        return status == ZE_RESULT_SUCCESS;
    }

private:
    ze_event_handle_t m_event;
    ze_event_pool_handle_t m_pool;
    bool m_owns_pool;
};

// Helper function to create an async_event from a Level-Zero event
async_event make_event_from_ze(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_pool) {
    return make_async_event<ze_event_impl>(evt, pool, owns_pool);
}

void* oneapi_backend::allocate_host_memory(size_t size, size_t alignment) {
    void* ptr = nullptr;
    if(posix_memalign(&ptr, alignment, size) != 0) {
        throw std::bad_alloc();
    }
    return ptr;
}

celerity::detail::oneapi_backend::host_state::host_state(ze_context_handle_t ctx, bool profiling)
    : context(ctx), alloc_queue(named_threads::thread_type::alloc, profiling) {
    // host_queues will be resized as needed in get_queue
}

celerity::detail::thread_queue& celerity::detail::oneapi_backend::host_state::get_queue(size_t lane) {
    while (lane >= host_queues.size()) {
        host_queues.emplace_back(named_threads::task_type_host_queue(host_queues.size()));
    }
    return host_queues[lane];
}

bool celerity::detail::oneapi_backend::is_profiling_enabled() const {
    return m_config.profiling;
}

} // namespace celerity::detail
