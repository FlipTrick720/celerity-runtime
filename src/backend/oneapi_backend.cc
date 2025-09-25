// oneapi_backend.cc

#include "backend/oneapi_backend.h"
#include "backend/sycl_backend.h"
#include "closure_hydrator.h"
#include "async_event.h"
#include "thread_queue.h"
#include "system_info.h"

#include <level_zero/ze_api.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <algorithm>
#include <variant>

#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <spdlog/spdlog.h>

namespace celerity::detail {

// --- force logging to TRACE unconditionally ----------------------------------
namespace {
struct _force_trace_log {
  _force_trace_log() {
    // Don’t crash if spdlog isn’t fully initialized yet; just best-effort set.
    try {
      spdlog::set_level(spdlog::level::trace);
      spdlog::info("oneAPI backend: forced spdlog level = TRACE (env not required)");
    } catch (...) {
      // ignore
    }
  }
} _force_trace_log_instance;
} // namespace

// --- tiny logging helper (at start of each function) -----------------------------------------------------
#define OAPI_LOG_ENTER() ::spdlog::debug("oneAPI backend: {}()", __func__)

// Forward decls
class ze_event_impl;
async_event make_event_from_ze_owned(ze_event_handle_t evt, ze_event_pool_handle_t pool);

// L0 error check (avoid fmt on ze_result_t directly)
static inline void ze_check(ze_result_t result, const char* where) {
  if (result != ZE_RESULT_SUCCESS) {
    spdlog::error("Level-Zero error in {}: code={}", where, static_cast<int>(result));
    throw std::runtime_error(std::string("Level-Zero error at ") + where +
                             " (code " + std::to_string(static_cast<int>(result)) + ")");
  }
}

// ----------------------------------------------------------------------------
// oneapi_backend::device_state helpers
// ----------------------------------------------------------------------------
sycl::queue& oneapi_backend::device_state::sycl_queue_for_lane(size_t lane, bool profiling) {
  OAPI_LOG_ENTER();
  while (sycl_lanes.size() <= lane) {
    spdlog::debug("oneAPI backend: creating SYCL queue for lane {} (profiling={})",
                  sycl_lanes.size(), profiling);

    // If profiling is off, discard SYCL per-op events to minimize native resource usage.
    sycl::property_list props = profiling
      ? sycl::property_list{
          sycl::property::queue::in_order{},
          sycl::property::queue::enable_profiling{}
        }
      : sycl::property_list{
          sycl::property::queue::in_order{},
          sycl::ext::oneapi::property::queue::discard_events{}
        };

    // Async handler (logs, never throws silently).
    auto ah = sycl::async_handler{[](sycl::exception_list el){
      for (auto& e : el) {
        try {
          std::rethrow_exception(e);
        } catch (const std::exception& ex) {
          spdlog::error("SYCL async handler: {}", ex.what());
        } catch (...) {
          spdlog::error("SYCL async handler: unknown exception");
        }
      }
    }};

    sycl_lanes.emplace_back(std::make_unique<sycl::queue>(sycl_ctx, sycl_dev, ah, props));
  }
  return *sycl_lanes[lane];
}

// ----------------------------------------------------------------------------
// ctor / dtor
// ----------------------------------------------------------------------------
oneapi_backend::oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                               const configuration& config)
: m_config(config) {
  OAPI_LOG_ENTER();

  // L0 init (idempotent)
  spdlog::debug("oneAPI backend: calling zeInit()");
  ze_check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");

  // Look for Driver that covers all devices
  uint32_t driver_count = 0;
  ze_check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
  std::vector<ze_driver_handle_t> drivers(driver_count);
  ze_check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet(handles)");

  for (auto drv : drivers) {
    uint32_t dev_count = 0;
    ze_check(zeDeviceGet(drv, &dev_count, nullptr), "zeDeviceGet(count)");
    std::vector<ze_device_handle_t> drv_devs(dev_count);
    ze_check(zeDeviceGet(drv, &dev_count, drv_devs.data()), "zeDeviceGet(list)");

    bool all_present = true;
    for (auto ud : devices) {
      if (std::find(drv_devs.begin(), drv_devs.end(), ud) == drv_devs.end()) { all_present = false; break; }
    }
    if (all_present) { m_driver = drv; break; }
  }
  if (!m_driver) {
    spdlog::error("oneAPI backend: No L0 driver covers the selected devices. "
                  "This usually means you’re not running on the Level-Zero plugin.");
    throw std::runtime_error("No Level-Zero driver supports all selected devices");
  }
  spdlog::debug("oneAPI backend: selected L0 driver {}", static_cast<void*>(m_driver));

  // Create L0 context
  ze_context_desc_t ctx_desc{ ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0 };
  ze_check(zeContextCreate(m_driver, &ctx_desc, &m_context), "zeContextCreate");
  spdlog::debug("oneAPI backend: created L0 context {}", static_cast<void*>(m_context));

  // Prepare device states
  m_devices.resize(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    device_state& st = m_devices[i];
    st.device  = devices[i];
    st.context = m_context;

    // Choose a compute-capable queue group
    uint32_t group_count = 0;
    ze_check(zeDeviceGetCommandQueueGroupProperties(st.device, &group_count, nullptr),
             "zeDeviceGetCommandQueueGroupProperties(count)");
    std::vector<ze_command_queue_group_properties_t> groups(group_count);
    for (auto& g : groups) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ze_check(zeDeviceGetCommandQueueGroupProperties(st.device, &group_count, groups.data()),
             "zeDeviceGetCommandQueueGroupProperties(list)");

    uint32_t compute_ordinal = UINT32_MAX;
    for (uint32_t gi = 0; gi < group_count; ++gi) {
      // erste COMPUTE-fähige Gruppe wählen
      if ((groups[gi].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && groups[gi].numQueues > 0) {
        compute_ordinal = gi; break;
      }
    }
    if (compute_ordinal == UINT32_MAX) throw std::runtime_error("No compute queue group found");

    // Queue  anlegen
    ze_command_queue_desc_t qdesc{};
    qdesc.stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qdesc.ordinal  = compute_ordinal;
    qdesc.index    = 0;
    qdesc.mode     = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_check(zeCommandQueueCreate(st.context, st.device, &qdesc, &st.queue), "zeCommandQueueCreate");
    st.queue_ordinal = compute_ordinal;
    spdlog::debug("oneAPI backend: created L0 queue {} (ordinal={}) on device {}",
                  static_cast<void*>(st.queue), st.queue_ordinal, static_cast<void*>(st.device));

    // SYCL Sicht für: device + context from native L0 handles
    st.sycl_dev = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(st.device);

    using ctx_in_t =
      sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context>;
    ctx_in_t ctx_in{ m_context,
                     std::vector<sycl::device>{ st.sycl_dev },
                     sycl::ext::oneapi::level_zero::ownership::keep };
    st.sycl_ctx = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(ctx_in);

    // Optional per-device submitter (not sure)
    if (m_config.per_device_submission_threads) {
      spdlog::debug("oneAPI backend: enabling submit thread for device index {}", i);
      st.submit_thread.emplace(named_threads::task_type_device_submitter(i));
    }
  }

  // Minimal system_info for Celerity needs
  m_system.num_devices = devices.size();
  m_system.devices.resize(devices.size());
  m_system.max_work_group_size = 0;

  m_system.memories.resize(2 + devices.size()); // user + host + N devices
  m_system.memories[user_memory_id].copy_peers.set(user_memory_id);
  m_system.memories[host_memory_id].copy_peers.set(host_memory_id);
  m_system.memories[user_memory_id].copy_peers.set(host_memory_id);
  m_system.memories[host_memory_id].copy_peers.set(user_memory_id);

  for (size_t i = 0; i < devices.size(); ++i) {
    const memory_id mid = first_device_memory_id + i;
    m_system.devices[i].native_memory = mid;
    m_system.memories[mid].copy_peers.set(mid);
    m_system.memories[mid].copy_peers.set(host_memory_id);
    m_system.memories[host_memory_id].copy_peers.set(mid);

    ze_device_compute_properties_t comp{};
    comp.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    ze_check(zeDeviceGetComputeProperties(devices[i], &comp), "zeDeviceGetComputeProperties");
    m_system.max_work_group_size = std::max(m_system.max_work_group_size,
                                            static_cast<size_t>(comp.maxTotalGroupSize));
  }

  m_host = std::make_unique<host_state>(m_context, m_config.profiling);
  spdlog::debug("oneAPI backend: ctor done (devices={}, max_wg={})",
                m_system.num_devices, m_system.max_work_group_size);
}

oneapi_backend::~oneapi_backend() {
  OAPI_LOG_ENTER();
  for (auto& st : m_devices) {
    st.submit_thread.reset(); // Threads beenden
    if (st.queue) {
      zeCommandQueueSynchronize(st.queue, UINT64_MAX); // warten
      zeCommandQueueDestroy(st.queue); // wenn fertig -> destroy
    }
  }
  m_host.reset();
  if (m_context) zeContextDestroy(m_context); // Context frei geben
}

// ----------------------------------------------------------------------------
// system info (Api)
// ----------------------------------------------------------------------------
const system_info& oneapi_backend::get_system_info() const { OAPI_LOG_ENTER(); return m_system; } // read-only
system_info&       oneapi_backend::get_system_info()       { OAPI_LOG_ENTER(); return m_system; } // read and write

// ----------------------------------------------------------------------------
// init (to see if all are runnig or idel)
// ----------------------------------------------------------------------------
void oneapi_backend::init() {
  OAPI_LOG_ENTER();
  for (auto& st : m_devices) {
    ze_result_t r = zeCommandQueueSynchronize(st.queue, 0); // mit L0 testen
    if (r != ZE_RESULT_SUCCESS && r != ZE_RESULT_NOT_READY) { // alles andere -> error
      throw std::runtime_error("Level-Zero queue sync failed in init()");
    }
  }
}

// ----------------------------------------------------------------------------
// memory
// ----------------------------------------------------------------------------
void* oneapi_backend::debug_alloc(size_t size) {
  OAPI_LOG_ENTER();
  ze_device_mem_alloc_desc_t dev_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ze_host_mem_alloc_desc_t   host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void* ptr = nullptr;
  ze_device_handle_t dev = m_devices.empty() ? nullptr : m_devices[0].device;
  ze_check(zeMemAllocShared(m_context, &dev_desc, &host_desc, size, 0, dev, &ptr), "zeMemAllocShared");
  spdlog::debug("oneAPI backend: debug_alloc(size={}) -> {}", size, ptr);
  return ptr;
}

void oneapi_backend::debug_free(void* ptr) {
  OAPI_LOG_ENTER();
  if (ptr) ze_check(zeMemFree(m_context, ptr), "zeMemFree(debug_free)");
}

async_event oneapi_backend::enqueue_host_alloc(size_t size, size_t alignment) {
  OAPI_LOG_ENTER();
  return m_host->alloc_queue.submit([this, size, alignment]() -> void* {
    spdlog::debug("oneAPI backend: host_alloc(size={}, align={})", size, alignment);
    ze_host_mem_alloc_desc_t host_desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* p = nullptr;
    ze_check(zeMemAllocHost(m_context, &host_desc, size, alignment, &p), "zeMemAllocHost");
    return p;
  });
}

async_event oneapi_backend::enqueue_host_free(void* ptr) {
  OAPI_LOG_ENTER();
  return m_host->alloc_queue.submit([this, ptr]() -> void* {
    spdlog::debug("oneAPI backend: host_free(ptr={})", ptr);
    if (ptr) ze_check(zeMemFree(m_context, ptr), "zeMemFree(host_free)");
    return nullptr;
  });
}

async_event oneapi_backend::enqueue_device_alloc(device_id did, size_t size, size_t alignment) {
  OAPI_LOG_ENTER();
  return enqueue_work(did, 0, [this, size, alignment, did](device_state& st) -> async_event {
    spdlog::debug("oneAPI backend: device_alloc(dev={}, size={}, align={})", did.value, size, alignment);
    ze_device_mem_alloc_desc_t dev_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* p = nullptr;
    ze_check(zeMemAllocDevice(st.context, &dev_desc, size, alignment, st.device, &p), "zeMemAllocDevice");
    return make_complete_event(p);
  });
}

async_event oneapi_backend::enqueue_device_free(device_id did, void* ptr) {
  OAPI_LOG_ENTER();
  return enqueue_work(did, 0, [this, ptr, did](device_state& st) -> async_event {
    spdlog::debug("oneAPI backend: device_free(dev={}, ptr={})", did.value, ptr);
    if (ptr) ze_check(zeMemFree(st.context, ptr), "zeMemFree(device_free)");
    return make_complete_event();
  });
}

// ----------------------------------------------------------------------------
// tasks
// ----------------------------------------------------------------------------
async_event oneapi_backend::enqueue_host_task( // cpu thread
  size_t host_lane, const host_task_launcher& launcher,
  std::vector<closure_hydrator::accessor_info> accessor_infos,
  const range<3>& global_range, const box<3>& execution_range,
  const communicator* collective_comm)
{
  OAPI_LOG_ENTER();
  auto& hydrator = closure_hydrator::get_instance();
  hydrator.arm(target::host_task, std::move(accessor_infos)); // arm macht Buffer und Subrange
  auto launch_hydrated = hydrator.hydrate<target::host_task>(launcher); // hydrate makes callable

  thread_queue& q = m_host->get_queue(host_lane); // Holt Host-Worker-Queue

  return q.submit([=, launch_hydrated = std::move(launch_hydrated)]() { // gibt Arbeit asynchron an Host-Queue
    spdlog::debug("oneAPI backend: running host_task on lane {}", host_lane);
    launch_hydrated(global_range, execution_range, collective_comm);
    return nullptr;
  });
}

async_event oneapi_backend::enqueue_device_kernel( // gpu queue
  device_id did, size_t lane, const device_kernel_launcher& launch,
  std::vector<closure_hydrator::accessor_info> accessor_infos,
  const box<3>& execution_range, const std::vector<void*>& reduction_ptrs)
{
  OAPI_LOG_ENTER();
  return enqueue_work(did, lane,
      [this, did, lane, launch, execution_range, reduction_ptrs,
       acc_infos = std::move(accessor_infos)](device_state& st) mutable -> async_event { // auf device-State
    spdlog::debug("oneAPI backend: device_kernel(dev={}, lane={})", did.value, lane);
    auto& q = st.sycl_queue_for_lane(lane, m_config.profiling); // get queue

    sycl::event ev = q.submit([&, acc_infos = std::move(acc_infos)](sycl::handler& cgh) mutable {
      auto& hydrator = closure_hydrator::get_instance();
      hydrator.arm(target::device, std::move(acc_infos)); // arm macht Buffer und Subrange
      auto launch_h = hydrator.hydrate<target::device>(cgh, launch);
      launch_h(cgh, execution_range, reduction_ptrs);
    });

    celerity::detail::sycl_backend_detail::flush(q); // to get to nativ runtime
    using sy_ev = celerity::detail::sycl_backend_detail::sycl_event;
    return make_async_event<sy_ev>(std::move(ev), m_config.profiling);
  });
}

// ----------------------------------------------------------------------------
// copies
// ----------------------------------------------------------------------------
// for host to host on cpu with up to 3D line by line copy
async_event oneapi_backend::enqueue_host_copy(size_t host_lane, // welcher Host-Thread-Lane
                                              const void* source,
                                              void* dest,
                                              const region_layout& src_layout, // orga of src_mem
                                              const region_layout& dst_layout, // orga of dst_mem
                                              const region<3>& copy_region, // offset und "size"
                                              size_t elem_size) {
  OAPI_LOG_ENTER();
  thread_queue& tq = m_host->get_queue(host_lane); // get right queue
  return tq.submit([=]() -> void* { // asynchron in queue
    spdlog::debug("oneAPI backend: host_copy(lane={}, bytes≈{})",
                  host_lane, copy_region.get_boxes()[0].get_area() * elem_size);
    const auto box  = copy_region.get_boxes()[0]; // extract info from box
    const auto off  = box.get_min(); // Start-Koordinate
    const auto ext  = box.get_range(); // Breite/Höhe/Tiefe = Spalten/Zeilen/Ebenen
    const auto s    = std::get<strided_layout>(src_layout).allocation.get_range(); // in sx, sy, sz // sollte strided sein
    const auto d    = std::get<strided_layout>(dst_layout).allocation.get_range(); // in dx, dy, dz

    const auto src = static_cast<const char*>(source); // Byte-Pointer cast
    auto       dst = static_cast<char*>(dest); // Byte-Pointer cast
    const size_t sx = s[0], sy = s[1]; // for Zeilenadressierung
    const size_t dx = d[0], dy = d[1]; // for Zeilenadressierung

    // Row-Major-Flattening => index = ((z * height) + y) * width + x
    // size_t so shouldn't overflow
    for (size_t z = 0; z < ext[2]; ++z) {
      for (size_t y = 0; y < ext[1]; ++y) {
        size_t src_off = ((off[2]+z) * sy + (off[1]+y)) * sx + off[0];
        size_t dst_off = ((off[2]+z) * dy + (off[1]+y)) * dx + off[0];
        std::memcpy(dst + dst_off*elem_size, src + src_off*elem_size, ext[0]*elem_size); // (ext[0]*elem_size) macht Byte pro Zeile
      } // Probleme wenn Overlap von src und dst
    }
    return nullptr;
  });
}

// for gpu for linearisiert or 3D/strided
async_event oneapi_backend::enqueue_device_copy(device_id did, //which device by id
                                                size_t lane,
                                                const void* source,
                                                void* dest,
                                                const region_layout& src_layout, // orga of src_mem
                                                const region_layout& dst_layout, // orga of dst_mem
                                                const region<3>& copy_region, // offset und "size"
                                                size_t elem_size) {
  OAPI_LOG_ENTER();
  return enqueue_work(did, lane, [this, did, lane, source, dest,
                                  src_layout, dst_layout, copy_region, elem_size](device_state& st) -> async_event {
    const size_t bytes = copy_region.get_boxes()[0].get_area() * elem_size; // for Logs

    // Linearized regions -> straight memcpy
    if (std::holds_alternative<linearized_layout>(src_layout) &&
        std::holds_alternative<linearized_layout>(dst_layout)) { // beide Seiten sind linear
        const auto box = copy_region.get_boxes()[0];
        const size_t n = box.get_area() * elem_size;
        const size_t src_off = std::get<linearized_layout>(src_layout).offset_bytes;
        const size_t dst_off = std::get<linearized_layout>(dst_layout).offset_bytes;
        const void* src = static_cast<const char*>(source) + src_off;
        void*       dst = static_cast<char*>(dest)         + dst_off;

        if (!m_config.profiling) {
          spdlog::trace("oneAPI backend: device_copy linearized (blocking) n={} bytes", n);
          ze_command_list_handle_t cl = nullptr;
          ze_command_list_desc_t cl_desc{ ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC };
          cl_desc.commandQueueGroupOrdinal = st.queue_ordinal;
          ze_check(zeCommandListCreate(st.context, st.device, &cl_desc, &cl), "zeCommandListCreate"); // make Command-List
          ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, n, /*event*/nullptr, 0, nullptr), // do Copy
                   "zeCommandListAppendMemoryCopy");
          ze_check(zeCommandListClose(cl), "zeCommandListClose"); // schließen
          ze_check(zeCommandQueueExecuteCommandLists(st.queue, 1, &cl, nullptr), // submitten
                   "zeCommandQueueExecuteCommandLists");
          zeCommandListDestroy(cl); // destroy
          ze_check(zeCommandQueueSynchronize(st.queue, UINT64_MAX), "zeCommandQueueSynchronize"); // warten bis fertig
          return make_complete_event();
        } else {
          spdlog::trace("oneAPI backend: device_copy linearized (SYCL/profiling) n={} bytes", n);
          auto& q = st.sycl_queue_for_lane(lane, true); // profiling = true
          sycl::event ev = q.memcpy(dst, src, n);
          celerity::detail::sycl_backend_detail::flush(q); // to get to nativ runtime
          using sy_ev = celerity::detail::sycl_backend_detail::sycl_event;
          return make_async_event<sy_ev>(std::move(ev), true); // profiling = true
        }
    }

    // else: for 3D region copy
    spdlog::debug("oneAPI backend: device_copy(dev={}, lane={}): 3D copy ({} bytes approx, profiling={})",
                  did.value, lane, bytes, m_config.profiling);

    // If profiling is off, use a blocking L0 submission with no events to
    // avoid per-copy event pool pressure
    if (!m_config.profiling) {
      ze_command_list_handle_t cl = nullptr;
      ze_command_list_desc_t cl_desc{ ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC };
      cl_desc.commandQueueGroupOrdinal = st.queue_ordinal;
      ze_check(zeCommandListCreate(st.context, st.device, &cl_desc, &cl), "zeCommandListCreate");

      const auto box = copy_region.get_boxes()[0]; // extract info from box
      const auto off = box.get_min(); // Start-Koordinate
      const auto ext = box.get_range(); // Breite/Höhe/Tiefe = Spalten/Zeilen/Ebenen

      ze_copy_region_t src_region{
        static_cast<uint32_t>(off[0] * elem_size),
        static_cast<uint32_t>(off[1]),
        static_cast<uint32_t>(off[2]),
        static_cast<uint32_t>(ext[0] * elem_size),
        static_cast<uint32_t>(ext[1]),
        static_cast<uint32_t>(ext[2])
      };
      ze_copy_region_t dst_region = src_region;

      const auto s = std::get<strided_layout>(src_layout).allocation.get_range(); // in sx, sy, sz // sollte strided sein
      const auto d = std::get<strided_layout>(dst_layout).allocation.get_range(); // in dx, dy, dz

      ze_check(zeCommandListAppendMemoryCopyRegion(
        cl,
        dest, &dst_region, static_cast<uint32_t>(d[0] * elem_size), static_cast<uint32_t>(d[0] * d[1] * elem_size),
        source, &src_region, static_cast<uint32_t>(s[0] * elem_size), static_cast<uint32_t>(s[0] * s[1] * elem_size),
        /*event*/nullptr, 0, nullptr), "zeCommandListAppendMemoryCopyRegion(no-event)");

      ze_check(zeCommandListClose(cl), "zeCommandListClose"); // close
      ze_check(zeCommandQueueExecuteCommandLists(st.queue, 1, &cl, nullptr), "zeCommandQueueExecuteCommandLists"); // execute
      zeCommandListDestroy(cl); // destroy
      ze_check(zeCommandQueueSynchronize(st.queue, UINT64_MAX), "zeCommandQueueSynchronize"); // synchronize
      return make_complete_event();
    }

    // else: Profiling ON path: create a tiny host-visible pool for exactly one event (kept owned by async_event)
    ze_command_list_handle_t cl = nullptr;
    ze_command_list_desc_t cl_desc{ ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC };
    cl_desc.commandQueueGroupOrdinal = st.queue_ordinal;
    ze_check(zeCommandListCreate(st.context, st.device, &cl_desc, &cl), "zeCommandListCreate");

    ze_event_pool_handle_t pool = nullptr;
    ze_event_handle_t      evt  = nullptr;

    //create Pool
    ze_event_pool_desc_t pool_desc{};
    pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pool_desc.count = 1;
    ze_check(zeEventPoolCreate(st.context, &pool_desc, 1, &st.device, &pool), "zeEventPoolCreate");

    // create Event
    ze_event_desc_t evt_desc{};
    evt_desc.stype  = ZE_STRUCTURE_TYPE_EVENT_DESC;
    evt_desc.index  = 0;
    evt_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    evt_desc.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
    ze_check(zeEventCreate(pool, &evt_desc, &evt), "zeEventCreate");

    const auto box = copy_region.get_boxes()[0]; // extract info from box
    const auto off = box.get_min();  // Start-Koordinate
    const auto ext = box.get_range(); // Breite/Höhe/Tiefe = Spalten/Zeilen/Ebenen

    ze_copy_region_t src_region{
      static_cast<uint32_t>(off[0] * elem_size),
      static_cast<uint32_t>(off[1]),
      static_cast<uint32_t>(off[2]),
      static_cast<uint32_t>(ext[0] * elem_size),
      static_cast<uint32_t>(ext[1]),
      static_cast<uint32_t>(ext[2])
    };
    ze_copy_region_t dst_region = src_region;

    const auto s = std::get<strided_layout>(src_layout).allocation.get_range(); // in sx, sy, sz // sollte strided sein
    const auto d = std::get<strided_layout>(dst_layout).allocation.get_range(); // in dx, dy, dz

    ze_check(zeCommandListAppendMemoryCopyRegion(
      cl,
      dest, &dst_region, static_cast<uint32_t>(d[0] * elem_size), static_cast<uint32_t>(d[0] * d[1] * elem_size),
      source, &src_region, static_cast<uint32_t>(s[0] * elem_size), static_cast<uint32_t>(s[0] * s[1] * elem_size),
      evt, 0, nullptr), "zeCommandListAppendMemoryCopyRegion");

    ze_check(zeCommandListClose(cl), "zeCommandListClose"); // close
    ze_check(zeCommandQueueExecuteCommandLists(st.queue, 1, &cl, nullptr), "zeCommandQueueExecuteCommandLists"); // execute
    zeCommandListDestroy(cl); // destroy

    return make_event_from_ze_owned(evt, pool); // fertiges Event
  });
}

// ----------------------------------------------------------------------------
// async error poll
// ----------------------------------------------------------------------------
void oneapi_backend::check_async_errors() {
  OAPI_LOG_ENTER();
  for (auto& st : m_devices) {
    for (auto& qp : st.sycl_lanes) if (qp) qp->wait_and_throw(); // check with sycl
    ze_result_t r = zeCommandQueueSynchronize(st.queue, 0); // check with L0
    if (r != ZE_RESULT_SUCCESS && r != ZE_RESULT_NOT_READY) { // alles andere -> error
      throw std::runtime_error("Level-Zero asynchronous error detected");
    }
  }
}

// ----------------------------------------------------------------------------
// Event-Implementierung (wrappers)
// ----------------------------------------------------------------------------
class ze_event_impl final : public async_event_impl {
public:
  ze_event_impl(ze_event_handle_t evt, ze_event_pool_handle_t pool, bool owns_evt, bool owns_pool)
  : m_event(evt), m_pool(pool), m_owns_evt(owns_evt), m_owns_pool(owns_pool) { // ctor 2 mode: Own/Responsible
    spdlog::debug("oneAPI backend: ze_event_impl(evt={}, pool={}, owns_evt={}, owns_pool={})",
                  static_cast<void*>(evt), static_cast<void*>(pool), owns_evt, owns_pool);
  }

  ~ze_event_impl() override { // dtor
    spdlog::debug("oneAPI backend: ~ze_event_impl(evt={}, owns_evt={}, owns_pool={})",
                  static_cast<void*>(m_event), m_owns_evt, m_owns_pool);
    if (m_owns_evt && m_event) {
      zeEventHostSynchronize(m_event, UINT64_MAX); // warten
      zeEventDestroy(m_event); // destroy
    }
    if (m_owns_pool && m_pool) zeEventPoolDestroy(m_pool); // pool destroy
  }

  bool is_complete() override { // pollt das Event -> true/false
    ze_result_t status = zeEventQueryStatus(m_event);
    return status == ZE_RESULT_SUCCESS;
  }

private: // for Members
  ze_event_handle_t m_event{};
  ze_event_pool_handle_t m_pool{};
  bool m_owns_evt{};
  bool m_owns_pool{};
};

// -----
// Factory-Helpers
// -----
async_event make_event_from_native_sycl(ze_event_handle_t evt) {
  OAPI_LOG_ENTER();
  return make_async_event<ze_event_impl>(evt, /*pool*/nullptr, /*owns_evt*/false, /*owns_pool*/false);
}

async_event make_event_from_ze_owned(ze_event_handle_t evt, ze_event_pool_handle_t pool) {
  OAPI_LOG_ENTER();
  return make_async_event<ze_event_impl>(evt, pool, /*owns_evt*/true, /*owns_pool*/true);
}

// ----------------------------------------------------------------------------
// host_state(CPU)
// ----------------------------------------------------------------------------
oneapi_backend::host_state::host_state(ze_context_handle_t ctx, bool profiling)
: context(ctx), profiling(profiling), alloc_queue(named_threads::thread_type::alloc, profiling) { // ctor
  spdlog::debug("oneAPI backend: host_state(ctx={}, profiling={})", static_cast<void*>(ctx), profiling);
}

thread_queue& oneapi_backend::host_state::get_queue(size_t lane) { // queue getter //glaub nicht thread save
  OAPI_LOG_ENTER();
  while (lane >= host_queues.size()) {
    spdlog::debug("oneAPI backend: creating host queue for lane {}", host_queues.size());
    host_queues.emplace_back(named_threads::task_type_host_queue(host_queues.size()), profiling);
  }
  return host_queues[lane]; // host queue fur lane
}

// ----------------------------------------------------------------------------
// common work submit
// ----------------------------------------------------------------------------
async_event oneapi_backend::enqueue_work(device_id did, // which device by id
                                         size_t lane,
                                         std::function<async_event(device_state&)> work) {
  OAPI_LOG_ENTER();
  const size_t idx = static_cast<size_t>(did); // need an index
  if (idx >= m_devices.size()) throw std::runtime_error("enqueue_work: invalid device_id");
  device_state& st = m_devices[idx]; // associated device
  spdlog::debug("oneAPI backend: dispatching work(dev={}, lane={})", did.value, lane);
  return work(st); // meist exec Lambda mit device_state
}

} // namespace celerity::detail
