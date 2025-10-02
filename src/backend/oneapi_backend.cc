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

// --- tiny logging helper (at start of each function) -----------------------------------------------------
#define OAPI_LOG_ENTER() ::spdlog::debug("oneAPI backend: {}()", __func__)

// Forward decls
class ze_event_impl;
async_event make_event_from_ze_owned(ze_event_handle_t evt, ze_event_pool_handle_t pool);

// L0 error check (avoid fmt on ze_result_t directly)
static inline void ze_check(ze_result_t result, const char* where) {
  if (result != ZE_RESULT_SUCCESS) {
    spdlog::error("Level-Zero error in {}: code={}", where, static_cast<int>(result));
    throw std::runtime_error(std::string("Level-Zero error at ") + where + " (code " + std::to_string(static_cast<int>(result)) + ")");
  }
}

// ----------------------------------------------------------------------------
// oneapi_backend::device_state helpers
// ----------------------------------------------------------------------------
sycl::queue& oneapi_backend::device_state::sycl_queue_for_lane(size_t lane, bool profiling) {
  OAPI_LOG_ENTER();
  while (sycl_lanes.size() <= lane) {
    spdlog::debug("oneAPI backend: creating SYCL queue for lane {} (profiling={})", sycl_lanes.size(), profiling);

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
oneapi_backend::oneapi_backend(const std::vector<ze_device_handle_t>& devices, const configuration& config)
: m_config(config) {
  OAPI_LOG_ENTER();

  // L0 init (idempotent)
  spdlog::debug("oneAPI backend: calling zeInit()");
  ze_check(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit");

  // Find a driver that covers all provided devices
  uint32_t driver_count = 0;
  ze_check(zeDriverGet(&driver_count, nullptr), "zeDriverGet(count)");
  if(driver_count == 0) {
    throw std::runtime_error("No Level-Zero drivers found");
  }
  std::vector<ze_driver_handle_t> drivers(driver_count);
  ze_check(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet(handles)");

  for (auto drv : drivers) {
    uint32_t dev_count = 0;
    ze_check(zeDeviceGet(drv, &dev_count, nullptr), "zeDeviceGet(count)");
    std::vector<ze_device_handle_t> drv_devs(dev_count);
    ze_check(zeDeviceGet(drv, &dev_count, drv_devs.data()), "zeDeviceGet(list)");

    bool all_present = true;
    for (auto ud : devices) {
      if (std::find(drv_devs.begin(), drv_devs.end(), ud) == drv_devs.end()) {
        all_present = false; break;
      }
    }
    if (all_present) { m_driver = drv; break; }
  }
  if (!m_driver) {
    spdlog::error("oneAPI backend: No L0 driver covers the selected devices "
                  "(are we on the Level-Zero plugin?).");
    throw std::runtime_error("No Level-Zero driver supports all selected devices");
  }
  spdlog::debug("oneAPI backend: selected L0 driver {}", static_cast<void*>(m_driver));

  // Create one shared L0 context
  ze_context_desc_t ctx_desc{ ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0 };
  ze_check(zeContextCreate(m_driver, &ctx_desc, &m_context), "zeContextCreate");
  spdlog::debug("oneAPI backend: created L0 context {}", static_cast<void*>(m_context));

  // Prepare device states
  m_devices.resize(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    device_state& st = m_devices[i];
    st.device  = devices[i];
    st.context = m_context;

    // --- find Level Zero "compute" queue group ordinal for this device ---
    uint32_t compute_ordinal = UINT32_MAX;
    {
      uint32_t num_groups = 0;
      ze_check(zeDeviceGetCommandQueueGroupProperties(st.device, &num_groups, nullptr),
               "zeDeviceGetCommandQueueGroupProperties(count)");
      std::vector<ze_command_queue_group_properties_t> props(num_groups);
      for (auto &p : props) p = ze_command_queue_group_properties_t{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES};
      ze_check(zeDeviceGetCommandQueueGroupProperties(st.device, &num_groups, props.data()),
               "zeDeviceGetCommandQueueGroupProperties(props)");
      for (uint32_t g = 0; g < num_groups; ++g) {
        if ((props[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && props[g].numQueues > 0) {
          compute_ordinal = g;
          break;
        }
      }
    }
    if (compute_ordinal == UINT32_MAX) {
      throw std::runtime_error("No compute queue group found");
    }
    st.compute_ordinal = compute_ordinal;
    spdlog::debug("oneAPI backend: using compute queue ordinal {} on device {}",
                  st.compute_ordinal, static_cast<void*>(st.device));

    // --- SYCL interop objects from native L0 handles (ownership kept by us) ---
    // device
    st.sycl_dev = sycl::make_device<sycl::backend::ext_oneapi_level_zero>(st.device);

    // context
    using ctx_in_t =
      sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context>;
    ctx_in_t ctx_in{ m_context,
                     std::vector<sycl::device>{ st.sycl_dev },
                     sycl::ext::oneapi::level_zero::ownership::keep };
    st.sycl_ctx = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(ctx_in);

    // Optional per-device submitter thread
    if (m_config.per_device_submission_threads) {
      spdlog::debug("oneAPI backend: enabling submit thread for device index {}", i);
      st.submit_thread.emplace(named_threads::task_type_device_submitter(i));
    }
  }

  // Minimal system_info for Celerity
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

    ze_device_compute_properties_t comp{ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES};
    ze_check(zeDeviceGetComputeProperties(devices[i], &comp), "zeDeviceGetComputeProperties");
    m_system.max_work_group_size =
        std::max(m_system.max_work_group_size, static_cast<size_t>(comp.maxTotalGroupSize));
  }

  m_host = std::make_unique<host_state>(m_context, m_config.profiling);
  spdlog::debug("oneAPI backend: ctor done (devices={}, max_wg={})",
                m_system.num_devices, m_system.max_work_group_size);
}

oneapi_backend::~oneapi_backend() {
  OAPI_LOG_ENTER();
  for (auto& st : m_devices) {
    st.submit_thread.reset(); // Threads beenden
  }
  m_host.reset();
  if (m_context) zeContextDestroy(m_context); // Context frei geben
}

// ----------------------------------------------------------------------------
// system info (Api)
// ----------------------------------------------------------------------------
const system_info& oneapi_backend::get_system_info() const {
  OAPI_LOG_ENTER();
  return m_system;
} // read-only
system_info& oneapi_backend::get_system_info() {
  OAPI_LOG_ENTER();
  return m_system;
} // read and write

// ----------------------------------------------------------------------------
// init (to see if all are running or idle)
// ----------------------------------------------------------------------------
void oneapi_backend::init() {
  OAPI_LOG_ENTER();
  // Nichts zu tun: Wir nutzen nur SYCL-Queues
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

  return q.submit([=, launch_hydrated = std::move(launch_hydrated)]() {
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
       acc_infos = std::move(accessor_infos)](device_state& st) mutable -> async_event {
    spdlog::debug("oneAPI backend: device_kernel(dev={}, lane={})", did.value, lane);
    auto& q = st.sycl_queue_for_lane(lane, m_config.profiling); // get queue

    sycl::event ev = q.submit([&, acc_infos = std::move(acc_infos)](sycl::handler& cgh) mutable {
      auto& hydrator = closure_hydrator::get_instance();
      hydrator.arm(target::device, std::move(acc_infos)); // arm macht Buffer und Subrange
      auto launch_h = hydrator.hydrate<target::device>(cgh, launch);
      launch_h(cgh, execution_range, reduction_ptrs);
    });

    celerity::detail::sycl_backend_detail::flush(q); // to get to native runtime
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
    spdlog::debug("oneAPI backend: host_copy(lane={}, bytes≈{})", host_lane, copy_region.get_boxes()[0].get_area() * elem_size);
    const auto box  = copy_region.get_boxes()[0]; // extract info from box
    const auto off  = box.get_min(); // Start-Koordinate
    const auto ext  = box.get_range(); // Breite/Höhe/Tiefe = Spalten/Zeilen/Ebenen
    const auto s    = 
std::get<strided_layout>(src_layout).allocation.get_range(); // in sx, sy, sz // sollte strided sein
    const auto d    = 
std::get<strided_layout>(dst_layout).allocation.get_range(); // in dx, dy, dz

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

async_event oneapi_backend::enqueue_device_copy(device_id did,
                                                size_t lane,
                                                const void* source,
                                                void* dest,
                                                const region_layout& src_layout,
                                                const region_layout& dst_layout,
                                                const region<3>& copy_region,
                                                size_t elem_size) {
  OAPI_LOG_ENTER();
  return enqueue_work(did, lane, [this, did, lane, source, dest,
                                  src_layout, dst_layout, copy_region, elem_size](device_state& st) -> async_event {
    auto& q = st.sycl_queue_for_lane(lane, m_config.profiling);

    // --------- Fast path: linearized -> linearized ----------
    if (std::holds_alternative<linearized_layout>(src_layout) &&
        std::holds_alternative<linearized_layout>(dst_layout)) {

      const auto box = copy_region.get_boxes()[0];
      const size_t n_bytes = box.get_area() * elem_size;

      const size_t src_off = std::get<linearized_layout>(src_layout).offset_bytes;
      const size_t dst_off = std::get<linearized_layout>(dst_layout).offset_bytes;

      const void* src_ptr = static_cast<const char*>(source) + src_off;
      void*       dst_ptr = static_cast<char*>(dest)         + dst_off;

      spdlog::trace("oneAPI backend: device_copy linearized n={} bytes", n_bytes);
      sycl::event ev = q.memcpy(dst_ptr, src_ptr, n_bytes);
      celerity::detail::sycl_backend_detail::flush(q);
      using sy_ev = celerity::detail::sycl_backend_detail::sycl_event;
      return make_async_event<sy_ev>(std::move(ev), m_config.profiling);
    }

    // --------- General path: 3D/strided/mixed ----------
    const auto box = copy_region.get_boxes()[0];
    const auto off = box.get_min();
    const auto ext = box.get_range();

    const auto get_allocation_xyz = [&](const region_layout& L) -> std::array<size_t,3> {
      if (auto s = std::get_if<strided_layout>(&L)) {
        const auto R = s->allocation.get_range();
        return {R[0], R[1], R[2]};
      } else {
        return {ext[0], 1, 1};
      }
    };

    const auto S = get_allocation_xyz(src_layout);
    const auto D = get_allocation_xyz(dst_layout);

    // Capture only POD types for lambda
    const uint32_t ordinal = st.compute_ordinal;
    ze_device_handle_t device = st.device;
    ze_context_handle_t context = st.context;

    // Prepare event pool
    ze_event_pool_handle_t pool = nullptr;
    ze_event_handle_t      evt  = nullptr;
    {
      ze_event_pool_desc_t pool_desc{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
      pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
      pool_desc.count = 1;
      ze_check(zeEventPoolCreate(context, &pool_desc, 1, &device, &pool), "zeEventPoolCreate");

      ze_event_desc_t evt_desc{ZE_STRUCTURE_TYPE_EVENT_DESC};
      evt_desc.index  = 0;
      evt_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
      evt_desc.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
      ze_check(zeEventCreate(pool, &evt_desc, &evt), "zeEventCreate");
    }

    // Submit via SYCL queue to maintain ordering
    sycl::event submit = q.submit([source, dest, off, ext, S, D, elem_size, 
                                    ordinal, device, context, evt, &q](sycl::handler& cgh) {
      cgh.host_task([source, dest, off, ext, S, D, elem_size, 
                     ordinal, device, context, evt, &q](sycl::interop_handle ih) {
        // Try to get native queue - may fail on some SYCL implementations
        ze_command_list_handle_t imm_cl = nullptr;
        ze_command_queue_handle_t ze_q  = nullptr;
                    
        try {
          // trieing SYCL free-function instead of interop_handle (oneAPI 2025.0 bug workaround)
          // This preserves intended functionality: returns either queue or immediate CL.
          auto native_q = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);

          // Handle variant return type from get_native
          if (std::holds_alternative<ze_command_list_handle_t>(native_q)) {
            imm_cl = std::get<ze_command_list_handle_t>(native_q);
          } else if (std::holds_alternative<ze_command_queue_handle_t>(native_q)) {
            ze_q = std::get<ze_command_queue_handle_t>(native_q);
          }
        } catch (...) {
          // Fallback: get_native_queue not supported, extract from context
          // This is a workaround for buggy SYCL implementations
          spdlog::warn("oneAPI backend: get_native_queue failed, using fallback");
        }
      
        // Setup copy regions
        ze_copy_region_t src_region{
          static_cast<uint32_t>(off[0] * elem_size),
          static_cast<uint32_t>(off[1]),
          static_cast<uint32_t>(off[2]),
          static_cast<uint32_t>(ext[0] * elem_size),
          static_cast<uint32_t>(ext[1]),
          static_cast<uint32_t>(ext[2])
        };
        ze_copy_region_t dst_region = src_region;
      
        const uint32_t src_pitch_y = static_cast<uint32_t>(S[0] * elem_size);
        const uint32_t src_pitch_z = static_cast<uint32_t>(S[0] * S[1] * elem_size);
        const uint32_t dst_pitch_y = static_cast<uint32_t>(D[0] * elem_size);
        const uint32_t dst_pitch_z = static_cast<uint32_t>(D[0] * D[1] * elem_size);
      
        if (imm_cl) {
          // Immediate command list path
          ze_check(zeCommandListAppendMemoryCopyRegion(
                     imm_cl,
                     dest, &dst_region, dst_pitch_y, dst_pitch_z,
                     source, &src_region, src_pitch_y, src_pitch_z,
                     evt, 0, nullptr),
                   "zeCommandListAppendMemoryCopyRegion(immediate)");
        } else {
          // Regular queue path OR fallback: create our own command list
          if (!ze_q) {
            // Fallback: create a temporary queue
            ze_command_queue_desc_t q_desc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
            q_desc.ordinal = ordinal;
            q_desc.index = 0;
            q_desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
            ze_check(zeCommandQueueCreate(context, device, &q_desc, &ze_q),
                     "zeCommandQueueCreate(fallback)");
          }
        
          ze_command_list_desc_t cl_desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
          cl_desc.commandQueueGroupOrdinal = ordinal;
        
          ze_command_list_handle_t cl = nullptr;
          ze_check(zeCommandListCreate(context, device, &cl_desc, &cl),
                   "zeCommandListCreate");
        
          ze_check(zeCommandListAppendMemoryCopyRegion(
                     cl,
                     dest, &dst_region, dst_pitch_y, dst_pitch_z,
                     source, &src_region, src_pitch_y, src_pitch_z,
                     evt, 0, nullptr),
                   "zeCommandListAppendMemoryCopyRegion");
          
          ze_check(zeCommandListClose(cl), "zeCommandListClose");
          ze_check(zeCommandQueueExecuteCommandLists(ze_q, 1, &cl, nullptr),
                   "zeCommandQueueExecuteCommandLists");
          zeCommandListDestroy(cl);
          
          // Don't destroy ze_q - we don't know if it's from SYCL or our fallback
        }
      });
    });

    celerity::detail::sycl_backend_detail::flush(q);
    return make_event_from_ze_owned(evt, pool);
  });
}

// ----------------------------------------------------------------------------
// async error poll
// ----------------------------------------------------------------------------
void oneapi_backend::check_async_errors() {
  OAPI_LOG_ENTER();
  for (auto& st : m_devices) {
    // Nicht blockierend: liefert aufgelaufene async exceptions an den async_handler
    for (auto& qp : st.sycl_lanes) {
      if (qp) qp->throw_asynchronous();
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

private:
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
// host_state (CPU)
// ----------------------------------------------------------------------------
oneapi_backend::host_state::host_state(ze_context_handle_t ctx, bool profiling)
: context(ctx), profiling(profiling), 
  alloc_queue(named_threads::thread_type::alloc, profiling) { // ctor
  spdlog::debug("oneAPI backend: host_state(ctx={}, profiling={})", static_cast<void*>(ctx), profiling);
}

thread_queue& oneapi_backend::host_state::get_queue(size_t lane) { // queue getter
  OAPI_LOG_ENTER();
  while (lane >= host_queues.size()) {
    spdlog::debug("oneAPI backend: creating host queue for lane {}", host_queues.size());
    host_queues.emplace_back(named_threads::task_type_host_queue(host_queues.size()), profiling);
  }
  return host_queues[lane]; // host queue für lane
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
