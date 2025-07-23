// include/backend/oneapi_backend.h:
#pragma once

#include "backend/backend.h"
#include "backend/sycl_backend.h"
#include "async_event.h"
#include "system_info.h"
#include "thread_queue.h"
#include "closure_hydrator.h"
#include "dense_map.h"
#include <level_zero/ze_api.h>
#include <memory>
#include <vector>
#include <optional>
#include <atomic>

namespace celerity::detail {

class oneapi_backend final : public backend {
public:
  struct configuration {
    bool profiling = false;
    bool per_device_submission_threads = false;
  };

  oneapi_backend(const std::vector<ze_device_handle_t>& devices, const configuration& config);
  ~oneapi_backend() override;

// override the base's pure-virtual const method:
  const system_info& get_system_info() const override;
// provide a non-const overload (no override!):
  system_info&       get_system_info();

  void               init() override;
  void*              debug_alloc(size_t size) override;
  void               debug_free(void* ptr) override;

  async_event enqueue_host_alloc(size_t size, size_t alignment) override;
  async_event enqueue_device_alloc(device_id device, size_t size, size_t alignment) override;
  async_event enqueue_host_free(void* ptr) override;
  async_event enqueue_device_free(device_id device, void* ptr) override;

  async_event enqueue_host_task(size_t host_lane,
                                 const host_task_launcher& launcher,
                                 std::vector<closure_hydrator::accessor_info> infos,
                                 const range<3>& global_range,
                                 const box<3>& execution_range,
                                 const communicator* comm) override;

  async_event enqueue_device_kernel(device_id device,
                                    size_t lane,
                                    const device_kernel_launcher& launch,
                                    std::vector<closure_hydrator::accessor_info> infos,
                                    const box<3>& execution_range,
                                    const std::vector<void*>& reduction_ptrs) override;

  async_event enqueue_host_copy(size_t host_lane,
                                const void* source,
                                void* dest,
                                const region_layout& src_layout,
                                const region_layout& dst_layout,
                                const region<3>& copy_region,
                                size_t elem_size) override;

  async_event enqueue_device_copy(device_id device,
                                  size_t lane,
                                  const void* source,
                                  void* dest,
                                  const region_layout& src_layout,
                                  const region_layout& dst_layout,
                                  const region<3>& copy_region,
                                  size_t elem_size) override;

  void           check_async_errors() override;
  bool           is_profiling_enabled() const;
  const char*    name() const { return "oneAPI/Level-Zero"; }
  

private:
struct device_state {
    ze_context_handle_t        context = nullptr;
    ze_device_handle_t         device = nullptr;
    ze_command_queue_handle_t  queue = nullptr;
    std::atomic_flag           error_check_in_flight = ATOMIC_FLAG_INIT;
    std::optional<thread_queue> submit_thread;

    device_state() = default;
    device_state(const device_state&) = delete;
    device_state& operator=(const device_state&) = delete;
    
    // Custom move constructor to handle atomic_flag
    device_state(device_state&& other) noexcept 
        : context(other.context)
        , device(other.device)
        , queue(other.queue)
        , submit_thread(std::move(other.submit_thread)) {
        other.context = nullptr;
        other.device = nullptr;
        other.queue = nullptr;
    }

    // Custom move assignment operator
    device_state& operator=(device_state&& other) noexcept {
        if (this != &other) {
            context = other.context;
            device = other.device;
            queue = other.queue;
            submit_thread = std::move(other.submit_thread);
            other.context = nullptr;
            other.device = nullptr;
            other.queue = nullptr;
        }
        return *this;
    }
};

// Forward declarations
class ze_event_impl;
void* allocate_host_memory(size_t size, size_t alignment);

  struct host_state {
    ze_context_handle_t    context;
    thread_queue           alloc_queue;
    std::vector<thread_queue> host_queues;
    host_state(ze_context_handle_t ctx, bool profiling);
    thread_queue& get_queue(size_t lane);
  };

  system_info                                  m_system;
  dense_map<device_id, device_state>           m_devices;
  host_state                                   m_host;
  configuration                                m_config;

  // helper: record L0 event + wrap it in an async_event
  async_event record_and_wrap(ze_event_handle_t evt_before,
                              ze_event_handle_t evt_after);

  // enqueue arbitrary device work, possibly via submit threads
  async_event enqueue_work(device_id did,
                           size_t lane,
                           std::function<async_event(device_state&)> work);

  ze_driver_handle_t  m_driver   = nullptr;
  ze_context_handle_t m_context  = nullptr;
};

std::unique_ptr<backend>
make_oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                    const oneapi_backend::configuration& lvl0_cfg);

inline std::unique_ptr<backend>
make_oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                    const sycl_backend::configuration& sycl_cfg) {
  oneapi_backend::configuration lvl0_cfg;
  lvl0_cfg.profiling                   = sycl_cfg.profiling;
  lvl0_cfg.per_device_submission_threads = sycl_cfg.per_device_submission_threads;
  // this call picks the *real* overload, not itself
  return make_oneapi_backend(devices, lvl0_cfg);
}


} // namespace celerity::detail
