// include/backend/oneapi_backend.h:
#pragma once

#include "backend/backend.h"
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
#include <memory>
#include <spdlog/spdlog.h>

namespace celerity::detail {

class oneapi_backend final : public backend {
public:
  struct configuration {
    bool profiling = false;
    bool per_device_submission_threads = true;
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
    ze_device_handle_t          device{};
    ze_context_handle_t         context{};
    ze_command_queue_handle_t   queue{};

    sycl::device                sycl_dev;
    sycl::context               sycl_ctx{ sycl::property_list{} }; // Inline init to avoid explicit ctor issues
    std::vector<std::unique_ptr<sycl::queue>> sycl_lanes;
    std::optional<thread_queue> submit_thread;

    sycl::queue& sycl_queue_for_lane(size_t lane, bool profiling) {
        while (sycl_lanes.size() <= lane) {
            auto props = profiling
              ? sycl::property_list{ sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} }
              : sycl::property_list{ sycl::property::queue::in_order{} };
        
            // Async handler that logs errors from device tasks
            auto ah = [](sycl::exception_list elist) {
                for (const auto& e : elist) {
                    try {
                        std::rethrow_exception(e);
                    } catch (const sycl::exception& se) {
                        spdlog::error("SYCL async exception: {}", se.what());
                        // If you need more detail:
                        // spdlog::error("SYCL error code: {}", static_cast<int>(se.code().value()));
                    } catch (...) {
                        spdlog::error("SYCL async exception: <non-SYCL exception>");
                    }
                }
                // Do NOT call std::terminate() here; keep running so we can see logs.
            };
          
            sycl_lanes.emplace_back(std::make_unique<sycl::queue>(sycl_ctx, sycl_dev, ah, props));
        }
        return *sycl_lanes[lane];
    }
};

// Forward declarations
class ze_event_impl;
void* allocate_host_memory(size_t size, size_t alignment);

  struct host_state {
    ze_context_handle_t    context;
    bool                   profiling;
    thread_queue           alloc_queue;
    std::vector<thread_queue> host_queues;
    host_state(ze_context_handle_t ctx, bool profiling);
    thread_queue& get_queue(size_t lane);
  };

  std::unique_ptr<host_state>                  m_host;
  dense_map<device_id, device_state>           m_devices;
  system_info                                  m_system;
  configuration                                m_config;

  // enqueue arbitrary device work, possibly via submit threads
  async_event enqueue_work(device_id did,
                           size_t lane,
                           std::function<async_event(device_state&)> work);

  ze_driver_handle_t  m_driver   = nullptr;
  ze_context_handle_t m_context  = nullptr;
};

// Helper functions for event management
async_event make_event_from_native_sycl(ze_event_handle_t evt);
async_event make_event_from_ze_owned(ze_event_handle_t evt, ze_event_pool_handle_t pool);

std::unique_ptr<backend>
make_oneapi_backend(const std::vector<ze_device_handle_t>& devices,
                    const oneapi_backend::configuration& lvl0_cfg);


} // namespace celerity::detail
