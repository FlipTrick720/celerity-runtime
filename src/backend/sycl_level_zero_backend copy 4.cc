// VARIANT 4: Event Pooling + Async Recycling
// - Persistent event pool per device (1024 events)
// - Persistent command list pool per device
// - NO zeCommandQueueSynchronize() in hotpath!
// - Async event recycling via SYCL host_task
// - Should eliminate synchronization bottleneck

#include "backend/sycl_backend.h"
#include "async_event.h"
#include "grid.h"
#include "log.h"
#include "nd_memory.h"
#include "ranges.h"
#include "system_info.h"
#include "tracy.h"
#include "types.h"
#include "utils.h"

#include <cassert>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace celerity::detail::level_zero_backend_detail {

static inline void ze_check(ze_result_t result, const char* where) {
	if(result != ZE_RESULT_SUCCESS) {
		utils::panic("Level-Zero error in {}:: code={}", where, static_cast<int>(result));
	}
}

// Event pool manager
class event_pool_manager {
  public:
	event_pool_manager(ze_context_handle_t context, ze_device_handle_t device, size_t pool_size = 1024)
	    : m_context(context), m_device(device), m_pool_size(pool_size) {
		create_pool();
		CELERITY_DEBUG("[V4] Level-Zero event pool created with {} events", pool_size);
	}

	~event_pool_manager() {
		if(m_pool) {
			zeEventPoolDestroy(m_pool);
		}
	}

	ze_event_handle_t acquire_event() {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		if(!m_free_events.empty()) {
			auto event = m_free_events.back();
			m_free_events.pop_back();
			zeEventHostReset(event);
			return event;
		}
		
		if(m_next_index < m_pool_size) {
			ze_event_desc_t event_desc = {};
			event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
			event_desc.index = m_next_index++;
			event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
			
			ze_event_handle_t event = nullptr;
			ze_check(zeEventCreate(m_pool, &event_desc, &event), "zeEventCreate");
			return event;
		}
		
		CELERITY_WARN("[V4] Event pool exhausted (size={}), creating larger pool", m_pool_size);
		grow_pool();
		return acquire_event();
	}

	void release_event(ze_event_handle_t event) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_free_events.push_back(event);
	}

  private:
	void create_pool() {
		ze_event_pool_desc_t pool_desc = {};
		pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		pool_desc.count = m_pool_size;
		
		ze_check(zeEventPoolCreate(m_context, &pool_desc, 1, &m_device, &m_pool), "zeEventPoolCreate");
	}

	void grow_pool() {
		for(auto event : m_free_events) {
			zeEventDestroy(event);
		}
		m_free_events.clear();
		
		if(m_pool) {
			zeEventPoolDestroy(m_pool);
		}
		
		m_pool_size *= 2;
		m_next_index = 0;
		create_pool();
	}

	ze_context_handle_t m_context;
	ze_device_handle_t m_device;
	ze_event_pool_handle_t m_pool = nullptr;
	size_t m_pool_size;
	size_t m_next_index = 0;
	std::vector<ze_event_handle_t> m_free_events;
	std::mutex m_mutex;
};

// Command list pool manager
struct command_list_pool {
	ze_context_handle_t ctx{};
	ze_device_handle_t dev{};
	std::mutex mtx;
	std::vector<ze_command_list_handle_t> free_lists;
	
	command_list_pool(ze_context_handle_t c, ze_device_handle_t d) : ctx(c), dev(d) {
		CELERITY_DEBUG("[V4] Level-Zero command list pool initialized");
	}
	
	ze_command_list_handle_t acquire() {
		std::scoped_lock lk(mtx);
		if(!free_lists.empty()) {
			auto cl = free_lists.back();
			free_lists.pop_back();
			ze_check(zeCommandListReset(cl), "zeCommandListReset");
			return cl;
		}
		ze_command_list_desc_t desc{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
		ze_command_list_handle_t cl = nullptr;
		ze_check(zeCommandListCreate(ctx, dev, &desc, &cl), "zeCommandListCreate");
		return cl;
	}
	
	void release(ze_command_list_handle_t cl) {
		std::scoped_lock lk(mtx);
		free_lists.push_back(cl);
	}
	
	~command_list_pool() {
		std::scoped_lock lk(mtx);
		for(auto cl : free_lists) {
			zeCommandListDestroy(cl);
		}
		free_lists.clear();
	}
};

// RAII wrappers
class pooled_event {
  public:
	pooled_event(event_pool_manager& pool) : m_pool(&pool), m_event(pool.acquire_event()) {}
	
	~pooled_event() {
		if(m_event && m_pool) {
			m_pool->release_event(m_event);
		}
	}

	pooled_event(const pooled_event&) = delete;
	pooled_event& operator=(const pooled_event&) = delete;
	
	pooled_event(pooled_event&& other) noexcept : m_pool(other.m_pool), m_event(other.m_event) {
		other.m_pool = nullptr;
		other.m_event = nullptr;
	}

	ze_event_handle_t get() const { return m_event; }
	
	ze_event_handle_t release() {
		auto evt = m_event;
		m_event = nullptr;
		m_pool = nullptr;
		return evt;
	}

  private:
	event_pool_manager* m_pool;
	ze_event_handle_t m_event;
};

struct pooled_cmd_list {
	command_list_pool* pool{};
	ze_command_list_handle_t cl{};
	
	explicit pooled_cmd_list(command_list_pool& p) : pool(&p), cl(p.acquire()) {}
	
	~pooled_cmd_list() {
		if(pool && cl) pool->release(cl);
	}
	
	pooled_cmd_list(const pooled_cmd_list&) = delete;
	pooled_cmd_list& operator=(const pooled_cmd_list&) = delete;
	
	ze_command_list_handle_t get() const { return cl; }
};

void nd_copy_box_level_zero(sycl::queue& queue, event_pool_manager& event_pool, command_list_pool& cmdlist_pool,
    const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const box<3>& copy_box, const size_t elem_size, sycl::event& last_event) //
{
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	
	const auto src_range = source_box.get_range();
	const auto dst_range = dest_box.get_range();
	const auto copy_range = copy_box.get_range();
	const auto src_offset = copy_box.get_offset() - source_box.get_offset();
	const auto dst_offset = copy_box.get_offset() - dest_box.get_offset();
	
	const auto layout = layout_nd_copy(src_range, dst_range, src_offset, dst_offset, copy_range, elem_size);
	
	if(layout.contiguous_size == 0) return;
	
	auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
	auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
	
	pooled_event ze_event(event_pool);
	pooled_cmd_list cmd_list(cmdlist_pool);
	
	if(layout.num_complex_strides == 0) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dst_ptr, src_ptr, layout.contiguous_size, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
		CELERITY_TRACE("[V4] Level-Zero backend: contiguous copy {} bytes", layout.contiguous_size);
	} else if(layout.num_complex_strides == 1) {
		const auto& stride = layout.strides[0];
		const size_t width = layout.contiguous_size;
		const size_t height = stride.count;
		const size_t src_pitch = stride.source_stride;
		const size_t dst_pitch = stride.dest_stride;
		
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		
		ze_copy_region_t src_region = {0, 0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
		ze_copy_region_t dst_region = {0, 0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
		
		ze_check(zeCommandListAppendMemoryCopyRegion(cmd_list.get(), dst_ptr, &dst_region, dst_pitch, 0,
		                                             src_ptr, &src_region, src_pitch, 0, ze_event.get(), 0, nullptr), 
		         "zeCommandListAppendMemoryCopyRegion");
		
		CELERITY_TRACE("[V4] Level-Zero backend: 2D copy {}x{} bytes", width, height);
	} else {
		std::vector<std::tuple<size_t, size_t, size_t>> chunks;
		for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t size) {
			chunks.emplace_back(src_off, dst_off, size);
		});
		
		for(size_t i = 0; i < chunks.size(); ++i) {
			const auto& [src_off, dst_off, size] = chunks[i];
			const void* src_ptr = static_cast<const char*>(source_base) + src_off;
			void* dst_ptr = static_cast<char*>(dest_base) + dst_off;
			
			const bool is_last = (i == chunks.size() - 1);
			ze_event_handle_t event_to_use = is_last ? ze_event.get() : nullptr;
			
			ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dst_ptr, src_ptr, size, event_to_use, 0, nullptr), "zeCommandListAppendMemoryCopy");
		}
		
		CELERITY_TRACE("[V4] Level-Zero backend: 3D copy {} chunks", chunks.size());
	}
	
	// Execute command list (NO HOST SYNC!)
	ze_check(zeCommandListClose(cmd_list.get()), "zeCommandListClose");
	auto cmd_list_handle = cmd_list.get();
	ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list_handle, nullptr), "zeCommandQueueExecuteCommandLists");
	
	// Release Level Zero event from RAII - will be recycled by host_task
	auto ze_event_handle = ze_event.release();
	
	// Submit host_task to recycle event asynchronously
	queue.submit([&event_pool, ze_event_handle](sycl::handler& h) {
		h.host_task([&event_pool, ze_event_handle]() {
			// This runs AFTER copy completes, outside hotpath!
			ze_check(zeEventHostSynchronize(ze_event_handle, UINT64_MAX), "zeEventHostSynchronize");
			ze_check(zeEventHostReset(ze_event_handle), "zeEventHostReset");
			event_pool.release_event(ze_event_handle);
		});
	});
	
	// Create SYCL barrier event for dependency tracking
	last_event = queue.ext_oneapi_submit_barrier();
	
	// Command list automatically returned to pool via RAII
}

async_event nd_copy_device_level_zero(sycl::queue& queue, event_pool_manager& event_pool, command_list_pool& cmdlist_pool,
    const void* const source_base, void* const dest_base, const region_layout& source_layout, const region_layout& dest_layout,
    const region<3>& copy_region, const size_t elem_size, bool enable_profiling) //
{
	sycl::event last_event;
	
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [&queue, &event_pool, &cmdlist_pool, elem_size, &last_event](const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_box_level_zero(queue, event_pool, cmdlist_pool, source, dest, source_box, dest_box, copy_box, elem_size, last_event);
	    },
	    [&queue, &event_pool, &cmdlist_pool, &last_event](const void* const source, void* const dest, size_t size_bytes) {
		    CELERITY_TRACE("[V4] Level-Zero backend: linear copy {} bytes", size_bytes);
		    
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    
		    pooled_event ze_event(event_pool);
		    pooled_cmd_list cmd_list(cmdlist_pool);
		    
		    ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dest, source, size_bytes, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
		    ze_check(zeCommandListClose(cmd_list.get()), "zeCommandListClose");
		    auto cmd_list_handle = cmd_list.get();
		    ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list_handle, nullptr), "zeCommandQueueExecuteCommandLists");
		    
		    // Release event from RAII
		    auto ze_event_handle = ze_event.release();
		    
		    // Async recycling
		    queue.submit([&event_pool, ze_event_handle](sycl::handler& h) {
		        h.host_task([&event_pool, ze_event_handle]() {
		            ze_check(zeEventHostSynchronize(ze_event_handle, UINT64_MAX), "zeEventHostSynchronize");
		            ze_check(zeEventHostReset(ze_event_handle), "zeEventHostReset");
		            event_pool.release_event(ze_event_handle);
		        });
		    });
		    
		    last_event = queue.ext_oneapi_submit_barrier();
	    });
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

struct sycl_level_zero_backend::impl {
	struct per_device_state {
		std::unique_ptr<level_zero_backend_detail::event_pool_manager> ev_pool;
		std::unique_ptr<level_zero_backend_detail::command_list_pool> cl_pool;
	};
	std::vector<per_device_state> devs;
};

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config), m_impl(new impl()) {
	CELERITY_DEBUG("[V4] Level-Zero backend initialized with {} device(s)", devices.size());
	
	for(const auto& device : devices) {
		impl::per_device_state st;
		auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.get_platform().ext_oneapi_get_default_context());
		auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
		st.ev_pool = std::make_unique<level_zero_backend_detail::event_pool_manager>(ze_ctx, ze_dev, 1024);
		st.cl_pool = std::make_unique<level_zero_backend_detail::command_list_pool>(ze_ctx, ze_dev);
		m_impl->devs.push_back(std::move(st));
	}
	
	for(device_id i = 0; i < devices.size(); ++i) {
		for(device_id j = i + 1; j < devices.size(); ++j) {
			try {
				const auto ze_device_i = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
				const auto ze_device_j = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[j]);
				
				ze_bool_t can_access_ij = false;
				ze_bool_t can_access_ji = false;
				
				const auto result_ij = zeDeviceCanAccessPeer(ze_device_i, ze_device_j, &can_access_ij);
				const auto result_ji = zeDeviceCanAccessPeer(ze_device_j, ze_device_i, &can_access_ji);
				
				if(result_ij == ZE_RESULT_SUCCESS && result_ji == ZE_RESULT_SUCCESS && can_access_ij && can_access_ji) {
					const memory_id mid_i = first_device_memory_id + i;
					const memory_id mid_j = first_device_memory_id + j;
					get_system_info().memories[mid_i].copy_peers.set(mid_j);
					get_system_info().memories[mid_j].copy_peers.set(mid_i);
					CELERITY_DEBUG("[V4] Level-Zero backend: enabled peer access between D{} and D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("[V4] Level-Zero backend: failed to query peer access: {}", e.what());
			}
		}
	}
}

sycl_level_zero_backend::~sycl_level_zero_backend() = default;

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) //
{
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		auto& st = m_impl->devs[device];
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, *st.ev_pool, *st.cl_pool, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
