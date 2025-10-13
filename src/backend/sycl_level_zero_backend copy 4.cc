// VARIANT 4: Immediate Command Lists + Persistent Pool
// Goal: Use zeCommandListCreateImmediate (persistent per device)
// Implementation: Immediate command lists with persistent event pool
// Hypothesis: Significantly less host overhead per submit

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
#include <deque>
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

// ============================================================================
// VARIANT 4: Immediate Command Lists + Persistent Pool
// ============================================================================

class immediate_cmdlist_pool {
public:
	immediate_cmdlist_pool(ze_context_handle_t ctx, ze_device_handle_t dev) : m_ctx(ctx), m_dev(dev) {
		// Create persistent event pool
		ze_event_pool_desc_t pool_desc = {};
		pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		pool_desc.count = 1024;
		
		ze_check(zeEventPoolCreate(m_ctx, &pool_desc, 1, &m_dev, &m_event_pool), "zeEventPoolCreate");
		CELERITY_DEBUG("[V4] Immediate command list pool created with persistent event pool (1024 events)");
	}
	
	~immediate_cmdlist_pool() {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		// Destroy all events
		for(auto event : m_all_events) {
			zeEventDestroy(event);
		}
		
		// Destroy all immediate command lists
		for(auto cl : m_all_cmdlists) {
			zeCommandListDestroy(cl);
		}
		
		if(m_event_pool) {
			zeEventPoolDestroy(m_event_pool);
		}
		
		CELERITY_DEBUG("[V4] Destroyed {} events and {} immediate command lists", 
		               m_all_events.size(), m_all_cmdlists.size());
	}
	
	ze_event_handle_t get_event() {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		// Try to reuse from free list
		if(!m_free_events.empty()) {
			auto event = m_free_events.back();
			m_free_events.pop_back();
			zeEventHostReset(event);
			return event;
		}
		
		// Create new event if pool not exhausted
		if(m_next_event_index < 1024) {
			ze_event_desc_t event_desc = {};
			event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
			event_desc.index = m_next_event_index++;
			event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
			
			ze_event_handle_t event = nullptr;
			ze_check(zeEventCreate(m_event_pool, &event_desc, &event), "zeEventCreate");
			m_all_events.push_back(event);
			return event;
		}
		
		utils::panic("[V4] Event pool exhausted");
	}
	
	void return_event(ze_event_handle_t event) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_free_events.push_back(event);
	}
	
	ze_command_list_handle_t get_immediate_cmdlist() {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		// Try to reuse from free list
		if(!m_free_cmdlists.empty()) {
			auto cl = m_free_cmdlists.back();
			m_free_cmdlists.pop_back();
			// Immediate command lists don't need reset
			return cl;
		}
		
		// Create new immediate command list
		ze_command_queue_desc_t queue_desc = {};
		queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
		queue_desc.ordinal = 0;
		queue_desc.index = 0;
		queue_desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
		
		ze_command_list_handle_t cl = nullptr;
		ze_check(zeCommandListCreateImmediate(m_ctx, m_dev, &queue_desc, &cl), "zeCommandListCreateImmediate");
		m_all_cmdlists.push_back(cl);
		return cl;
	}
	
	void return_immediate_cmdlist(ze_command_list_handle_t cl) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_free_cmdlists.push_back(cl);
	}

private:
	ze_context_handle_t m_ctx;
	ze_device_handle_t m_dev;
	ze_event_pool_handle_t m_event_pool = nullptr;
	size_t m_next_event_index = 0;
	
	std::vector<ze_event_handle_t> m_all_events;
	std::vector<ze_event_handle_t> m_free_events;
	std::vector<ze_command_list_handle_t> m_all_cmdlists;
	std::vector<ze_command_list_handle_t> m_free_cmdlists;
	std::mutex m_mutex;
};

// Global pools
static std::vector<std::unique_ptr<immediate_cmdlist_pool>> g_immediate_pools;
static std::mutex g_pools_mutex;

void initialize_pools(const std::vector<sycl::device>& devices) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	g_immediate_pools.clear();
	
	for(const auto& device : devices) {
		auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.get_platform().ext_oneapi_get_default_context());
		auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
		g_immediate_pools.push_back(std::make_unique<immediate_cmdlist_pool>(ze_ctx, ze_dev));
	}
	
	CELERITY_DEBUG("[V4] Initialized {} immediate command list pools", devices.size());
}

void cleanup_pools() {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	g_immediate_pools.clear();
	CELERITY_DEBUG("[V4] Cleaned up pools");
}

immediate_cmdlist_pool& get_immediate_pool(device_id device) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	if(device >= g_immediate_pools.size()) {
		utils::panic("[V4] Invalid device ID for immediate pool: {}", device);
	}
	return *g_immediate_pools[device];
}

// RAII wrappers
class pooled_event {
public:
	pooled_event(device_id device) : m_device(device) {
		m_event = get_immediate_pool(device).get_event();
	}
	
	~pooled_event() {
		if(m_event) {
			get_immediate_pool(m_device).return_event(m_event);
		}
	}
	
	ze_event_handle_t get() const { return m_event; }
	
	pooled_event(const pooled_event&) = delete;
	pooled_event& operator=(const pooled_event&) = delete;

private:
	device_id m_device;
	ze_event_handle_t m_event = nullptr;
};

class pooled_immediate_cmdlist {
public:
	pooled_immediate_cmdlist(device_id device) : m_device(device) {
		m_cmdlist = get_immediate_pool(device).get_immediate_cmdlist();
	}
	
	~pooled_immediate_cmdlist() {
		if(m_cmdlist) {
			get_immediate_pool(m_device).return_immediate_cmdlist(m_cmdlist);
		}
	}
	
	ze_command_list_handle_t get() const { return m_cmdlist; }
	
	pooled_immediate_cmdlist(const pooled_immediate_cmdlist&) = delete;
	pooled_immediate_cmdlist& operator=(const pooled_immediate_cmdlist&) = delete;

private:
	device_id m_device;
	ze_command_list_handle_t m_cmdlist = nullptr;
};

// ============================================================================
// Copy Functions (using immediate command lists)
// ============================================================================

void nd_copy_box_level_zero(sycl::queue& queue, device_id device, const void* const source_base, void* const dest_base,
    const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box, const size_t elem_size, sycl::event& last_event) {
	
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
	
	// Use pooled event and immediate command list
	pooled_event ze_event(device);
	pooled_immediate_cmdlist cmd_list(device);
	
	if(layout.num_complex_strides == 0) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		// Immediate command list - executes immediately, no close/execute needed
		ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dst_ptr, src_ptr, layout.contiguous_size, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
		CELERITY_TRACE("[V4] Level-Zero backend: contiguous copy {} bytes (immediate)", layout.contiguous_size);
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
		
		CELERITY_TRACE("[V4] Level-Zero backend: 2D copy {}x{} bytes (immediate)", width, height);
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
		
		CELERITY_TRACE("[V4] Level-Zero backend: 3D copy {} chunks (immediate)", chunks.size());
	}
	
	// Immediate command lists execute synchronously, but we still sync to be safe
	ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
	
	last_event = queue.ext_oneapi_submit_barrier();
}

async_event nd_copy_device_level_zero(sycl::queue& queue, device_id device, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size, bool enable_profiling) {
	
	sycl::event last_event;
	
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [&queue, device, elem_size, &last_event](const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_box_level_zero(queue, device, source, dest, source_box, dest_box, copy_box, elem_size, last_event);
	    },
	    [&queue, device, &last_event](const void* const source, void* const dest, size_t size_bytes) {
		    CELERITY_TRACE("[V4] Level-Zero backend: linear copy {} bytes (immediate)", size_bytes);
		    
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    
		    pooled_event ze_event(device);
		    pooled_immediate_cmdlist cmd_list(device);
		    
		    ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dest, source, size_bytes, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
		    ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
		    
		    last_event = queue.ext_oneapi_submit_barrier();
	    });
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("[V4] Level-Zero backend initialized with {} device(s)", devices.size());
	
	level_zero_backend_detail::initialize_pools(devices);
	
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

sycl_level_zero_backend::~sycl_level_zero_backend() {
	level_zero_backend_detail::cleanup_pools();
}

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) {
	
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, device, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
