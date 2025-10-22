//Version: v1_event_pooling
//Text: Persistent event pool with freelist reuse (512 events default)

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
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>
#include <queue>
#include <mutex>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <libenvpp/env.hpp>

namespace celerity::detail::level_zero_backend_detail {

// Level Zero error checking helper
static inline void ze_check(ze_result_t result, const char* where) {
	if(result != ZE_RESULT_SUCCESS) {
		utils::panic("Level-Zero error in {}:: code={}", where, static_cast<int>(result));
	}
}

// Event pool manager (one per device)
struct event_pool_manager {
	ze_event_pool_handle_t pool = nullptr;
	std::vector<ze_event_handle_t> events;
	std::queue<size_t> free_indices;
	std::mutex mutex;
	size_t peak_usage = 0;
	size_t total_acquires = 0;
	
	// Make non-copyable and non-movable (contains std::mutex)
	event_pool_manager() = default;
	event_pool_manager(const event_pool_manager&) = delete;
	event_pool_manager& operator=(const event_pool_manager&) = delete;
	event_pool_manager(event_pool_manager&&) = delete;
	event_pool_manager& operator=(event_pool_manager&&) = delete;
	
	void initialize(ze_context_handle_t context, ze_device_handle_t device, size_t pool_size) {
		// Create persistent pool
		ze_event_pool_desc_t pool_desc = {};
		pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		pool_desc.count = pool_size;
		
		ze_check(zeEventPoolCreate(context, &pool_desc, 1, &device, &pool), "zeEventPoolCreate");
		
		// Pre-create all events
		events.resize(pool_size);
		for (size_t i = 0; i < pool_size; ++i) {
			ze_event_desc_t event_desc = {};
			event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
			event_desc.index = i;
			event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
			
			ze_check(zeEventCreate(pool, &event_desc, &events[i]), "zeEventCreate");
			free_indices.push(i);
		}
		
		CELERITY_DEBUG("Level-Zero V1: Created event pool with {} events", pool_size);
	}
	
	size_t acquire() {
		std::lock_guard<std::mutex> lock(mutex);
		
		if (free_indices.empty()) {
			CELERITY_WARN("Level-Zero event pool exhausted! Consider increasing CELERITY_L0_EVENT_POOL_SIZE");
			utils::panic("Event pool exhausted");
		}
		
		size_t idx = free_indices.front();
		free_indices.pop();
		
		total_acquires++;
		size_t in_use = events.size() - free_indices.size();
		peak_usage = std::max(peak_usage, in_use);
		
		return idx;
	}
	
	void release(size_t idx) {
		// Reset event for reuse
		ze_check(zeEventHostReset(events[idx]), "zeEventHostReset");
		
		std::lock_guard<std::mutex> lock(mutex);
		free_indices.push(idx);
	}
	
	ze_event_handle_t get_event(size_t idx) {
		return events[idx];
	}
	
	void cleanup() {
		CELERITY_DEBUG("Level-Zero V1 event pool stats: peak usage {}/{}, total acquires {}", 
		              peak_usage, events.size(), total_acquires);
		
		for (auto event : events) {
			if (event) zeEventDestroy(event);
		}
		if (pool) zeEventPoolDestroy(pool);
	}
};

// Global pools (one per device) - use unique_ptr because event_pool_manager contains std::mutex
static std::vector<std::unique_ptr<event_pool_manager>> g_event_pools;
static std::mutex g_pools_mutex;
static bool g_pools_initialized = false;

// Initialize pools at backend startup
void initialize_event_pools(const std::vector<sycl::device>& devices, ze_context_handle_t context) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	
	if (g_pools_initialized) return;
	
	// FIXED: Register environment variable with libenvpp to prevent "unused" warnings
	static bool env_registered = false;
	if (!env_registered) {
		auto pref = env::prefix("CELERITY");
		pref.register_variable<size_t>("L0_EVENT_POOL_SIZE");
		env_registered = true;
	}
	
	// Get pool size from environment
	const char* env_size = std::getenv("CELERITY_L0_EVENT_POOL_SIZE");
	size_t pool_size = env_size ? std::atoi(env_size) : 512;
	
	if (env_size) {
		CELERITY_DEBUG("Level-Zero V1: Using CELERITY_L0_EVENT_POOL_SIZE={}", pool_size);
	}
	
	g_event_pools.reserve(devices.size());
	
	for (size_t i = 0; i < devices.size(); ++i) {
		auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
		g_event_pools.emplace_back(std::make_unique<event_pool_manager>());
		g_event_pools[i]->initialize(context, ze_device, pool_size);
	}
	
	g_pools_initialized = true;
}

void cleanup_event_pools() {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	for (auto& pool : g_event_pools) {
		pool->cleanup();
	}
	g_event_pools.clear();
	g_pools_initialized = false;
}

// Helper to perform box-based copy using native Level Zero operations
void nd_copy_box_level_zero(sycl::queue& queue, device_id device, const void* const source_base, void* const dest_base, 
    const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box, const size_t elem_size, sycl::event& last_event) {
	
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	
	// compute layout/strides/offsets
	const auto src_range = source_box.get_range();
	const auto dst_range = dest_box.get_range();
	const auto copy_range = copy_box.get_range();
	const auto src_offset = copy_box.get_offset() - source_box.get_offset();
	const auto dst_offset = copy_box.get_offset() - dest_box.get_offset();
	
	const auto layout = layout_nd_copy(src_range, dst_range, src_offset, dst_offset, copy_range, elem_size);
	
	// Zero-work short-circuit
	if(layout.contiguous_size == 0) {
		last_event = queue.ext_oneapi_submit_barrier();
		return;
	}
	
	// Get native Level Zero handles
	auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
	auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
	auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
	auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
	
	// Acquire event from pool
	size_t event_idx = g_event_pools[device]->acquire();
	ze_event_handle_t ze_event = g_event_pools[device]->get_event(event_idx);
	
	// Create command list
	ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
	ze_command_list_handle_t cmd_list = nullptr;
	ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
	
	if(layout.num_complex_strides == 0) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, layout.contiguous_size, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
		CELERITY_TRACE("Level-Zero V1: contiguous copy {} bytes", layout.contiguous_size);
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
		
		ze_check(zeCommandListAppendMemoryCopyRegion(cmd_list, dst_ptr, &dst_region, dst_pitch, 0,
		                                             src_ptr, &src_region, src_pitch, 0, ze_event, 0, nullptr), 
		         "zeCommandListAppendMemoryCopyRegion");
		
		CELERITY_TRACE("Level-Zero V1: 2D copy {}x{} bytes", width, height);
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
			ze_event_handle_t event_to_use = is_last ? ze_event : nullptr;
			
			ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, size, event_to_use, 0, nullptr), "zeCommandListAppendMemoryCopy");
		}
		
		CELERITY_TRACE("Level-Zero V1: 3D copy {} chunks", chunks.size());
	}
	
	// Execute
	ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
	ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
	ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
	
	// Clean up command list
	ze_check(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
	
	// Release event back to pool
	g_event_pools[device]->release(event_idx);
	
	// Create SYCL barrier event
	last_event = queue.ext_oneapi_submit_barrier();
}

// Helper function for n-dimensional device copy
async_event nd_copy_device_level_zero(sycl::queue& queue, device_id device, const void* const source_base, void* const dest_base, 
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, 
    const size_t elem_size, bool enable_profiling) {
	
	sycl::event last_event;
	
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [&queue, device, elem_size, &last_event](const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_box_level_zero(queue, device, source, dest, source_box, dest_box, copy_box, elem_size, last_event);
	    },
	    [&queue, device, &last_event](const void* const source, void* const dest, size_t size_bytes) {
		    // Zero-work short-circuit
		    if (size_bytes == 0) {
			    last_event = queue.ext_oneapi_submit_barrier();
			    return;
		    }
		    
		    CELERITY_TRACE("Level-Zero V1: linear copy {} bytes", size_bytes);
		    
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
		    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
		    
		    // Acquire event from pool
		    size_t event_idx = g_event_pools[device]->acquire();
		    ze_event_handle_t ze_event = g_event_pools[device]->get_event(event_idx);
		    
		    ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
		    ze_command_list_handle_t cmd_list = nullptr;
		    ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
		    ze_check(zeCommandListAppendMemoryCopy(cmd_list, dest, source, size_bytes, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
		    ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
		    ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
		    ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
		    
		    ze_check(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
		    
		    // Release event back to pool
		    g_event_pools[device]->release(event_idx);
		    
		    last_event = queue.ext_oneapi_submit_barrier();
	    });
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("Level-Zero V1 backend initialized with {} device(s)", devices.size());
	
	// Initialize event pools
	auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[0].get_platform().ext_oneapi_get_default_context());
	level_zero_backend_detail::initialize_event_pools(devices, ze_context);
	
	// Query and enable peer-to-peer access
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
					CELERITY_DEBUG("Level-Zero V1: enabled peer access between D{} and D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("Level-Zero V1: failed to query peer access: {}", e.what());
			}
		}
	}
}

sycl_level_zero_backend::~sycl_level_zero_backend() {
	level_zero_backend_detail::cleanup_event_pools();
}

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) {
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, device, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
