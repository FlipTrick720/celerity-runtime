//Version: v3_batch_fence
//Text: Fence-based batching for massive bandwidth improvements (batch multiple ops, single fence sync)

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
#include <utility>
#include <vector>
#include <queue>
#include <mutex>
#include <chrono>

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

// Batch manager for fence-based batching (NEW in V3)
struct batch_manager {
	ze_command_list_handle_t batch_list = nullptr;
	ze_fence_handle_t fence = nullptr;
	ze_command_queue_handle_t queue = nullptr;
	std::mutex mutex;
	size_t pending_ops = 0;
	size_t batch_threshold_ops = 8;     // Batch after N operations
	size_t batch_threshold_us = 100;    // Or after N microseconds
	std::chrono::steady_clock::time_point batch_start_time;
	size_t total_batches = 0;
	size_t total_ops_batched = 0;
	
	void initialize(ze_context_handle_t context, ze_device_handle_t device, ze_command_queue_handle_t q) {
		queue = q;
		
		// Create batch command list
		ze_command_list_desc_t desc = {};
		desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		desc.flags = 0;  // Regular command list (not immediate)
		
		ze_check(zeCommandListCreate(context, device, &desc, &batch_list), "zeCommandListCreate");
		
		// Create fence for batch synchronization
		ze_fence_desc_t fence_desc = {};
		fence_desc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
		fence_desc.flags = 0;
		
		ze_check(zeFenceCreate(queue, &fence_desc, &fence), "zeFenceCreate");
		
		// Get thresholds from environment
		const char* env_ops = std::getenv("CELERITY_L0_BATCH_THRESHOLD_OPS");
		const char* env_us = std::getenv("CELERITY_L0_BATCH_THRESHOLD_US");
		
		if (env_ops) batch_threshold_ops = std::atoi(env_ops);
		if (env_us) batch_threshold_us = std::atoi(env_us);
		
		batch_start_time = std::chrono::steady_clock::now();
		
		CELERITY_DEBUG("Level-Zero V3: Created batch manager (ops={}, us={})", batch_threshold_ops, batch_threshold_us);
	}
	
	bool should_flush() {
		if (pending_ops >= batch_threshold_ops) return true;
		
		auto now = std::chrono::steady_clock::now();
		auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - batch_start_time).count();
		return elapsed_us >= static_cast<long>(batch_threshold_us);
	}
	
	void add_operation() {
		pending_ops++;
	}
	
	void flush_batch() {
		if (pending_ops == 0) return;
		
		// Close and execute batch
		ze_check(zeCommandListClose(batch_list), "zeCommandListClose");
		ze_check(zeCommandQueueExecuteCommandLists(queue, 1, &batch_list, fence), "zeCommandQueueExecuteCommandLists");
		
		// Wait for completion
		ze_check(zeFenceHostSynchronize(fence, UINT64_MAX), "zeFenceHostSynchronize");
		
		// Reset for next batch
		ze_check(zeFenceReset(fence), "zeFenceReset");
		ze_check(zeCommandListReset(batch_list), "zeCommandListReset");
		
		// Update stats
		total_batches++;
		total_ops_batched += pending_ops;
		
		CELERITY_TRACE("Level-Zero V3: flushed batch with {} operations", pending_ops);
		
		pending_ops = 0;
		batch_start_time = std::chrono::steady_clock::now();
	}
	
	ze_command_list_handle_t get_batch_list() {
		return batch_list;
	}
	
	void cleanup() {
		// Flush any pending operations
		if (pending_ops > 0) {
			flush_batch();
		}
		
		CELERITY_DEBUG("Level-Zero V3 batch stats: {} batches, {} ops total (avg {:.1f} ops/batch)", 
		              total_batches, total_ops_batched, 
		              total_batches > 0 ? static_cast<double>(total_ops_batched) / total_batches : 0.0);
		
		if (fence) ze_check(zeFenceDestroy(fence), "zeFenceDestroy");
		if (batch_list) ze_check(zeCommandListDestroy(batch_list), "zeCommandListDestroy");
	}
};

// Global managers
static std::vector<batch_manager> g_batch_managers;
static std::mutex g_pools_mutex;
static bool g_pools_initialized = false;

void initialize_batching(const std::vector<sycl::device>& devices, ze_context_handle_t context) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	if (g_pools_initialized) return;
	
	g_batch_managers.resize(devices.size());
	
	for (size_t i = 0; i < devices.size(); ++i) {
		auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
		
		// Create a temporary queue to get queue handle
		sycl::queue temp_queue(devices[i]);
		auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(temp_queue);
		auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		
		g_batch_managers[i].initialize(context, ze_device, ze_queue);
	}
	
	CELERITY_DEBUG("Level-Zero V3: Initialized {} devices with batching", devices.size());
	g_pools_initialized = true;
}

void cleanup_batching() {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	for (auto& batch : g_batch_managers) {
		batch.cleanup();
	}
	g_batch_managers.clear();
	g_pools_initialized = false;
}

// Helper to perform box-based copy with batching
void nd_copy_box_level_zero_batch(sycl::queue& queue, device_id device, const void* const source_base, void* const dest_base, 
    const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box, const size_t elem_size, sycl::event& last_event) {
	
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	
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
	
	// BATCH MODE: No per-op events, use fence
	std::lock_guard<std::mutex> lock(g_batch_managers[device].mutex);
	ze_command_list_handle_t batch_list = g_batch_managers[device].get_batch_list();
	
	if(layout.num_complex_strides == 0) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		ze_check(zeCommandListAppendMemoryCopy(batch_list, dst_ptr, src_ptr, layout.contiguous_size, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
		CELERITY_TRACE("Level-Zero V3: batch contiguous copy {} bytes", layout.contiguous_size);
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
		
		ze_check(zeCommandListAppendMemoryCopyRegion(batch_list, dst_ptr, &dst_region, dst_pitch, 0,
		                                             src_ptr, &src_region, src_pitch, 0, nullptr, 0, nullptr), 
		         "zeCommandListAppendMemoryCopyRegion");
		
		CELERITY_TRACE("Level-Zero V3: batch 2D copy {}x{} bytes", width, height);
	} else {
		std::vector<std::tuple<size_t, size_t, size_t>> chunks;
		for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t size) {
			chunks.emplace_back(src_off, dst_off, size);
		});
		
		for(const auto& [src_off, dst_off, size] : chunks) {
			const void* src_ptr = static_cast<const char*>(source_base) + src_off;
			void* dst_ptr = static_cast<char*>(dest_base) + dst_off;
			
			ze_check(zeCommandListAppendMemoryCopy(batch_list, dst_ptr, src_ptr, size, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
		}
		
		CELERITY_TRACE("Level-Zero V3: batch 3D copy {} chunks", chunks.size());
	}
	
	// Add to batch
	g_batch_managers[device].add_operation();
	
	// Check if we should flush
	if (g_batch_managers[device].should_flush()) {
		g_batch_managers[device].flush_batch();
	}
	
	// Create SYCL barrier event
	last_event = queue.ext_oneapi_submit_barrier();
}

// Helper function for n-dimensional device copy with batching
async_event nd_copy_device_level_zero_batch(sycl::queue& queue, device_id device, const void* const source_base, void* const dest_base, 
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, 
    const size_t elem_size, bool enable_profiling) {
	
	sycl::event last_event;
	
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [&queue, device, elem_size, &last_event](const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_box_level_zero_batch(queue, device, source, dest, source_box, dest_box, copy_box, elem_size, last_event);
	    },
	    [&queue, device, &last_event](const void* const source, void* const dest, size_t size_bytes) {
		    // Zero-work short-circuit
		    if (size_bytes == 0) {
			    last_event = queue.ext_oneapi_submit_barrier();
			    return;
		    }
		    
		    CELERITY_TRACE("Level-Zero V3: batch linear copy {} bytes", size_bytes);
		    
		    std::lock_guard<std::mutex> lock(g_batch_managers[device].mutex);
		    ze_command_list_handle_t batch_list = g_batch_managers[device].get_batch_list();
		    ze_check(zeCommandListAppendMemoryCopy(batch_list, dest, source, size_bytes, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
		    
		    g_batch_managers[device].add_operation();
		    
		    if (g_batch_managers[device].should_flush()) {
			    g_batch_managers[device].flush_batch();
		    }
		    
		    last_event = queue.ext_oneapi_submit_barrier();
	    });
	
	// Force flush any pending batch operations
	{
		std::lock_guard<std::mutex> lock(g_batch_managers[device].mutex);
		g_batch_managers[device].flush_batch();
	}
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("Level-Zero V3 backend initialized with {} device(s)", devices.size());
	
	// Initialize batch managers
	auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[0].get_platform().ext_oneapi_get_default_context());
	level_zero_backend_detail::initialize_batching(devices, ze_context);
	
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
					CELERITY_DEBUG("Level-Zero V3: enabled peer access between D{} and D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("Level-Zero V3: failed to query peer access: {}", e.what());
			}
		}
	}
}

sycl_level_zero_backend::~sycl_level_zero_backend() {
	level_zero_backend_detail::cleanup_batching();
}

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) {
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		return level_zero_backend_detail::nd_copy_device_level_zero_batch(
		    queue, device, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
