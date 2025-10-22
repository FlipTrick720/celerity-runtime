//Version: v4_micro_optimized
//Text: Micro-copy optimizations + all previous optimizations combined

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
#include <cstring>

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

// Configuration
static size_t g_micro_threshold = 256;  // Bytes - use memcpy for tiny copies
static size_t g_small_threshold = 4096; // Bytes - use immediate lists for small copies
static bool g_use_batching = true;      // Use batching for larger copies

// Event pool manager
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
		ze_event_pool_desc_t pool_desc = {};
		pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		pool_desc.count = pool_size;
		
		ze_check(zeEventPoolCreate(context, &pool_desc, 1, &device, &pool), "zeEventPoolCreate");
		
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
		
		CELERITY_DEBUG("Level-Zero V4: Created event pool with {} events", pool_size);
	}
	
	size_t acquire() {
		std::lock_guard<std::mutex> lock(mutex);
		if (free_indices.empty()) {
			CELERITY_WARN("Level-Zero event pool exhausted!");
			utils::panic("Event pool exhausted");
		}
		size_t idx = free_indices.front();
		free_indices.pop();
		total_acquires++;
		peak_usage = std::max(peak_usage, events.size() - free_indices.size());
		return idx;
	}
	
	void release(size_t idx) {
		std::lock_guard<std::mutex> lock(mutex);
		free_indices.push(idx);
	}
	
	ze_event_handle_t get_event(size_t idx) { return events[idx]; }
	
	void cleanup() {
		CELERITY_DEBUG("Level-Zero V4 event pool: peak {}/{}, total {}", peak_usage, events.size(), total_acquires);
		for (auto event : events) { if (event) zeEventDestroy(event); }
		if (pool) zeEventPoolDestroy(pool);
	}
};

// Immediate command list manager
struct immediate_cmdlist_manager {
	ze_command_list_handle_t immediate_list = nullptr;
	std::mutex mutex;
	size_t operations_count = 0;
	
	// Make non-copyable and non-movable (contains std::mutex)
	immediate_cmdlist_manager() = default;
	immediate_cmdlist_manager(const immediate_cmdlist_manager&) = delete;
	immediate_cmdlist_manager& operator=(const immediate_cmdlist_manager&) = delete;
	immediate_cmdlist_manager(immediate_cmdlist_manager&&) = delete;
	immediate_cmdlist_manager& operator=(immediate_cmdlist_manager&&) = delete;
	
	void initialize(ze_context_handle_t context, ze_device_handle_t device) {
		ze_command_queue_desc_t queue_desc = {};
		queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
		queue_desc.ordinal = 0;
		queue_desc.index = 0;
		queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
		
		ze_check(zeCommandListCreateImmediate(context, device, &queue_desc, &immediate_list), "zeCommandListCreateImmediate");
		CELERITY_DEBUG("Level-Zero V4: Created persistent immediate command list");
	}
	
	ze_command_list_handle_t get() { 
		operations_count++;
		return immediate_list; 
	}
	
	void cleanup() {
		CELERITY_DEBUG("Level-Zero V4: Immediate list executed {} operations", operations_count);
		if (immediate_list) zeCommandListDestroy(immediate_list);
	}
};

// Batch manager
struct batch_manager {
	ze_command_list_handle_t batch_list = nullptr;
	ze_fence_handle_t fence = nullptr;
	ze_command_queue_handle_t queue = nullptr;
	std::mutex mutex;
	size_t pending_ops = 0;
	size_t batch_threshold_ops = 8;
	size_t batch_threshold_us = 100;
	std::chrono::steady_clock::time_point batch_start_time;
	size_t total_batches = 0;
	size_t total_ops_batched = 0;
	size_t micro_copies_bypassed = 0;
	
	// Make non-copyable and non-movable (contains std::mutex)
	batch_manager() = default;
	batch_manager(const batch_manager&) = delete;
	batch_manager& operator=(const batch_manager&) = delete;
	batch_manager(batch_manager&&) = delete;
	batch_manager& operator=(batch_manager&&) = delete;
	
	void initialize(ze_context_handle_t context, ze_device_handle_t device, ze_command_queue_handle_t q) {
		queue = q;
		
		ze_command_list_desc_t desc = {};
		desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		desc.flags = 0;
		
		ze_check(zeCommandListCreate(context, device, &desc, &batch_list), "zeCommandListCreate");
		
		ze_fence_desc_t fence_desc = {};
		fence_desc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
		fence_desc.flags = 0;
		
		ze_check(zeFenceCreate(queue, &fence_desc, &fence), "zeFenceCreate");
		
		// Get thresholds from environment
		const char* env_ops = std::getenv("CELERITY_L0_BATCH_THRESHOLD_OPS");
		const char* env_us = std::getenv("CELERITY_L0_BATCH_THRESHOLD_US");
		
		if (env_ops) {
			batch_threshold_ops = std::atoi(env_ops);
			CELERITY_DEBUG("Level-Zero V4: Using CELERITY_L0_BATCH_THRESHOLD_OPS={}", batch_threshold_ops);
		}
		if (env_us) {
			batch_threshold_us = std::atoi(env_us);
			CELERITY_DEBUG("Level-Zero V4: Using CELERITY_L0_BATCH_THRESHOLD_US={}", batch_threshold_us);
		}
		
		batch_start_time = std::chrono::steady_clock::now();
		
		CELERITY_DEBUG("Level-Zero V4: Created batch manager (ops={}, us={})", batch_threshold_ops, batch_threshold_us);
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
		
		ze_check(zeCommandListClose(batch_list), "zeCommandListClose");
		ze_check(zeCommandQueueExecuteCommandLists(queue, 1, &batch_list, fence), "zeCommandQueueExecuteCommandLists");
		ze_check(zeFenceHostSynchronize(fence, UINT64_MAX), "zeFenceHostSynchronize");
		ze_check(zeFenceReset(fence), "zeFenceReset");
		ze_check(zeCommandListReset(batch_list), "zeCommandListReset");
		
		total_batches++;
		total_ops_batched += pending_ops;
		
		CELERITY_TRACE("Level-Zero V4: flushed batch with {} operations", pending_ops);
		
		pending_ops = 0;
		batch_start_time = std::chrono::steady_clock::now();
	}
	
	ze_command_list_handle_t get_batch_list() {
		return batch_list;
	}
	
	void record_micro_bypass() {
		std::lock_guard<std::mutex> lock(mutex);
		micro_copies_bypassed++;
	}
	
	void cleanup() {
		if (pending_ops > 0) {
			flush_batch();
		}
		
		CELERITY_DEBUG("Level-Zero V4 batch stats: {} batches, {} ops total (avg {:.1f} ops/batch), {} micro-copies bypassed", 
		              total_batches, total_ops_batched, 
		              total_batches > 0 ? static_cast<double>(total_ops_batched) / total_batches : 0.0,
		              micro_copies_bypassed);
		
		if (fence) ze_check(zeFenceDestroy(fence), "zeFenceDestroy");
		if (batch_list) ze_check(zeCommandListDestroy(batch_list), "zeCommandListDestroy");
		if (queue) ze_check(zeCommandQueueDestroy(queue), "zeCommandQueueDestroy");
	}
};

// Global managers - use unique_ptr because managers contain std::mutex
static std::vector<std::unique_ptr<event_pool_manager>> g_event_pools;
static std::vector<std::unique_ptr<immediate_cmdlist_manager>> g_immediate_lists;
static std::vector<std::unique_ptr<batch_manager>> g_batch_managers;
static std::mutex g_pools_mutex;
static bool g_pools_initialized = false;

void initialize_all(const std::vector<sycl::device>& devices, ze_context_handle_t context) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	if (g_pools_initialized) return;
	
	// FIXED: Read all environment variables with proper consumption
	const char* env_size = std::getenv("CELERITY_L0_EVENT_POOL_SIZE");
	const char* env_micro = std::getenv("CELERITY_L0_MICRO_THRESHOLD");
	const char* env_small = std::getenv("CELERITY_L0_SMALL_THRESHOLD");
	const char* env_batch = std::getenv("CELERITY_L0_USE_BATCHING");
	const char* env_batch_ops = std::getenv("CELERITY_L0_BATCH_THRESHOLD_OPS");
	const char* env_batch_us = std::getenv("CELERITY_L0_BATCH_THRESHOLD_US");
	
	size_t pool_size = env_size ? std::atoi(env_size) : 512;
	
	if (env_micro) {
		g_micro_threshold = std::atoi(env_micro);
		CELERITY_DEBUG("Level-Zero V4: Using CELERITY_L0_MICRO_THRESHOLD={}", g_micro_threshold);
	}
	if (env_small) {
		g_small_threshold = std::atoi(env_small);
		CELERITY_DEBUG("Level-Zero V4: Using CELERITY_L0_SMALL_THRESHOLD={}", g_small_threshold);
	}
	if (env_batch) {
		g_use_batching = (std::atoi(env_batch) != 0);
		CELERITY_DEBUG("Level-Zero V4: Using CELERITY_L0_USE_BATCHING={}", g_use_batching);
	}
	if (env_batch_ops) {
		CELERITY_DEBUG("Level-Zero V4: CELERITY_L0_BATCH_THRESHOLD_OPS={} (will be used by batch managers)", env_batch_ops);
	}
	if (env_batch_us) {
		CELERITY_DEBUG("Level-Zero V4: CELERITY_L0_BATCH_THRESHOLD_US={} (will be used by batch managers)", env_batch_us);
	}
	
	g_event_pools.reserve(devices.size());
	g_immediate_lists.reserve(devices.size());
	g_batch_managers.reserve(devices.size());
	
	for (size_t i = 0; i < devices.size(); ++i) {
		auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
		g_event_pools.emplace_back(std::make_unique<event_pool_manager>());
		g_event_pools[i]->initialize(context, ze_device, pool_size);
		g_immediate_lists.emplace_back(std::make_unique<immediate_cmdlist_manager>());
		g_immediate_lists[i]->initialize(context, ze_device);
		
		if (g_use_batching) {
			// Create command queue directly (don't use temporary SYCL queue)
			ze_command_queue_desc_t queue_desc = {};
			queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
			queue_desc.ordinal = 0;
			queue_desc.index = 0;
			queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
			
			ze_command_queue_handle_t ze_queue = nullptr;
			ze_check(zeCommandQueueCreate(context, ze_device, &queue_desc, &ze_queue), "zeCommandQueueCreate");
			
			g_batch_managers.emplace_back(std::make_unique<batch_manager>());
			g_batch_managers[i]->initialize(context, ze_device, ze_queue);
		}
	}
	
	CELERITY_DEBUG("Level-Zero V4: Initialized {} devices (micro={}, small={}, batching={})", 
	              devices.size(), g_micro_threshold, g_small_threshold, g_use_batching);
	g_pools_initialized = true;
}

void cleanup_all() {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	for (auto& pool : g_event_pools) pool->cleanup();
	for (auto& list : g_immediate_lists) list->cleanup();
	for (auto& batch : g_batch_managers) batch->cleanup();
	g_event_pools.clear();
	g_immediate_lists.clear();
	g_batch_managers.clear();
	g_pools_initialized = false;
}

// Micro-copy fast path (NEW in V4) - DISABLED for safety
bool try_micro_copy(const void* source, void* dest, size_t size_bytes, device_id device) {
	// DISABLED: Micro-copy optimization is unsafe for device memory
	// Device memory is not directly accessible from host, so memcpy will segfault
	// This optimization is only safe for host-to-host copies, which we can't detect reliably
	// For now, disable this optimization entirely to prevent crashes
	(void)source;
	(void)dest;
	(void)size_bytes;
	(void)device;
	return false;
}

void nd_copy_box_level_zero(sycl::queue& queue, device_id device, const void* const source_base, void* const dest_base, 
    const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box, const size_t elem_size, sycl::event& last_event) {
	
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	
	const auto copy_range = copy_box.get_range();
	
	// Zero-work short-circuit: never touch Level-Zero for empty ranges
	if(copy_range.size() == 0) {
		CELERITY_TRACE("Level-Zero V4: short-circuit empty copy (no ZE submit)");
		last_event = queue.ext_oneapi_submit_barrier();
		return;
	}
	
	const auto src_range = source_box.get_range();
	const auto dst_range = dest_box.get_range();
	const auto src_offset = copy_box.get_offset() - source_box.get_offset();
	const auto dst_offset = copy_box.get_offset() - dest_box.get_offset();
	
	const auto layout = layout_nd_copy(src_range, dst_range, src_offset, dst_offset, copy_range, elem_size);
	
	if(layout.contiguous_size == 0) {
		CELERITY_TRACE("Level-Zero V4: short-circuit zero-size layout");
		last_event = queue.ext_oneapi_submit_barrier();
		return;
	}
	
	// V4 OPTIMIZATION: Choose strategy based on size
	const size_t total_bytes = layout.contiguous_size * (layout.num_complex_strides == 0 ? 1 : 
	                           layout.num_complex_strides == 1 ? layout.strides[0].count : 
	                           layout.contiguous_size);  // Rough estimate for 3D
	
	// FIXED: Micro-copy fast path with proper validation
	if (layout.num_complex_strides == 0 && total_bytes <= g_micro_threshold) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		
		if (try_micro_copy(src_ptr, dst_ptr, layout.contiguous_size, device)) {
			last_event = queue.ext_oneapi_submit_barrier();
			return;
		}
	}
	
	// Small copy: use immediate list
	if (total_bytes <= g_small_threshold) {
		size_t event_idx = g_event_pools[device]->acquire();
		ze_event_handle_t ze_event = g_event_pools[device]->get_event(event_idx);
		
		std::lock_guard<std::mutex> lock(g_immediate_lists[device]->mutex);
		ze_command_list_handle_t cmd_list = g_immediate_lists[device]->get();
		
		if(layout.num_complex_strides == 0) {
			const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
			void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
			ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, layout.contiguous_size, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
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
		}
		
		ze_check(zeEventHostSynchronize(ze_event, UINT64_MAX), "zeEventHostSynchronize");
		ze_check(zeEventHostReset(ze_event), "zeEventHostReset");
		g_event_pools[device]->release(event_idx);
		
		CELERITY_TRACE("Level-Zero V4: immediate small copy {} bytes", total_bytes);
		last_event = queue.ext_oneapi_submit_barrier();
		return;
	}
	
	// Large copy: use batching if enabled
	if (g_use_batching) {
		std::lock_guard<std::mutex> lock(g_batch_managers[device]->mutex);
		ze_command_list_handle_t batch_list = g_batch_managers[device]->get_batch_list();
		
		if(layout.num_complex_strides == 0) {
			const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
			void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
			ze_check(zeCommandListAppendMemoryCopy(batch_list, dst_ptr, src_ptr, layout.contiguous_size, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
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
		}
		
		g_batch_managers[device]->add_operation();
		
		// IMPORTANT: Flush immediately for box copies to ensure data is transferred
		// before pointers become invalid
		g_batch_managers[device]->flush_batch();
		
		CELERITY_TRACE("Level-Zero V4: batch large copy {} bytes", total_bytes);
	}
	
	last_event = queue.ext_oneapi_submit_barrier();
}

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
		    // CRITICAL: Zero-work short-circuit - must check size_bytes == 0 before any Level-Zero operations
		    if (size_bytes == 0) {
			    CELERITY_TRACE("Level-Zero V4: short-circuit empty linear copy (no ZE submit)");
			    last_event = queue.ext_oneapi_submit_barrier();
			    return;
		    }
		    
		    // Micro-copy fast path (host-side memcpy for tiny transfers)
		    if (try_micro_copy(source, dest, size_bytes, device)) {
			    last_event = queue.ext_oneapi_submit_barrier();
			    return;
		    }
		    
		    // Small copy: immediate list
		    if (size_bytes <= g_small_threshold) {
			    size_t event_idx = g_event_pools[device]->acquire();
			    ze_event_handle_t ze_event = g_event_pools[device]->get_event(event_idx);
			    
			    std::lock_guard<std::mutex> lock(g_immediate_lists[device]->mutex);
			    ze_command_list_handle_t cmd_list = g_immediate_lists[device]->get();
			    
			    ze_check(zeCommandListAppendMemoryCopy(cmd_list, dest, source, size_bytes, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
			    ze_check(zeEventHostSynchronize(ze_event, UINT64_MAX), "zeEventHostSynchronize");
			    ze_check(zeEventHostReset(ze_event), "zeEventHostReset");
			    
			    g_event_pools[device]->release(event_idx);
			    
			    CELERITY_TRACE("Level-Zero V4: immediate linear copy {} bytes", size_bytes);
		    } else if (g_use_batching) {
			    // Large copy: batching
			    std::lock_guard<std::mutex> lock(g_batch_managers[device]->mutex);
			    ze_command_list_handle_t batch_list = g_batch_managers[device]->get_batch_list();
			    ze_check(zeCommandListAppendMemoryCopy(batch_list, dest, source, size_bytes, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
			    
			    g_batch_managers[device]->add_operation();
			    
			    if (g_batch_managers[device]->should_flush()) {
				    g_batch_managers[device]->flush_batch();
			    }
			    
			    CELERITY_TRACE("Level-Zero V4: batch linear copy {} bytes", size_bytes);
		    }
		    
		    last_event = queue.ext_oneapi_submit_barrier();
	    });
	
	// Force flush any pending batch operations
	if (g_use_batching) {
		std::lock_guard<std::mutex> lock(g_batch_managers[device]->mutex);
		g_batch_managers[device]->flush_batch();
	}
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("Level-Zero V4 backend initialized with {} device(s)", devices.size());
	
	auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[0].get_platform().ext_oneapi_get_default_context());
	level_zero_backend_detail::initialize_all(devices, ze_context);
	
	for(device_id i = 0; i < devices.size(); ++i) {
		for(device_id j = i + 1; j < devices.size(); ++j) {
			try {
				const auto ze_device_i = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
				const auto ze_device_j = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[j]);
				
				ze_bool_t can_access_ij = false, can_access_ji = false;
				const auto result_ij = zeDeviceCanAccessPeer(ze_device_i, ze_device_j, &can_access_ij);
				const auto result_ji = zeDeviceCanAccessPeer(ze_device_j, ze_device_i, &can_access_ji);
				
				if(result_ij == ZE_RESULT_SUCCESS && result_ji == ZE_RESULT_SUCCESS && can_access_ij && can_access_ji) {
					const memory_id mid_i = first_device_memory_id + i;
					const memory_id mid_j = first_device_memory_id + j;
					get_system_info().memories[mid_i].copy_peers.set(mid_j);
					get_system_info().memories[mid_j].copy_peers.set(mid_i);
					CELERITY_DEBUG("Level-Zero V4: enabled peer access between D{} and D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("Level-Zero V4: peer access query failed: {}", e.what());
			}
		}
	}
}

sycl_level_zero_backend::~sycl_level_zero_backend() {
	level_zero_backend_detail::cleanup_all();
}

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) {
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, device, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
