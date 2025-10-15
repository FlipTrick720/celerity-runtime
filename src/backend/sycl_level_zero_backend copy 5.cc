//Version: v5_Automatic_Batching
//Text: Automatically batch multiple copy operations before synchronizing. Queue-based batching with configurable batch size and timeout.

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
#include <chrono>
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
// VARIANT 5: Automatic Batching System
// ============================================================================

// Configuration for batching behavior
struct batching_config {
	size_t max_batch_size = 64;           // Max operations before forced sync
	size_t small_transfer_threshold = 8192; // Transfers < 8KB are "small"
	size_t large_batch_size = 32;         // Batch size for large transfers
	bool enable_adaptive = true;           // Adapt batch size based on transfer size
};

// Pending copy operation
struct pending_copy {
	ze_command_list_handle_t cmdlist;
	ze_event_handle_t event;
	ze_event_pool_handle_t event_pool;
	sycl::event sycl_event;
	size_t transfer_size;
};

// Per-device batching queue
class batching_queue {
public:
	batching_queue(ze_context_handle_t ctx, ze_device_handle_t dev, const batching_config& config)
	    : m_ctx(ctx), m_dev(dev), m_config(config) {
		// Create event pool
		ze_event_pool_desc_t pool_desc = {};
		pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		pool_desc.count = 1024;
		
		ze_check(zeEventPoolCreate(m_ctx, &pool_desc, 1, &m_dev, &m_event_pool), "zeEventPoolCreate");
		
		CELERITY_DEBUG("[V5-Batch] Batching queue created (max_batch={}, threshold={}KB)",
		               config.max_batch_size, config.small_transfer_threshold / 1024);
	}
	
	~batching_queue() {
		flush_all();
		
		// Clean up events
		for(auto event : m_all_events) {
			zeEventDestroy(event);
		}
		
		// Clean up command lists
		for(auto cl : m_all_cmdlists) {
			zeCommandListDestroy(cl);
		}
		
		if(m_event_pool) {
			zeEventPoolDestroy(m_event_pool);
		}
		
		CELERITY_DEBUG("[V5-Batch] Batching queue destroyed (total_ops={}, batches={}, avg_batch_size={})",
		               m_total_ops, m_batch_count, m_batch_count > 0 ? m_total_ops / m_batch_count : 0);
	}
	
	// Add a copy operation to the batch
	void enqueue_copy(ze_command_list_handle_t cmdlist, ze_event_handle_t event, 
	                  ze_event_pool_handle_t event_pool, sycl::event sycl_evt, size_t transfer_size) {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		pending_copy copy;
		copy.cmdlist = cmdlist;
		copy.event = event;
		copy.event_pool = event_pool;
		copy.sycl_event = sycl_evt;
		copy.transfer_size = transfer_size;
		
		m_pending.push_back(copy);
		m_total_ops++;
		
		// Determine if we should flush based on batch size
		size_t batch_threshold = get_batch_threshold(transfer_size);
		
		if(m_pending.size() >= batch_threshold) {
			flush_locked();
		}
	}
	
	// Flush all pending operations
	void flush_all() {
		std::lock_guard<std::mutex> lock(m_mutex);
		flush_locked();
	}
	
	// Get statistics
	void log_stats() const {
		CELERITY_DEBUG("[V5-Batch] Stats: total_ops={}, batches={}, avg_batch={:.1f}, max_batch={}",
		               m_total_ops, m_batch_count, 
		               m_batch_count > 0 ? static_cast<double>(m_total_ops) / m_batch_count : 0.0,
		               m_max_batch_size);
	}

private:
	void flush_locked() {
		if(m_pending.empty()) return;
		
		// Wait for all events in batch
		for(auto& copy : m_pending) {
			copy.sycl_event.wait();
		}
		
		// Clean up resources
		for(auto& copy : m_pending) {
			zeEventDestroy(copy.event);
			zeEventPoolDestroy(copy.event_pool);
			zeCommandListDestroy(copy.cmdlist);
		}
		
		// Update statistics
		m_batch_count++;
		m_max_batch_size = std::max(m_max_batch_size, m_pending.size());
		
		CELERITY_TRACE("[V5-Batch] Flushed batch of {} operations", m_pending.size());
		
		m_pending.clear();
	}
	
	size_t get_batch_threshold(size_t transfer_size) const {
		if(!m_config.enable_adaptive) {
			return m_config.max_batch_size;
		}
		
		// Adaptive batching: smaller batches for large transfers
		if(transfer_size < m_config.small_transfer_threshold) {
			return m_config.max_batch_size;
		} else {
			return m_config.large_batch_size;
		}
	}

	ze_context_handle_t m_ctx;
	ze_device_handle_t m_dev;
	ze_event_pool_handle_t m_event_pool = nullptr;
	batching_config m_config;
	
	std::deque<pending_copy> m_pending;
	std::mutex m_mutex;
	
	// Statistics
	size_t m_total_ops = 0;
	size_t m_batch_count = 0;
	size_t m_max_batch_size = 0;
	
	// Resource pools (for future optimization)
	std::vector<ze_event_handle_t> m_all_events;
	std::vector<ze_command_list_handle_t> m_all_cmdlists;
};

// Global batching queues
static std::vector<std::unique_ptr<batching_queue>> g_batching_queues;
static std::mutex g_queues_mutex;
static batching_config g_batching_config;

void initialize_batching(const std::vector<sycl::device>& devices, const batching_config& config = batching_config{}) {
	std::lock_guard<std::mutex> lock(g_queues_mutex);
	g_batching_queues.clear();
	g_batching_config = config;
	
	for(const auto& device : devices) {
		auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.get_platform().ext_oneapi_get_default_context());
		auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
		g_batching_queues.push_back(std::make_unique<batching_queue>(ze_ctx, ze_dev, config));
	}
	
	CELERITY_DEBUG("[V5-Batch] Initialized {} batching queues", devices.size());
}

void cleanup_batching() {
	std::lock_guard<std::mutex> lock(g_queues_mutex);
	
	// Log statistics before cleanup
	for(size_t i = 0; i < g_batching_queues.size(); ++i) {
		CELERITY_DEBUG("[V5-Batch] Device {} statistics:", i);
		g_batching_queues[i]->log_stats();
	}
	
	g_batching_queues.clear();
	CELERITY_DEBUG("[V5-Batch] Cleaned up batching queues");
}

batching_queue& get_batching_queue(device_id device) {
	std::lock_guard<std::mutex> lock(g_queues_mutex);
	if(device >= g_batching_queues.size()) {
		utils::panic("[V5-Batch] Invalid device ID for batching queue: {}", device);
	}
	return *g_batching_queues[device];
}

// ============================================================================
// Copy Functions (with automatic batching)
// ============================================================================

// Helper to create Level Zero event
std::pair<ze_event_handle_t, ze_event_pool_handle_t> create_level_zero_event(ze_context_handle_t context, ze_device_handle_t device) {
	ze_event_pool_desc_t pool_desc = {};
	pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
	pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
	pool_desc.count = 1;
	
	ze_event_pool_handle_t event_pool = nullptr;
	ze_check(zeEventPoolCreate(context, &pool_desc, 1, &device, &event_pool), "zeEventPoolCreate");
	
	ze_event_desc_t event_desc = {};
	event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
	event_desc.index = 0;
	event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
	event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
	
	ze_event_handle_t event = nullptr;
	ze_check(zeEventCreate(event_pool, &event_desc, &event), "zeEventCreate");
	
	return {event, event_pool};
}

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
	auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
	auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
	
	// Create event and command list
	auto [ze_event, ze_event_pool] = create_level_zero_event(ze_context, ze_device);
	
	ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
	ze_command_list_handle_t cmd_list = nullptr;
	ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
	
	size_t transfer_size = layout.contiguous_size;
	
	if(layout.num_complex_strides == 0) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, layout.contiguous_size, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
		CELERITY_TRACE("[V5-Batch] Level-Zero backend: contiguous copy {} bytes", layout.contiguous_size);
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
		
		transfer_size = width * height;
		CELERITY_TRACE("[V5-Batch] Level-Zero backend: 2D copy {}x{} bytes", width, height);
	} else {
		std::vector<std::tuple<size_t, size_t, size_t>> chunks;
		for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t size) {
			chunks.emplace_back(src_off, dst_off, size);
		});
		
		size_t total_size = 0;
		for(size_t i = 0; i < chunks.size(); ++i) {
			const auto& [src_off, dst_off, size] = chunks[i];
			const void* src_ptr = static_cast<const char*>(source_base) + src_off;
			void* dst_ptr = static_cast<char*>(dest_base) + dst_off;
			
			const bool is_last = (i == chunks.size() - 1);
			ze_event_handle_t event_to_use = is_last ? ze_event : nullptr;
			
			ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, size, event_to_use, 0, nullptr), "zeCommandListAppendMemoryCopy");
			total_size += size;
		}
		
		transfer_size = total_size;
		CELERITY_TRACE("[V5-Batch] Level-Zero backend: 3D copy {} chunks, {} bytes total", chunks.size(), total_size);
	}
	
	ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
	ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
	
	// Create SYCL event
	last_event = queue.ext_oneapi_submit_barrier();
	
	// Add to batching queue (will sync when batch is full)
	get_batching_queue(device).enqueue_copy(cmd_list, ze_event, ze_event_pool, last_event, transfer_size);
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
		    CELERITY_TRACE("[V5-Batch] Level-Zero backend: linear copy {} bytes", size_bytes);
		    
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
		    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
		    
		    auto [ze_event, ze_event_pool] = create_level_zero_event(ze_context, ze_device);
		    
		    ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
		    ze_command_list_handle_t cmd_list = nullptr;
		    ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
		    ze_check(zeCommandListAppendMemoryCopy(cmd_list, dest, source, size_bytes, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
		    ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
		    ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
		    
		    last_event = queue.ext_oneapi_submit_barrier();
		    
		    // Add to batching queue
		    get_batching_queue(device).enqueue_copy(cmd_list, ze_event, ze_event_pool, last_event, size_bytes);
	    });
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("[V5-Batch] Level-Zero backend initialized with {} device(s)", devices.size());
	
	// Initialize batching system with default config
	level_zero_backend_detail::batching_config batch_config;
	batch_config.max_batch_size = 64;              // Batch up to 64 small operations
	batch_config.small_transfer_threshold = 8192;  // 8 KB threshold
	batch_config.large_batch_size = 16;            // Smaller batches for large transfers
	batch_config.enable_adaptive = true;           // Enable adaptive batching
	
	level_zero_backend_detail::initialize_batching(devices, batch_config);
	
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
					CELERITY_DEBUG("[V5-Batch] Level-Zero backend: enabled peer access between D{} and D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("[V5-Batch] Level-Zero backend: failed to query peer access: {}", e.what());
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
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, device, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
