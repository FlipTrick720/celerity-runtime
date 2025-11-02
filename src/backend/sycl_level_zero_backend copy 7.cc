//Version: v7_async_fence_optimized
//Text: Async execution with fence tracking, double-buffered command lists, lazy sync
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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>
#include <vector>
#include <chrono>
#include <unordered_map>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace celerity::detail::level_zero_backend_detail {

static inline void ze_check(ze_result_t r, const char* where) {
	if(r != ZE_RESULT_SUCCESS) utils::panic("Level-Zero error in {}: code={}", where, static_cast<int>(r));
}

// ============================================================================
// Configuration
// ============================================================================
static inline size_t env_or(const char* name, size_t def) {
	if(const char* v = std::getenv(name)) {
		try { return static_cast<size_t>(std::stoull(v)); }
		catch(...) {}
	}
	return def;
}

static const size_t g_pool_size = env_or("CELERITY_L0_EVENT_POOL_SIZE", 1024);
static const size_t g_batch_small_ops = env_or("CELERITY_L0_BATCH_SMALL_OPS", 64);
static const size_t g_batch_medium_ops = env_or("CELERITY_L0_BATCH_MEDIUM_OPS", 16);
static const size_t g_batch_large_ops = env_or("CELERITY_L0_BATCH_LARGE_OPS", 4);
static const size_t g_batch_timeout_us = env_or("CELERITY_L0_BATCH_TIMEOUT_US", 100);

// Size thresholds for batching strategy
static constexpr size_t SMALL_COPY_THRESHOLD = 4096;      // 4KB
static constexpr size_t MEDIUM_COPY_THRESHOLD = 1048576;  // 1MB

// ============================================================================
// Event Pool: Thread-safe persistent event management
// ============================================================================
struct event_pool {
	ze_event_pool_handle_t pool = nullptr;
	std::vector<ze_event_handle_t> events;
	std::queue<size_t> free_idx;
	std::mutex mtx;
	size_t peak_usage = 0;
	size_t total_acquires = 0;

	void init(ze_context_handle_t ctx, ze_device_handle_t dev, size_t count) {
		ze_event_pool_desc_t d{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
		d.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		d.count = static_cast<uint32_t>(count);
		ze_check(zeEventPoolCreate(ctx, &d, 1, &dev, &pool), "zeEventPoolCreate");

		events.resize(count);
		for(size_t i = 0; i < count; ++i) {
			ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC};
			ed.index = static_cast<uint32_t>(i);
			ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
			ze_check(zeEventCreate(pool, &ed, &events[i]), "zeEventCreate");
			free_idx.push(i);
		}
		CELERITY_DEBUG("L0 event pool created: {} events", count);
	}

	size_t acquire() {
		std::lock_guard lk(mtx);
		if(free_idx.empty()) {
			CELERITY_WARN("L0 event pool exhausted (size={}), consider increasing CELERITY_L0_EVENT_POOL_SIZE", events.size());
			utils::panic("Event pool exhausted");
		}
		const auto i = free_idx.front();
		free_idx.pop();
		++total_acquires;
		peak_usage = std::max(peak_usage, events.size() - free_idx.size());
		ze_check(zeEventHostReset(events[i]), "zeEventHostReset");
		return i;
	}

	void release(size_t i) {
		std::lock_guard lk(mtx);
		free_idx.push(i);
	}

	ze_event_handle_t get(size_t i) const { return events[i]; }

	void destroy() {
		CELERITY_DEBUG("L0 event pool stats: peak {}/{}, total acquires {}", peak_usage, events.size(), total_acquires);
		for(auto& e : events) if(e) zeEventDestroy(e);
		if(pool) zeEventPoolDestroy(pool);
	}
};

// ============================================================================
// Async Batch Manager: Double-buffered with fence tracking
// ============================================================================
class async_batch_manager {
public:
	async_batch_manager() = default;
	
	void init(ze_context_handle_t ctx, ze_device_handle_t dev) {
		m_context = ctx;
		m_device = dev;
		
		// Create double-buffered command lists
		ze_command_list_desc_t d{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
		ze_check(zeCommandListCreate(ctx, dev, &d, &m_cl[0]), "zeCommandListCreate");
		ze_check(zeCommandListCreate(ctx, dev, &d, &m_cl[1]), "zeCommandListCreate");
		
		// Create fence for async tracking
		ze_fence_desc_t fd{ZE_STRUCTURE_TYPE_FENCE_DESC};
		ze_check(zeFenceCreate(nullptr, &fd, &m_fence), "zeFenceCreate");
		
		m_batch_start = std::chrono::steady_clock::now();
	}

	// Size-aware batch threshold
	size_t get_batch_threshold(size_t copy_size) const {
		if(copy_size < SMALL_COPY_THRESHOLD) return g_batch_small_ops;
		if(copy_size < MEDIUM_COPY_THRESHOLD) return g_batch_medium_ops;
		return g_batch_large_ops;
	}

	bool should_flush(size_t next_copy_size) const {
		if(m_pending_ops == 0) return false;
		
		const auto threshold = get_batch_threshold(next_copy_size);
		if(m_pending_ops >= threshold) return true;
		
		if(g_batch_timeout_us > 0) {
			const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - m_batch_start).count();
			if(static_cast<size_t>(elapsed) >= g_batch_timeout_us) return true;
		}
		
		return false;
	}

	// Append 1D copy
	void append_copy_1d(const void* src, void* dst, size_t bytes) {
		// Wait for previous batch if needed
		if(m_in_flight) wait_for_completion();
		
		if(m_pending_ops == 0) m_batch_start = std::chrono::steady_clock::now();
		
		auto cl = m_cl[m_current_buffer];
		ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr), 
		         "zeCommandListAppendMemoryCopy");
		++m_pending_ops;
		m_total_bytes += bytes;
	}

	// Append 2D copy (OPTIMIZED)
	void append_copy_2d(const void* src, void* dst, 
	                    size_t width, size_t height, 
	                    size_t src_pitch, size_t dst_pitch) {
		// Wait for previous batch if needed
		if(m_in_flight) wait_for_completion();
		
		if(m_pending_ops == 0) m_batch_start = std::chrono::steady_clock::now();
		
		ze_copy_region_t src_region{};
		src_region.originX = 0;
		src_region.originY = 0;
		src_region.originZ = 0;
		src_region.width = static_cast<uint32_t>(width);
		src_region.height = static_cast<uint32_t>(height);
		src_region.depth = 1;

		ze_copy_region_t dst_region{};
		dst_region.originX = 0;
		dst_region.originY = 0;
		dst_region.originZ = 0;
		dst_region.width = static_cast<uint32_t>(width);
		dst_region.height = static_cast<uint32_t>(height);
		dst_region.depth = 1;

		auto cl = m_cl[m_current_buffer];
		ze_check(zeCommandListAppendMemoryCopyRegion(
			cl, dst, &dst_region, static_cast<uint32_t>(dst_pitch), 0,
			src, &src_region, static_cast<uint32_t>(src_pitch), 0,
			nullptr, 0, nullptr), "zeCommandListAppendMemoryCopyRegion");
		
		++m_pending_ops;
		m_total_bytes += width * height;
	}

	// Submit batch ASYNC (no blocking!)
	void submit_async(ze_command_queue_handle_t zeq) {
		if(m_pending_ops == 0) return;
		
		// Wait for previous batch if still in flight
		if(m_in_flight) wait_for_completion();
		
		auto cl = m_cl[m_current_buffer];
		
		ze_check(zeCommandListClose(cl), "zeCommandListClose");
		ze_check(zeFenceReset(m_fence), "zeFenceReset");
		ze_check(zeCommandQueueExecuteCommandLists(zeq, 1, &cl, m_fence), 
		         "zeCommandQueueExecuteCommandLists");
		
		// Mark as in-flight (DON'T WAIT!)
		m_in_flight = true;
		m_in_flight_queue = zeq;
		
		++m_total_batches;
		m_total_ops += m_pending_ops;
		
		CELERITY_TRACE("L0 batch submitted ASYNC: {} ops, {} bytes", m_pending_ops, m_total_bytes);
		
		m_pending_ops = 0;
		m_total_bytes = 0;
		m_batch_start = std::chrono::steady_clock::now();
		
		// Swap buffers for next batch
		m_current_buffer = 1 - m_current_buffer;
	}

	// Ensure completion (called before SYCL barrier)
	void ensure_completion() {
		if(m_in_flight) {
			wait_for_completion();
		}
	}

	size_t pending_ops() const { return m_pending_ops; }

	void destroy() {
		// Wait for any in-flight operations
		if(m_in_flight) wait_for_completion();
		
		if(m_pending_ops > 0) {
			CELERITY_WARN("L0 batch destroyed with {} pending operations - data loss possible", m_pending_ops);
		}
		
		CELERITY_DEBUG("L0 batch stats: {} batches, {} ops total (avg {:.1f} ops/batch)",
		              m_total_batches, m_total_ops,
		              m_total_batches > 0 ? static_cast<double>(m_total_ops) / m_total_batches : 0.0);
		
		if(m_fence) zeFenceDestroy(m_fence);
		if(m_cl[0]) zeCommandListDestroy(m_cl[0]);
		if(m_cl[1]) zeCommandListDestroy(m_cl[1]);
	}

private:
	ze_context_handle_t m_context = nullptr;
	ze_device_handle_t m_device = nullptr;
	ze_command_list_handle_t m_cl[2] = {nullptr, nullptr};  // Double-buffered
	ze_fence_handle_t m_fence = nullptr;
	int m_current_buffer = 0;
	
	bool m_in_flight = false;
	ze_command_queue_handle_t m_in_flight_queue = nullptr;
	
	size_t m_pending_ops = 0;
	size_t m_total_bytes = 0;
	std::chrono::steady_clock::time_point m_batch_start;
	
	size_t m_total_batches = 0;
	size_t m_total_ops = 0;

	void wait_for_completion() {
		if(!m_in_flight) return;
		
		// Wait on fence (only blocks when necessary)
		ze_check(zeFenceHostSynchronize(m_fence, UINT64_MAX), "zeFenceHostSynchronize");
		
		// Reset the command list that just completed
		auto completed_cl = m_cl[1 - m_current_buffer];
		ze_check(zeCommandListReset(completed_cl), "zeCommandListReset");
		
		m_in_flight = false;
		m_in_flight_queue = nullptr;
	}
};

// ============================================================================
// Per-Lane State: Each SYCL queue lane gets its own batch manager
// ============================================================================
struct lane_state {
	async_batch_manager batch;
	std::mutex mtx;  // Protects batch operations
	
	void init(ze_context_handle_t ctx, ze_device_handle_t dev) {
		batch.init(ctx, dev);
	}
	
	void destroy() {
		batch.destroy();
	}
};

// ============================================================================
// Per-Device State
// ============================================================================
struct device_state {
	event_pool pool;
	ze_context_handle_t context = nullptr;
	ze_device_handle_t device = nullptr;
	
	// Lane-specific state (indexed by device_lane from enqueue_device_copy)
	std::unordered_map<size_t, std::unique_ptr<lane_state>> lanes;
	std::mutex lanes_mtx;
	
	void init(ze_context_handle_t ctx, ze_device_handle_t dev, size_t pool_size) {
		context = ctx;
		device = dev;
		pool.init(ctx, dev, pool_size);
	}
	
	lane_state& get_or_create_lane(size_t lane_id) {
		std::lock_guard lk(lanes_mtx);
		auto it = lanes.find(lane_id);
		if(it == lanes.end()) {
			auto ls = std::make_unique<lane_state>();
			ls->init(context, device);
			it = lanes.emplace(lane_id, std::move(ls)).first;
			CELERITY_DEBUG("L0 lane state created: device={}, lane={}", static_cast<const void*>(device), lane_id);
		}
		return *it->second;
	}
	
	void destroy() {
		for(auto& [lid, ls] : lanes) {
			ls->destroy();
		}
		lanes.clear();
		pool.destroy();
	}
};

// ============================================================================
// Backend Implementation State
// ============================================================================
struct backend_impl {
	std::vector<std::unique_ptr<device_state>> devices;
	
	void init(const std::vector<sycl::device>& sycl_devices, ze_context_handle_t ctx) {
		devices.reserve(sycl_devices.size());
		for(size_t i = 0; i < sycl_devices.size(); ++i) {
			auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_devices[i]);
			auto ds = std::make_unique<device_state>();
			ds->init(ctx, zedev, g_pool_size);
			devices.push_back(std::move(ds));
		}
	}
	
	void destroy() {
		for(auto& ds : devices) {
			ds->destroy();
		}
		devices.clear();
	}
};

// ============================================================================
// Copy Engine: Handles all copy operations with 2D/3D optimization
// ============================================================================
class copy_engine {
public:
	explicit copy_engine(backend_impl* impl) : m_impl(impl) {}

	void execute_copy(sycl::queue& sq, device_id dev_id, size_t lane_id,
	                  const void* src_base, void* dst_base,
	                  const region_layout& src_layout, const region_layout& dst_layout,
	                  const region<3>& copy_region, size_t elem_size) {
		
		// Extract SYCL's native Level-Zero queue (maintains ordering!)
		auto zeq_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sq);
		auto zeq = std::get<ze_command_queue_handle_t>(zeq_variant);
		
		auto& dev_state = *m_impl->devices[dev_id];
		auto& lane = dev_state.get_or_create_lane(lane_id);
		
		std::lock_guard lk(lane.mtx);
		
		// Dispatch with optimized 2D/3D paths
		dispatch_nd_region_copy(
			src_base, dst_base, src_layout, dst_layout, copy_region, elem_size,
			[&](const void* src, void* dst, const box<3>& src_box, 
			    const box<3>& dst_box, const box<3>& copy_box) {
				execute_box_copy(lane, zeq, src, dst, src_box, dst_box, copy_box, elem_size);
			},
			[&](const void* src, void* dst, size_t bytes) {
				execute_linear_copy(lane, zeq, src, dst, bytes);
			});
		
		// Final flush for this copy operation (async)
		flush_lane(lane, zeq);
		
		// CRITICAL: Ensure completion before SYCL barrier
		lane.batch.ensure_completion();
	}

private:
	backend_impl* m_impl;

	void execute_box_copy(lane_state& lane, ze_command_queue_handle_t zeq,
	                      const void* src_base, void* dst_base,
	                      const box<3>& src_box, const box<3>& dst_box,
	                      const box<3>& copy_box, size_t elem_size) {
		
		assert(src_box.covers(copy_box));
		assert(dst_box.covers(copy_box));
		
		const auto layout = layout_nd_copy(
			src_box.get_range(), dst_box.get_range(),
			copy_box.get_offset() - src_box.get_offset(),
			copy_box.get_offset() - dst_box.get_offset(),
			copy_box.get_range(), elem_size);
		
		if(layout.contiguous_size == 0) return;
		
		// Optimization: Use native 2D copy for strided layouts
		if(layout.num_complex_strides == 1) {
			const auto& stride = layout.strides[0];
			const size_t width = layout.contiguous_size;
			const size_t height = stride.count;
			
			const void* src_ptr = static_cast<const std::byte*>(src_base) + layout.offset_in_source;
			void* dst_ptr = static_cast<std::byte*>(dst_base) + layout.offset_in_dest;
			
			// Check if we should flush before this operation
			if(lane.batch.should_flush(width * height)) {
				lane.batch.submit_async(zeq);
			}
			
			lane.batch.append_copy_2d(src_ptr, dst_ptr, width, height, 
			                         stride.source_stride, stride.dest_stride);
			
			CELERITY_TRACE("L0 2D copy: {}x{} bytes (src_pitch={}, dst_pitch={})",
			              width, height, stride.source_stride, stride.dest_stride);
			return;
		}
		
		// Fallback: 1D chunks for 0-stride and 2-stride cases
		for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t chunk) {
			const void* src_ptr = static_cast<const std::byte*>(src_base) + src_off;
			void* dst_ptr = static_cast<std::byte*>(dst_base) + dst_off;
			
			if(lane.batch.should_flush(chunk)) {
				lane.batch.submit_async(zeq);
			}
			
			lane.batch.append_copy_1d(src_ptr, dst_ptr, chunk);
		});
	}

	void execute_linear_copy(lane_state& lane, ze_command_queue_handle_t zeq,
	                        const void* src, void* dst, size_t bytes) {
		if(bytes == 0) return;
		
		if(lane.batch.should_flush(bytes)) {
			lane.batch.submit_async(zeq);
		}
		
		lane.batch.append_copy_1d(src, dst, bytes);
	}

	void flush_lane(lane_state& lane, ze_command_queue_handle_t zeq) {
		// Submit any pending operations (async, non-blocking)
		lane.batch.submit_async(zeq);
	}
};

} // namespace celerity::detail::level_zero_backend_detail

// ============================================================================
// Public Backend Interface
// ============================================================================
namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(
	const std::vector<sycl::device>& devices,
	const sycl_backend::configuration& config)
	: sycl_backend(devices, config)
{
	using namespace level_zero_backend_detail;
	
	// Initialize implementation state
	auto impl = std::make_unique<backend_impl>();
	
	auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
		devices[0].get_platform().ext_oneapi_get_default_context());
	
	impl->init(devices, ctx);
	
	// Store as opaque pointer (to avoid exposing internal types in header)
	m_l0_impl = impl.release();
	
	// Peer access detection
	for(device_id i = 0; i < devices.size(); ++i) {
		for(device_id j = i + 1; j < devices.size(); ++j) {
			try {
				const auto di = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
				const auto dj = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[j]);
				
				ze_bool_t ij = false, ji = false;
				const auto rij = zeDeviceCanAccessPeer(di, dj, &ij);
				const auto rji = zeDeviceCanAccessPeer(dj, di, &ji);
				
				if(rij == ZE_RESULT_SUCCESS && rji == ZE_RESULT_SUCCESS && ij && ji) {
					const memory_id mi = first_device_memory_id + i;
					const memory_id mj = first_device_memory_id + j;
					get_system_info().memories[mi].copy_peers.set(mj);
					get_system_info().memories[mj].copy_peers.set(mi);
					CELERITY_DEBUG("L0 peer access enabled: D{} <-> D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("L0 peer access query failed: {}", e.what());
			}
		}
	}
	
	CELERITY_DEBUG("L0 v7 async backend initialized: {} device(s), pool_size={}, batch_small={}, batch_medium={}, batch_large={}, timeout={}us",
	              devices.size(), g_pool_size, g_batch_small_ops, g_batch_medium_ops, g_batch_large_ops, g_batch_timeout_us);
}

sycl_level_zero_backend::~sycl_level_zero_backend() {
	if(m_l0_impl) {
		auto impl = static_cast<level_zero_backend_detail::backend_impl*>(m_l0_impl);
		impl->destroy();
		delete impl;
		m_l0_impl = nullptr;
	}
}

async_event sycl_level_zero_backend::enqueue_device_copy(
	device_id device, size_t device_lane,
	const void* src_base, void* dst_base,
	const region_layout& src_layout, const region_layout& dst_layout,
	const region<3>& copy_region, size_t elem_size)
{
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& sq) {
		using namespace level_zero_backend_detail;
		
		// Get backend implementation (thread-safe, immutable after construction)
		auto impl = static_cast<backend_impl*>(m_l0_impl);
		
		// Create copy engine (lightweight, no state)
		copy_engine engine(impl);
		
		// Execute copy with proper SYCL queue ordering
		engine.execute_copy(sq, device, device_lane, 
		                   src_base, dst_base, 
		                   src_layout, dst_layout, 
		                   copy_region, elem_size);
		
		// Return SYCL barrier event (maintains dependency chain)
		sycl_backend_detail::flush(sq);
		auto event = sq.ext_oneapi_submit_barrier();
		
		return make_async_event<sycl_backend_detail::sycl_event>(
			std::move(event), is_profiling_enabled());
	});
}

} // namespace celerity::detail
