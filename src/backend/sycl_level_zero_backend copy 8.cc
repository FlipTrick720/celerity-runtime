//Version: v8_aggressive_batch
//Text: Aggressive batching with deferred synchronization and persistent command lists

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
// NEW: Much more aggressive batching
static const size_t g_batch_ops = env_or("CELERITY_L0_BATCH_OPS", 128);  // Batch up to 128 ops
static const size_t g_batch_timeout_us = env_or("CELERITY_L0_BATCH_TIMEOUT_US", 500);  // 500us timeout

// ============================================================================
// Event Pool
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
		CELERITY_DEBUG("L0 v7: event pool created: {} events", count);
	}

	size_t acquire() {
		std::lock_guard lk(mtx);
		if(free_idx.empty()) {
			CELERITY_WARN("L0 event pool exhausted (size={})", events.size());
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
		CELERITY_DEBUG("L0 v7: event pool stats: peak {}/{}, total {}", peak_usage, events.size(), total_acquires);
		for(auto& e : events) if(e) zeEventDestroy(e);
		if(pool) zeEventPoolDestroy(pool);
	}
};

// ============================================================================
// Batch Manager: Aggressive batching with deferred sync
// ============================================================================
class batch_manager {
public:
	batch_manager() = default;
	
	void init(ze_context_handle_t ctx, ze_device_handle_t dev, ze_command_queue_handle_t q) {
		m_context = ctx;
		m_device = dev;
		m_queue = q;
		
		// Create persistent command list
		ze_command_list_desc_t d{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
		ze_check(zeCommandListCreate(ctx, dev, &d, &m_cl), "zeCommandListCreate");
		
		m_batch_start = std::chrono::steady_clock::now();
		
		CELERITY_DEBUG("L0 v7: batch manager initialized (ops={}, timeout={}us)", g_batch_ops, g_batch_timeout_us);
	}

	bool should_flush() const {
		if(m_pending_ops >= g_batch_ops) return true;
		
		if(g_batch_timeout_us > 0 && m_pending_ops > 0) {
			const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::steady_clock::now() - m_batch_start).count();
			if(static_cast<size_t>(elapsed) >= g_batch_timeout_us) return true;
		}
		
		return false;
	}

	// Append 1D copy
	void append_copy_1d(const void* src, void* dst, size_t bytes) {
		if(m_pending_ops == 0) m_batch_start = std::chrono::steady_clock::now();
		ze_check(zeCommandListAppendMemoryCopy(m_cl, dst, src, bytes, nullptr, 0, nullptr), 
		         "zeCommandListAppendMemoryCopy");
		++m_pending_ops;
		m_total_bytes += bytes;
	}

	// Append 2D copy
	void append_copy_2d(const void* src, void* dst, 
	                    size_t width, size_t height, 
	                    size_t src_pitch, size_t dst_pitch) {
		if(m_pending_ops == 0) m_batch_start = std::chrono::steady_clock::now();
		
		ze_copy_region_t src_region{};
		src_region.width = static_cast<uint32_t>(width);
		src_region.height = static_cast<uint32_t>(height);
		src_region.depth = 1;

		ze_copy_region_t dst_region{};
		dst_region.width = static_cast<uint32_t>(width);
		dst_region.height = static_cast<uint32_t>(height);
		dst_region.depth = 1;

		ze_check(zeCommandListAppendMemoryCopyRegion(
			m_cl, dst, &dst_region, static_cast<uint32_t>(dst_pitch), 0,
			src, &src_region, static_cast<uint32_t>(src_pitch), 0,
			nullptr, 0, nullptr), "zeCommandListAppendMemoryCopyRegion");
		
		++m_pending_ops;
		m_total_bytes += width * height;
	}

	// NEW: Submit batch WITHOUT sync (async)
	void submit_batch_async() {
		if(m_pending_ops == 0) return;
		
		ze_check(zeCommandListClose(m_cl), "zeCommandListClose");
		ze_check(zeCommandQueueExecuteCommandLists(m_queue, 1, &m_cl, nullptr), 
		         "zeCommandQueueExecuteCommandLists");
		
		// NO SYNC - let operations run in background
		
		ze_check(zeCommandListReset(m_cl), "zeCommandListReset");
		
		++m_total_batches;
		m_total_ops += m_pending_ops;
		
		CELERITY_TRACE("L0 v7: batch submitted async: {} ops, {} bytes", m_pending_ops, m_total_bytes);
		
		m_pending_ops = 0;
		m_total_bytes = 0;
		m_batch_start = std::chrono::steady_clock::now();
	}

	// NEW: Sync only when needed (end of copy operation)
	void sync() {
		if(m_pending_ops > 0) {
			submit_batch_async();
		}
		// Sync the queue to ensure all submitted batches complete
		ze_check(zeCommandQueueSynchronize(m_queue, UINT64_MAX), "zeCommandQueueSynchronize");
	}

	size_t pending_ops() const { return m_pending_ops; }

	void destroy() {
		if(m_pending_ops > 0) {
			CELERITY_WARN("L0 v7: batch destroyed with {} pending ops", m_pending_ops);
			sync();
		}
		
		CELERITY_DEBUG("L0 v7: batch stats: {} batches, {} ops total (avg {:.1f} ops/batch)",
		              m_total_batches, m_total_ops,
		              m_total_batches > 0 ? static_cast<double>(m_total_ops) / m_total_batches : 0.0);
		
		if(m_cl) zeCommandListDestroy(m_cl);
	}

private:
	ze_context_handle_t m_context = nullptr;
	ze_device_handle_t m_device = nullptr;
	ze_command_queue_handle_t m_queue = nullptr;
	ze_command_list_handle_t m_cl = nullptr;
	
	size_t m_pending_ops = 0;
	size_t m_total_bytes = 0;
	std::chrono::steady_clock::time_point m_batch_start;
	
	size_t m_total_batches = 0;
	size_t m_total_ops = 0;
};

// ============================================================================
// Per-Lane State
// ============================================================================
struct lane_state {
	batch_manager batch;
	std::mutex mtx;
	
	void init(ze_context_handle_t ctx, ze_device_handle_t dev, ze_command_queue_handle_t q) {
		batch.init(ctx, dev, q);
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
	ze_command_queue_handle_t queue = nullptr;  // Single queue per device
	
	std::unordered_map<size_t, std::unique_ptr<lane_state>> lanes;
	std::mutex lanes_mtx;
	
	void init(ze_context_handle_t ctx, ze_device_handle_t dev, size_t pool_size) {
		context = ctx;
		device = dev;
		pool.init(ctx, dev, pool_size);
		
		// Create single command queue per device
		ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
		qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
		ze_check(zeCommandQueueCreate(ctx, dev, &qd, &queue), "zeCommandQueueCreate");
	}
	
	lane_state& get_or_create_lane(size_t lane_id) {
		std::lock_guard lk(lanes_mtx);
		auto it = lanes.find(lane_id);
		if(it == lanes.end()) {
			auto ls = std::make_unique<lane_state>();
			ls->init(context, device, queue);  // Share queue
			it = lanes.emplace(lane_id, std::move(ls)).first;
			CELERITY_DEBUG("L0 v7: lane state created: device={}, lane={}", static_cast<const void*>(device), lane_id);
		}
		return *it->second;
	}
	
	void destroy() {
		for(auto& [lid, ls] : lanes) {
			ls->destroy();
		}
		lanes.clear();
		if(queue) zeCommandQueueDestroy(queue);
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
// Copy Engine
// ============================================================================
class copy_engine {
public:
	explicit copy_engine(backend_impl* impl) : m_impl(impl) {}

	void execute_copy(sycl::queue& sq, device_id dev_id, size_t lane_id,
	                  const void* src_base, void* dst_base,
	                  const region_layout& src_layout, const region_layout& dst_layout,
	                  const region<3>& copy_region, size_t elem_size) {
		
		auto& dev_state = *m_impl->devices[dev_id];
		auto& lane = dev_state.get_or_create_lane(lane_id);
		
		std::lock_guard lk(lane.mtx);
		
		// Dispatch with optimized 2D paths
		dispatch_nd_region_copy(
			src_base, dst_base, src_layout, dst_layout, copy_region, elem_size,
			[&](const void* src, void* dst, const box<3>& src_box, 
			    const box<3>& dst_box, const box<3>& copy_box) {
				execute_box_copy(lane, src, dst, src_box, dst_box, copy_box, elem_size);
			},
			[&](const void* src, void* dst, size_t bytes) {
				execute_linear_copy(lane, src, dst, bytes);
			});
		
		// CRITICAL: Sync at end of copy operation
		lane.batch.sync();
	}

private:
	backend_impl* m_impl;

	void execute_box_copy(lane_state& lane, const void* src_base, void* dst_base,
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
		
		// 2D optimization
		if(layout.num_complex_strides == 1) {
			const auto& stride = layout.strides[0];
			const size_t width = layout.contiguous_size;
			const size_t height = stride.count;
			
			const void* src_ptr = static_cast<const std::byte*>(src_base) + layout.offset_in_source;
			void* dst_ptr = static_cast<std::byte*>(dst_base) + layout.offset_in_dest;
			
			if(lane.batch.should_flush()) {
				lane.batch.submit_batch_async();
			}
			
			lane.batch.append_copy_2d(src_ptr, dst_ptr, width, height, 
			                         stride.source_stride, stride.dest_stride);
			
			CELERITY_TRACE("L0 v7: 2D copy: {}x{} bytes", width, height);
			return;
		}
		
		// Fallback: 1D chunks
		for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t chunk) {
			const void* src_ptr = static_cast<const std::byte*>(src_base) + src_off;
			void* dst_ptr = static_cast<std::byte*>(dst_base) + dst_off;
			
			if(lane.batch.should_flush()) {
				lane.batch.submit_batch_async();
			}
			
			lane.batch.append_copy_1d(src_ptr, dst_ptr, chunk);
		});
	}

	void execute_linear_copy(lane_state& lane, const void* src, void* dst, size_t bytes) {
		if(bytes == 0) return;
		
		if(lane.batch.should_flush()) {
			lane.batch.submit_batch_async();
		}
		
		lane.batch.append_copy_1d(src, dst, bytes);
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
	
	auto impl = std::make_unique<backend_impl>();
	
	auto ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
		devices[0].get_platform().ext_oneapi_get_default_context());
	
	impl->init(devices, ctx);
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
					CELERITY_DEBUG("L0 v7: peer access enabled: D{} <-> D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("L0 v7: peer access query failed: {}", e.what());
			}
		}
	}
	
	CELERITY_DEBUG("L0 v7 backend initialized: {} device(s), pool={}, batch_ops={}, timeout={}us",
	              devices.size(), g_pool_size, g_batch_ops, g_batch_timeout_us);
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
		
		auto impl = static_cast<backend_impl*>(m_l0_impl);
		copy_engine engine(impl);
		
		engine.execute_copy(sq, device, device_lane, 
		                   src_base, dst_base, 
		                   src_layout, dst_layout, 
		                   copy_region, elem_size);
		
		sycl_backend_detail::flush(sq);
		auto event = sq.ext_oneapi_submit_barrier();
		
		return make_async_event<sycl_backend_detail::sycl_event>(
			std::move(event), is_profiling_enabled());
	});
}

} // namespace celerity::detail