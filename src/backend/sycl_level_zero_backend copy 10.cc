//Version: v10_host_sync_optimized
//Text: Event-pool overflow + no SYCL barrier + native timing

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
#include <utility>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <cstdlib>
#include <algorithm>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace celerity::detail::level_zero_backend_detail {

// Level Zero error checking helper
static inline void ze_check(ze_result_t result, const char* where) {
	if(result != ZE_RESULT_SUCCESS) {
		utils::panic("Level-Zero error in {}:: code={}", where, static_cast<int>(result));
	}
}

// Persistent pooling (per context+device for events, per queue for cmd lists)
struct ctx_dev_key {
    ze_context_handle_t ctx;
    ze_device_handle_t dev;
    bool operator==(const ctx_dev_key& other) const { return ctx == other.ctx && dev == other.dev; }
};

struct ctx_dev_key_hash {
    std::size_t operator()(const ctx_dev_key& k) const {
        return std::hash<void*>{}(k.ctx) ^ (std::hash<void*>{}(k.dev) << 1);
    }
};

struct event_pool_mgr {
    ze_context_handle_t ctx = nullptr;
    ze_device_handle_t dev = nullptr;
    ze_event_pool_handle_t pool = nullptr;
    std::vector<ze_event_handle_t> events;
    std::queue<size_t> free_indices;
    size_t peak = 0;
    std::mutex mtx;
};

struct cmdlist_mgr {
    ze_command_list_handle_t list = nullptr;
    std::mutex mtx;
};

static std::unordered_map<ctx_dev_key, event_pool_mgr, ctx_dev_key_hash> g_event_pools;
static std::unordered_map<ze_command_queue_handle_t, cmdlist_mgr> g_cmdlists;
static std::mutex g_global_mtx;

static size_t env_size_t(const char* name, size_t def) {
    if(const char* v = std::getenv(name)) { long x = std::strtol(v, nullptr, 10); if(x > 0) return static_cast<size_t>(x); }
    return def;
}

static bool env_bool(const char* name, bool def) {
    const char* v = std::getenv(name);
    if(!v) return def;
    const std::string s(v);
    if(s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON") return true;
    if(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF") return false;
    return def;
}

static event_pool_mgr& ensure_event_pool(ze_context_handle_t ctx, ze_device_handle_t dev) {
    std::lock_guard<std::mutex> lock(g_global_mtx);
    ctx_dev_key key{ctx, dev};
    auto it = g_event_pools.find(key);
    if(it != g_event_pools.end()) return it->second;
    
    // Construct directly in the map to avoid moving non-movable mutex
    auto& mgr = g_event_pools[key];
    mgr.ctx = ctx;
    mgr.dev = dev;
    const size_t pool_size = env_size_t("CELERITY_L0_EVENT_POOL_SIZE", 512);
    ze_event_pool_desc_t pool_desc{};
    pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pool_desc.count = pool_size;
    ze_check(zeEventPoolCreate(ctx, &pool_desc, 1, &dev, &mgr.pool), "zeEventPoolCreate");
    mgr.events.resize(pool_size);
    for(size_t i = 0; i < pool_size; ++i) {
        ze_event_desc_t e{};
        e.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
        e.index = static_cast<uint32_t>(i);
        e.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        e.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        ze_check(zeEventCreate(mgr.pool, &e, &mgr.events[i]), "zeEventCreate");
        mgr.free_indices.push(i);
    }
    return mgr;
}

struct acquired_event {
    ze_event_handle_t handle = nullptr;
    bool from_pool = true;
    size_t pool_index = 0;
    ze_event_pool_handle_t temp_pool = nullptr; // non-null when from_pool==false
};

static acquired_event acquire_event(event_pool_mgr& mgr) {
    std::lock_guard<std::mutex> lock(mgr.mtx);
    if(!mgr.free_indices.empty()) {
        const auto idx = mgr.free_indices.front();
        mgr.free_indices.pop();
        mgr.peak = std::max(mgr.peak, mgr.events.size() - mgr.free_indices.size());
        ze_check(zeEventHostReset(mgr.events[idx]), "zeEventHostReset");
        return acquired_event{mgr.events[idx], true, idx, nullptr};
    }

    // Pool exhausted: Fall back to a one-shot temporary event to avoid panicking, if allowed
    const bool allow_overflow = env_bool("CELERITY_L0_ALLOW_EVENT_POOL_OVERFLOW", true);
    if(!allow_overflow) {
        utils::panic("L0 event pool exhausted (size={})", mgr.events.size());
    }

    ze_event_pool_desc_t pool_desc{};
    pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    pool_desc.count = 1;
    ze_event_pool_handle_t temp_pool = nullptr;
    ze_check(zeEventPoolCreate(mgr.ctx, &pool_desc, 1, &mgr.dev, &temp_pool), "zeEventPoolCreate(temp)");

    ze_event_desc_t e{};
    e.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    e.index = 0;
    e.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    e.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    ze_event_handle_t ev = nullptr;
    ze_check(zeEventCreate(temp_pool, &e, &ev), "zeEventCreate(temp)");
    return acquired_event{ev, false, 0, temp_pool};
}

static void release_event(event_pool_mgr& mgr, const acquired_event& ev) {
    if(ev.from_pool) {
        std::lock_guard<std::mutex> lock(mgr.mtx);
        mgr.free_indices.push(ev.pool_index);
    } else {
        // destroy one-shot event+pool
        if(ev.handle) zeEventDestroy(ev.handle);
        if(ev.temp_pool) zeEventPoolDestroy(ev.temp_pool);
    }
}

static cmdlist_mgr& ensure_cmdlist(ze_context_handle_t ctx, ze_device_handle_t dev, ze_command_queue_handle_t ze_queue) {
    std::lock_guard<std::mutex> lock(g_global_mtx);
    auto it = g_cmdlists.find(ze_queue);
    if(it != g_cmdlists.end()) return it->second;
    
    // Construct directly in the map to avoid moving non-movable mutex
    auto& mgr = g_cmdlists[ze_queue];
    ze_command_list_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    desc.flags = 0;
    // Use default group; we execute on the same SYCL queue to preserve ordering
    ze_check(zeCommandListCreate(ctx, dev, &desc, &mgr.list), "zeCommandListCreate");
    return mgr;
}

// Simple async_event that carries a native execution time if profiling is enabled
class native_timed_event final : public async_event_impl {
  public:
    explicit native_timed_event(std::optional<std::chrono::nanoseconds> t) : m_time(std::move(t)) {}
    bool is_complete() override { return true; }
    std::optional<std::chrono::nanoseconds> get_native_execution_time() override { return m_time; }

  private:
    std::optional<std::chrono::nanoseconds> m_time;
};

// Level Zero event wrapper for proper SYCL integration
class level_zero_event final : public async_event_impl {
  public:
	level_zero_event(ze_event_handle_t event, ze_event_pool_handle_t pool) : m_event(event), m_pool(pool) {}
	
	~level_zero_event() override {
		if(m_event) {
			zeEventDestroy(m_event);
		}
		if(m_pool) {
			zeEventPoolDestroy(m_pool);
		}
	}
	
	bool is_complete() override {
		const auto result = zeEventQueryStatus(m_event);
		return result == ZE_RESULT_SUCCESS;
	}
	
  private:
	ze_event_handle_t m_event;
	ze_event_pool_handle_t m_pool;
};

// Helper to create Level Zero event that integrates with SYCL
std::pair<ze_event_handle_t, ze_event_pool_handle_t> create_level_zero_event(ze_context_handle_t context, ze_device_handle_t device) {
	// Create event pool
	ze_event_pool_desc_t pool_desc = {};
	pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
	pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
	pool_desc.count = 1;
	
	ze_event_pool_handle_t event_pool = nullptr;
	ze_check(zeEventPoolCreate(context, &pool_desc, 1, &device, &event_pool), "zeEventPoolCreate");
	
	// Create event
	ze_event_desc_t event_desc = {};
	event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
	event_desc.index = 0;
	event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
	event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
	
	ze_event_handle_t event = nullptr;
	ze_check(zeEventCreate(event_pool, &event_desc, &event), "zeEventCreate");
	
	return {event, event_pool};
}

// Helper to perform box-based copy using native Level Zero operations
void nd_copy_box_level_zero(sycl::queue& queue, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const box<3>& copy_box, const size_t elem_size, sycl::event& last_event) //
{
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	
	// compute layout/strides/offsets
	const auto src_range = source_box.get_range();
	const auto dst_range = dest_box.get_range();
	const auto copy_range = copy_box.get_range();
	const auto src_offset = copy_box.get_offset() - source_box.get_offset();
	const auto dst_offset = copy_box.get_offset() - dest_box.get_offset();
	
	const auto layout = layout_nd_copy(src_range, dst_range, src_offset, dst_offset, copy_range, elem_size);
	
	if(layout.contiguous_size == 0) return;
	
	// Get native Level Zero handles
	auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
	auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
	auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
	auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
	
    // Persistent resources
    auto& pool = ensure_event_pool(ze_context, ze_device);
    const auto ev = acquire_event(pool);
    ze_event_handle_t ze_event = ev.handle;
    auto& cl_mgr = ensure_cmdlist(ze_context, ze_device, ze_queue);
    ze_command_list_handle_t cmd_list = cl_mgr.list;
	
	if(layout.num_complex_strides == 0) {
		// 1) Contiguous: single blit
		// Single contiguous copy
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, layout.contiguous_size, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
		CELERITY_TRACE("Level-Zero backend: contiguous copy {} bytes", layout.contiguous_size);
	} else if(layout.num_complex_strides == 1) {
		// 2) 2D region copy
		// Optimized 2D copy using Level Zero's native 2D copy operation
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
		
		CELERITY_TRACE("Level-Zero backend: 2D copy {}x{} bytes (src_pitch={}, dst_pitch={})", width, height, src_pitch, dst_pitch);
	} else {
		// 3) 3D: many 1D copies (signal event on the LAST chunk only)
		// Multiple 1D copies for complex 3D layouts
		// First, collect all chunks to know which is the last one
		std::vector<std::tuple<size_t, size_t, size_t>> chunks;
		for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t size) {
			chunks.emplace_back(src_off, dst_off, size);
		});
		
		// Now append all copies, signaling event only on the last one
		for(size_t i = 0; i < chunks.size(); ++i) {
			const auto& [src_off, dst_off, size] = chunks[i];
			const void* src_ptr = static_cast<const char*>(source_base) + src_off;
			void* dst_ptr = static_cast<char*>(dest_base) + dst_off;
			
			// Signal event ONLY on the last chunk
			const bool is_last = (i == chunks.size() - 1);
			ze_event_handle_t event_to_use = is_last ? ze_event : nullptr;
			
			ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, size, event_to_use, 0, nullptr), "zeCommandListAppendMemoryCopy");
		}
		
		CELERITY_TRACE("Level-Zero backend: 3D copy {} chunks of {} bytes", chunks.size(), layout.contiguous_size);
	}
	
    // Execute and wait for just the event; then reset list for reuse
    ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
    ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
    ze_check(zeEventHostSynchronize(ze_event, UINT64_MAX), "zeEventHostSynchronize");
    ze_check(zeCommandListReset(cmd_list), "zeCommandListReset");
    release_event(pool, ev);

    // We already synchronized on host; return a dummy event (we'll report completion separately)
    last_event = sycl::event{}; // unused – completion is reported via async_event below
}

// Helper function for n-dimensional device copy using native Level Zero
async_event nd_copy_device_level_zero(sycl::queue& queue, const void* const source_base, void* const dest_base, const region_layout& source_layout,
    const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size, bool enable_profiling) //
{
	sycl::event last_event;
	std::optional<std::chrono::nanoseconds> native_time = std::nullopt;

	// Use dispatch_nd_region_copy to handle all layout combinations
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
		// box path
	    [&queue, elem_size, &last_event, enable_profiling, &native_time](const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    const auto t0 = std::chrono::steady_clock::now();
		    nd_copy_box_level_zero(queue, source, dest, source_box, dest_box, copy_box, elem_size, last_event);
		    const auto t1 = std::chrono::steady_clock::now();
		    if(enable_profiling) native_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
	    },
		// linear path
	    [&queue, &last_event, enable_profiling, &native_time](const void* const source, void* const dest, size_t size_bytes) {
		    CELERITY_TRACE("Level-Zero backend: linear copy {} bytes", size_bytes);
		    
		    // Get native Level Zero handles
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
		    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
		    
            // Persistent resources and execution
            auto& pool = ensure_event_pool(ze_context, ze_device);
            const auto ev = acquire_event(pool);
            ze_event_handle_t ze_event = ev.handle;
            auto& cl_mgr = ensure_cmdlist(ze_context, ze_device, ze_queue);
            ze_command_list_handle_t cmd_list = cl_mgr.list;
            const auto t0 = std::chrono::steady_clock::now();
            ze_check(zeCommandListAppendMemoryCopy(cmd_list, dest, source, size_bytes, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
            ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
            ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
            ze_check(zeEventHostSynchronize(ze_event, UINT64_MAX), "zeEventHostSynchronize");
            ze_check(zeCommandListReset(cmd_list), "zeCommandListReset");
            release_event(pool, ev);
            const auto t1 = std::chrono::steady_clock::now();
            if(enable_profiling) native_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
            // No SYCL barrier needed; we already synchronized
            last_event = sycl::event{};
        });

    // Flush any pending operations in the queue path (no-ops for L0-only path).
    sycl_backend_detail::flush(queue);

    return make_async_event<native_timed_event>(native_time);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("Level-Zero backend initialized with {} device(s)", devices.size());
	
	// Note: Error handling is provided by the base class:
	// - SYCL async_handler captures exceptions from device operations
	// - check_async_errors() inherited from sycl_backend calls throw_asynchronous() on queues
	// - All SYCL operations return events that can be queried for completion
	
	// Query and enable peer-to-peer access between devices
	// Level Zero devices on the same driver can typically access each other's memory
	// Not sure if Possible but could be so we test
	for(device_id i = 0; i < devices.size(); ++i) {
		for(device_id j = i + 1; j < devices.size(); ++j) {
			try {
				// Get native Level Zero device handles
				const auto ze_device_i = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
				const auto ze_device_j = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[j]);
				
				// Query peer access capabilities
				ze_bool_t can_access_ij = false;
				ze_bool_t can_access_ji = false;
				
				const auto result_ij = zeDeviceCanAccessPeer(ze_device_i, ze_device_j, &can_access_ij);
				const auto result_ji = zeDeviceCanAccessPeer(ze_device_j, ze_device_i, &can_access_ji);
				
				if(result_ij == ZE_RESULT_SUCCESS && result_ji == ZE_RESULT_SUCCESS && can_access_ij && can_access_ji) {
					// Both devices can access each other - enable peer access
					const memory_id mid_i = first_device_memory_id + i;
					const memory_id mid_j = first_device_memory_id + j;
					get_system_info().memories[mid_i].copy_peers.set(mid_j);
					get_system_info().memories[mid_j].copy_peers.set(mid_i);
					CELERITY_DEBUG("Level-Zero backend: enabled peer access between D{} and D{}", i, j);
				} else {
					CELERITY_DEBUG("Level-Zero backend: no peer access between D{} and D{}, device-to-device copies will be staged in host memory", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("Level-Zero backend: failed to query peer access between D{} and D{}: {}", i, j, e.what());
			}
		}
	}
}

sycl_level_zero_backend::~sycl_level_zero_backend() = default;

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) //
{
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
