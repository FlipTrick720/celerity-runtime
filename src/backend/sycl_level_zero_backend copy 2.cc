// VARIANT 2: Persistent Event Pool (Professor's suggestion)
// Goal: One persistent ze_event_pool per device, events created/destroyed per copy
// Implementation: Pool persists, but events are created fresh and destroyed after each operation
// Hypothesis: Small to moderate gain from avoiding pool creation overhead

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

// ============================================================================
// VARIANT 2: Persistent Event Pool (Professor's suggestion)
// ============================================================================

class persistent_event_pool {
public:
	persistent_event_pool(ze_context_handle_t ctx, ze_device_handle_t dev) : m_ctx(ctx), m_dev(dev) {
		ze_event_pool_desc_t pool_desc = {};
		pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		pool_desc.count = 1; // Only 1 event per pool, but pool persists
		
		ze_check(zeEventPoolCreate(m_ctx, &pool_desc, 1, &m_dev, &m_pool), "zeEventPoolCreate");
		CELERITY_DEBUG("[V2] Persistent event pool created (1 event capacity)");
	}
	
	~persistent_event_pool() {
		if(m_pool) {
			zeEventPoolDestroy(m_pool);
		}
	}
	
	// Create a fresh event from the persistent pool
	ze_event_handle_t create_event() {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		ze_event_desc_t event_desc = {};
		event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
		event_desc.index = 0; // Always use index 0 since we only have 1 slot
		event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
		event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
		
		ze_event_handle_t event = nullptr;
		ze_check(zeEventCreate(m_pool, &event_desc, &event), "zeEventCreate");
		return event;
	}
	
	ze_event_pool_handle_t get_pool() const { return m_pool; }

private:
	ze_context_handle_t m_ctx;
	ze_device_handle_t m_dev;
	ze_event_pool_handle_t m_pool = nullptr;
	std::mutex m_mutex;
};

// Global persistent pools (one per device)
static std::vector<std::unique_ptr<persistent_event_pool>> g_persistent_pools;
static std::mutex g_pools_mutex;

void initialize_pools(const std::vector<sycl::device>& devices) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	g_persistent_pools.clear();
	
	for(const auto& device : devices) {
		auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.get_platform().ext_oneapi_get_default_context());
		auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
		g_persistent_pools.push_back(std::make_unique<persistent_event_pool>(ze_ctx, ze_dev));
	}
	
	CELERITY_DEBUG("[V2] Initialized {} persistent event pools", devices.size());
}

void cleanup_pools() {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	g_persistent_pools.clear();
	CELERITY_DEBUG("[V2] Cleaned up persistent pools");
}

persistent_event_pool& get_persistent_pool(device_id device) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	if(device >= g_persistent_pools.size()) {
		utils::panic("[V2] Invalid device ID for persistent pool: {}", device);
	}
	return *g_persistent_pools[device];
}

// RAII wrapper for events created from persistent pool
class transient_event {
public:
	transient_event(device_id device) : m_device(device) {
		m_event = get_persistent_pool(device).create_event();
	}
	
	~transient_event() {
		if(m_event) {
			zeEventDestroy(m_event); // Destroy event, but pool persists
		}
	}
	
	ze_event_handle_t get() const { return m_event; }
	
	transient_event(const transient_event&) = delete;
	transient_event& operator=(const transient_event&) = delete;

private:
	device_id m_device;
	ze_event_handle_t m_event = nullptr;
};

// ============================================================================
// Copy Functions (using transient events from persistent pool)
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
	auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
	auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
	
	// Create transient event from persistent pool (event destroyed after use, pool persists)
	transient_event ze_event(device);
	
	// Create command list (not pooled in this variant)
	ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
	ze_command_list_handle_t cmd_list = nullptr;
	ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
	
	if(layout.num_complex_strides == 0) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, layout.contiguous_size, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
		CELERITY_TRACE("[V2] Level-Zero backend: contiguous copy {} bytes", layout.contiguous_size);
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
		                                             src_ptr, &src_region, src_pitch, 0, ze_event.get(), 0, nullptr), 
		         "zeCommandListAppendMemoryCopyRegion");
		
		CELERITY_TRACE("[V2] Level-Zero backend: 2D copy {}x{} bytes", width, height);
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
			
			ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, size, event_to_use, 0, nullptr), "zeCommandListAppendMemoryCopy");
		}
		
		CELERITY_TRACE("[V2] Level-Zero backend: 3D copy {} chunks", chunks.size());
	}
	
	ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
	ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
	ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
	ze_check(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
	
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
		    CELERITY_TRACE("[V2] Level-Zero backend: linear copy {} bytes", size_bytes);
		    
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
		    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
		    
		    transient_event ze_event(device);
		    
		    ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
		    ze_command_list_handle_t cmd_list = nullptr;
		    ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
		    ze_check(zeCommandListAppendMemoryCopy(cmd_list, dest, source, size_bytes, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
		    ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
		    ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
		    ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
		    ze_check(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
		    
		    last_event = queue.ext_oneapi_submit_barrier();
	    });
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("[V2] Level-Zero backend initialized with {} device(s)", devices.size());
	
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
					CELERITY_DEBUG("[V2] Level-Zero backend: enabled peer access between D{} and D{}", i, j);
				}
			} catch(const std::exception& e) {
				CELERITY_WARN("[V2] Level-Zero backend: failed to query peer access: {}", e.what());
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
