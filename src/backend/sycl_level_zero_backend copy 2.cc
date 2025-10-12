// VARIANT 2: Event + Command List Pooling
// Goal: Test if pooling BOTH events AND command lists provides additional benefit
// Implementation: Global pools for both events and command lists per device

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
// VARIANT 2: Event + Command List Pool Management
// ============================================================================

class event_pool {
public:
	event_pool(ze_context_handle_t ctx, ze_device_handle_t dev) : m_ctx(ctx), m_dev(dev) {
		ze_event_pool_desc_t pool_desc = {};
		pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		pool_desc.count = 1024;
		
		ze_check(zeEventPoolCreate(m_ctx, &pool_desc, 1, &m_dev, &m_pool), "zeEventPoolCreate");
		CELERITY_DEBUG("[V2] Event pool created with 1024 events");
	}
	
	~event_pool() {
		if(m_pool) {
			zeEventPoolDestroy(m_pool);
		}
	}
	
	ze_event_handle_t get_event() {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		if(!m_free_events.empty()) {
			auto event = m_free_events.back();
			m_free_events.pop_back();
			zeEventHostReset(event);
			return event;
		}
		
		if(m_next_index < 1024) {
			ze_event_desc_t event_desc = {};
			event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
			event_desc.index = m_next_index++;
			event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
			
			ze_event_handle_t event = nullptr;
			ze_check(zeEventCreate(m_pool, &event_desc, &event), "zeEventCreate");
			return event;
		}
		
		utils::panic("[V2] Event pool exhausted");
	}
	
	void return_event(ze_event_handle_t event) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_free_events.push_back(event);
	}

private:
	ze_context_handle_t m_ctx;
	ze_device_handle_t m_dev;
	ze_event_pool_handle_t m_pool = nullptr;
	size_t m_next_index = 0;
	std::vector<ze_event_handle_t> m_free_events;
	std::mutex m_mutex;
};

class cmdlist_pool {
public:
	cmdlist_pool(ze_context_handle_t ctx, ze_device_handle_t dev) : m_ctx(ctx), m_dev(dev) {
		CELERITY_DEBUG("[V2] Command list pool created");
	}
	
	~cmdlist_pool() {
		std::lock_guard<std::mutex> lock(m_mutex);
		for(auto cl : m_free_cmdlists) {
			zeCommandListDestroy(cl);
		}
	}
	
	ze_command_list_handle_t get_cmdlist() {
		std::lock_guard<std::mutex> lock(m_mutex);
		
		if(!m_free_cmdlists.empty()) {
			auto cl = m_free_cmdlists.back();
			m_free_cmdlists.pop_back();
			zeCommandListReset(cl);
			return cl;
		}
		
		ze_command_list_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
		ze_command_list_handle_t cl = nullptr;
		ze_check(zeCommandListCreate(m_ctx, m_dev, &desc, &cl), "zeCommandListCreate");
		return cl;
	}
	
	void return_cmdlist(ze_command_list_handle_t cl) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_free_cmdlists.push_back(cl);
	}

private:
	ze_context_handle_t m_ctx;
	ze_device_handle_t m_dev;
	std::vector<ze_command_list_handle_t> m_free_cmdlists;
	std::mutex m_mutex;
};

// Global pools
static std::vector<std::unique_ptr<event_pool>> g_event_pools;
static std::vector<std::unique_ptr<cmdlist_pool>> g_cmdlist_pools;
static std::mutex g_pools_mutex;

void initialize_pools(const std::vector<sycl::device>& devices) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	g_event_pools.clear();
	g_cmdlist_pools.clear();
	
	for(const auto& device : devices) {
		auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.get_platform().ext_oneapi_get_default_context());
		auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
		g_event_pools.push_back(std::make_unique<event_pool>(ze_ctx, ze_dev));
		g_cmdlist_pools.push_back(std::make_unique<cmdlist_pool>(ze_ctx, ze_dev));
	}
	
	CELERITY_DEBUG("[V2] Initialized {} event+cmdlist pools", devices.size());
}

void cleanup_pools() {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	g_event_pools.clear();
	g_cmdlist_pools.clear();
	CELERITY_DEBUG("[V2] Cleaned up pools");
}

event_pool& get_event_pool(device_id device) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	return *g_event_pools[device];
}

cmdlist_pool& get_cmdlist_pool(device_id device) {
	std::lock_guard<std::mutex> lock(g_pools_mutex);
	return *g_cmdlist_pools[device];
}

// RAII wrappers
class pooled_event {
public:
	pooled_event(device_id device) : m_device(device) {
		m_event = get_event_pool(device).get_event();
	}
	
	~pooled_event() {
		if(m_event) {
			get_event_pool(m_device).return_event(m_event);
		}
	}
	
	ze_event_handle_t get() const { return m_event; }
	
	pooled_event(const pooled_event&) = delete;
	pooled_event& operator=(const pooled_event&) = delete;

private:
	device_id m_device;
	ze_event_handle_t m_event = nullptr;
};

class pooled_cmdlist {
public:
	pooled_cmdlist(device_id device) : m_device(device) {
		m_cmdlist = get_cmdlist_pool(device).get_cmdlist();
	}
	
	~pooled_cmdlist() {
		if(m_cmdlist) {
			get_cmdlist_pool(m_device).return_cmdlist(m_cmdlist);
		}
	}
	
	ze_command_list_handle_t get() const { return m_cmdlist; }
	
	pooled_cmdlist(const pooled_cmdlist&) = delete;
	pooled_cmdlist& operator=(const pooled_cmdlist&) = delete;

private:
	device_id m_device;
	ze_command_list_handle_t m_cmdlist = nullptr;
};

// ============================================================================
// Copy Functions (using pooled events and command lists)
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
	
	// Use pooled event and command list
	pooled_event ze_event(device);
	pooled_cmdlist cmd_list(device);
	
	if(layout.num_complex_strides == 0) {
		const void* src_ptr = static_cast<const char*>(source_base) + layout.offset_in_source;
		void* dst_ptr = static_cast<char*>(dest_base) + layout.offset_in_dest;
		ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dst_ptr, src_ptr, layout.contiguous_size, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
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
		
		ze_check(zeCommandListAppendMemoryCopyRegion(cmd_list.get(), dst_ptr, &dst_region, dst_pitch, 0,
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
			
			ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dst_ptr, src_ptr, size, event_to_use, 0, nullptr), "zeCommandListAppendMemoryCopy");
		}
		
		CELERITY_TRACE("[V2] Level-Zero backend: 3D copy {} chunks", chunks.size());
	}
	
	ze_check(zeCommandListClose(cmd_list.get()), "zeCommandListClose");
	
	// Fix: Store cmd_list.get() in a variable before taking address
	ze_command_list_handle_t cl = cmd_list.get();
	ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cl, nullptr), "zeCommandQueueExecuteCommandLists");
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
		    CELERITY_TRACE("[V2] Level-Zero backend: linear copy {} bytes", size_bytes);
		    
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    
		    pooled_event ze_event(device);
		    pooled_cmdlist cmd_list(device);
		    
		    ze_check(zeCommandListAppendMemoryCopy(cmd_list.get(), dest, source, size_bytes, ze_event.get(), 0, nullptr), "zeCommandListAppendMemoryCopy");
		    ze_check(zeCommandListClose(cmd_list.get()), "zeCommandListClose");
		    
		    ze_command_list_handle_t cl = cmd_list.get();
		    ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cl, nullptr), "zeCommandQueueExecuteCommandLists");
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
