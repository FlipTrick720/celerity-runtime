//Version: v5_No_Sync
//Text: Remove zeCommandQueueSynchronize calls to test async behavior

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
	
	// Create Level Zero event for synchronization
	auto [ze_event, ze_event_pool] = create_level_zero_event(ze_context, ze_device);
	
	// Create command list for batched operations
	ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
	ze_command_list_handle_t cmd_list = nullptr;
	ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
	
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
	
	// Execute the command list
	ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
	ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
	
	// V5: NO SYNC - Let SYCL handle synchronization through events
	// ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
	
	// Clean up/Destroy Level Zero resources
	ze_check(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
	zeEventDestroy(ze_event);
	zeEventPoolDestroy(ze_event_pool);
	
	// Create SYCL barrier event to integrate with SYCL's event system
	last_event = queue.ext_oneapi_submit_barrier();
}

// Helper function for n-dimensional device copy using native Level Zero
async_event nd_copy_device_level_zero(sycl::queue& queue, const void* const source_base, void* const dest_base, const region_layout& source_layout,
    const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size, bool enable_profiling) //
{
	sycl::event last_event;
	
	// Use dispatch_nd_region_copy to handle all layout combinations
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
		// box path
	    [&queue, elem_size, &last_event](const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_box_level_zero(queue, source, dest, source_box, dest_box, copy_box, elem_size, last_event);
	    },
		// linear path
	    [&queue, &last_event](const void* const source, void* const dest, size_t size_bytes) {
		    CELERITY_TRACE("Level-Zero backend: linear copy {} bytes", size_bytes);
		    
		    // Get native Level Zero handles
		    auto ze_queue_variant = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
		    auto ze_queue = std::get<ze_command_queue_handle_t>(ze_queue_variant);
		    auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
		    auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
		    
		    // Create Level Zero event for synchronization
		    auto [ze_event, ze_event_pool] = create_level_zero_event(ze_context, ze_device);
		    
		    // Create and execute command list for simple copy
		    ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0};
		    ze_command_list_handle_t cmd_list = nullptr;
		    ze_check(zeCommandListCreate(ze_context, ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
		    ze_check(zeCommandListAppendMemoryCopy(cmd_list, dest, source, size_bytes, ze_event, 0, nullptr), "zeCommandListAppendMemoryCopy");
		    ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
		    ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
		    
		    // V5: NO SYNC - Let SYCL handle synchronization through events
		    // ze_check(zeCommandQueueSynchronize(ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
		    
			// Clean up/Destroy Level Zero resources
		    ze_check(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
		    zeEventDestroy(ze_event);
		    zeEventPoolDestroy(ze_event_pool);
		    
		    // Create SYCL barrier event to integrate with SYCL's event system
		    last_event = queue.ext_oneapi_submit_barrier();
	    });
	
	sycl_backend_detail::flush(queue);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(last_event), enable_profiling);
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
    : sycl_backend(devices, config) {
	CELERITY_DEBUG("[V5-NoSync] Level-Zero backend initialized with {} device(s)", devices.size());
	
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

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) //
{
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail