#pragma once

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

// Level Zero error checking
inline void ze_check(ze_result_t result, const char* where) {
	if(result != ZE_RESULT_SUCCESS) {
		throw std::runtime_error(std::string("Level-Zero error in ") + where + ": code=" + std::to_string(static_cast<int>(result)));
	}
}

// Lightweight Level Zero queue wrapper for benchmarking
class level_zero_queue {
public:
	level_zero_queue(int device_index = 0) {
		// Get SYCL device to extract Level Zero handles
		std::vector<sycl::device> gpus;
		for (auto& d: sycl::device::get_devices(sycl::info::device_type::gpu)) {
			if(d.get_backend() == sycl::backend::ext_oneapi_level_zero) {
				gpus.push_back(d);
			}
		}
		
		if (gpus.empty()) {
			throw std::runtime_error("No Level Zero GPU device found");
		}
		if (device_index < 0 || device_index >= int(gpus.size())) {
			throw std::runtime_error("GPU index out of range");
		}
		
		m_sycl_device = gpus[device_index];
		m_sycl_context = sycl::context(m_sycl_device);
		
		// Extract native Level Zero handles
		m_ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(m_sycl_context);
		m_ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(m_sycl_device);
		
		// Create Level Zero command queue
		ze_command_queue_desc_t queue_desc = {};
		queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
		queue_desc.ordinal = 0;
		queue_desc.index = 0;
		queue_desc.flags = 0;
		queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
		queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
		
		ze_check(zeCommandQueueCreate(m_ze_context, m_ze_device, &queue_desc, &m_ze_queue), "zeCommandQueueCreate");
		
		m_device_name = m_sycl_device.get_info<sycl::info::device::name>();
	}
	
	~level_zero_queue() {
		if(m_ze_queue) {
			zeCommandQueueDestroy(m_ze_queue);
		}
	}
	
	// Disable copy/move for simplicity
	level_zero_queue(const level_zero_queue&) = delete;
	level_zero_queue& operator=(const level_zero_queue&) = delete;
	
	// Memory allocation using SYCL (compatible with Level Zero)
	void* malloc_device(size_t bytes) {
		return sycl::malloc_device(bytes, m_sycl_device, m_sycl_context);
	}
	
	void* malloc_host(size_t bytes) {
		return sycl::malloc_host(bytes, m_sycl_context);
	}
	
	void free(void* ptr) {
		sycl::free(ptr, m_sycl_context);
	}
	
	// Copy operation using native Level Zero
	void copy(const void* src, void* dst, size_t bytes) {
		// Create command list
		ze_command_list_desc_t cmd_list_desc = {};
		cmd_list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		cmd_list_desc.commandQueueGroupOrdinal = 0;
		
		ze_command_list_handle_t cmd_list = nullptr;
		ze_check(zeCommandListCreate(m_ze_context, m_ze_device, &cmd_list_desc, &cmd_list), "zeCommandListCreate");
		
		// Append memory copy
		ze_check(zeCommandListAppendMemoryCopy(cmd_list, dst, src, bytes, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy");
		
		// Close and execute
		ze_check(zeCommandListClose(cmd_list), "zeCommandListClose");
		ze_check(zeCommandQueueExecuteCommandLists(m_ze_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
		
		// Synchronize
		ze_check(zeCommandQueueSynchronize(m_ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
		
		// Cleanup
		ze_check(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
	}
	
	// Synchronization
	void wait() {
		ze_check(zeCommandQueueSynchronize(m_ze_queue, UINT64_MAX), "zeCommandQueueSynchronize");
	}
	
	// Device info
	const std::string& get_device_name() const { return m_device_name; }
	sycl::backend get_backend() const { return sycl::backend::ext_oneapi_level_zero; }
	
private:
	sycl::device m_sycl_device;
	sycl::context m_sycl_context;
	ze_context_handle_t m_ze_context = nullptr;
	ze_device_handle_t m_ze_device = nullptr;
	ze_command_queue_handle_t m_ze_queue = nullptr;
	std::string m_device_name;
};

} // namespace bench
