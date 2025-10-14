#pragma once
#include <sycl/sycl.hpp>
#include <string>
#include <iostream>

namespace bench {

struct DeviceInfo {
	std::string name;
	sycl::backend backend;
};

inline sycl::queue make_queue(int device_index, bool profiling=true) {
	std::vector<sycl::device> gpus;
	for (auto& d: sycl::device::get_devices(sycl::info::device_type::gpu)) gpus.push_back(d);
	if (gpus.empty()) throw std::runtime_error("No GPU device found");
	if (device_index<0 || device_index>=int(gpus.size())) throw std::runtime_error("GPU index out of range");
	sycl::device dev = gpus[device_index];
	if (profiling) {
		return sycl::queue{dev, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
	} else {
		return sycl::queue{dev, {sycl::property::queue::in_order{}}};
	}
}

inline DeviceInfo device_info(const sycl::queue& q){
	auto dev = q.get_device();
	DeviceInfo di;
	di.name = dev.get_info<sycl::info::device::name>();
	di.backend = dev.get_backend();
	return di;
}

inline const char* backend_name(sycl::backend be) {
	switch(be){
		case sycl::backend::ext_oneapi_level_zero: return "level_zero";
		case sycl::backend::ext_oneapi_cuda: return "cuda";
		case sycl::backend::opencl: return "opencl";
		case sycl::backend::native_cpu: return "native_cpu";
		default: return "other";
	}
}

inline void* alloc_host(size_t bytes, sycl::queue& q, bool pinned){
	if (pinned) return sycl::aligned_alloc_host(4096, bytes, q);
	// fallback pageable
	return std::aligned_alloc(4096, ((bytes+4095)/4096)*4096);
}

inline void free_host(void* p, sycl::queue& q, bool pinned){
	if (pinned) sycl::free(p, q);
	else std::free(p);
}

} // namespace bench
