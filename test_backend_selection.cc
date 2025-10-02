#include <iostream>
#include <sycl/sycl.hpp>

int main() {
    try {
        // Get all devices
        auto devices = sycl::device::get_devices();
        
        std::cout << "=== Device Backend Analysis ===" << std::endl;
        
        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            
            std::cout << "\nDevice " << i << ":" << std::endl;
            std::cout << "  Name: " << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
            std::cout << "  Type: ";
            
            auto device_type = device.get_info<sycl::info::device::device_type>();
            switch(device_type) {
                case sycl::info::device_type::cpu: std::cout << "CPU"; break;
                case sycl::info::device_type::gpu: std::cout << "GPU"; break;
                case sycl::info::device_type::accelerator: std::cout << "ACCELERATOR"; break;
                default: std::cout << "UNKNOWN"; break;
            }
            std::cout << std::endl;
            
            // Check backend
            std::cout << "  Backend: ";
            auto backend = device.get_backend();
            
            if (backend == sycl::backend::opencl) {
                std::cout << "OpenCL";
            }
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
            else if (backend == sycl::backend::ext_oneapi_level_zero) {
                std::cout << "Level Zero (OneAPI)";
            }
#endif
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA  
            else if (backend == sycl::backend::ext_oneapi_cuda) {
                std::cout << "CUDA (OneAPI)";
            }
#endif
            else {
                std::cout << "Unknown (" << static_cast<int>(backend) << ")";
            }
            std::cout << std::endl;
            
            // Check if it's the default device
            auto default_device = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (!default_device.empty() && device == default_device[0]) {
                std::cout << "  *** This is the default GPU device ***" << std::endl;
            }
        }
        
        // Check what device Celerity would select
        std::cout << "\n=== Default GPU Device ===" << std::endl;
        try {
            auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (!gpu_devices.empty()) {
                const auto& gpu = gpu_devices[0];
                std::cout << "Default GPU: " << gpu.get_info<sycl::info::device::name>() << std::endl;
                std::cout << "Backend: ";
                auto backend = gpu.get_backend();
                
                if (backend == sycl::backend::opencl) {
                    std::cout << "OpenCL";
                }
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
                else if (backend == sycl::backend::ext_oneapi_level_zero) {
                    std::cout << "Level Zero (OneAPI) - SHOULD USE SPECIALIZED BACKEND";
                }
#endif
                else {
                    std::cout << "Other (" << static_cast<int>(backend) << ")";
                }
                std::cout << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Error getting GPU device: " << e.what() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}