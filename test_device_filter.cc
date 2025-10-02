#include <iostream>
#include <sycl/sycl.hpp>

int main() {
    std::cout << "=== Testing SYCL_DEVICE_FILTER effectiveness ===" << std::endl;
    
    // Check environment
    const char* filter = std::getenv("SYCL_DEVICE_FILTER");
    std::cout << "SYCL_DEVICE_FILTER = " << (filter ? filter : "not set") << std::endl;
    
    const char* ur_disable = std::getenv("UR_DISABLE_ADAPTERS");
    std::cout << "UR_DISABLE_ADAPTERS = " << (ur_disable ? ur_disable : "not set") << std::endl;
    
    // Get all platforms
    auto platforms = sycl::platform::get_platforms();
    std::cout << "\nFound " << platforms.size() << " platforms:" << std::endl;
    
    for (size_t i = 0; i < platforms.size(); ++i) {
        const auto& platform = platforms[i];
        std::cout << "\nPlatform " << i << ": " << platform.get_info<sycl::info::platform::name>() << std::endl;
        
        auto devices = platform.get_devices();
        std::cout << "  Devices: " << devices.size() << std::endl;
        
        for (size_t j = 0; j < devices.size(); ++j) {
            const auto& device = devices[j];
            std::cout << "    Device " << j << ": " << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "      Backend: ";
            
            auto backend = device.get_backend();
            if (backend == sycl::backend::opencl) {
                std::cout << "OpenCL";
            }
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
            else if (backend == sycl::backend::ext_oneapi_level_zero) {
                std::cout << "Level Zero";
            }
#endif
            else {
                std::cout << "Other (" << static_cast<int>(backend) << ")";
            }
            std::cout << std::endl;
        }
    }
    
    // Test GPU device selection specifically
    std::cout << "\n=== GPU Device Selection Test ===" << std::endl;
    try {
        auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        std::cout << "Found " << gpu_devices.size() << " GPU devices:" << std::endl;
        
        for (size_t i = 0; i < gpu_devices.size(); ++i) {
            const auto& device = gpu_devices[i];
            std::cout << "  GPU " << i << ": " << device.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "    Backend: ";
            
            auto backend = device.get_backend();
            if (backend == sycl::backend::opencl) {
                std::cout << "OpenCL (PROBLEM!)";
            }
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
            else if (backend == sycl::backend::ext_oneapi_level_zero) {
                std::cout << "Level Zero (GOOD!)";
            }
#endif
            else {
                std::cout << "Other (" << static_cast<int>(backend) << ")";
            }
            std::cout << std::endl;
        }
        
        if (!gpu_devices.empty()) {
            std::cout << "\nDefault GPU device would be: " << gpu_devices[0].get_info<sycl::info::device::name>() << std::endl;
            auto backend = gpu_devices[0].get_backend();
            if (backend == sycl::backend::opencl) {
                std::cout << "*** PROBLEM: Default GPU is OpenCL, not Level Zero! ***" << std::endl;
            }
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
            else if (backend == sycl::backend::ext_oneapi_level_zero) {
                std::cout << "*** SUCCESS: Default GPU is Level Zero! ***" << std::endl;
            }
#endif
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error getting GPU devices: " << e.what() << std::endl;
    }
    
    return 0;
}