// Standalone stress test for Level Zero event pool manager
// This is NOT part of the regular test suite - compile and run manually
//
// Compile:
//   icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device dg2" \
//        -I../include -I../vendor/matchbox/include \
//        -std=c++20 -o level_zero_pool_stress x_level_zero_pool_stress_tests.cc \
//        -lze_loader
//
// Run:
//   SYCL_DEVICE_FILTER=level_zero:gpu ./level_zero_pool_stress

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

int main() {
    std::cout << "=== Level Zero Event Pool Stress Test ===" << std::endl;
    
    try {
        // Select Level Zero GPU
        sycl::queue q{sycl::gpu_selector_v};
        
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Backend: " << q.get_device().get_backend() << std::endl;
        
        constexpr size_t N = 1 << 20; // 1 MiB
        constexpr int REPS = 500;     // Many short-lived copies
        
        std::cout << "Allocating " << N << " bytes on device..." << std::endl;
        
        std::vector<char> host_src(N, 42);
        std::vector<char> host_dst(N, 0);
        
        char* dev_src = static_cast<char*>(sycl::malloc_device(N, q));
        char* dev_dst = static_cast<char*>(sycl::malloc_device(N, q));
        
        if(!dev_src || !dev_dst) {
            std::cerr << "Failed to allocate device memory!" << std::endl;
            return 1;
        }
        
        // Initial copy to device
        q.memcpy(dev_src, host_src.data(), N).wait();
        
        std::cout << "Running " << REPS << " small device-to-device copies..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Stress test: many small, short-lived copies
        for(int i = 0; i < REPS; ++i) {
            size_t sz = 4096;  // 4KB chunks
            size_t off = (i % (N - sz));
            
            // This creates a new event for each copy
            // With the old implementation, this would create/destroy a pool each time
            // With the new implementation, events are reused from the pool
            q.memcpy(dev_dst + off, dev_src + off, sz).wait();
            
            if((i + 1) % 100 == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << REPS << " copies" << std::endl;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Average: " << (duration.count() / (double)REPS) << " ms per copy" << std::endl;
        
        // Verify data
        q.memcpy(host_dst.data(), dev_dst, N).wait();
        
        bool has_data = std::any_of(host_dst.begin(), host_dst.end(), [](char c){ return c == 42; });
        
        // Cleanup
        sycl::free(dev_src, q);
        sycl::free(dev_dst, q);
        
        if(has_data) {
            std::cout << "✓ Data verification passed" << std::endl;
            std::cout << "✓ Event pool stress test PASSED" << std::endl;
            return 0;
        } else {
            std::cerr << "✗ Data verification FAILED" << std::endl;
            return 1;
        }
        
    } catch(const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
