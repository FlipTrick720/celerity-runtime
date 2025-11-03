#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    const size_t size = 64 * 1024 * 1024; // 64MB
    
    auto h = sycl::malloc_host<char>(size, q);
    auto d = sycl::malloc_device<char>(size, q);
    
    // Warmup
    for(int i=0; i<3; i++) q.memcpy(d, h, size).wait();
    
    // Measure H2D
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<20; i++) q.memcpy(d, h, size).wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double secs = std::chrono::duration<double>(t1-t0).count() / 20.0;
    double gbps = (size / 1e9) / secs;
    
    std::cout << "H2D Bandwidth: " << gbps << " GB/s\n";
    std::cout << (gbps > 15.0 ? "✓ PASS - Resizable BAR working!\n" 
                                : "✗ FAIL - Still limited to 16MB BAR\n");
    
    sycl::free(h, q);
    sycl::free(d, q);
    return gbps > 15.0 ? 0 : 1;
}
