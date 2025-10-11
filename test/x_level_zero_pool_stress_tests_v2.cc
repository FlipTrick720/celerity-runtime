// Enhanced stress test to identify bottlenecks
// Tests different scenarios to isolate performance issues

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

void run_test(const char* name, int reps, size_t copy_size, bool sync_each) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Reps: " << reps << ", Size: " << copy_size << " bytes, Sync: " << (sync_each ? "each" : "batch") << std::endl;
    
    sycl::queue q{sycl::gpu_selector_v};
    
    constexpr size_t N = 1 << 20; // 1 MiB
    
    char* dev_src = static_cast<char*>(sycl::malloc_device(N, q));
    char* dev_dst = static_cast<char*>(sycl::malloc_device(N, q));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if(sync_each) {
        // Synchronous: wait after each copy (stresses event/cmdlist creation)
        for(int i = 0; i < reps; ++i) {
            size_t off = (i % (N - copy_size));
            q.memcpy(dev_dst + off, dev_src + off, copy_size).wait();
        }
    } else {
        // Asynchronous: batch all copies, then wait once
        std::vector<sycl::event> events;
        for(int i = 0; i < reps; ++i) {
            size_t off = (i % (N - copy_size));
            events.push_back(q.memcpy(dev_dst + off, dev_src + off, copy_size));
        }
        sycl::event::wait(events);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Avg: " << (duration.count() / (double)reps) << " ms per copy" << std::endl;
    
    sycl::free(dev_src, q);
    sycl::free(dev_dst, q);
}

int main() {
    std::cout << "=== Level Zero Backend Bottleneck Analysis ===" << std::endl;
    
    try {
        sycl::queue q{sycl::gpu_selector_v};
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        // Test 1: Small copies, sync each (original stress test)
        run_test("Small copies, sync each (original)", 500, 4096, true);
        
        // Test 2: Small copies, batched (tests if sync is the bottleneck)
        run_test("Small copies, batched", 500, 4096, false);
        
        // Test 3: Larger copies, sync each (tests if copy size matters)
        run_test("Large copies, sync each", 100, 65536, true);
        
        // Test 4: Many tiny copies (extreme stress)
        run_test("Tiny copies, sync each", 1000, 1024, true);
        
        // Test 5: Very large batch (tests pool exhaustion)
        run_test("Huge batch, async", 2000, 4096, false);
        
        std::cout << "\n=== Analysis ===" << std::endl;
        std::cout << "If 'batched' is much faster than 'sync each':" << std::endl;
        std::cout << "  → Bottleneck is synchronization (zeCommandQueueSynchronize)" << std::endl;
        std::cout << "If 'sync each' times are similar:" << std::endl;
        std::cout << "  → Bottleneck is command list creation/destruction" << std::endl;
        std::cout << "If 'huge batch' is slow:" << std::endl;
        std::cout << "  → Event pool exhaustion or memory issue" << std::endl;
        
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
