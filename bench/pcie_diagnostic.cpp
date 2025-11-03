#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

// Measure raw PCIe bandwidth with various strategies
class PCIeDiagnostic {
public:
    struct Result {
        std::string test_name;
        double bandwidth_gbps;
        double latency_us;
        bool passed;
    };

    PCIeDiagnostic(sycl::queue& q) : queue(q) {
        device = queue.get_device();
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
    }

    std::vector<Result> run_all_tests(size_t size_mb = 64) {
        std::vector<Result> results;
        const size_t size = size_mb * 1024 * 1024;
        
        results.push_back(test_h2d_pinned(size));
        results.push_back(test_d2h_pinned(size));
        results.push_back(test_bidirectional(size));
        results.push_back(test_multi_queue(size));
        results.push_back(test_async_overlap(size));
        
        return results;
    }

private:
    sycl::queue queue;
    sycl::device device;

    Result test_h2d_pinned(size_t size) {
        auto host_ptr = sycl::malloc_host<char>(size, queue);
        auto dev_ptr = sycl::malloc_device<char>(size, queue);
        
        // Warmup
        for(int i = 0; i < 3; i++) {
            queue.memcpy(dev_ptr, host_ptr, size).wait();
        }
        
        // Measure
        const int reps = 10;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < reps; i++) {
            queue.memcpy(dev_ptr, host_ptr, size).wait();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avg_us = total_us / reps;
        double bw_gbps = (size / 1e9) / (avg_us / 1e6);
        
        sycl::free(host_ptr, queue);
        sycl::free(dev_ptr, queue);
        
        return {"H2D Pinned (sync)", bw_gbps, avg_us, bw_gbps > 12.0};
    }

    Result test_d2h_pinned(size_t size) {
        auto host_ptr = sycl::malloc_host<char>(size, queue);
        auto dev_ptr = sycl::malloc_device<char>(size, queue);
        
        for(int i = 0; i < 3; i++) {
            queue.memcpy(host_ptr, dev_ptr, size).wait();
        }
        
        const int reps = 10;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < reps; i++) {
            queue.memcpy(host_ptr, dev_ptr, size).wait();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avg_us = total_us / reps;
        double bw_gbps = (size / 1e9) / (avg_us / 1e6);
        
        sycl::free(host_ptr, queue);
        sycl::free(dev_ptr, queue);
        
        return {"D2H Pinned (sync)", bw_gbps, avg_us, bw_gbps > 12.0};
    }

    Result test_bidirectional(size_t size) {
        auto h2d_src = sycl::malloc_host<char>(size, queue);
        auto d2h_dst = sycl::malloc_host<char>(size, queue);
        auto dev1 = sycl::malloc_device<char>(size, queue);
        auto dev2 = sycl::malloc_device<char>(size, queue);
        
        // Warmup
        queue.memcpy(dev1, h2d_src, size).wait();
        queue.memcpy(d2h_dst, dev2, size).wait();
        
        // Measure simultaneous H2D + D2H
        const int reps = 10;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < reps; i++) {
            auto e1 = queue.memcpy(dev1, h2d_src, size);
            auto e2 = queue.memcpy(d2h_dst, dev2, size);
            sycl::event::wait({e1, e2});
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avg_us = total_us / reps;
        double total_bw_gbps = (2 * size / 1e9) / (avg_us / 1e6);
        
        sycl::free(h2d_src, queue);
        sycl::free(d2h_dst, queue);
        sycl::free(dev1, queue);
        sycl::free(dev2, queue);
        
        return {"Bidirectional (H2D+D2H)", total_bw_gbps, avg_us, total_bw_gbps > 20.0};
    }

    Result test_multi_queue(size_t size) {
        const int num_queues = 4;
        std::vector<sycl::queue> queues;
        for(int i = 0; i < num_queues; i++) {
            queues.emplace_back(device, sycl::property::queue::in_order{});
        }
        
        std::vector<char*> host_ptrs(num_queues);
        std::vector<char*> dev_ptrs(num_queues);
        
        for(int i = 0; i < num_queues; i++) {
            host_ptrs[i] = sycl::malloc_host<char>(size, queues[i]);
            dev_ptrs[i] = sycl::malloc_device<char>(size, queues[i]);
        }
        
        // Warmup
        for(int i = 0; i < num_queues; i++) {
            queues[i].memcpy(dev_ptrs[i], host_ptrs[i], size).wait();
        }
        
        // Measure interleaved copies
        const int reps = 10;
        auto t0 = std::chrono::high_resolution_clock::now();
        for(int r = 0; r < reps; r++) {
            std::vector<sycl::event> events;
            for(int i = 0; i < num_queues; i++) {
                events.push_back(queues[i].memcpy(dev_ptrs[i], host_ptrs[i], size));
            }
            sycl::event::wait(events);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avg_us = total_us / reps;
        double total_bw_gbps = (num_queues * size / 1e9) / (avg_us / 1e6);
        
        for(int i = 0; i < num_queues; i++) {
            sycl::free(host_ptrs[i], queues[i]);
            sycl::free(dev_ptrs[i], queues[i]);
        }
        
        return {"Multi-Queue (4x interleaved)", total_bw_gbps, avg_us, total_bw_gbps > 15.0};
    }

    Result test_async_overlap(size_t size) {
        const int num_buffers = 4;
        std::vector<char*> host_ptrs(num_buffers);
        std::vector<char*> dev_ptrs(num_buffers);
        
        for(int i = 0; i < num_buffers; i++) {
            host_ptrs[i] = sycl::malloc_host<char>(size, queue);
            dev_ptrs[i] = sycl::malloc_device<char>(size, queue);
        }
        
        // Warmup
        for(int i = 0; i < num_buffers; i++) {
            queue.memcpy(dev_ptrs[i], host_ptrs[i], size).wait();
        }
        
        // Measure pipelined async copies
        const int reps = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        
        std::vector<sycl::event> pipeline;
        for(int r = 0; r < reps; r++) {
            int buf_idx = r % num_buffers;
            pipeline.push_back(queue.memcpy(dev_ptrs[buf_idx], host_ptrs[buf_idx], size));
        }
        sycl::event::wait(pipeline);
        
        auto t1 = std::chrono::high_resolution_clock::now();
        
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double avg_us = total_us / reps;
        double bw_gbps = (size / 1e9) / (avg_us / 1e6);
        
        for(int i = 0; i < num_buffers; i++) {
            sycl::free(host_ptrs[i], queue);
            sycl::free(dev_ptrs[i], queue);
        }
        
        return {"Async Pipeline (4 buffers)", bw_gbps, avg_us, bw_gbps > 15.0};
    }
};

int main() {
    try {
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if(devices.empty()) {
            std::cerr << "No GPU found\n";
            return 1;
        }
        
        sycl::queue q{devices[0], sycl::property::queue::in_order{}};
        
        std::cout << "=== PCIe Bandwidth Diagnostic ===\n";
        std::cout << "Target: Gen4 x16 = 32 GB/s theoretical\n";
        std::cout << "Expected: 20-25 GB/s practical\n\n";
        
        PCIeDiagnostic diag(q);
        auto results = diag.run_all_tests(64);
        
        std::cout << "\n=== Results ===\n";
        std::cout << std::setw(35) << std::left << "Test"
                  << std::setw(15) << "Bandwidth"
                  << std::setw(15) << "Latency"
                  << "Status\n";
        std::cout << std::string(70, '-') << "\n";
        
        for(const auto& r : results) {
            std::cout << std::setw(35) << std::left << r.test_name
                      << std::setw(15) << (std::to_string(r.bandwidth_gbps) + " GB/s")
                      << std::setw(15) << (std::to_string(r.latency_us) + " µs")
                      << (r.passed ? "✓ PASS" : "✗ FAIL") << "\n";
        }
        
        std::cout << "\n=== Analysis ===\n";
        bool any_failed = false;
        for(const auto& r : results) {
            if(!r.passed) {
                any_failed = true;
                std::cout << "⚠ " << r.test_name << " underperforming\n";
                if(r.test_name.find("H2D") != std::string::npos || 
                   r.test_name.find("D2H") != std::string::npos) {
                    if(r.bandwidth_gbps < 12.0) {
                        std::cout << "  → Check: PCIe link status, Resizable BAR, IOMMU\n";
                    }
                }
            }
        }
        
        if(!any_failed) {
            std::cout << "✓ All tests passed - PCIe is performing well\n";
            std::cout << "  → Your 10.5 GB/s in Celerity benchmarks suggests\n";
            std::cout << "    the bottleneck is in the Level-Zero driver or\n";
            std::cout << "    Celerity's copy engine integration\n";
        }
        
        return any_failed ? 1 : 0;
        
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}