# Implementation Plan

- [x] 1. Update backend type enumeration and enumerator
  - Add `level_zero` to `sycl_backend_type` enum in `sycl_backend.h`
  - Update `sycl_backend_enumerator::compatible_backends()` to detect Level Zero devices
  - Update `sycl_backend_enumerator::available_backends()` to include level_zero when compiled with support
  - Update `sycl_backend_enumerator::is_specialized()` to return true for level_zero
  - Update `sycl_backend_enumerator::get_priority()` to assign appropriate priority to level_zero
  - _Requirements: 5.2, 5.3_

- [x] 2. Create sycl_level_zero_backend class structure
  - Create new file `src/backend/sycl_level_zero_backend.cc`
  - Implement `sycl_level_zero_backend` class inheriting from `sycl_backend`
  - Implement constructor that calls base class constructor with devices and config
  - Implement destructor (can be default since base class handles cleanup)
  - Add forward declarations and helper namespace `level_zero_backend_detail`
  - _Requirements: 1.1, 1.2, 5.1_

- [x] 3. Implement basic device copy with linearized layout fast path
  - Implement `enqueue_device_copy()` method signature matching base class
  - Use `enqueue_device_work()` pattern from base class to get SYCL queue
  - Detect linearized → linearized layout case
  - Use SYCL `queue.memcpy()` for contiguous copies
  - Return `sycl_event` wrapped in async_event using `make_async_event<sycl_event>()`
  - Call `sycl_backend_detail::flush(queue)` after submission
  - _Requirements: 2.1, 2.5, 3.1_

- [x] 4. Implement 2D/3D strided copy operations
  - Add helper function `nd_copy_device_level_zero()` in detail namespace
  - Use `dispatch_nd_region_copy()` utility from `nd_memory.h` to decompose region
  - Implement copy using SYCL parallel_for with optimized work-group sizing
  - Calculate proper 3D indices from linear work-item id
  - Handle different element sizes (1, 4, 8 bytes) with optimized paths
  - Ensure proper memory access patterns for coalescing
  - _Requirements: 2.2, 2.3, 2.4_

- [x] 5. Add peer-to-peer device copy support
  - In constructor, query Level Zero for peer access capabilities between devices
  - Use `sycl::get_native<sycl::backend::ext_oneapi_level_zero>()` to get native device handles
  - Call Level Zero APIs to check and enable peer access
  - Update `system_info.memories[].copy_peers` bitset for device pairs with peer access
  - Log devices without peer access for diagnostics
  - _Requirements: 4.4_

- [x] 6. Integrate with backend factory and build system
  - Update `make_sycl_backend()` factory function in `sycl_backend.cc` to handle `level_zero` case
  - Add conditional compilation guards using `CELERITY_DETAIL_BACKEND_LEVEL_ZERO_ENABLED`
  - Update `CMakeLists.txt` in `src/backend/` to include new source file
  - Add Level Zero backend declaration to `sycl_backend.h` header
  - _Requirements: 5.1, 5.2_

- [x] 7. Remove old oneapi_backend implementation
  - Delete `include/backend/oneapi_backend.h`
  - Delete `src/backend/oneapi_backend.cc`
  - Remove any references to `oneapi_backend` class from build files
  - Update any documentation referencing the old backend
  - _Requirements: 5.4_

- [x] 8. Add error handling and diagnostics
  - Ensure SYCL async_handler is properly configured in base class usage
  - Add Level Zero error checking helper if using native APIs
  - Verify `check_async_errors()` inherited from base class works correctly
  - Add debug logging for copy operations using `CELERITY_DEBUG` macro
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 9. Validate with test suite
  - Run `all_tests` executable with Level Zero backend selected
  - Verify no infinite loops in `check_async_errors()`
  - Confirm all accessor tests pass
  - Check for proper cleanup and no memory leaks
  - Validate multi-device scenarios with 2 Intel Arc GPUs
  - _Requirements: 3.2, 4.1, 4.2, 5.4_

- [ ] 10. Performance optimization and validation
  - Benchmark copy operations against native Level Zero performance
  - Profile to ensure no unexpected overhead
  - Optimize work-group sizes for parallel_for copies if needed
  - Compare with CUDA backend performance characteristics
  - _Requirements: 2.2, 2.3_
