# oneAPI Backend Refactoring - Implementation Summary

## Overview

The oneAPI backend has been completely refactored to fix the infinite loop issue and align with the CUDA backend architecture. The new implementation inherits from `sycl_backend` instead of directly implementing the `backend` interface, which provides better integration with Celerity's runtime and eliminates the manual resource management that was causing issues.

## Key Changes

### 1. Architecture Change
- **Old**: `oneapi_backend` directly implemented `backend` interface
- **New**: `sycl_level_zero_backend` inherits from `sycl_backend` base class
- **Benefit**: Leverages existing infrastructure for queue management, host operations, and event handling

### 2. Files Changed

#### Created:
- `src/backend/sycl_level_zero_backend.cc` - New Level Zero backend implementation

#### Modified:
- `include/backend/sycl_backend.h` - Added `sycl_level_zero_backend` class declaration
- `src/backend/sycl_backend.cc` - Updated factory function to use new backend
- `src/backend/CMakeLists.txt` - Updated to compile new source file

#### Deleted:
- `include/backend/oneapi_backend.h` - Old implementation
- `src/backend/oneapi_backend.cc` - Old implementation

### 3. Root Cause of Infinite Loop

The original implementation had a critical issue in `enqueue_device_copy`:
- Used a `parallel_for` kernel for element-wise copying
- The kernel didn't properly synchronize with the SYCL queue's event system
- `check_async_errors()` was called repeatedly while waiting for completion
- Each call logged a debug message, creating millions of log lines
- Test timed out after 5 minutes (exit status 124)

### 4. Solution

The new implementation:
- Uses SYCL's built-in `queue.memcpy()` for contiguous copies
- Uses `dispatch_nd_region_copy()` utility for strided copies
- Properly returns `sycl_event` wrapped in `async_event`
- Calls `sycl_backend_detail::flush(queue)` after submissions
- Inherits `check_async_errors()` from base class which only checks for exceptions

## Implementation Details

### Copy Operations

**Fast Path (Linearized → Linearized):**
```cpp
auto event = queue.memcpy(dst_ptr, src_ptr, total_bytes);
sycl_backend_detail::flush(queue);
return make_async_event<sycl_backend_detail::sycl_event>(std::move(event), enable_profiling);
```

**Strided Path (2D/3D):**
- Uses `dispatch_nd_region_copy()` to decompose regions
- Implements parallel_for kernel with proper indexing
- Optimized for 1, 4, and 8-byte element sizes
- Returns barrier event after dispatch

### Peer-to-Peer Support

The constructor queries Level Zero for peer access capabilities:
```cpp
zeDeviceCanAccessPeer(ze_device_i, ze_device_j, &can_access_ij)
```

If both devices can access each other, updates `system_info.memories[].copy_peers` bitset.

### Error Handling

- SYCL async_handler captures exceptions from device operations
- `check_async_errors()` inherited from base class calls `throw_asynchronous()` on queues
- All SYCL operations return events that can be queried for completion
- Level Zero API calls in peer access detection are wrapped in try-catch

## Compilation Fixes Applied

Two compilation errors were fixed:

1. **Removed old include**: Deleted `#include "backend/oneapi_backend.h"` from `sycl_backend.cc`
2. **Removed duplicate class declaration**: The `sycl_level_zero_backend` class was declared in both the header and .cc file - removed from .cc file

## Testing Instructions

### Build on Server

```bash
cd /home/malte.braig/testApproach/celerity-runtime
mkdir -p build && cd build
cmake .. -DCELERITY_ENABLE_LEVEL_ZERO_BACKEND=ON
make -j$(nproc)
```

### Run Tests

```bash
# Run all tests
./test/all_tests

# Run specific test that was failing
./test/all_tests "accessor supports multi-dimensional subscript operator - 2"

# Run with verbose logging
CELERITY_LOG_LEVEL=debug SPDLOG_LEVEL=debug ./test/all_tests
```

### Expected Results

1. **No infinite loop**: Test should complete in seconds, not timeout
2. **No millions of log lines**: Should see normal debug output
3. **Test passes**: "accessor supports multi-dimensional subscript operator - 2" should pass
4. **Exit status 0**: Not 124 (timeout)

### Verification Checklist

- [ ] Build completes without errors
- [ ] Tests run without infinite loops
- [ ] `check_async_errors()` is called a reasonable number of times (< 100)
- [ ] All accessor tests pass
- [ ] No memory leaks (check with valgrind if needed)
- [ ] Multi-device tests work with 2 Intel Arc GPUs

## Debugging Tips

If issues persist:

1. **Check backend selection**:
   ```bash
   CELERITY_LOG_LEVEL=info ./test/all_tests 2>&1 | grep "backend"
   ```
   Should see: "Using level_zero backend for the selected devices"

2. **Monitor copy operations**:
   ```bash
   CELERITY_LOG_LEVEL=trace ./test/all_tests 2>&1 | grep "Level-Zero backend"
   ```

3. **Check for SYCL errors**:
   ```bash
   UR_LOG_LEVEL=warning ./test/all_tests
   ```

4. **Verify device detection**:
   ```bash
   sycl-ls
   ```
   Should show Level Zero devices

## Performance Notes

- Linearized copies use direct `memcpy` - should be optimal
- Strided copies use parallel_for - may benefit from work-group tuning
- Peer access is automatically detected and enabled where supported
- Profiling overhead is minimal when disabled (uses `discard_events` property)

## Next Steps

After successful testing:

1. Run full test suite to ensure no regressions
2. Benchmark copy performance vs. native Level Zero
3. Test with real Celerity applications
4. Consider optimizing work-group sizes for strided copies
5. Explore using Level Zero's native copy commands for potential performance gains

## Rollback Plan

If the new implementation has issues, you can temporarily revert by:

```bash
git checkout HEAD~1 -- include/backend/oneapi_backend.h src/backend/oneapi_backend.cc
git checkout HEAD~1 -- src/backend/CMakeLists.txt
# Remove new files
rm src/backend/sycl_level_zero_backend.cc
# Rebuild
```

## Contact

If you encounter issues, provide:
- Full build log
- Test output (first 100 and last 100 lines)
- Output of `sycl-ls`
- Celerity version and commit hash
- DPC++ compiler version
