# Level Zero Backend Copy Fix

## Issue Identified

The backend copy tests were failing with incorrect data placement. The test showed:

**Expected layout** (with proper striding):
```
{ 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16 }
```

**Actual result** (data copied to wrong locations):
```
{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
```

## Root Cause

The original implementation had two critical flaws:

1. **Misunderstanding of `dispatch_nd_region_copy`**: I thought this function would call my lambda once with the full region, but it actually decomposes the region into boxes and calls the lambda for each box with adjusted pointers.

2. **Incorrect use of dispatch pattern**: I was trying to use `dispatch_nd_region_copy` which is designed for host-side copies, but for device copies we should directly iterate over the boxes and use `layout_nd_copy` to determine the optimal copy strategy.

## Solution

The fix properly handles strided copies by:

1. **Iterating over copy region boxes**: Each box in the copy region is processed separately
2. **Using `layout_nd_copy` utility**: This function analyzes the copy and determines if it's contiguous or requires strided access
3. **Choosing the right strategy**:
   - **Contiguous**: Use SYCL `queue.memcpy()` for optimal performance
   - **Strided**: Use `parallel_for` kernel with proper 3D-to-linear index conversion

### Key Changes

```cpp
// For each box in the copy region
for(const auto& copy_box : copy_region.get_boxes()) {
    // Calculate source and destination ranges and offsets
    const auto layout = layout_nd_copy(src_range, dst_range, src_offset, dst_offset, copy_range, elem_size);
    
    if(layout.num_complex_strides == 0) {
        // Contiguous - use memcpy
        queue.memcpy(dst_ptr, src_ptr, layout.contiguous_size);
    } else {
        // Strided - use parallel_for with proper indexing
        queue.submit([=](sycl::handler& cgh) {
            cgh.parallel_for(total_elements, [=](sycl::id<1> idx) {
                // Convert linear index to 3D
                // Calculate source and dest indices with proper strides
                // Copy element
            });
        });
    }
}
```

## Testing Instructions

Build and run the tests again:

```bash
cd build
make -j
./test/all_tests "backend copies work correctly"
```

### Expected Results

The backend copy tests should now pass:
- ✅ host to device copies
- ✅ device to host copies  
- ✅ device to itself copies
- ✅ All layout combinations (linearized, strided, mixed)

### What Should Work Now

1. **No infinite loop** ✅ (already fixed in previous iteration)
2. **Correct data placement** ✅ (this fix)
3. **Proper striding** ✅ (respects destination layout)
4. **All copy directions** ✅ (H2D, D2H, D2D)

## Remaining Test Failures

There are still 2 other test failures unrelated to the backend:

1. **Affinity test**: Log level mismatch (warning vs info) - not a backend issue
2. **Side-effect test**: Missing exception - not a backend issue  
3. **Fence test**: Data mismatch - likely related to copy fix, should be resolved

The fence test failure (`fences extract data from buffers`) is likely caused by the same copy bug and should be fixed by this change.

## Performance Notes

- Contiguous copies use direct `memcpy` - optimal
- Strided copies use parallel_for - could be optimized further with:
  - Better work-group sizing
  - Using Level Zero native copy commands
  - Vectorized loads/stores

## Next Steps

1. Test on server with Intel Arc GPUs
2. Verify all backend tests pass
3. Check fence test now passes
4. If all tests pass, we're done with core functionality!
5. Optional: Performance optimization of strided copies

## Debug Commands

If issues persist:

```bash
# Run only backend tests
./test/backend_tests

# Run with trace logging
CELERITY_LOG_LEVEL=trace ./test/backend_tests 2>&1 | grep "Level-Zero"

# Check specific test
./test/all_tests "backend copies work correctly on all source- and destination layouts"
```
