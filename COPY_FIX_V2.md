# Level Zero Backend Copy Fix V2

## Issue

The previous fix still had the same problem - data was being copied to wrong locations. The logs showed:

```
[trace] Level-Zero backend: contiguous copy 4 bytes
[trace] Level-Zero backend: contiguous copy 16 bytes
... (many contiguous copies)
[trace] Level-Zero backend: strided copy 16 elements of 4 bytes
```

Most copies were being classified as "contiguous" when they should have been "strided".

## Root Cause

I was trying to manually extract boxes from layouts and iterate over the copy region, but I was doing it incorrectly. The key insight is that **`dispatch_nd_region_copy` already does all this work for us!**

The `dispatch_nd_region_copy` function:
1. Handles all layout combinations (strided/strided, strided/linearized, linearized/strided, linearized/linearized)
2. Decomposes the copy region into boxes
3. Calls the appropriate lambda for each box with **adjusted pointers and correct box parameters**

## Solution

Follow the exact pattern used by the CUDA backend:

```cpp
dispatch_nd_region_copy(
    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
    // Box copy lambda - called for strided layouts
    [&](const void* source, void* dest, const box<3>& source_box, 
        const box<3>& dest_box, const box<3>& copy_box) {
        // Use layout_nd_copy to analyze this specific box copy
        // Then either memcpy (contiguous) or parallel_for (strided)
    },
    // Linear copy lambda - called for linearized layouts
    [&](const void* source, void* dest, size_t size_bytes) {
        queue.memcpy(dest, source, size_bytes);
    }
);
```

The key differences from my previous attempt:

1. **Don't manually iterate** over `copy_region.get_boxes()` - let `dispatch_nd_region_copy` do it
2. **Don't try to extract boxes** from layouts - `dispatch_nd_region_copy` provides them
3. **Use the provided pointers** - they're already adjusted for linearized layouts
4. **Call `layout_nd_copy` inside the box lambda** - this analyzes each individual box copy

## Implementation

Created a helper function `nd_copy_box_level_zero` that:
1. Takes source_box, dest_box, and copy_box parameters (provided by dispatch)
2. Calls `layout_nd_copy` to analyze the copy
3. Uses memcpy for contiguous, parallel_for for strided

The main function `nd_copy_device_level_zero` simply calls `dispatch_nd_region_copy` with the two lambdas.

## Why This Works

For the failing test case:
- `source_box = [3,4,0] - [7,8,1]` (allocation, size 4x4x1)
- `dest_box = [3,2,0] - [7,8,1]` (allocation, size 4x6x1)
- `copy_box = [3,4,0] - [7,8,1]` (what to copy, size 4x4x1)

`dispatch_nd_region_copy` will:
1. Recognize both are strided layouts
2. Call the box lambda with these exact boxes
3. Our code calculates:
   - `src_offset = [0,0,0]` (copy starts at source origin)
   - `dst_offset = [0,2,0]` (copy starts at Y=2 in dest)
   - `src_range = [4,4,1]`
   - `dst_range = [4,6,1]` ← Different Y dimension!
4. `layout_nd_copy` sees different dimensions → returns strided layout
5. We use parallel_for with correct indexing

## Testing

```bash
cd build
make -j
./test/backend_tests
```

Should now pass all copy tests:
- ✅ host to device
- ✅ device to host
- ✅ device to itself
- ✅ All layout combinations

## What Changed

**Before**: Manually iterating, extracting boxes incorrectly, calling layout_nd_copy at wrong level

**After**: Using `dispatch_nd_region_copy` properly, letting it handle layout decomposition, calling `layout_nd_copy` for each box

This is exactly how the CUDA backend works, just adapted for SYCL instead of native CUDA calls.
