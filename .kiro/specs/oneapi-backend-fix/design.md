# Design Document: oneAPI Backend Fix

## Overview

The oneAPI backend will be refactored to follow the same architectural pattern as the CUDA backend by inheriting from `sycl_backend` instead of directly implementing the `backend` interface. This approach leverages the existing infrastructure for SYCL queue management, host operations, and event handling while allowing specialization for Level Zero-specific optimizations.

The key insight is that the CUDA backend successfully uses native CUDA operations for device copies while delegating everything else to the SYCL backend base class. We'll apply the same pattern for Level Zero.

## Architecture

### Class Hierarchy

```
backend (interface)
  └── sycl_backend (abstract base)
        ├── sycl_generic_backend
        ├── sycl_cuda_backend
        └── sycl_level_zero_backend (NEW - replaces oneapi_backend)
```

### Key Design Decisions

1. **Inherit from sycl_backend**: This provides automatic handling of:
   - Host memory allocation/deallocation
   - Host task execution
   - Device kernel submission
   - SYCL queue management with per-device lanes
   - Event wrapping and profiling
   - Async error checking

2. **Specialize only device_copy**: Like the CUDA backend, we only override `enqueue_device_copy` to provide optimized implementations for 2D/3D copies using Level Zero or SYCL operations.

3. **Use SYCL interop for Level Zero access**: When needed, use `sycl::get_native<sycl::backend::ext_oneapi_level_zero>()` to access native handles.

4. **Remove manual Level Zero management**: The base class handles SYCL devices and contexts, eliminating the need for manual Level Zero driver/context management.

## Components and Interfaces

### 1. sycl_level_zero_backend Class

```cpp
class sycl_level_zero_backend final : public sycl_backend {
public:
    sycl_level_zero_backend(const std::vector<sycl::device>& devices, 
                            const sycl_backend::configuration& config);
    
    async_event enqueue_device_copy(device_id device, size_t device_lane,
                                    const void* source_base, void* dest_base,
                                    const region_layout& source_layout,
                                    const region_layout& dest_layout,
                                    const region<3>& copy_region,
                                    size_t elem_size) override;
};
```

### 2. Level Zero Copy Implementation

The device copy implementation will follow this strategy:

**Fast Path (Linearized Layouts):**
- Use SYCL `queue.memcpy()` for contiguous 1D copies
- This maps directly to Level Zero's `zeCommandListAppendMemoryCopy`

**2D/3D Strided Copies:**
- Option A: Use SYCL `queue.ext_oneapi_copy_2d()` / `copy_3d()` if available
- Option B: Use Level Zero native `zeCommandListAppendMemoryCopyRegion` via interop
- Option C: Fall back to optimized parallel_for kernel with proper work-group sizing

### 3. Helper Functions

```cpp
namespace level_zero_backend_detail {
    // Perform n-dimensional copy using Level Zero native operations
    async_event nd_copy_device_level_zero(sycl::queue& queue,
                                          const void* source_base,
                                          void* dest_base,
                                          const region_layout& source_layout,
                                          const region_layout& dest_layout,
                                          const region<3>& copy_region,
                                          size_t elem_size,
                                          bool enable_profiling);
}
```

### 4. Backend Factory Integration

Update `make_sycl_backend` in `sycl_backend.cc`:

```cpp
std::unique_ptr<backend> make_sycl_backend(
    const sycl_backend_type type,
    const std::vector<sycl::device>& devices,
    const sycl_backend::configuration& config) 
{
    switch(type) {
        case sycl_backend_type::generic:
            return std::make_unique<sycl_generic_backend>(devices, config);
        case sycl_backend_type::cuda:
            return std::make_unique<sycl_cuda_backend>(devices, config);
        case sycl_backend_type::level_zero:
            return std::make_unique<sycl_level_zero_backend>(devices, config);
    }
}
```

### 5. Backend Enumerator Updates

Update `sycl_backend_enumerator` to recognize Level Zero devices:

```cpp
std::vector<backend_type> compatible_backends(const sycl::device& device) const {
    std::vector<backend_type> result;
    
    if(device.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        result.push_back(sycl_backend_type::level_zero);
    }
    // ... other backends
    
    result.push_back(sycl_backend_type::generic);
    return result;
}
```

## Data Models

### Memory Layout Handling

The copy implementation must handle three layout types:

1. **linearized_layout**: Contiguous memory with byte offset
   - Use direct memcpy
   
2. **strided_layout**: 3D allocation with row-major ordering
   - Use 2D/3D copy operations or strided kernel
   
3. **Mixed layouts**: One linearized, one strided
   - Decompose into appropriate copy operations

### Copy Region Representation

```cpp
struct copy_params {
    const void* source_base;
    void* dest_base;
    box<3> source_box;      // allocation dimensions
    box<3> dest_box;        // allocation dimensions  
    box<3> copy_box;        // region to copy
    size_t elem_size;
};
```

## Error Handling

### SYCL Async Handler

The base class already provides async error handling through SYCL's async_handler mechanism. Our implementation will:

1. Let SYCL queues capture exceptions via async_handler
2. Call `queue.throw_asynchronous()` in `check_async_errors()`
3. Log errors through Celerity's logging system

### Level Zero Error Checking

When using native Level Zero operations:

```cpp
static void ze_check(ze_result_t result, const char* context) {
    if(result != ZE_RESULT_SUCCESS) {
        utils::panic("Level-Zero error in {}: code={}", context, 
                     static_cast<int>(result));
    }
}
```

## Testing Strategy

### Unit Tests

1. **Copy Operations**: Test all combinations of layout types
   - linearized → linearized
   - linearized → strided
   - strided → linearized
   - strided → strided

2. **Event Handling**: Verify async_event completion tracking

3. **Multi-device**: Test with 2 Intel Arc GPUs

### Integration Tests

1. Run existing Celerity test suite with Level Zero backend
2. Verify no infinite loops in `check_async_errors()`
3. Confirm proper cleanup on shutdown

### Performance Validation

1. Compare copy bandwidth with native Level Zero benchmarks
2. Ensure 2D/3D copies are not slower than element-wise fallback
3. Verify profiling overhead is minimal when disabled

## Implementation Notes

### Avoiding the Infinite Loop

The current implementation's infinite loop occurs because:
1. `enqueue_device_copy` submits a parallel_for kernel
2. The kernel doesn't properly synchronize
3. `check_async_errors()` is called repeatedly while waiting
4. Each call logs a debug message, creating millions of log lines

**Solution**: Use SYCL's built-in memcpy operations which properly integrate with the queue's event system, ensuring `check_async_errors()` only checks for actual errors without spinning.

### DPC++ Specific Considerations

DPC++ provides:
- `sycl::queue::memcpy()` for 1D copies
- `sycl::queue::ext_oneapi_memcpy2d()` for 2D copies (if available)
- `sycl::get_native<sycl::backend::ext_oneapi_level_zero>()` for native handle access

We'll prefer SYCL operations over native Level Zero to maintain portability and leverage DPC++'s optimizations.

### Peer-to-Peer Copy Support

Like the CUDA backend, we need to detect and enable peer access between devices:

```cpp
sycl_level_zero_backend::sycl_level_zero_backend(
    const std::vector<sycl::device>& devices,
    const sycl_backend::configuration& config) 
    : sycl_backend(devices, config) 
{
    // Query Level Zero for peer access capabilities
    // Enable peer access where supported
    // Update system_info.memories[].copy_peers accordingly
}
```

## Migration Path

1. **Phase 1**: Create new `sycl_level_zero_backend` class
2. **Phase 2**: Implement basic device copy with SYCL memcpy
3. **Phase 3**: Add optimized 2D/3D copy paths
4. **Phase 4**: Remove old `oneapi_backend` files
5. **Phase 5**: Update build system and documentation

## File Structure

```
include/backend/
  ├── backend.h                    (unchanged)
  ├── sycl_backend.h              (add level_zero backend type)
  └── oneapi_backend.h            (DELETE)

src/backend/
  ├── sycl_backend.cc             (update factory)
  ├── sycl_generic_backend.cc     (unchanged)
  ├── sycl_cuda_backend.cc        (unchanged)
  ├── sycl_level_zero_backend.cc  (NEW)
  └── oneapi_backend.cc           (DELETE)
```

## Dependencies

- DPC++ compiler with Level Zero support
- Level Zero headers (ze_api.h)
- SYCL Level Zero backend extension headers
- Existing Celerity backend infrastructure

## Open Questions

1. Does DPC++ provide `ext_oneapi_memcpy2d` or similar for 2D copies?
   - If yes, use it for strided copies
   - If no, implement using parallel_for with proper work-group sizing

2. Should we use Level Zero command lists directly for copies?
   - Pro: Maximum control and potential performance
   - Con: More complex, bypasses SYCL's queue management
   - Decision: Start with SYCL operations, optimize later if needed

3. How to handle peer access detection?
   - Query Level Zero device properties
   - Test actual copy performance between devices
   - Conservative approach: assume no peer access initially
