# How to Update Backend Version

## Quick Guide

When you make changes to a backend, update the first two lines of the source file:

### Level Zero Backend (`sycl_level_zero_backend.cc`)

```cpp
//Version: v2_optimized
//Text: Added batch command list submission for reduced overhead
#include "backend/sycl_backend.h"
...
```

### CUDA Backend (`sycl_cuda_backend.cc`)

```cpp
//Version: v2_optimized
//Text: Improved stream synchronization
#include "backend/sycl_backend.h"
...
```

## Format Rules

1. **Line 1**: `//Version: <tag>`
   - Short identifier (e.g., `v1_baseline`, `v2_batch_opt`, `v3_async`)
   - No spaces in tag (use underscores)
   - Used for result directory naming

2. **Line 2**: `//Text: <description>`
   - Brief description of this version
   - Can have spaces
   - Appears in metadata.txt

## Automatic Detection

The `bench/scripts/run_matrix.sh` script automatically reads these lines:

```bash
# Runs automatically detect version from source
./bench/scripts/run_matrix.sh results
# → results/results_v2_optimized_20251014_153045/

# Manual override still possible
BACKEND_TAG="custom_test" ./bench/scripts/run_matrix.sh results
# → results/results_custom_test_20251014_153045/
```

## Version Naming Convention

### Suggested Format: `v <number>_<short_description>`

Examples:
- `v1_baseline` - Initial implementation
- `v2_batch_opt` - Batch optimization
- `v3_async_copy` - Async copy engines
- `v4_mem_pool` - Memory pool optimization
- `v5_event_opt` - Event handling optimization

### For Experiments:
- `exp_feature_x` - Experimental feature X
- `test_approach_y` - Testing approach Y
- `debug_issue_123` - Debugging specific issue

## Checking Current Version

```bash
# Check Level Zero version
head -2 src/backend/sycl_level_zero_backend.cc

# Check CUDA version
head -2 src/backend/sycl_cuda_backend.cc
```

## Example Git History

```
commit b7e4d2a
    Level Zero: Add batch command list optimization
    
    //Version: v2_batch_opt
    //Text: Batch command lists to reduce submission overhead
    
    Results: 37% reduction in small transfer overhead

commit a3f2c1b
    Level Zero: Initial implementation
    
    //Version: v1_baseline
    //Text: Initial implementation
```

## Manual Override

If you need to override the auto-detected version:

```bash
# Override both
BACKEND_TAG="custom_test" \
BACKEND_NOTES="Testing specific configuration" \
./scripts/run_matrix.sh results

# Override just tag
BACKEND_TAG="quick_test" ./scripts/run_matrix.sh results

# Override just notes
BACKEND_NOTES="Testing on different hardware" ./scripts/run_matrix.sh results
```

The environment variables always take precedence over source file annotations.
