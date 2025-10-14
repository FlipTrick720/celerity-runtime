# Celerity Backend Benchmarking Suite

Micro-benchmarks for comparing SYCL backend performance (Level Zero vs CUDA).

## Quick Start

### Build
```bash
cd bench
cmake -S . -B build
cmake --build build -j
```

### Run All Tests (Automated)
```bash
./scripts/run_matrix.sh out
# Results in ./out/ with timestamps
```

### Run Individual Tests

**Level Zero:**
```bash
SYCL_DEVICE_FILTER=level_zero:gpu ./build/memcpy_linear \
  --csv l0_memcpy.csv --min 1024 --max $((1<<26)) --steps 16 --secs 1

SYCL_DEVICE_FILTER=level_zero:gpu ./build/event_overhead \
  --csv l0_overhead.csv
```

**CUDA:**
```bash
SYCL_DEVICE_FILTER=cuda:gpu ./build/memcpy_linear \
  --csv cuda_memcpy.csv --min 1024 --max $((1<<26)) --steps 16 --secs 1

SYCL_DEVICE_FILTER=cuda:gpu ./build/event_overhead \
  --csv cuda_overhead.csv
```

## Benchmarks

### memcpy_linear
Tests memory copy bandwidth for:
- **H2D**: Host to Device
- **D2H**: Device to Host  
- **D2D**: Device to Device

Measures bandwidth in GiB/s across geometric size progression.

### event_overhead
Measures SYCL event submission overhead:
- Empty kernel submission
- Tiny (1-byte) memcpy

Useful for understanding backend scheduling costs.

## Command-Line Options

```
--csv <file>        Write CSV results to <file>
--no-human          Suppress human table output
--batch             Enqueue all ops then one wait (vs. sync after each)
--no-pin            Use pageable host memory
--min <bytes>       Minimum size (default 1024)
--max <bytes>       Maximum size (default 1048576)
--steps <n>         Geometric steps between min..max (default 13)
--secs <s>          Time budget per size (default 1s)
--reps-cap <n>      Upper cap for reps (default 5000)
--dev <i>           GPU device index (default 0)
--verbose           Print extra info
--help              Show this help
```

## Directory Structure

```
bench/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── common/
│   ├── bench_util.hpp       # Argument parsing, timing, CSV output
│   └── sycl_helpers.hpp     # SYCL queue/device utilities
├── micro/
│   ├── memcpy_linear.cpp    # Memory copy bandwidth test
│   └── event_overhead.cpp   # Event submission overhead test
└── scripts/
    └── run_matrix.sh        # Automated test runner
```

## Requirements

- SYCL compiler (Intel oneAPI DPC++, AdaptiveCpp, etc.)
- CMake 3.16+
- C++17 compiler
- GPU hardware (Intel Arc/Data Center GPU for Level Zero, NVIDIA for CUDA)

