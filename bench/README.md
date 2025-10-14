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

## Output Formats

### Human-Readable (Console)
```
=== memcpy_linear ===
Device: Intel(R) Arc(TM) A770 Graphics
Backend: level_zero
Mode: sync-each, HostPinned: yes

D2D  size=1024       reps=5000   avg(us)=2.345     GiB/s=0.42   [sync] [pinned]
H2D  size=1024       reps=5000   avg(us)=3.456     GiB/s=0.28   [sync] [pinned]
...
```

### CSV Output
```csv
bench,backend,device,mode,pinned,op,bytes,reps,avg_us,gib_per_s
memcpy_linear,level_zero,"Intel(R) Arc(TM) A770 Graphics",sync,yes,D2D,1024,5000,2.345,0.420000
...
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

## Reproducibility Tips

For publication-quality measurements:

- **Fix CPU Affinity** (Linux): `taskset -c 0-15 ./build/memcpy_linear ...`
- **Force Level Zero**: `export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO`
- **Disable OpenCL** (optional): `export UR_DISABLE_ADAPTERS=OPENCL`
- **Test Both Modes**: Run with `--batch` and without (sync-each)
- **Compare Host Memory**: Test `--no-pin` vs default pinned memory
- **Disable Turbo Boost**: For consistent clock speeds (system-dependent)
- **Multiple Runs**: Average 3-5 runs to account for variance
- **Thermal Stability**: Let GPU warm up, ensure adequate cooling

## Notes

- Benchmarks use `SYCL_DEVICE_FILTER` for backend selection
- Profiling enabled by default for accurate timing
- Automatic repetition calibration based on time budget
- Supports both pinned and pageable host memory
- Batch mode tests enqueue overhead vs synchronization cost

## Future Extensions

- `memcpy_nd.cpp`: 2D/3D pitched/box copies
- `kernel_reduce.cpp`: Reduction benchmarks
- `kernel_stencil.cpp`: Stencil pattern benchmarks
- `summarize.py`: CSV analysis and plotting
- BabelStream integration for industry-standard comparison
