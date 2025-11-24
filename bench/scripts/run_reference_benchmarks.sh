#!/usr/bin/env bash
# Run reference benchmarks (L0 Native + Generic SYCL) once per test run
# These are hardware-dependent, not backend-variant-dependent

set -euo pipefail

echo "========================================="
echo "Running Reference Benchmarks"
echo "========================================="
echo ""

# Ensure oneAPI is loaded (needed to run SYCL executables)
if [ -z "${ONEAPI_ROOT:-}" ]; then
    echo "Loading oneAPI environment..."
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        set +u  # Temporarily disable -u for oneAPI script
        source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
        set -u
        echo "✓ oneAPI loaded"
    else
        echo "⚠️  Warning: oneAPI not found at /opt/intel/oneapi/setvars.sh"
        echo "   Benchmarks may fail if SYCL libraries are not in LD_LIBRARY_PATH"
    fi
else
    echo "✓ oneAPI already loaded"
fi
echo ""

# Check if we're in bench directory
if [[ ! -f "CMakeLists.txt" ]]; then
    if [[ -d "bench" ]]; then
        cd bench
    else
        echo "Error: Cannot find bench directory"
        exit 1
    fi
fi

# Ensure benchmarks are built
if [[ ! -f "build/memcpy_linear" ]] || [[ ! -f "build/memcpy_linear_l0" ]]; then
    echo "Building benchmarks..."
    ./build_bench.sh
fi

# Create reference results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REF_DIR="reference_results_${TIMESTAMP}"
mkdir -p "$REF_DIR"

echo "Reference results will be saved to: $REF_DIR"
echo ""

# Set reproducibility environment
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

# =========================
# 1. L0 Native Benchmark
# =========================
echo "=== Running L0 Native Benchmark ==="
echo "This measures hardware capability using direct Level Zero API"
echo ""

# Run full matrix for L0 Native
for mode_flag in "" "--batch"; do
    for pin_flag in "" "--no-pin"; do
        mode_name=$([ -z "$mode_flag" ] && echo "sync" || echo "batch")
        pin_name=$([ -z "$pin_flag" ] && echo "pinned" || echo "pageable")
        
        csv_name="l0_native_memcpy_${mode_name}_${pin_name}_${TIMESTAMP}.csv"
        
        echo "Running L0 Native: ${mode_name} + ${pin_name}"
        ./build/memcpy_linear_l0 \
            --dev 0 \
            --min 1024 \
            --max $((1<<26)) \
            --steps 16 \
            --secs 1 \
            --csv "${REF_DIR}/${csv_name}" \
            ${mode_flag} ${pin_flag} || echo "Warning: L0 Native benchmark failed"
    done
done

echo ""
echo "✓ L0 Native benchmarks complete"
echo ""

# =========================
# 2. Generic SYCL Benchmark
# =========================
echo "=== Running Generic SYCL Benchmark ==="
echo "This measures baseline SYCL performance (OpenCL backend)"
echo ""

# Try to force generic/OpenCL backend
# Method 1: Use OpenCL device filter
export SYCL_DEVICE_FILTER=opencl:gpu
export ONEAPI_DEVICE_SELECTOR=opencl:gpu

# Check if OpenCL device is available
if ./build/memcpy_linear --dev 0 --min 1024 --max 4096 --steps 1 --no-human > /dev/null 2>&1; then
    echo "OpenCL backend available, running generic SYCL benchmarks..."
    
    # Run full matrix for Generic SYCL
    for mode_flag in "" "--batch"; do
        for pin_flag in "" "--no-pin"; do
            mode_name=$([ -z "$mode_flag" ] && echo "sync" || echo "batch")
            pin_name=$([ -z "$pin_flag" ] && echo "pinned" || echo "pageable")
            
            csv_name="generic_memcpy_${mode_name}_${pin_name}_${TIMESTAMP}.csv"
            
            echo "Running Generic SYCL: ${mode_name} + ${pin_name}"
            ./build/memcpy_linear \
                --dev 0 \
                --min 1024 \
                --max $((1<<26)) \
                --steps 16 \
                --secs 1 \
                --csv "${REF_DIR}/${csv_name}" \
                ${mode_flag} ${pin_flag} || echo "Warning: Generic SYCL benchmark failed"
        done
    done
    
    echo ""
    echo "✓ Generic SYCL benchmarks complete"
else
    echo "⚠️  OpenCL backend not available, skipping Generic SYCL benchmarks"
    echo "   This is OK - you can compare L0 Backend vs L0 Native only"
fi

# Restore environment
unset SYCL_DEVICE_FILTER
unset ONEAPI_DEVICE_SELECTOR
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

echo ""
echo "========================================="
echo "Reference Benchmarks Complete"
echo "========================================="
echo ""
echo "Results saved to: $REF_DIR"
echo ""

# Check if any CSV files were created
csv_count=$(ls -1 "$REF_DIR"/*.csv 2>/dev/null | wc -l)
if [ "$csv_count" -gt 0 ]; then
    echo "Files:"
    ls -lh "$REF_DIR"/*.csv
    echo ""
    echo "These reference results can be copied to each variant directory"
    echo "for consistent comparison across all backend variants."
else
    echo "⚠️  WARNING: No CSV files were created!"
    echo "   Reference benchmarks failed to run properly."
    echo "   Check that:"
    echo "   - oneAPI environment is loaded"
    echo "   - Benchmarks are built (./build_bench.sh)"
    echo "   - Level Zero drivers are installed"
    exit 1
fi
