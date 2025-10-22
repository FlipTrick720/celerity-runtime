#!/usr/bin/env bash
# Test all backend variants
# Cycles through variant implementations, builds, tests, and benchmarks each
# Creates output structure compatible with analyze_results.py and compare_versions.py

set -euo pipefail

echo "========================================="
echo "Level Zero Backend Variant Testing"
echo "========================================="
echo ""

# =========================
#  ENVIRONMENT CONFIGURATION
# =========================

# Base L0 tuning variables
L0_POOL_SIZE=512
L0_USE_BATCHING=1
L0_SMALL_THRESHOLD=4096
L0_MICRO_THRESHOLD=256
L0_BATCH_THRESHOLD_OPS=8
L0_BATCH_THRESHOLD_US=100

# Function: unset all CELERITY_L0_* variables for clean testing
unset_l0_env() {
  unset CELERITY_L0_EVENT_POOL_SIZE
  unset CELERITY_L0_USE_BATCHING
  unset CELERITY_L0_SMALL_THRESHOLD
  unset CELERITY_L0_MICRO_THRESHOLD
  unset CELERITY_L0_BATCH_THRESHOLD_OPS
  unset CELERITY_L0_BATCH_THRESHOLD_US
}

# Function: export all CELERITY_L0_* variables for benchmarking
export_l0_env() {
  export CELERITY_L0_EVENT_POOL_SIZE=${L0_POOL_SIZE}
  export CELERITY_L0_USE_BATCHING=${L0_USE_BATCHING}
  export CELERITY_L0_SMALL_THRESHOLD=${L0_SMALL_THRESHOLD}
  export CELERITY_L0_MICRO_THRESHOLD=${L0_MICRO_THRESHOLD}
  export CELERITY_L0_BATCH_THRESHOLD_OPS=${L0_BATCH_THRESHOLD_OPS}
  export CELERITY_L0_BATCH_THRESHOLD_US=${L0_BATCH_THRESHOLD_US}
}

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="variant_test_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Backup original files
echo "Backing up original backend files..."
cp src/backend/sycl_level_zero_backend.cc "$RESULTS_DIR/backend_original.cc"
cp include/backend/sycl_backend.h "$RESULTS_DIR/backend_h_original.h"

# Define variants
declare -A VARIANTS
VARIANTS[baseline]="src/backend/sycl_level_zero_backend.cc"
VARIANTS[variant1]="src/backend/sycl_level_zero_backend copy 1.cc"
VARIANTS[variant2]="src/backend/sycl_level_zero_backend copy 2.cc"
VARIANTS[variant3]="src/backend/sycl_level_zero_backend copy 3.cc"
VARIANTS[variant4]="src/backend/sycl_level_zero_backend copy 4.cc"
VARIANTS[variant5]="src/backend/sycl_level_zero_backend copy 5.cc"

declare -A VARIANT_HEADERS
VARIANT_HEADERS[baseline]="include/backend/sycl_backend.h"
VARIANT_HEADERS[variant1]="include/backend/sycl_backend copy 1.h"
VARIANT_HEADERS[variant2]="include/backend/sycl_backend copy 1.h"
VARIANT_HEADERS[variant3]="include/backend/sycl_backend copy 1.h"
VARIANT_HEADERS[variant4]="include/backend/sycl_backend copy 1.h"
VARIANT_HEADERS[variant5]="include/backend/sycl_backend copy 1.h"

# Test each variant
for variant in baseline variant1 variant2 variant3 variant4 variant5; do
    echo ""
    echo "========================================="
    echo "Testing: $variant"
    echo "========================================="
    
    # Environment variables are now handled by unset_l0_env() for tests
    # and export_l0_env() for benchmarks - no need to set them here
    
    if [[ "$variant" != "baseline" ]]; then
        echo "Installing $variant files..."
        cp "${VARIANTS[$variant]}" src/backend/sycl_level_zero_backend.cc
        cp "${VARIANT_HEADERS[$variant]}" include/backend/sycl_backend.h
    else
        echo "Using baseline (no file changes needed)..."
    fi

    # Backend version/tag - read from source file
    BACKEND_SOURCE="src/backend/sycl_level_zero_backend.cc"
    BACKEND_TAG=""
    BACKEND_NOTES=""

    if [[ -f "${BACKEND_SOURCE}" ]]; then
    	# Read first line: //Version: v1_baseline
    	BACKEND_TAG=$(grep -m1 "^//Version:" "${BACKEND_SOURCE}" 2>/dev/null | sed 's/^\/\/Version: *//' | tr -d '\r\n' || echo "untagged")
    	# Read second line: //Text: Initial implementation
    	BACKEND_NOTES=$(grep -m1 "^//Text:" "${BACKEND_SOURCE}" 2>/dev/null | sed 's/^\/\/Text: *//' | tr -d '\r\n' || echo "no_notes_found")
    fi

    # Create variant log directory
    VARIANT_DIR="$RESULTS_DIR/$BACKEND_TAG"
    mkdir -p "$VARIANT_DIR"
    
    # Build Celerity
    echo "Building Celerity..."
    if ./build_celerity.sh > "$VARIANT_DIR/build.log" 2>&1; then
        echo "✓ Celerity build successful"
    else
        echo "✗ Celerity build failed - check $VARIANT_DIR/build.log"
        continue
    fi
    
    # Build benchmarks (only once, first variant)
    if [[ "$variant" == "baseline" ]]; then
        echo "Building benchmarks..."
        if (cd bench && ./build_bench.sh) > "$VARIANT_DIR/bench_build.log" 2>&1; then
            echo "✓ Benchmark build successful"
        else
            echo "✗ Benchmark build failed - check $VARIANT_DIR/bench_build.log"
            continue
        fi
    else
        echo "✓ Using existing benchmark build"
    fi
    
    # Run tests with clean environment (no CELERITY_L0_* vars)
    echo "Running tests with clean environment (no CELERITY_L0_* vars)..."
    unset_l0_env
    if ./run_test.sh --profile test > "$VARIANT_DIR/tests.log" 2>&1; then
        echo "✓ Tests passed"
        if [ -f "$VARIANT_DIR/tests.log" ]; then
            tail -n 16 "$VARIANT_DIR/tests.log"
        fi
    else
        echo "⚠️  Tests not available or failed - check $VARIANT_DIR/tests.log"
        echo "    (Continuing with benchmarks...)"
    fi
    
    # Cool down
    echo "Cool down (30s)..."
    sleep 30
    
    # Run benchmarks with Level Zero tuning variables
    echo "Running benchmarks with CELERITY_L0_* vars exported:"
    export_l0_env
    env | grep CELERITY_L0_
    
    # Set reproducibility environment
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    
    # Ensure bench.log directory exists
    mkdir -p "$(dirname "$VARIANT_DIR/bench.log")"
    
    # Run benchmarks and save to variant-specific directory in bench/results/
    cd bench
    if ENABLE_CUDA=no taskset -c 0-15 ./scripts/run_matrix.sh results > "../$VARIANT_DIR/bench.log" 2>&1; then
        echo "✓ Benchmarks complete"        
    else
        echo "✗ Benchmarks failed - check $VARIANT_DIR/bench.log"
    fi
    cd ..
       
    echo "$variant complete!"
done

# Restore original files
echo ""
echo "Restoring original backend files..."
cp "$RESULTS_DIR/backend_original.cc" src/backend/sycl_level_zero_backend.cc
cp "$RESULTS_DIR/backend_h_original.h" include/backend/sycl_backend.h

echo ""
echo "========================================="
echo "All Variants Complete!"
echo "========================================="
echo ""
echo "Test logs: $RESULTS_DIR/"
for variant in baseline variant1 variant2 variant3 variant4 variant5; do
    if [ -d "$RESULTS_DIR/$variant" ]; then
        echo "  $variant/"
        echo "    ├── build.log"
        echo "    ├── tests.log"
        echo "    ├── bench_build.log (if first variant)"
        echo "    └── bench.log"
    fi
done

echo ""
echo "Benchmark results: bench/results/"

# Final cleanup - unset L0 environment variables
unset_l0_env
echo "✓ Environment cleaned up"
