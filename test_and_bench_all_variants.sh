#!/usr/bin/env bash
# Test all backend variants
# Cycles through variant implementations, builds, tests, and benchmarks each
# Creates output structure compatible with analyze_results.py and compare_versions.py

set -euo pipefail

echo "========================================="
echo "Level Zero Backend Variant Testing"
echo "========================================="
echo ""

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
    
    if [[ "$variant" != "baseline" ]]; then
        echo "Installing $variant files..."
        cp "${VARIANTS[$variant]}" src/backend/sycl_level_zero_backend.cc
        cp "${VARIANT_HEADERS[$variant]}" include/backend/sycl_backend.h
    else
        echo "Using baseline (no file changes needed)..."
    fi

    # Backend version/tag - read from source file
    BACKEND_SOURCE="../src/backend/sycl_level_zero_backend.cc"
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
    
    # Run tests (optional - don't fail if tests aren't built)
    echo "Running tests..."
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
    
    # Run benchmarks - directly call run_matrix.sh to control output location
    echo "Running benchmarks..."
    
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
FOUND_VARIANTS=()
for variant in baseline variant1 variant2 variant3 variant4 variant5; do
    result_dir=$(ls -d bench/results/results_${variant}_* 2>/dev/null | tail -1)
    if [ -n "$result_dir" ]; then
        echo "  ${variant}: ${result_dir}"
        FOUND_VARIANTS+=("\"${variant}\"")
    fi
done

echo ""
echo "========================================="
echo "Next Steps - Local Analysis"
echo "========================================="
echo ""
echo "1. Download results from server:"
echo "   scp -r server:path/to/bench/results/ bench/"
echo ""
echo "2. Edit compare_bench.sh and set:"
echo "   VERSIONS=(${FOUND_VARIANTS[@]})"
echo ""
echo "3. Run analysis:"
echo "   ./compare_bench.sh"
echo ""
echo "This will generate:"
echo "  - Individual plots: bench/plots_<variant>/"
echo "  - Comparison plots: bench/comparison_all/"
echo "  - Summary tables: bench/plots_*/summary_statistics.csv"
echo ""
