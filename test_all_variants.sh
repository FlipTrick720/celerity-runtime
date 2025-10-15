#!/usr/bin/env bash
# Test all backend variants
# Cycles through variant implementations, builds, tests, and benchmarks each

set -euo pipefail

echo "========================================="
echo "Level Zero Backend Variant Testing"
echo "========================================="
echo ""

# Create results directory
RESULTS_DIR="variant_test_results_$(date +%Y%m%d_%H%M%S)"
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

declare -A VARIANT_HEADERS
VARIANT_HEADERS[baseline]="include/backend/sycl_backend.h"
VARIANT_HEADERS[variant1]="include/backend/sycl_backend copy 1.h"
VARIANT_HEADERS[variant2]="include/backend/sycl_backend copy 1.h"
VARIANT_HEADERS[variant3]="include/backend/sycl_backend copy 1.h"
VARIANT_HEADERS[variant4]="include/backend/sycl_backend copy 1.h"

# Test each variant
for variant in baseline variant1 variant2 variant3 variant4; do
    echo ""
    echo "========================================="
    echo "Testing: $variant"
    echo "========================================="
    
    # Create variant log directory
    VARIANT_DIR="$RESULTS_DIR/$variant"
    mkdir -p "$VARIANT_DIR"
    
    # Copy variant files
    echo "Installing $variant files..."
    cp "${VARIANTS[$variant]}" src/backend/sycl_level_zero_backend.cc
    cp "${VARIANT_HEADERS[$variant]}" include/backend/sycl_backend.h
    
    # Update version tag in source
    sed -i "1s|.*|//Version: ${variant}|" src/backend/sycl_level_zero_backend.cc
    sed -i "2s|.*|//Text: Automated test of ${variant}|" src/backend/sycl_level_zero_backend.cc
    
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
        if (cd bench && ./build_bench.sh > /dev/null 2>&1); then
            echo "✓ Benchmark build successful"
        else
            echo "✗ Benchmark build failed"
            continue
        fi
    else
        echo "✓ Using existing benchmark build"
    fi
    
    # Run tests
    echo "Running tests..."
    if ./run_test.sh --profile test > "$VARIANT_DIR/tests.log" 2>&1; then
        echo "✓ Tests passed"
    else
        echo "✗ Tests failed - check $VARIANT_DIR/tests.log"
    fi
    tail -n 16 "$VARIANT_DIR/tests.log"
    
    # Cool down
    echo "Cool down (30s)..."
    sleep 30
    
    # Run benchmarks (calls run_bench.sh logic)
    echo "Running benchmarks..."
    if ./run_bench.sh > "$VARIANT_DIR/bench.log" 2>&1; then
        echo "✓ Benchmarks complete"
        echo "  Results: bench/results/results_${variant}_*"
    else
        echo "✗ Benchmarks failed - check $VARIANT_DIR/bench.log"
    fi
    
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
for variant in baseline variant1 variant2 variant3 variant4; do
    if [ -d "$RESULTS_DIR/$variant" ]; then
        echo "  $variant/"
        echo "    ├── build.log"
        echo "    ├── tests.log"
        echo "    └── bench.log"
    fi
done

echo ""
echo "Benchmark results: bench/results/"
for variant in baseline variant1 variant2 variant3 variant4; do
    result_dir=$(ls -d bench/results/results_${variant}_* 2>/dev/null | tail -1)
    if [ -n "$result_dir" ]; then
        echo "  ${variant}: ${result_dir}"
    fi
done

echo ""
echo "Next steps:"
echo "  1. Download bench/results/ to local machine"
echo "  2. Edit compare_bench.sh:"
echo "     VERSIONS=(\"baseline\" \"variant1\" \"variant2\" \"variant3\" \"variant4\")"
echo "  3. Run: ./compare_bench.sh"
echo ""
