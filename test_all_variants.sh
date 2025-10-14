#!/bin/bash

set -e

echo "========================================="
echo "Level Zero Backend Variant Testing"
echo "========================================="
echo ""

# Ensure test scripts are executable
echo "Setting execute permissions on test scripts..."
chmod +x test/stress_v1.sh test/stress_v2.sh test/l0_microbench.sh build_celerity.sh run_test.sh

# Create results directory
RESULTS_DIR="variant_test_results_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Backup originals
echo "Backing up original backend files..."
cp src/backend/sycl_level_zero_backend.cc "$RESULTS_DIR/sycl_level_zero_backend_original.cc.backup"
cp include/backend/sycl_backend.h "$RESULTS_DIR/sycl_backend_original.h.backup"

# Define variants (both .cc and .h files)
declare -A VARIANT_CC
declare -A VARIANT_H
VARIANT_CC[baseline]="src/backend/sycl_level_zero_backend.cc"
VARIANT_H[baseline]="include/backend/sycl_backend.h"
VARIANT_CC[variant1]="src/backend/sycl_level_zero_backend copy 1.cc"
VARIANT_H[variant1]="include/backend/sycl_backend copy 1.h"
VARIANT_CC[variant2]="src/backend/sycl_level_zero_backend copy 2.cc"
VARIANT_H[variant2]="include/backend/sycl_backend copy 1.h"
VARIANT_CC[variant3]="src/backend/sycl_level_zero_backend copy 3.cc"
VARIANT_H[variant3]="include/backend/sycl_backend copy 1.h"
VARIANT_CC[variant4]="src/backend/sycl_level_zero_backend copy 4.cc"
VARIANT_H[variant4]="include/backend/sycl_backend copy 1.h"

# Number of iterations for each test
NUM_ITERATIONS=5

# Test each variant
for variant in baseline variant1 variant2 variant3 variant4; do
    echo "========================================="
    echo "Testing: $variant"
    echo "========================================="
    
    source_cc="${VARIANT_CC[$variant]}"
    source_h="${VARIANT_H[$variant]}"
    
    # Copy source files
    if [ "$variant" != "baseline" ] || [ ! -f "src/backend/sycl_level_zero_backend.cc" ]; then
        echo "Copying $source_cc to src/backend/sycl_level_zero_backend.cc"
        cp "$source_cc" src/backend/sycl_level_zero_backend.cc
        echo "Copying $source_h to include/backend/sycl_backend.h"
        cp "$source_h" include/backend/sycl_backend.h
    fi
    
    # Create variant-specific directory for logs
    VARIANT_LOG_DIR="$RESULTS_DIR/${variant}"
    mkdir -p "$VARIANT_LOG_DIR"
    
    # Build
    echo "Building..."
    if ! ./build_celerity.sh > "$VARIANT_LOG_DIR/build.log" 2>&1; then
        echo "ERROR: Build failed for $variant"
        echo "Check $VARIANT_LOG_DIR/build.log for details"
        continue
    fi
    echo "Build successful!"
    
    # Run run_test
    echo "Running all_tests..."
    if ./run_test.sh --profile test > "$VARIANT_LOG_DIR/all_tests.log" 2>&1; then
        echo "✓ all_tests complete"
    else
        echo "✗ all_tests failed"
    fi
    tail -n 14 "$VARIANT_LOG_DIR/all_tests.log"

    # Run performance benchmarks
    #cd test
    
    # Stress test v1
    #for i in $(seq 1 $NUM_ITERATIONS); do
    #    echo "Running stress test v1 (iteration $i/$NUM_ITERATIONS)..."
    #    if ./stress_v1.sh > "../$RESULTS_DIR/${variant}_stress_v1_${i}.txt" 2>&1; then
    #        echo "✓ Stress v1 iteration $i complete"
    #    else
    #        echo "✗ Stress v1 iteration $i failed"
    #    fi
    #done
    
    # Stress test v2
    #for i in $(seq 1 $NUM_ITERATIONS); do
    #    echo "Running stress test v2 (iteration $i/$NUM_ITERATIONS)..."
    #    if ./stress_v2.sh > "../$RESULTS_DIR/${variant}_stress_v2_${i}.txt" 2>&1; then
    #        echo "✓ Stress v2 iteration $i complete"
    #    else
    #        echo "✗ Stress v2 iteration $i failed"
    #    fi
    #done
    
    # Microbench
    #for i in $(seq 1 $NUM_ITERATIONS); do
    #    echo "Running microbench (iteration $i/$NUM_ITERATIONS)..."
    #    if ./l0_microbench.sh > "../$RESULTS_DIR/${variant}_microbench_${i}.txt" 2>&1; then
    #        echo "✓ Microbench iteration $i complete"
    #    else
    #        echo "✗ Microbench iteration $i failed"
    #    fi
    #done
    
    #cd ..
    
    # Backend Benchmarks (NO PYTHON - saves directly to bench/results/)
    echo "Running backend benchmarks..."
    echo "Cool Down Phase (30s)..."
    sleep 30
    
    cd bench
    
    # Set reproducibility environment
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    
    # Run benchmark - saves directly to bench/results/ (same format as run_bench.sh)
    echo "Benchmarking ${variant}..."
    BACKEND_TAG="${variant}" \
    BACKEND_NOTES="Automated test of ${variant}" \
    ENABLE_CUDA=no \
    taskset -c 0-15 ./scripts/run_matrix.sh results 2>&1 | tee "../$VARIANT_LOG_DIR/bench.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Backend benchmark complete"
        echo "  Results saved to: bench/results/results_${variant}_*"
        echo "  Results saved to: $VARIANT_DIR/bench/"
    else
        echo "✗ Backend benchmark failed"
    fi
    
    cd ..
    
    echo ""
    echo "$variant testing complete!"
    echo ""
    sleep 5
done

# Restore originals
echo "Restoring original backend files..."
cp "$RESULTS_DIR/sycl_level_zero_backend_original.cc.backup" src/backend/sycl_level_zero_backend.cc
cp "$RESULTS_DIR/sycl_backend_original.h.backup" include/backend/sycl_backend.h

echo "========================================="
echo "All runs complete!"
echo "========================================="
echo ""
echo "Test logs saved to: $RESULTS_DIR/"
for variant in baseline variant1 variant2 variant3 variant4; do
    if [ -d "$RESULTS_DIR/${variant}" ]; then
        echo "  - $variant: $RESULTS_DIR/${variant}/"
        echo "      ├── build.log"
        echo "      ├── all_tests.log"
        echo "      └── bench.log"
    fi
done
echo ""
echo "Benchmark results saved to: bench/results/"
for variant in baseline variant1 variant2 variant3 variant4; do
    result_dir=$(ls -d bench/results/results_${variant}_* 2>/dev/null | tail -1)
    if [ -n "$result_dir" ]; then
        echo "  - ${variant}: ${result_dir}"
    fi
done
echo ""
echo "To analyze locally:"
echo "  1. Download bench/results/ directory"
echo "  2. Edit compare_bench.sh to set VERSIONS=(\"baseline\" \"variant1\" \"variant2\" ...)"
echo "  3. Run: ./compare_bench.sh"
echo ""
echo "Optional: Download test logs from $RESULTS_DIR/ for debugging"
echo ""
