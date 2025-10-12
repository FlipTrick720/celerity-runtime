#!/bin/bash
# Automated testing script for all Level Zero backend variants
# Usage: ./test_all_variants.sh

set -e

echo "========================================="
echo "Level Zero Backend Variant Testing"
echo "========================================="
echo ""

# Ensure test scripts are executable
echo "Setting execute permissions on test scripts..."
chmod +x test/stress_v1.sh test/stress_v2.sh test/l0_microbench.sh build_celerity.sh run_test.sh

# Create results directory
RESULTS_DIR="variant_test_results_$(date +%Y%m%d_%H%M%S)"
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
    
    # Build
    echo "Building..."
    if ! ./build_celerity.sh > "$RESULTS_DIR/build_${variant}.log" 2>&1; then
        echo "ERROR: Build failed for $variant"
        echo "Check $RESULTS_DIR/build_${variant}.log for details"
        continue
    fi
    echo "Build successful!"
    
    # Run run_test (full test suite for correctness verification)
    echo "Running all_tests..."
    if ./run_test.sh --profile test > "$RESULTS_DIR/${variant}_all_tests.txt" 2>&1; then
        echo "✓ all_tests complete"
    else
        echo "✗ all_tests failed"
    fi

    # Run performance benchmarks
    echo "Running stress test v1..."
    cd test
    if ./stress_v1.sh > "../$RESULTS_DIR/${variant}_stress_v1.txt" 2>&1; then
        echo "✓ Stress v1 complete"
    else
        echo "✗ Stress v1 failed"
    fi
    
    echo "Running stress test v2..."
    if ./stress_v2.sh > "../$RESULTS_DIR/${variant}_stress_v2.txt" 2>&1; then
        echo "✓ Stress v2 complete"
    else
        echo "✗ Stress v2 failed"
    fi
    
    echo "Running microbench..."
    if ./l0_microbench.sh > "../$RESULTS_DIR/${variant}_microbench.txt" 2>&1; then
        echo "✓ Microbench complete"
    else
        echo "✗ Microbench failed"
    fi
    
    cd ..
    
    echo ""
    echo "$variant testing complete!"
    echo ""
    sleep 2
done

# Restore originals
echo "Restoring original backend files..."
cp "$RESULTS_DIR/sycl_level_zero_backend_original.cc.backup" src/backend/sycl_level_zero_backend.cc
cp "$RESULTS_DIR/sycl_backend_original.h.backup" include/backend/sycl_backend.h

echo "========================================="
echo "All tests complete!"
echo "========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To analyze results:"
echo "  grep 'Time:' $RESULTS_DIR/*_stress_*.txt"
echo "  grep 'GiB/s' $RESULTS_DIR/*_microbench.txt"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR"
echo "  2. Create comparison table"
echo "  3. Write response to professor"
echo ""
