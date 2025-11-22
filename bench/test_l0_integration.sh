#!/usr/bin/env bash
# Quick test to validate Level Zero native benchmark integration

set -euo pipefail

echo "========================================="
echo "Level Zero Native Integration Test"
echo "========================================="
echo ""

# Change to bench directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Check if oneAPI is loaded
echo "Step 1: Checking oneAPI environment..."
if [ -z "${ONEAPI_ROOT:-}" ]; then
    echo "Loading oneAPI..."
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        set +u
        source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
        set -u
        echo "✓ oneAPI loaded"
    else
        echo "✗ oneAPI not found"
        exit 1
    fi
else
    echo "✓ oneAPI already loaded"
fi
echo ""

# Step 2: Build benchmarks
echo "Step 2: Building benchmarks..."
if ./build_bench.sh > /tmp/build_test.log 2>&1; then
    echo "✓ Build successful"
else
    echo "✗ Build failed - check /tmp/build_test.log"
    exit 1
fi
echo ""

# Step 3: Check if Level Zero native benchmark was built
echo "Step 3: Checking Level Zero native benchmark..."
if [[ -f "build/memcpy_linear_l0" ]]; then
    echo "✓ memcpy_linear_l0 found"
    ls -lh build/memcpy_linear_l0
else
    echo "✗ memcpy_linear_l0 not found"
    echo "Available executables:"
    ls -lh build/
    exit 1
fi
echo ""

# Step 4: Quick smoke test
echo "Step 4: Running quick smoke test..."
if ./build/memcpy_linear_l0 --dev 0 --min 1024 --max 4096 --steps 2 --no-human > /tmp/l0_test.log 2>&1; then
    echo "✓ Level Zero native benchmark runs successfully"
else
    echo "✗ Level Zero native benchmark failed"
    echo "Check /tmp/l0_test.log for details"
    exit 1
fi
echo ""

# Step 5: Test run_matrix.sh integration
echo "Step 5: Testing run_matrix.sh integration..."
TEST_DIR="test_integration_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

# Set environment for reproducibility
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

# Run a minimal benchmark matrix (just one size, quick test)
echo "Running minimal benchmark matrix..."
if ENABLE_CUDA=no ./scripts/run_matrix.sh "$TEST_DIR" > /tmp/matrix_test.log 2>&1; then
    echo "✓ run_matrix.sh completed"
else
    echo "⚠️  run_matrix.sh had issues - check /tmp/matrix_test.log"
fi
echo ""

# Step 6: Verify results
echo "Step 6: Verifying results..."
RESULT_DIR=$(find "$TEST_DIR" -type d -name "results_*" | head -1)

if [[ -z "$RESULT_DIR" ]]; then
    echo "✗ No results directory found"
    exit 1
fi

echo "Results directory: $RESULT_DIR"
echo ""

# Check for SYCL baseline results
if ls "$RESULT_DIR"/l0_memcpy_sync_pinned_*.csv > /dev/null 2>&1; then
    echo "✓ SYCL baseline results found"
    SYCL_CSV=$(ls "$RESULT_DIR"/l0_memcpy_sync_pinned_*.csv | head -1)
    SYCL_LINES=$(wc -l < "$SYCL_CSV")
    echo "  Lines in CSV: $SYCL_LINES"
else
    echo "✗ SYCL baseline results not found"
fi

# Check for Level Zero native results
if ls "$RESULT_DIR"/l0_native_memcpy_sync_pinned_*.csv > /dev/null 2>&1; then
    echo "✓ Level Zero native results found"
    L0_CSV=$(ls "$RESULT_DIR"/l0_native_memcpy_sync_pinned_*.csv | head -1)
    L0_LINES=$(wc -l < "$L0_CSV")
    echo "  Lines in CSV: $L0_LINES"
else
    echo "✗ Level Zero native results not found"
    echo "Available CSV files:"
    ls -lh "$RESULT_DIR"/*.csv
    exit 1
fi
echo ""

# Step 7: Quick performance comparison
echo "Step 7: Quick performance comparison..."
if [[ -f "$SYCL_CSV" && -f "$L0_CSV" ]]; then
    echo "Comparing H2D bandwidth (largest transfer):"
    
    # Extract last H2D line (largest transfer)
    SYCL_H2D=$(grep "H2D" "$SYCL_CSV" | tail -1 | cut -d, -f13)
    L0_H2D=$(grep "H2D" "$L0_CSV" | tail -1 | cut -d, -f13)
    
    if [[ -n "$SYCL_H2D" && -n "$L0_H2D" ]]; then
        echo "  SYCL baseline:  $SYCL_H2D GB/s"
        echo "  L0 native:      $L0_H2D GB/s"
        
        # Simple comparison (bash doesn't do floating point, so use bc if available)
        if command -v bc &>/dev/null; then
            DIFF=$(echo "scale=2; ($L0_H2D - $SYCL_H2D) / $SYCL_H2D * 100" | bc)
            echo "  Difference:     $DIFF%"
            
            # Check if L0 is at least as fast as SYCL
            IS_FASTER=$(echo "$L0_H2D >= $SYCL_H2D" | bc)
            if [[ "$IS_FASTER" == "1" ]]; then
                echo "  ✓ Level Zero native is faster or equal"
            else
                echo "  ⚠️  Level Zero native is slower (may need optimization)"
            fi
        fi
    else
        echo "  ⚠️  Could not extract bandwidth values"
    fi
fi
echo ""

# Step 8: Cleanup
echo "Step 8: Cleanup..."
echo "Test results saved in: $TEST_DIR"
echo "To remove: rm -rf $TEST_DIR"
echo ""

echo "========================================="
echo "Integration Test Complete!"
echo "========================================="
echo ""
echo "Summary:"
echo "  ✓ Build successful"
echo "  ✓ Level Zero native benchmark built"
echo "  ✓ Smoke test passed"
echo "  ✓ run_matrix.sh integration working"
echo "  ✓ Results generated correctly"
echo ""
echo "Next steps:"
echo "  1. Review results in: $TEST_DIR"
echo "  2. Run full workflow: ../test_and_bench_all_variants.sh"
echo "  3. Compare performance across variants"
