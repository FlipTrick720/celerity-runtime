#!/bin/bash
# Quick test to validate Level Zero benchmark build and basic functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_l0"

echo "=== Level Zero Benchmark Build Test ==="
echo ""

# Check if oneAPI is sourced
if ! command -v icpx &> /dev/null; then
    echo "Error: icpx compiler not found"
    echo "Please source oneAPI environment:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

echo "✓ Intel oneAPI compiler found: $(which icpx)"

# Check for Level Zero headers
if ! echo '#include <level_zero/ze_api.h>' | icpx -fsycl -x c++ -c - -o /dev/null 2>/dev/null; then
    echo "✗ Level Zero headers not found"
    echo "Install with: sudo apt install level-zero-dev"
    exit 1
fi

echo "✓ Level Zero headers found"

# Check for SYCL devices
echo ""
echo "Checking SYCL devices..."
if command -v sycl-ls &> /dev/null; then
    sycl-ls | grep -i "level_zero\|gpu" || echo "Warning: No Level Zero GPU devices found"
else
    echo "Warning: sycl-ls not found, skipping device check"
fi

# Build
echo ""
echo "Building benchmarks..."
cd "${SCRIPT_DIR}"
./build_l0_bench.sh

# Quick smoke test
echo ""
echo "Running quick smoke test..."
if [ -f "${BUILD_DIR}/memcpy_linear_l0" ]; then
    echo "Testing Level Zero benchmark with minimal parameters..."
    "${BUILD_DIR}/memcpy_linear_l0" --dev 0 --min 1024 --max 4096 --steps 2 --no-human || {
        echo "✗ Level Zero benchmark failed"
        exit 1
    }
    echo "✓ Level Zero benchmark executed successfully"
else
    echo "✗ memcpy_linear_l0 not built"
    exit 1
fi

echo ""
echo "=== Build Test Complete ==="
echo "Ready to run full benchmarks:"
echo "  ./run_l0_comparison.sh 0"
