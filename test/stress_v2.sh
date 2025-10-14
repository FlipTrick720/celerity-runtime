#!/usr/bin/env bash

set -e

# Find the newest build directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR=$(ls -td "$REPO_ROOT"/build_* 2>/dev/null | head -1)

if [ -z "$BUILD_DIR" ]; then
    echo "Error: No build directory found. Run ./build_celerity.sh first."
    exit 1
fi

echo "=== Building Enhanced Stress Test v2 ==="
echo "Using build directory: $BUILD_DIR"

# Source oneAPI environment
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
fi

# Build in the test directory of the newest build
TEST_BUILD_DIR="$BUILD_DIR/test"
mkdir -p "$TEST_BUILD_DIR"
cd "$TEST_BUILD_DIR"

echo "Compiling x_level_zero_pool_stress_tests_v2.cc..."
icpx -fsycl -fsycl-targets=spir64_gen \
     -Xsycl-target-backend=spir64_gen "-device dg2" \
     -I"$REPO_ROOT/include" \
     -I"$REPO_ROOT/vendor/matchbox/include" \
     -std=c++20 \
     -O2 \
     -o stress_v2 \
     "$SCRIPT_DIR/x_level_zero_pool_stress_tests_v2.cc" \
     -lze_loader

if [ ! -f stress_v2 ]; then
    echo "Error: Compilation failed"
    exit 1
fi

chmod +x stress_v2

echo "✓ Build successful: $TEST_BUILD_DIR/stress_v2"
echo ""
echo "Running stress test..."

SYCL_DEVICE_FILTER=level_zero:gpu ./stress_v2
