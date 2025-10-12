#!/bin/bash
# Build and run Level Zero Microbench

set -e

# Find the newest build directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR=$(ls -td "$REPO_ROOT"/build_* 2>/dev/null | head -1)

if [ -z "$BUILD_DIR" ]; then
    echo "Error: No build directory found. Run ./build_celerity.sh first."
    exit 1
fi

echo "=== Building Level Zero Microbench ==="
echo "Using build directory: $BUILD_DIR"

# Source oneAPI environment
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
fi

# Build in the test directory of the newest build
TEST_BUILD_DIR="$BUILD_DIR/test"
mkdir -p "$TEST_BUILD_DIR"
cd "$TEST_BUILD_DIR"

echo "Compiling x_celerity_l0_microbench.cpp..."
icpx -fsycl -O3 -std=c++17 "$SCRIPT_DIR/x_celerity_l0_microbench.cpp" -o l0mbench

if [ ! -f l0mbench ]; then
    echo "Error: Compilation failed"
    exit 1
fi

chmod +x l0mbench

echo "✓ Build successful: $TEST_BUILD_DIR/l0mbench"
echo ""
echo "Running benchmarks..."

# Run variants
echo "=== Default (sync-each, pinned) ==="
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --csv l0_default.csv

echo ""
echo "=== Pageable host memory ==="
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --no-pin --csv l0_pageable.csv

echo ""
echo "=== Batch mode ==="
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --batch --csv l0_batch.csv

echo ""
echo "=== Extended range (1KB - 64KB) ==="
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --min 1024 --max $((64*1024)) --steps 10 --secs 1 --csv l0_1k_64k.csv

echo ""
echo "✓ All benchmarks complete. CSV files in: $TEST_BUILD_DIR"

