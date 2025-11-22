#!/bin/bash
# Build script for Level Zero native backend benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_l0"

echo "=== Building Level Zero Native Backend Benchmark ==="
echo "Build directory: ${BUILD_DIR}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_INTEL_GPU_AOT=OFF

# Build
echo ""
echo "Building..."
cmake --build . --parallel $(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executables:"
echo "  - ${BUILD_DIR}/memcpy_linear (SYCL baseline)"
echo "  - ${BUILD_DIR}/memcpy_linear_l0 (Level Zero native)"
echo ""
echo "Run comparison:"
echo "  ${BUILD_DIR}/memcpy_linear --dev 0"
echo "  ${BUILD_DIR}/memcpy_linear_l0 --dev 0"
