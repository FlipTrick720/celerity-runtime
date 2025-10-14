#!/usr/bin/env bash
set -e

# Ensure oneAPI is loaded
if [ -z "$ONEAPI_ROOT" ]; then
    echo "Warning: ONEAPI_ROOT not set. Attempting to source oneAPI..."
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        source /opt/intel/oneapi/setvars.sh
    else
        echo "Error: oneAPI not found. Please run: source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
fi

# Get Git SHA if available
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build directory
BUILD_DIR="${1:-build}"

echo "Building benchmarks with DPC++ compiler..."
echo "Git SHA: ${GIT_SHA}"

# Configure with icpx (Intel DPC++)
cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DGIT_SHA="${GIT_SHA}"

# Build
cmake --build "${BUILD_DIR}" -j

echo "Build complete! Executables in ${BUILD_DIR}/"
