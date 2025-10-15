#!/usr/bin/env bash
set -euo pipefail

# Change to bench directory if not already there
if [[ ! -f "CMakeLists.txt" ]]; then
    if [[ -d "bench" ]]; then
        cd bench
    else
        echo "Error: Cannot find bench directory"
        exit 1
    fi
fi

echo "========================================="
echo "Building Benchmark Executables"
echo "========================================="

# Build directory - use "build" to match run_matrix.sh expectations
BUILD_DIR="build"

# Clean old build
if [[ -d "${BUILD_DIR}" ]]; then
    echo "Cleaning old build..."
    rm -rf "${BUILD_DIR}"
fi

# Ensure oneAPI is loaded
if [ -z "${ONEAPI_ROOT:-}" ]; then
    echo "Loading oneAPI environment..."
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        # Temporarily disable -u flag for oneAPI script
        set +u
        source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
        set -u
        echo "✓ oneAPI loaded"
    else
        echo "Error: oneAPI not found at /opt/intel/oneapi/setvars.sh"
        echo "Please run: source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
else
    echo "✓ oneAPI already loaded"
fi

# Get Git SHA if available
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "Git SHA: ${GIT_SHA}"

# Configure and build
echo "Configuring and building..."
cmake -S . -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo "Executables:"
ls -lh "${BUILD_DIR}"/memcpy_linear "${BUILD_DIR}"/event_overhead
echo ""
