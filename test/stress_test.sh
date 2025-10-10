#!/usr/bin/env bash
# Build script for standalone Level Zero event pool stress test

set -e

echo "=== Building Level Zero Event Pool Stress Test ==="

# Source oneAPI environment
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "Sourcing oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
fi

# Compile
echo "Compiling x_level_zero_pool_stress_tests.cc..."
icpx -fsycl -fsycl-targets=spir64_gen \
     -Xsycl-target-backend=spir64_gen "-device dg2" \
     -I../include \
     -I../vendor/matchbox/include \
     -std=c++20 \
     -O2 \
     -o level_zero_pool_stress \
     x_level_zero_pool_stress_tests.cc \
     -lze_loader

echo "✓ Build successful: ./level_zero_pool_stress"
echo ""
echo "Run:"

SYCL_DEVICE_FILTER=level_zero:gpu ./level_zero_pool_stress
