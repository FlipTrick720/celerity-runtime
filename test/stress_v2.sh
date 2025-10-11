#!/usr/bin/env bash

set -e

echo "=== Building Enhanced Stress Test v2 ==="

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh
fi

echo "Compiling x_level_zero_pool_stress_tests_v2.cc..."
icpx -fsycl -fsycl-targets=spir64_gen \
     -Xsycl-target-backend=spir64_gen "-device dg2" \
     -std=c++20 \
     -O2 \
     -o stress_v2 \
     x_level_zero_pool_stress_tests_v2.cc \
     -lze_loader

chmod +x stress_v2

echo "✓ Build successful: ./stress_v2"
echo ""
echo "Run with:"
echo "  SYCL_DEVICE_FILTER=level_zero:gpu ./stress_v2"

SYCL_DEVICE_FILTER=level_zero:gpu ./stress_v2
