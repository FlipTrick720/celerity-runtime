#!/usr/bin/env bash

set -e

echo "=== Building Level Zero Event Pool Stress Test ==="

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh
fi

echo "Compiling x_level_zero_pool_stress_tests.cc..."
icpx -fsycl -fsycl-targets=spir64_gen \
     -Xsycl-target-backend=spir64_gen "-device dg2" \
     -I../include \
     -I../vendor/matchbox/include \
     -std=c++20 \
     -O2 \
     -o stress_v1 \
     x_level_zero_pool_stress_tests.cc \
     -lze_loader

chmod +x stress_v1

echo "✓ Build successful: ./stress_v1"
echo ""
echo "Run:"

SYCL_DEVICE_FILTER=level_zero:gpu ./stress_v1
