#!/bin/bash

# Make the compile script executable
chmod +x compile_debug.sh

# Compile the device filter test
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -std=c++20 -DSYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO test_device_filter.cc -o test_device_filter

# Run with the same environment as the test script
export SYCL_DEVICE_FILTER=level_zero:gpu
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

./test_device_filter
