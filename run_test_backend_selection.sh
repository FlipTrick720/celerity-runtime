#!/bin/bash

source /opt/intel/oneapi/setvars.sh

# Compile the backend selection debug tool
icpx -fsycl -std=c++20 -DSYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO test_backend_selection.cc -o test_backend_selection

# Run it with the same environment
./test_backend_selection
