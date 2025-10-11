# Build (oneAPI)
echo "=== Building Level Zero l0 Microbench ==="

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh
fi

echo "Compiling x_level_zero_pool_stress_tests.cc..."

dpcpp -O3 -std=c++17 x_celerity_l0_microbench.cpp -o l0mbench

# Level Zero wählen + CSV schreiben
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --csv l0.csv

chmod +x l0mbench

echo "✓ Build successful: ./l0mbench"
echo ""
echo "Run:"

# Varianten:
# - ohne Host-Pinning
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --no-pin --csv y_l0_pageable.csv
# - Batch-Mode (alle Kopien enqueuen, dann eine Wait)
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --batch --csv y_l0_batch.csv
# - größere Spanne
SYCL_DEVICE_FILTER=level_zero:gpu ./l0mbench --min 1024 --max $((64*1024)) --steps 10 --secs 1 --csv y_l0_1k_64k.csv

