#!/bin/bash
# Test both GPUs to see if there's a performance difference

# Load oneAPI (don't exit on error)
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
fi

echo "========================================"
echo "Testing Both Arc A770 GPUs"
echo "========================================"
echo ""

# Test GPU 0
echo "Testing GPU 0 (9a:00.0)..."
SYCL_DEVICE_FILTER=level_zero:gpu:0 ./bench/build/memcpy_linear \
  --csv gpu0_test.csv --min 1024 --max 67108864 --steps 16 --secs 1

echo ""
echo "Testing GPU 1 (b3:00.0)..."
SYCL_DEVICE_FILTER=level_zero:gpu:1 ./bench/build/memcpy_linear \
  --csv gpu1_test.csv --min 1024 --max 67108864 --steps 16 --secs 1

echo ""
echo "========================================"
echo "RESULTS COMPARISON (64MB transfers)"
echo "========================================"
echo ""
echo "GPU 0:"
tail -3 gpu0_test.csv
echo ""
echo "GPU 1:"
tail -3 gpu1_test.csv
echo ""
echo "If one GPU is x16 and one is x8, we should see ~2x difference in H2D/D2H speeds"
