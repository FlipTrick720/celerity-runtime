# Build
cd bench

rm -rf build # Clean

cmake -S . -B build
cmake --build build -j

# Run automated tests (both backends)
# ./scripts/run_matrix.sh out

# Or run individual tests
echo "=== Test Run: ==="
SYCL_DEVICE_FILTER=level_zero:gpu ./build/memcpy_linear
echo "=== Fertig ==="



