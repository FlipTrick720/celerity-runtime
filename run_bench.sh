#!/usr/bin/env bash
# Run benchmarks on server
# Saves CSVs directly to bench/results/ for later analysis

set -euo pipefail

# Read version from backend source
BACKEND_SOURCE="src/backend/sycl_level_zero_backend.cc"
VERSION="unknown"
if [[ -f "${BACKEND_SOURCE}" ]]; then
    VERSION=$(grep -m1 "^//Version:" "${BACKEND_SOURCE}" 2>/dev/null | sed 's/^\/\/Version: *//' | tr -d '\r\n' || echo "unknown")
fi

echo "========================================="
echo "Running Backend Benchmarks"
echo "========================================="
echo "Backend Version: ${VERSION}"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Build benchmarks first
echo "Building benchmarks..."
cd bench
if ! ./build_bench.sh > /dev/null 2>&1; then
    echo "Error: Benchmark build failed"
    echo "Run manually: cd bench && ./build_bench.sh"
    exit 1
fi
echo "✓ Build complete"
echo ""

# Reproducibility settings
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

# Run benchmark - saves directly to bench/results/
echo "Starting benchmark run..."
ENABLE_CUDA=no taskset -c 0-15 ./scripts/run_matrix.sh results

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
echo "Results saved to: bench/results/results_${VERSION}_*"
echo ""
echo "To analyze locally:"
echo "  1. Download bench/results/ directory"
echo "  2. Edit compare_bench.sh to set VERSIONS=(\"${VERSION}\")"
echo "  3. Run: ./compare_bench.sh"
echo ""

