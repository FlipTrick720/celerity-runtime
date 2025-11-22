#!/bin/bash
# Run comparison between SYCL and Level Zero native backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_l0"

if [ ! -f "${BUILD_DIR}/memcpy_linear_l0" ]; then
    echo "Error: memcpy_linear_l0 not found. Run build_l0_bench.sh first."
    exit 1
fi

DEVICE=${1:-0}
SIZE_MIN=${2:-1024}
SIZE_MAX=${3:-$((64*1024*1024))}
STEPS=${4:-20}

echo "=== Level Zero Backend Comparison ==="
echo "Device: ${DEVICE}"
echo "Size range: ${SIZE_MIN} - ${SIZE_MAX} bytes"
echo "Steps: ${STEPS}"
echo ""

# Create results directory
RESULTS_DIR="${SCRIPT_DIR}/results_l0_comparison"
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Running SYCL baseline benchmark..."
"${BUILD_DIR}/memcpy_linear" \
    --dev ${DEVICE} \
    --min ${SIZE_MIN} \
    --max ${SIZE_MAX} \
    --steps ${STEPS} \
    --csv "${RESULTS_DIR}/sycl_${TIMESTAMP}.csv" \
    | tee "${RESULTS_DIR}/sycl_${TIMESTAMP}.txt"

echo ""
echo "Running Level Zero native benchmark..."
"${BUILD_DIR}/memcpy_linear_l0" \
    --dev ${DEVICE} \
    --min ${SIZE_MIN} \
    --max ${SIZE_MAX} \
    --steps ${STEPS} \
    --csv "${RESULTS_DIR}/l0_${TIMESTAMP}.csv" \
    | tee "${RESULTS_DIR}/l0_${TIMESTAMP}.txt"

echo ""
echo "=== Results saved to ${RESULTS_DIR} ==="
echo "SYCL baseline: ${RESULTS_DIR}/sycl_${TIMESTAMP}.csv"
echo "Level Zero native: ${RESULTS_DIR}/l0_${TIMESTAMP}.csv"
echo ""
echo "To compare results, check the GiB/s columns in the output files."
