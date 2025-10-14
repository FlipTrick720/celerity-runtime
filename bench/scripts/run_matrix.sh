#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-out}"
ENABLE_CUDA="${ENABLE_CUDA:-auto}"  # auto, yes, no

mkdir -p "${OUT_DIR}"

date_tag=$(date +"%Y%m%d_%H%M%S")
build_dir="${OUT_DIR}/build_${date_tag}"
mkdir -p "${build_dir}"

cmake -S "$(dirname "$0")/.." -B "${build_dir}"
cmake --build "${build_dir}" -j

run_backend() {
	local backend="$1"
	local exe="$2"
	local suffix="$3"
	local extra="$4"   # zusätzliche Flags (z.B. --batch, --no-pin)
	local csv="$5"
	
	echo "Running ${exe} on backend=${backend} ${suffix} -> ${csv}"
	SYCL_DEVICE_FILTER="${backend}:gpu" "${build_dir}/${exe}" \
		--csv "${OUT_DIR}/${csv}" --min 1024 --max $((1<<26)) --steps 16 --secs 1 ${extra} || true
}

# Level Zero - Full Matrix (sync/batch × pinned/pageable)
run_backend level_zero memcpy_linear "[sync pinned]"   ""                 "l0_memcpy_sync_pinned_${date_tag}.csv"
run_backend level_zero memcpy_linear "[sync pageable]" "--no-pin"         "l0_memcpy_sync_pageable_${date_tag}.csv"
run_backend level_zero memcpy_linear "[batch pinned]"  "--batch"          "l0_memcpy_batch_pinned_${date_tag}.csv"
run_backend level_zero memcpy_linear "[batch pageable]" "--batch --no-pin" "l0_memcpy_batch_pageable_${date_tag}.csv"

run_backend level_zero event_overhead "[default]" "" "l0_event_overhead_${date_tag}.csv"

# CUDA - Full Matrix (optional)
if [[ "${ENABLE_CUDA}" == "yes" ]] || [[ "${ENABLE_CUDA}" == "auto" ]]; then
	echo ""
	echo "=== Testing CUDA backend ==="
	if SYCL_DEVICE_FILTER=cuda:gpu "${build_dir}/memcpy_linear" --help &>/dev/null; then
		run_backend cuda memcpy_linear "[sync pinned]"   ""                   "cuda_memcpy_sync_pinned_${date_tag}.csv"
		run_backend cuda memcpy_linear "[sync pageable]" "--no-pin"           "cuda_memcpy_sync_pageable_${date_tag}.csv"
		run_backend cuda memcpy_linear "[batch pinned]"  "--batch"            "cuda_memcpy_batch_pinned_${date_tag}.csv"
		run_backend cuda memcpy_linear "[batch pageable]" "--batch --no-pin"  "cuda_memcpy_batch_pageable_${date_tag}.csv"
		run_backend cuda event_overhead "[default]" "" "cuda_event_overhead_${date_tag}.csv"
	else
		echo "No CUDA device found, skipping CUDA tests"
	fi
fi

echo ""
echo "Done. CSVs in ${OUT_DIR}/"
