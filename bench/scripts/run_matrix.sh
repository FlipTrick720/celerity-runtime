#!/usr/bin/env bash
set -euo pipefail

# Ensure oneAPI is loaded (needed to run SYCL executables)
if [ -z "${ONEAPI_ROOT:-}" ]; then
    echo "Loading oneAPI environment..."
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        set +u  # Temporarily disable -u for oneAPI script
        source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
        set -u
        echo "✓ oneAPI loaded"
    else
        echo "⚠️  Warning: oneAPI not found at /opt/intel/oneapi/setvars.sh"
        echo "   Benchmarks may fail if SYCL libraries are not in LD_LIBRARY_PATH"
    fi
else
    echo "✓ oneAPI already loaded"
fi
echo ""

OUT_DIR="${1:-out}"
ENABLE_CUDA="${ENABLE_CUDA:-auto}"  # auto, yes, no

# Backend version/tag - read from source file
BACKEND_SOURCE="../src/backend/sycl_level_zero_backend.cc"
BACKEND_TAG=""
BACKEND_NOTES=""

if [[ -f "${BACKEND_SOURCE}" ]]; then
	# Read first line: //Version: v1_baseline
	BACKEND_TAG=$(grep -m1 "^//Version:" "${BACKEND_SOURCE}" 2>/dev/null | sed 's/^\/\/Version: *//' | tr -d '\r\n' || echo "")
	# Read second line: //Text: Initial implementation
	BACKEND_NOTES=$(grep -m1 "^//Text:" "${BACKEND_SOURCE}" 2>/dev/null | sed 's/^\/\/Text: *//' | tr -d '\r\n' || echo "")
fi

# Auto-detect git info if available
GIT_SHA=""
GIT_BRANCH=""
GIT_DIRTY=""
if git rev-parse --git-dir > /dev/null 2>&1; then
	GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
	GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
	if [[ -n $(git status --porcelain 2>/dev/null) ]]; then
		GIT_DIRTY="-dirty"
	fi
fi

# Build result directory name with optional tag
date_tag=$(date +"%Y%m%d_%H%M%S")
if [[ -n "${BACKEND_TAG}" ]]; then
	result_dir="${OUT_DIR}/results_${BACKEND_TAG}_${date_tag}"
else
	result_dir="${OUT_DIR}/results_${date_tag}"
fi

mkdir -p "${result_dir}"

# Write metadata file
cat > "${result_dir}/metadata.txt" << EOF
# Benchmark Run Metadata
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Date Tag: ${date_tag}
Backend Tag: ${BACKEND_TAG:-none}
Backend Notes: ${BACKEND_NOTES:-none}
Git SHA: ${GIT_SHA}${GIT_DIRTY}
Git Branch: ${GIT_BRANCH}
Hostname: $(hostname)
User: $(whoami)
SYCL Implementation: ${CELERITY_SYCL_IMPL:-auto-detected}
EOF

echo "=== Benchmark Run Info ==="
cat "${result_dir}/metadata.txt"
echo ""

# Use shared build directory (created by build_bench.sh)
build_dir="build"

# Check if benchmarks are built
if [[ ! -f "${build_dir}/memcpy_linear" ]] || [[ ! -f "${build_dir}/event_overhead" ]]; then
	echo "Building benchmarks..."
	cmake -S "$(dirname "$0")/.." -B "${build_dir}"
	cmake --build "${build_dir}" -j
	echo "✓ Build complete"
else
	echo "✓ Using existing benchmark build (${build_dir}/)"
fi
echo ""

run_backend() {
	local backend="$1"
	local exe="$2"
	local suffix="$3"
	local extra="$4"   # zusätzliche Flags (z.B. --batch, --no-pin)
	local csv="$5"
	
	echo "Running ${exe} on backend=${backend} ${suffix} -> ${csv}"
	SYCL_DEVICE_FILTER="${backend}:gpu" "${build_dir}/${exe}" \
		--csv "${result_dir}/${csv}" --min 1024 --max $((1<<26)) --steps 16 --secs 1 ${extra} || true
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
	echo "=== Checking for CUDA backend ==="
	
	# Check if CUDA GPU is actually available
	# Try to list devices and grep for CUDA
	cuda_available=false
	if command -v sycl-ls &>/dev/null; then
		if sycl-ls 2>/dev/null | grep -i "cuda" &>/dev/null; then
			cuda_available=true
		fi
	fi
	
	# Alternative: try nvidia-smi
	if ! $cuda_available && command -v nvidia-smi &>/dev/null; then
		if nvidia-smi &>/dev/null; then
			cuda_available=true
		fi
	fi
	
	if $cuda_available || [[ "${ENABLE_CUDA}" == "yes" ]]; then
		echo "CUDA device detected, running CUDA tests..."
		run_backend cuda memcpy_linear "[sync pinned]"   ""                   "cuda_memcpy_sync_pinned_${date_tag}.csv"
		run_backend cuda memcpy_linear "[sync pageable]" "--no-pin"           "cuda_memcpy_sync_pageable_${date_tag}.csv"
		run_backend cuda memcpy_linear "[batch pinned]"  "--batch"            "cuda_memcpy_batch_pinned_${date_tag}.csv"
		run_backend cuda memcpy_linear "[batch pageable]" "--batch --no-pin"  "cuda_memcpy_batch_pageable_${date_tag}.csv"
		run_backend cuda event_overhead "[default]" "" "cuda_event_overhead_${date_tag}.csv"
	else
		echo "No CUDA device found (checked sycl-ls and nvidia-smi), skipping CUDA tests"
		echo "To force CUDA tests: ENABLE_CUDA=yes ./scripts/run_matrix.sh"
	fi
fi

echo ""
echo "=== Benchmark Complete ==="
echo "Results directory: ${result_dir}"
echo "Metadata: ${result_dir}/metadata.txt"
echo "CSVs: ${result_dir}/*.csv"
echo ""
echo "To analyze: python3 bench/scripts/analyze_results.py ${result_dir}"
