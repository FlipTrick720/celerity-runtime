#!/usr/bin/env bash
#SBATCH --job-name=all_tests
#SBATCH --partition=intel
#SBATCH --gres=gpu:ARC770:1
#SBATCH --time=00:10:00
#SBATCH --output=z_output.log
#SBATCH --error=z_error.log

# =============================================================================
# run_test.sh — run celerity's `all_tests` with controlled oneAPI/L0 env
#
# Usage
#   ./run_test.sh [--profile clean|test|verbose|noisy|debug] [--gdb] [--] [gtest-args...]
#
# Profiles
#   clean   : Level Zero only, minimal layers (default). Good for normal runs.
#   test    : Level Zero only, clean output like professional CI. Recommended.
#   verbose : Level Zero + detailed UR tracing (old 'test' profile).
#   noisy   : Very verbose logging/tracing from UR/L0 for debugging.
#   debug   : Maximum debugging + auto-GDB for crash investigation.
#
# Examples
#   sbatch run_test.sh --profile test
#   sbatch run_test.sh --profile test --gdb
#
# Notes
# - We *unset* common SYCL/UR/L0 vars first to avoid spill-overs between runs.
# - GDB is **off by default**; enable with `--gdb` for useful backtraces.
# - Picks the newest build dir like ~/celerity-runtime/build_2025* unless BUILD_DIR is set.
# - Hard-fails if the run falls back to the generic/OpenCL backend.
# =============================================================================

set -Eeuo pipefail

# -------- args --------
PROFILE="${PROFILE:-clean}"
USE_GDB=0
TEST_FILTER=""

usage() {
  sed -n '1,40p' "$0" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { echo "Missing value for --profile" >&2; exit 2; }
      PROFILE="$2"; shift 2 ;;
    --gdb)
      USE_GDB=1; shift ;;
    --filter)
      [[ $# -ge 2 ]] || { echo "Missing value for --filter" >&2; exit 2; }
      TEST_FILTER="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --) shift; break ;;
    *)  break ;;
  esac
done
# Remaining "$@" (if any) are passed to ./all_tests

# -------- logging --------
ts="$(date +%Y%m%d_%H%M%S)"
logdir="${LOG_DIR:-$PWD/test_logs}"
mkdir -p "$logdir"
runlog="$logdir/run_${ts}.log"
exec > >(tee -a "$runlog") 2>&1

echo ":: initializing oneAPI environment ..."
set +u
source /opt/intel/oneapi/setvars.sh || echo "WARN: setvars.sh returned non-zero"
set -u

# -------- env cleanup (avoid spill-over) --------
unset_vars=(
  ONEAPI_DEVICE_SELECTOR
  SYCL_DEVICE_FILTER
  SYCL_BE
  SYCL_UR_TRACE
  UR_ENABLE_LAYERS
  UR_LOG_LEVEL
  UR_ADAPTERS_FORCE_ORDER
  UR_DISABLE_ADAPTERS
  ZE_DEBUG
  ZE_AFFINITY_MASK
  ZE_ENABLE_PCI_ID_DEVICE_ORDER
)
for v in "${unset_vars[@]}"; do unset "$v" || true; done

# -------- apply profile --------
case "$PROFILE" in
  clean)
    # Minimal layers, L0 GPU only
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ENABLE_LAYERS=""
    export UR_LOG_LEVEL=warning
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    export ZE_AFFINITY_MASK=0
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    ;;
  test)
    # Level Zero only, similar to professional CI setup
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    export ONEAPI_DEVICE_SELECTOR=level_zero:*
    export ZE_AFFINITY_MASK=0
    # Test-friendly logging (respects build type)
    export UR_ENABLE_LAYERS="VALIDATION"
    export UR_LOG_LEVEL=warning
    # Don't override CELERITY_LOG_LEVEL - let it use build-type defaults
    # (debug builds expect debug logs, release builds expect info logs)
    export SPDLOG_LEVEL=debug
    ;;
  verbose)
    # More detailed logging (old 'test' profile)
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    export SYCL_BE=PI_LEVEL_ZERO
    export ZE_AFFINITY_MASK=0
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    export SYCL_PREFER_UR=1
    export UR_ENABLE_LAYERS="LOGGING;VALIDATION"
    export UR_LOG_LEVEL=debug
    export SYCL_UR_TRACE=2
    export SPDLOG_LEVEL=trace
    export CELERITY_LOG_LEVEL=trace
    ;;
  noisy)
    # Verbose tracing/validation
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ENABLE_LAYERS="LOGGING;VALIDATION"
    export UR_LOG_LEVEL=debug
    export SYCL_UR_TRACE=2
    export ZE_DEBUG=4
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    export CELERITY_LOG_LEVEL=trace
    ;;
  debug)
    # Maximum debugging for crash investigation
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    export SYCL_BE=PI_LEVEL_ZERO
    export ZE_AFFINITY_MASK=0
    # Enhanced debugging
    export UR_ENABLE_LAYERS="LOGGING;VALIDATION;TRACING"
    export UR_LOG_LEVEL=debug
    export SYCL_UR_TRACE=2
    export ZE_DEBUG=1
    export SPDLOG_LEVEL=debug
    export CELERITY_LOG_LEVEL=debug
    # Memory debugging
    export MALLOC_CHECK_=2
    export MALLOC_PERTURB_=42
    # Force GDB for this profile
    USE_GDB=1
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE'. Use clean|test|verbose|noisy|debug" >&2
    exit 2
    ;;
esac

# -------- celerity/spdlog verbosity --------
export CELERITY_LOG_LEVEL="${CELERITY_LOG_LEVEL:-info}"
export SPDLOG_LEVEL="${SPDLOG_LEVEL:-info}"

# -------- symbolization & stacks --------
ulimit -c unlimited || true
if [[ -r /lib/x86_64-linux-gnu/libSegFault.so ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/x86_64-linux-gnu/libSegFault.so"
  export SEGFAULT_SIGNALS="abrt segv"
fi
export CXXFLAGS="${CXXFLAGS:-} -fno-omit-frame-pointer"
export LDFLAGS="${LDFLAGS:-} -rdynamic"

# -------- sanity info --------
echo ":: PROFILE:       $PROFILE"
echo ":: UR layers:     ${UR_ENABLE_LAYERS:-unset}"
echo ":: UR loglevel:   ${UR_LOG_LEVEL:-unset}"
echo ":: ZE debug:      ${ZE_DEBUG:-unset}"
echo ":: SYCL filter:   ${SYCL_DEVICE_FILTER:-unset}"
echo ":: ZE affinity:   ${ZE_AFFINITY_MASK:-unset}"
echo ":: CELERITY_LOG_LEVEL: ${CELERITY_LOG_LEVEL:-unset}"
echo ":: SPDLOG_LEVEL:  ${SPDLOG_LEVEL:-unset}"
echo ":: core_pattern:  $(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo '?')"

echo ":: effective env (UR/ZE/SYCL):"
env | egrep '^(UR_|ZE_|SYCL_|ONEAPI_)' | sort

# Optional device info
if command -v sycl-ls >/dev/null 2>&1; then
  echo; echo ":: sycl-ls:"
  sycl-ls --ignore-device-selectors || true
fi
if command -v zeinfo >/dev/null 2>&1; then
  echo; echo ":: zeinfo:"
  zeinfo || true
fi

# -------- pick test dir --------
echo ":: current directory: $(pwd)"

# Auto-detect celerity-runtime directory from various common locations
if [[ -z "${BUILD_DIR:-}" ]]; then
  # Search paths in order of preference
  search_paths=(
    "$(pwd)"                                    # Current directory
    "$(pwd)/celerity-runtime"                  # Subdirectory of current
    "$(dirname "$(pwd)")/celerity-runtime"     # Sibling directory
    "$HOME/celerity-runtime"                   # Home directory
    "$HOME/testApproach/celerity-runtime"      # Your testApproach folder
    "$HOME/mainApproach/celerity-runtime"      # Your mainApproach folder
  )
  
  for base_dir in "${search_paths[@]}"; do
    if [[ -d "$base_dir" ]]; then
      # Look for the newest build directory in this location
      candidate=$(ls -d "$base_dir"/build_2025* 2>/dev/null | sort | tail -n1 || true)
      if [[ -n "$candidate" && -d "$candidate" ]]; then
        BUILD_DIR="$candidate"
        echo ":: found build directory: $BUILD_DIR"
        break
      fi
    fi
  done
fi

TEST_DIR="${TEST_DIR:-${BUILD_DIR:+$BUILD_DIR/test}}"
echo ":: using test directory: ${TEST_DIR:-<none>}"

if [[ -z "$TEST_DIR" ]]; then
  echo "ERROR: Could not find celerity-runtime build directory"
  echo "Searched in:"
  printf "  %s\n" "${search_paths[@]}"
  echo "Set BUILD_DIR environment variable to override"
  exit 1
fi

cd "${TEST_DIR}" || { echo "ERROR: cannot cd to ${TEST_DIR}"; exit 1; }

echo ":: checking if all_tests exists..."
if [[ ! -f ./all_tests ]]; then
  echo "ERROR: all_tests executable not found in $(pwd)"
  echo "Available files:"; ls -la
  exit 1
fi
echo ":: all_tests executable found"

# -------- run --------
echo; echo "all_tests ($( ((USE_GDB)) && echo 'with GDB' || echo 'direct') )"
if [[ $# -gt 0 ]]; then
  echo ":: test arguments: $*"
else
  echo ":: no test arguments (running all tests)"
fi

# Prepare test arguments
test_args=("$@")
if [[ -n "$TEST_FILTER" ]]; then
  test_args+=("--gtest_filter=$TEST_FILTER")
  echo ":: applying test filter: $TEST_FILTER"
fi

set +e
if (( USE_GDB )) && command -v gdb >/dev/null 2>&1; then
  echo ":: running under GDB (batch mode) for backtrace"
  timeout 300 gdb -q -batch \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "set print thread-events off" \
    -ex "run" \
    -ex "bt full" \
    -ex "thread apply all bt full" \
    -ex "info threads" \
    -ex "info registers" \
    -ex "info sharedlibrary" \
    --args ./all_tests "${test_args[@]}"
  status=$?
else
  echo ":: running without GDB (5 minute timeout)..."
  timeout 300 ./all_tests "${test_args[@]}"
  status=$?
fi
set -e

echo ":: all_tests exited with status $status"

# -------- guard against generic backend fallback --------
# Check for actual backend fallback (not just misleading platform names)
if grep -E -q 'falling back to generic' "$runlog"; then
  echo ":: HARD-FAIL: Celerity fell back to generic backend."
  echo ":: This indicates no backend specialization is available for the selected devices."
  echo ":: Check device compatibility and backend configuration."
  exit 3
fi

# Check if we're actually using OpenCL instead of Level Zero (real failure case)
if grep -E -q 'Using platform.*Intel.*OpenCL.*' "$runlog" && ! grep -E -q 'oneAPI Unified Runtime over Level-Zero' "$runlog"; then
  echo ":: HARD-FAIL: Only OpenCL backend detected, Level Zero not available."
  echo ":: Ensure SYCL_DEVICE_FILTER=level_zero:gpu and UR_DISABLE_ADAPTERS=OPENCL are in effect."
  exit 3
fi

# -------- better coredump analyses --------
if [[ $status -ne 0 ]]; then
  echo; echo ":: CRASH ANALYSIS"
  
  # Check for core dumps
  if command -v coredumpctl >/dev/null 2>&1; then
    echo ":: coredumpctl summary for all_tests:"
    coredumpctl info all_tests || true
    
    # Get the latest core dump for detailed analysis
    latest_core=$(coredumpctl list all_tests --no-pager --no-legend | tail -n1 | awk '{print $5}' 2>/dev/null || true)
    if [[ -n "$latest_core" ]]; then
      echo ":: Latest core dump PID: $latest_core"
      echo ":: Detailed backtrace from core dump:"
      coredumpctl gdb all_tests --batch \
        -ex "set pagination off" \
        -ex "bt full" \
        -ex "thread apply all bt full" \
        -ex "info threads" \
        -ex "info registers" \
        -ex "quit" 2>/dev/null || true
    fi
  fi
  
  # Check for local core files
  echo ":: Checking for local core files:"
  find . -name "core*" -type f -newer ./all_tests 2>/dev/null | head -5 || true
  
  # Memory/resource analysis
  echo ":: System resource status:"
  echo "  Memory: $(free -h | grep '^Mem:' || echo 'unavailable')"
  echo "  GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'N/A (not NVIDIA)')"
  
  # Check for common crash patterns in logs
  echo ":: Crash pattern analysis:"
  if grep -E -i "(segmentation fault|segfault|sigsegv|sigabrt|assertion.*failed|abort|terminate)" "$runlog" | head -5; then
    echo "  Found crash indicators in logs"
  else
    echo "  No obvious crash patterns in logs"
  fi
fi

# -------- final summary --------
echo
echo "=========================================="
echo ":: TEST RUN COMPLETE"
echo ":: Exit status: $status"
echo ":: Log file: $runlog"
if [[ $status -eq 0 ]]; then
  echo ":: RESULT: SUCCESS"
else
  echo ":: RESULT: FAILURE"
fi
echo "=========================================="
echo; echo ":: logs saved to $runlog"
