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
#   ./run_test.sh [--profile clean|test|noisy] [--gdb] [--] [gtest-args...]
#
# Profiles
#   clean : Level Zero only, minimal layers (default). Good for normal runs.
#   test  : Level Zero only + explicit BE hints (PI_LEVEL_ZERO), simple pinning.
#   noisy : Very verbose logging/tracing from UR/L0 for debugging.
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
    -h|--help)
      usage; exit 0 ;;
    --) shift; break ;;
    *)  break ;;
  esac
done
# Remaining "$@" (if any) are passed to ./all_tests

# -------- logging --------
ts="$(date +%Y%m%d_%H%M%S)"
logdir="${LOG_DIR:-$PWD/build_test_logs}"
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
    # Level Zero only, explicit plugin selection hints
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
    export UR_ENABLE_LAYERS="LOGGING;VALIDATION;TRACING"
    export UR_LOG_LEVEL=debug
    export SYCL_UR_TRACE=1
    export ZE_DEBUG=4
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    ;;
  *)
    echo "Unknown PROFILE='$PROFILE'. Use clean|test|noisy" >&2
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
BUILD_DIR="${BUILD_DIR:-$(ls -d ~/celerity-runtime/build_2025* 2>/dev/null | sort | tail -n1 || true)}"
TEST_DIR="${TEST_DIR:-${BUILD_DIR:+$BUILD_DIR/test}}"
echo ":: using test directory: ${TEST_DIR:-<none>}"
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

set +e
if (( USE_GDB )) && command -v gdb >/dev/null 2>&1; then
  echo ":: running under GDB (batch mode) for backtrace"
  gdb -q -batch \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "set print thread-events off" \
    -ex "run" \
    -ex "bt full" \
    -ex "thread apply all bt full" \
    -ex "info threads" \
    -ex "info registers" \
    -ex "info sharedlibrary" \
    --args ./all_tests "$@"
  status=$?
else
  echo ":: running without GDB..."
  ./all_tests "$@"
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

# -------- coredump summary (if failed) --------
if [[ $status -ne 0 ]] && command -v coredumpctl >/dev/null 2>&1; then
  echo; echo ":: coredumpctl summary for all_tests (if any):"
  coredumpctl info all_tests || true
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
