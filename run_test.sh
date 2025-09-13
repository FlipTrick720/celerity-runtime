#!/usr/bin/env bash
#SBATCH --job-name=all_tests_test
#SBATCH --partition=intel
#SBATCH --gres=gpu:ARC770:1
#SBATCH --time=00:10:00
#SBATCH --output=output.log
#SBATCH --error=error.log

set -Eeuo pipefail

# -------------------------------
# Profile selection
#   Choose with:  ./run_test.sh --profile clean           "test name"
#                  or set PROFILE=clean|immediate|pin0|idx0|noisy
# Defaults to: clean
# -------------------------------
PROFILE="${PROFILE:-clean}"

# ---- logging ----
ts="$(date +%Y%m%d_%H%M%S)"
logdir="${LOG_DIR:-$PWD/build_test_logs}"
mkdir -p "$logdir"
runlog="$logdir/run_${ts}.log"
exec > >(tee -a "$runlog") 2>&1

echo ":: initializing oneAPI environment ..."
set +u
source /opt/intel/oneapi/setvars.sh || echo "WARN: setvars.sh returned non-zero"
set -u

# Clear any leftovers that often bite
unset ONEAPI_DEVICE_SELECTOR
unset SYCL_UR_TRACE
unset ZE_DEBUG
unset ZE_AFFINITY_MASK

# -------------------------------
# Apply profile
# -------------------------------
case "$PROFILE" in
  clean)
    # Minimal layers, no tile pinning, L0 GPU (no index), L0-only adapters.
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ENABLE_LAYERS=""
    export UR_LOG_LEVEL=warning
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    ;;

  immediate)
    # clean + immediate command lists for copy/submit pressure relief
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ENABLE_LAYERS=""
    export UR_LOG_LEVEL=warning
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    export UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1   # legacy env some builds honor
    ;;

  pin0)
    # clean + pin to GPU0.tile0 (can help or hurt; try if you want strict pin)
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export ZE_AFFINITY_MASK=0.0
    export UR_ENABLE_LAYERS=""
    export UR_LOG_LEVEL=warning
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    ;;

  idx0)
    # If you WANT explicit device index — some Celerity builds dislike the ':0', so use with care.
    export SYCL_DEVICE_FILTER=level_zero:gpu:0
    export UR_ENABLE_LAYERS=""
    export UR_LOG_LEVEL=warning
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    ;;

  noisy)
    # Verbose UR/ZE tracing (for deep debugging)
    export SYCL_DEVICE_FILTER=level_zero:gpu
    export UR_ENABLE_LAYERS="LOGGING;VALIDATION;TRACING"
    export UR_LOG_LEVEL=debug
    export SYCL_UR_TRACE=1
    export ZE_DEBUG=4
    export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
    export UR_DISABLE_ADAPTERS=OPENCL
    ;;

  *)
    echo "Unknown PROFILE='$PROFILE'. Use clean|immediate|pin0|idx0|noisy"
    exit 2
    ;;
esac

# ---- celerity/spdlog verbosity ----
export CELERITY_LOG_LEVEL="${CELERITY_LOG_LEVEL:-info}"
export SPDLOG_LEVEL="${SPDLOG_LEVEL:-info}"

# ---- symbolization & stacks ----
ulimit -c unlimited || true
if [[ -r /lib/x86_64-linux-gnu/libSegFault.so ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/x86_64-linux-gnu/libSegFault.so"
  export SEGFAULT_SIGNALS="abrt segv"
fi
export CXXFLAGS="${CXXFLAGS:-} -fno-omit-frame-pointer"
export LDFLAGS="${LDFLAGS:-} -rdynamic"

# ---- sanity info ----
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

# ---- sycl-ls (optional) ----
if command -v sycl-ls >/dev/null 2>&1; then
  echo; echo ":: sycl-ls:"
  sycl-ls --ignore-device-selectors || true
fi

# ---- zeinfo (optional) ----
if command -v zeinfo >/dev/null 2>&1; then
  echo; echo ":: zeinfo:"
  zeinfo || true
fi

# ---- build dir (auto-detect latest unless BUILD_DIR is set) ----
echo ":: current directory: $(pwd)"
BUILD_DIR="${BUILD_DIR:-$(ls -d ~/testApproach/celerity-runtime/build_2025* 2>/dev/null | sort | tail -n1)}"
TEST_DIR="${TEST_DIR:-$BUILD_DIR/test}"
echo ":: using test directory: ${TEST_DIR}"
cd "${TEST_DIR}" || { echo "ERROR: cannot cd to ${TEST_DIR}"; exit 1; }

echo ":: checking if all_tests exists..."
if [[ ! -f ./all_tests ]]; then
  echo "ERROR: all_tests executable not found in $(pwd)"
  echo "Available files:"
  ls -la
  exit 1
fi
echo ":: all_tests executable found"

# ---- run (GDB mode by default) ----
echo; echo "all_tests (under gdb for backtrace)"
if [[ $# -gt 0 && "$1" == "--profile" ]]; then
  shift; shift || true
fi
if [[ $# -gt 0 ]]; then
  echo ":: test arguments: $*"
else
  echo ":: no test arguments (running all tests)"
fi

set +e
if command -v gdb >/dev/null 2>&1 && [[ -z "${NO_GDB:-}" ]]; then
  echo ":: running with GDB for backtrace... (set NO_GDB=1 to disable)"
  gdb -q -batch \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "run" \
    -ex "bt full" \
    -ex "thread apply all bt full" \
    -ex "info registers" \
    --args ./all_tests "$@"
  status=$?
else
  echo ":: running without GDB..."
  ./all_tests "$@"
  status=$?
fi
set -e
echo ":: all_tests exited with status $status"

# ---- coredump fallback ----
if [[ $status -ne 0 ]] && command -v coredumpctl >/dev/null 2>&1; then
  echo; echo ":: coredumpctl summary for all_tests (if any):"
  coredumpctl info all_tests || true
fi

# ---- final summary ----
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
