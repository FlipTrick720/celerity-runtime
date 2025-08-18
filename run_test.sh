#!/usr/bin/env bash
#SBATCH --job-name=all_tests_test
#SBATCH --partition=intel
#SBATCH --gres=gpu:ARC770:1
#SBATCH --time=00:10:00
#SBATCH --output=output.log
#SBATCH --error=error.log

set -Eeuo pipefail

# ---- logging ----
ts="$(date +%Y%m%d_%H%M%S)"
logdir="${LOG_DIR:-$PWD/build_test_logs}"
mkdir -p "$logdir"
runlog="$logdir/run_${ts}.log"
exec > >(tee -a "$runlog") 2>&1

echo ":: initializing oneAPI environment ..."
# Make setvars errors non-fatal but visible
set +u
source /opt/intel/oneapi/setvars.sh || echo "WARN: setvars.sh returned non-zero"
set -u

# ---- device selection ----
export ONEAPI_DEVICE_SELECTOR="level_zero:*"
export SYCL_DEVICE_FILTER=level_zero:gpu:0
export ZE_AFFINITY_MASK=0.0
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

# ---- UR / Level Zero diagnostics ----
export UR_ENABLE_LAYERS="LOGGING;VALIDATION;TRACING"
export SYCL_UR_TRACE=${SYCL_UR_TRACE:-2}
export UR_LOG_LEVEL=${UR_LOG_LEVEL:-debug}
export ZE_ENABLE_VALIDATION_LAYER=1
export ZE_ENABLE_TRACING_LAYER=1
export ZE_ENABLE_PARAMETER_VALIDATION=1
export ZE_DEBUG=${ZE_DEBUG:-4}

# ---- celerity/spdlog verbosity ----
export CELERITY_LOG_LEVEL=${CELERITY_LOG_LEVEL:-trace}
export SPDLOG_LEVEL=${SPDLOG_LEVEL:-trace}

# ---- symbolization & stacks ----
ulimit -c unlimited || true
# Load glibc’s libSegFault to print a stack on SIGABRT/SEGV
if [[ -r /lib/x86_64-linux-gnu/libSegFault.so ]]; then
  export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/x86_64-linux-gnu/libSegFault.so"
  export SEGFAULT_SIGNALS="abrt segv"
fi
# Keep frame pointers for clearer stacks if you rebuild
export CXXFLAGS="${CXXFLAGS:-} -fno-omit-frame-pointer"
export LDFLAGS="${LDFLAGS:-} -rdynamic"

# ---- sanity info ----
echo ":: UR layers:    $UR_ENABLE_LAYERS"
echo ":: ZE debug:     $ZE_DEBUG"
echo ":: SYCL filter:  ${SYCL_DEVICE_FILTER:-unset}"
echo ":: core_pattern: $(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo '?')"

# ---- sycl-ls (optional) ----
if command -v sycl-ls >/dev/null 2>&1; then
  echo; echo ":: sycl-ls:"
  sycl-ls --ignore-device-selectors || true
fi

# ---- build dir ----
cd ~/testApproach/celerity-runtime/build_2025-08-18_18-31-29/test/  # <- update this path

# ---- run (GDB mode by default to catch the abort) ----
echo; echo "all_tests (under gdb for backtrace)"
set +e
if command -v gdb >/dev/null 2>&1; then
  gdb -q -batch \
    -ex "set pagination off" \
    -ex "set confirm off" \
    -ex "run" \
    -ex "bt full" \
    -ex "thread apply all bt full" \
    -ex "info registers" \
    --args ./all_tests "$@"
else
  ./all_tests "$@"
fi
set -e
echo ":: all_tests exited with status $status"

# ---- coredump fallback (systemd) ----
if [[ $status -ne 0 ]] && command -v coredumpctl >/dev/null 2>&1; then
  echo; echo ":: coredumpctl summary for all_tests (if any):"
  coredumpctl info all_tests || true
fi

echo; echo ":: logs saved to $runlog"
