#!/usr/bin/env bash
#SBATCH --job-name=all_tests_test
#SBATCH --partition=intel
#SBATCH --gres=gpu:ARC770:1
#SBATCH --time=00:10:00
#SBATCH --output=output.log
#SBATCH --error=error.log

set -Eeuo pipefail

# 0) Capture everything to a single log (and still print to console)
ts="$(date +%Y%m%d_%H%M%S)"
logdir="${LOG_DIR:-$PWD}"
mkdir -p "$logdir"
runlog="$logdir/run_${ts}.log"
exec > >(tee -a "$runlog") 2>&1

echo ":: initializing oneAPI environment ..."
source /opt/intel/oneapi/setvars.sh

# 1) Device selection (yours)
export ONEAPI_DEVICE_SELECTOR="level_zero:*"
export SYCL_DEVICE_FILTER=level_zero:gpu:0
export ZE_AFFINITY_MASK=0.0
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

# 2) Unified Runtime / Level Zero diagnostics (balanced but helpful)
# UR trace (high-level SYCL<->UR calls)
export SYCL_UR_TRACE=2
export UR_LOG_LEVEL=debug
# Turn on UR/L0 validation & tracing layers if present
export UR_ENABLE_LAYERS=LOGGING;VALIDATION;TRACING
export ZE_ENABLE_VALIDATION_LAYER=1
export ZE_ENABLE_TRACING_LAYER=1
# More verbose L0 debug (0..4). You already used 4
export ZE_DEBUG=${ZE_DEBUG:-4}
# Parameter validation at L0
export ZE_ENABLE_PARAMETER_VALIDATION=1

# 3) Celerity/Spdlog verbosity
export CELERITY_LOG_LEVEL=${CELERITY_LOG_LEVEL:-trace}
export SPDLOG_LEVEL=${SPDLOG_LEVEL:-trace}

# 4) Make sure we get symbols & core dumps
ulimit -c unlimited || true
# Prefer keeping frame pointers for nicer stacks if you rebuild:
# export CFLAGS="${CFLAGS:-} -fno-omit-frame-pointer"
# export CXXFLAGS="${CXXFLAGS:-} -fno-omit-frame-pointer"
# LD flag that helps gdb find symbols of the main binary
export LDFLAGS="${LDFLAGS:-} -rdynamic"

# Optional: list devices seen by SYCL (nice sanity check)
if command -v sycl-ls >/dev/null 2>&1; then
  echo; echo ":: sycl-ls:"
  sycl-ls || true
fi

# 5) Go to your build dir
cd ~/testApproach/celerity-runtime/build_2025-08-17_20-42-57/test/

# 6) Run tests
echo; echo "all_tests"
set +e
./all_tests "$@"
status=$?
set -e
echo ":: all_tests exited with status $status"

# 7) If it crashed, try to auto-print a backtrace
if [[ $status -ne 0 ]]; then
  echo; echo ":: attempting post-mortem backtrace"
  # Find newest core file in this dir (SLURM setups may store cores elsewhere)
  corefile="$(ls -t core* 2>/dev/null | head -n1 || true)"
  if [[ -n "${corefile:-}" && -f "$corefile" ]] && command -v gdb >/dev/null 2>&1; then
    echo ":: using core file: $corefile"
    gdb -q -batch \
      -ex "set pagination off" \
      -ex "thread apply all bt full" \
      -ex "info registers" \
      -ex "quit" \
      ./all_tests "$corefile" || true
  elif command -v coredumpctl >/dev/null 2>&1; then
    echo ":: using coredumpctl (systemd)"
    coredumpctl info "$(basename "$(pwd)")/all_tests" || true
    coredumpctl gdb  "$(basename "$(pwd)")/all_tests" -q -batch \
      -ex "set pagination off" \
      -ex "thread apply all bt full" \
      -ex "info registers" \
      -ex "quit" || true
  else
    echo ":: no core file found or gdb unavailable; consider running under: gdb --args ./all_tests"
  fi
fi

echo; echo ":: logs saved to $runlog"
