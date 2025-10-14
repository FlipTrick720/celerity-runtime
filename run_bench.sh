cd bench
export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL
taskset -c 0-15 ./scripts/run_matrix.sh results_run1
