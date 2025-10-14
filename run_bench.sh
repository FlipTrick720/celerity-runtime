cd bench

export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

# Run 1
taskset -c 0-15 ./scripts/run_matrix.sh results_$(date +%Y%m%d_%H%M%S)

# Run 2
echo "=== Cool Down ==="
sleep 300 #(Cool Down Phase)
taskset -c 0-15 ./scripts/run_matrix.sh results_$(date +%Y%m%d_%H%M%S)

# Run 3
echo "=== Cool Down ==="
sleep 300 #(Cool Down Phase)
taskset -c 0-15 ./scripts/run_matrix.sh results_$(date +%Y%m%d_%H%M%S)
