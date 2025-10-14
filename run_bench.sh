cd bench

export UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
export UR_DISABLE_ADAPTERS=OPENCL

# Run 1
taskset -c 0-15 ./scripts/run_matrix.sh results_$(date +%Y%m%d_%H%M%S)
echo "=== Cool Down ==="
sleep 300 #(Cool Down Phase)

# Run 2
taskset -c 0-15 ./scripts/run_matrix.sh results_$(date +%Y%m%d_%H%M%S)
echo "=== Cool Down ==="
sleep 300 #(Cool Down Phase)

# Run 3
taskset -c 0-15 ./scripts/run_matrix.sh results_$(date +%Y%m%d_%H%M%S)
echo "=== Cool Down ==="
sleep 300 #(Cool Down Phase)
