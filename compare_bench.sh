#!/usr/bin/env bash
# Analyze and compare benchmark results locally
# Run this on your local machine after downloading results from server

set -euo pipefail

# ============================================
# CONFIGURATION
# ============================================
# For test_all_variants.sh results, use variant names:
VERSIONS=(
    "v0_baseline"
    "v1_event_pooling"
    "v2_immediate_pool"
    "v3_batch_fence"
    "v4_micro_optimized"
    "v5_No_Sync"
)


# Check if we're in the right directory
if [[ ! -d "bench/scripts" ]]; then
    echo "Error: Must run from repository root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

cd bench

# Check if results exist
if [[ ! -d "results" ]]; then
    echo "Error: bench/results/ directory not found"
    echo ""
    echo "Please prepare results first:"
    echo "  1. Download from server"
    echo "  2. Move benchmark directories to bench/results/"
    echo ""
    echo "Example for test_all_variants.sh results:"
    echo "  mv variant_test_results_*/baseline/bench results/results_baseline_\$(date +%Y%m%d)"
    echo "  mv variant_test_results_*/variant1/bench results/results_variant1_\$(date +%Y%m%d)"
    echo ""
    echo "Example for run_bench.sh results:"
    echo "  mv downloaded_results/results_v1_baseline_* results/"
    echo ""
    exit 1
fi

echo "========================================="
echo "Analyzing Benchmark Results"
echo "========================================="
echo ""

# ============================================
# Analyze each version separately
# ============================================
echo "=== Analyzing Individual Versions ==="
echo ""

for version in "${VERSIONS[@]}"; do
    result_dirs=(results/results_${version}_*)
    
    if [[ -d "${result_dirs[0]}" ]]; then
        echo "Analyzing ${version}..."
        latest_dir="${result_dirs[-1]}"  # Get last (most recent) directory
        python3 scripts/analyze_results.py \
            "${latest_dir}" \
            --output "plots_${version}"
        echo "  ✓ Plots saved to: bench/plots_${version}/"
        echo ""
    else
        echo "  ⚠️  No results found for ${version}"
        echo ""
    fi
done

# ============================================
# Compare all versions
# ============================================
if [[ ${#VERSIONS[@]} -ge 2 ]]; then
    echo "=== Comparing All Versions ==="
    echo ""
    
    # Build comparison command with only the newest directory for each version
    compare_cmd="python3 scripts/compare_versions.py"
    found_versions=0
    for version in "${VERSIONS[@]}"; do
        result_dirs=(results/results_${version}_*)
        if [[ -d "${result_dirs[0]}" ]]; then
            # Use the last (most recent) directory
            latest_dir="${result_dirs[-1]}"
            compare_cmd="${compare_cmd} ${latest_dir}"
            found_versions=$((found_versions + 1))
        fi
    done
    compare_cmd="${compare_cmd} --output comparison_all"
    
    if [[ ${found_versions} -ge 2 ]]; then
        echo "Running comparison..."
        eval "${compare_cmd}"
        echo "  ✓ Comparison saved to: bench/comparison_all/"
        echo ""
    else
        echo "  ⚠️  Need at least 2 versions to compare (found ${found_versions})"
        echo ""
    fi
fi

# ============================================
# Summary
# ============================================
echo "========================================="
echo "Analysis Complete!"
echo "========================================="
