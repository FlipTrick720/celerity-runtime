#!/usr/bin/env bash
# Analyze and compare benchmark results locally
# Run this on your local machine after downloading results from server

set -euo pipefail

# ============================================
# CONFIGURATION
# ============================================
# For test_all_variants.sh results, use variant names:
VERSIONS=(
    "baseline"
    "variant1"
    "variant2"
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
        python3 scripts/analyze_results.py \
            "results/results_${version}_"* \
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
    
    # Build comparison command
    compare_cmd="python3 scripts/compare_versions.py"
    for version in "${VERSIONS[@]}"; do
        result_dirs=(results/results_${version}_*)
        if [[ -d "${result_dirs[0]}" ]]; then
            compare_cmd="${compare_cmd} results/results_${version}_*"
        fi
    done
    compare_cmd="${compare_cmd} --output comparison_all"
    
    echo "Running comparison..."
    eval "${compare_cmd}"
    echo "  ✓ Comparison saved to: bench/comparison_all/"
    echo ""
fi

# ============================================
# Summary
# ============================================
echo "========================================="
echo "Analysis Complete!"
echo "========================================="
echo ""
echo "Individual version plots:"
for version in "${VERSIONS[@]}"; do
    if [[ -d "plots_${version}" ]]; then
        echo "  - ${version}: bench/plots_${version}/"
    fi
done

if [[ -d "comparison_all" ]]; then
    echo ""
    echo "Version comparison:"
    echo "  - All versions: bench/comparison_all/"
fi

echo ""
echo "Key files to check:"
echo "  - plots_*/mode_comparison_level_zero.png"
echo "  - plots_*/peak_bandwidth_summary.png"
echo "  - plots_*/summary_statistics.csv"
echo "  - comparison_all/version_comparison_*.png"
echo "  - comparison_all/version_comparison.csv"
echo ""
