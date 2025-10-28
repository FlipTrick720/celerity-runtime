
## Workflow Summary

### 1. Benchmark on Server
```bash
./run_bench.sh
# or
./test_and_bench_all_variants.sh
```

### 2. Download Results

### 3. Configure Analysis
Edit `compare_bench.sh`:
```bash
VERSIONS=(
    "v0_baseline"
    "v1_optimized"
)
```

### 4. Analyze Locally
```bash
./compare_bench.sh
```

### 5. Review Results
- Individual plots: `bench/plots_<version>/`
- Comparison plots: `bench/comparison_all/`
- Summary tables: `bench/plots_*/summary_statistics.csv`
