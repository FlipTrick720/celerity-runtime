#!/usr/bin/env python3
"""
Quick test to check if SYCL vs Native detection is working.
"""

import pandas as pd
from pathlib import Path
import sys

def test_detection(results_dir):
    """Test implementation detection."""
    results_path = Path(results_dir)
    csv_files = list(results_path.rglob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    print()
    
    implementations = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check what columns we have
            if csv_file.name.endswith('metadata.txt'):
                continue
                
            print(f"File: {csv_file.name}")
            print(f"  Columns: {', '.join(df.columns.tolist())}")
            
            if 'backend' in df.columns and len(df) > 0:
                backend_val = df['backend'].iloc[0]
                print(f"  Backend column value: '{backend_val}'")
                
                # Determine implementation
                if 'native' in backend_val.lower():
                    impl = 'L0 Native'
                elif 'cuda' in backend_val.lower():
                    impl = 'CUDA'
                else:
                    impl = 'SYCL'
                
                print(f"  Detected implementation: {impl}")
                implementations[impl] = implementations.get(impl, 0) + 1
            
            if 'bench' in df.columns and len(df) > 0:
                bench_val = df['bench'].iloc[0]
                print(f"  Bench column value: '{bench_val}'")
            
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    print("=" * 60)
    print("Summary:")
    for impl, count in implementations.items():
        print(f"  {impl}: {count} files")
    
    if 'SYCL' in implementations and 'L0 Native' in implementations:
        print()
        print("✓ Both SYCL and L0 Native detected!")
        print("  Comparison plots should be generated.")
    else:
        print()
        print("✗ Missing implementation data")
        print(f"  Found: {', '.join(implementations.keys())}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 test_plot_detection.py <results_dir>")
        print("Example: python3 test_plot_detection.py results/results_v0_baseline_20251124_110213")
        sys.exit(1)
    
    test_detection(sys.argv[1])
