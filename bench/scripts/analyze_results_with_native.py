#!/usr/bin/env python3
"""
Analyze and plot Celerity backend benchmark results including Level Zero native comparison.
Generates publication-quality plots comparing SYCL baseline vs Level Zero native.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import argparse

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def load_metadata(results_dir):
    """Load metadata from results directory if available."""
    results_path = Path(results_dir)
    metadata_file = results_path / "metadata.txt"
    
    metadata = {}
    if metadata_file.exists():
        print("\n=== Benchmark Metadata ===")
        with open(metadata_file, 'r') as f:
            content = f.read()
            print(content)
            for line in content.split('\n'):
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
        return metadata
    return metadata

def load_all_csvs(results_dir):
    """Load all CSV files from results directory, distinguishing SYCL vs L0 native."""
    results_path = Path(results_dir)
    
    metadata = load_metadata(results_dir)
    csv_files = list(results_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None, metadata
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Determine implementation type from filename
            filename = csv_file.name
            if 'l0_native_' in filename or 'native_' in filename:
                df['implementation'] = 'Level Zero Native'
            elif 'cuda_' in filename:
                df['implementation'] = 'CUDA'
            else:
                df['implementation'] = 'SYCL Baseline'
            
            # Add backend version info
            if metadata and 'Backend Tag' in metadata:
                df['backend_version'] = metadata['Backend Tag']
            else:
                dir_name = csv_file.parent.name
                if 'results_' in dir_name:
                    parts = dir_name.split('_')
                    if len(parts) >= 2 and parts[1].startswith('v'):
                        version_parts = []
                        for i in range(1, len(parts)):
                            if parts[i].isdigit() and len(parts[i]) == 8:
                                break
                            version_parts.append(parts[i])
                        df['backend_version'] = '_'.join(version_parts) if version_parts else 'unknown'
                    else:
                        df['backend_version'] = 'unknown'
                else:
                    df['backend_version'] = 'unknown'
            
            dfs.append(df)
            print(f"  Loaded: {csv_file.name} ({df['implementation'].iloc[0]})")
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
    
    if not dfs:
        return None, metadata
    
    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal rows: {len(combined)}")
    if 'implementation' in combined.columns:
        implementations = combined['implementation'].unique()
        print(f"Implementations found: {', '.join(implementations)}")
    if 'backend' in combined.columns:
        backends = combined['backend'].unique()
        print(f"Backends found: {', '.join(backends)}")
    
    return combined, metadata

def plot_sycl_vs_native_comparison(df, output_dir, metadata=None):
    """Plot SYCL baseline vs Level Zero native comparison."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    
    if memcpy_df.empty:
        print("No memcpy data found")
        return
    
    # Check if we have both SYCL and native data
    implementations = memcpy_df['implementation'].unique()
    if 'SYCL Baseline' not in implementations or 'Level Zero Native' not in implementations:
        print(f"⚠️  Missing implementation data. Found: {', '.join(implementations)}")
        print("   Skipping SYCL vs Native comparison")
        return
    
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    version_str = ""
    if metadata and 'Backend Tag' in metadata:
        version_str = f" ({metadata['Backend Tag']})"
    
    operations = ['D2D', 'H2D', 'D2H']
    
    for op in operations:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{op} Bandwidth: SYCL Baseline vs Level Zero Native{version_str}', 
                    fontsize=16, fontweight='bold')
        
        modes = [
            ('sync', 'yes', 'Sync + Pinned'),
            ('sync', 'no', 'Sync + Pageable'),
            ('batch', 'yes', 'Batch + Pinned'),
            ('batch', 'no', 'Batch + Pageable')
        ]
        
        for idx, (mode, pinned, title) in enumerate(modes):
            ax = axes[idx // 2, idx % 2]
            
            for impl in ['SYCL Baseline', 'Level Zero Native']:
                data = memcpy_df[
                    (memcpy_df['op'] == op) &
                    (memcpy_df['implementation'] == impl) &
                    (memcpy_df['mode'] == mode) &
                    (memcpy_df['pinned'] == pinned)
                ]
                
                if not data.empty:
                    grouped = data.groupby('size_kib')['gib_per_s'].median().reset_index()
                    linestyle = '-' if impl == 'Level Zero Native' else '--'
                    linewidth = 2.5 if impl == 'Level Zero Native' else 2.0
                    ax.plot(grouped['size_kib'], grouped['gib_per_s'], 
                           marker='o', linewidth=linewidth, markersize=6,
                           linestyle=linestyle, label=impl, alpha=0.9)
            
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Transfer Size (KiB)')
            ax.set_ylabel('Bandwidth (GiB/s)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, which='both')
            ax.legend()
        
        plt.tight_layout()
        output_file = output_dir / f'sycl_vs_native_{op.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_native_overhead_analysis(df, output_dir):
    """Plot overhead comparison: SYCL vs Native for small transfers."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    # Focus on small transfers (< 1 MiB)
    small_df = memcpy_df[memcpy_df['size_kib'] <= 1024].copy()
    
    # Check if we have both implementations
    implementations = small_df['implementation'].unique()
    if 'SYCL Baseline' not in implementations or 'Level Zero Native' not in implementations:
        print("⚠️  Missing implementation data for overhead analysis")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Small Transfer Overhead: SYCL vs Level Zero Native (≤1 MiB)', 
                fontsize=16, fontweight='bold')
    
    for op_idx, op in enumerate(['D2D', 'H2D', 'D2H']):
        ax = axes[op_idx]
        op_data = small_df[small_df['op'] == op]
        
        # Compare SYCL vs Native (sync + pinned for clarity)
        for impl in ['SYCL Baseline', 'Level Zero Native']:
            data = op_data[
                (op_data['implementation'] == impl) &
                (op_data['mode'] == 'sync') &
                (op_data['pinned'] == 'yes')
            ]
            
            if not data.empty:
                grouped = data.groupby('size_kib')['avg_us'].median().reset_index()
                linestyle = '-' if impl == 'Level Zero Native' else '--'
                ax.plot(grouped['size_kib'], grouped['avg_us'],
                       marker='o', linewidth=2, markersize=6,
                       linestyle=linestyle, label=impl)
        
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Transfer Size (KiB)')
        ax.set_ylabel('Latency (μs)')
        ax.set_title(f'{op} Latency')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    output_file = output_dir / 'sycl_vs_native_overhead.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speedup_analysis(df, output_dir):
    """Plot speedup of Level Zero Native over SYCL Baseline."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    
    # Check if we have both implementations
    implementations = memcpy_df['implementation'].unique()
    if 'SYCL Baseline' not in implementations or 'Level Zero Native' not in implementations:
        print("⚠️  Missing implementation data for speedup analysis")
        return
    
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Level Zero Native Speedup over SYCL Baseline', fontsize=16, fontweight='bold')
    
    modes = [
        ('sync', 'yes', 'Sync + Pinned'),
        ('sync', 'no', 'Sync + Pageable'),
        ('batch', 'yes', 'Batch + Pinned'),
        ('batch', 'no', 'Batch + Pageable')
    ]
    
    for idx, (mode, pinned, title) in enumerate(modes):
        ax = axes[idx // 2, idx % 2]
        
        for op in ['D2D', 'H2D', 'D2H']:
            sycl_data = memcpy_df[
                (memcpy_df['op'] == op) &
                (memcpy_df['implementation'] == 'SYCL Baseline') &
                (memcpy_df['mode'] == mode) &
                (memcpy_df['pinned'] == pinned)
            ]
            
            native_data = memcpy_df[
                (memcpy_df['op'] == op) &
                (memcpy_df['implementation'] == 'Level Zero Native') &
                (memcpy_df['mode'] == mode) &
                (memcpy_df['pinned'] == pinned)
            ]
            
            if not sycl_data.empty and not native_data.empty:
                sycl_grouped = sycl_data.groupby('size_kib')['gib_per_s'].median()
                native_grouped = native_data.groupby('size_kib')['gib_per_s'].median()
                
                # Calculate speedup (native / sycl)
                common_sizes = sycl_grouped.index.intersection(native_grouped.index)
                speedup = native_grouped[common_sizes] / sycl_grouped[common_sizes]
                
                ax.plot(common_sizes, speedup, marker='o', linewidth=2, 
                       markersize=6, label=op, alpha=0.8)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Transfer Size (KiB)')
        ax.set_ylabel('Speedup (Native / SYCL)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    output_file = output_dir / 'native_speedup.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_comparison_summary(df, output_dir):
    """Generate summary table comparing SYCL vs Native."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    
    # Check if we have both implementations
    implementations = memcpy_df['implementation'].unique()
    if 'SYCL Baseline' not in implementations or 'Level Zero Native' not in implementations:
        print("⚠️  Missing implementation data for comparison summary")
        return None
    
    summary = []
    
    for op in ['D2D', 'H2D', 'D2H']:
        for mode in ['sync', 'batch']:
            for pinned in ['yes', 'no']:
                sycl_data = memcpy_df[
                    (memcpy_df['op'] == op) &
                    (memcpy_df['implementation'] == 'SYCL Baseline') &
                    (memcpy_df['mode'] == mode) &
                    (memcpy_df['pinned'] == pinned)
                ]
                
                native_data = memcpy_df[
                    (memcpy_df['op'] == op) &
                    (memcpy_df['implementation'] == 'Level Zero Native') &
                    (memcpy_df['mode'] == mode) &
                    (memcpy_df['pinned'] == pinned)
                ]
                
                if not sycl_data.empty and not native_data.empty:
                    sycl_peak = sycl_data['gib_per_s'].max()
                    native_peak = native_data['gib_per_s'].max()
                    speedup = (native_peak / sycl_peak - 1) * 100  # Percentage improvement
                    
                    summary.append({
                        'Operation': op,
                        'Mode': mode.capitalize(),
                        'Pinned': pinned.capitalize(),
                        'SYCL Peak (GiB/s)': f"{sycl_peak:.2f}",
                        'Native Peak (GiB/s)': f"{native_peak:.2f}",
                        'Improvement (%)': f"{speedup:+.1f}",
                    })
    
    summary_df = pd.DataFrame(summary)
    
    # Save as CSV
    output_file = output_dir / 'sycl_vs_native_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    # Save as Markdown
    output_file = output_dir / 'sycl_vs_native_summary.md'
    with open(output_file, 'w') as f:
        f.write("# SYCL Baseline vs Level Zero Native Comparison\n\n")
        f.write("| " + " | ".join(summary_df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(summary_df.columns)) + " |\n")
        for _, row in summary_df.iterrows():
            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
    print(f"Saved: {output_file}")
    
    return summary_df

# Import original functions from analyze_results.py
def plot_bandwidth_comparison(df, output_dir, metadata=None):
    """Original bandwidth comparison (all backends/implementations)."""
    from analyze_results import plot_bandwidth_comparison as original_plot
    original_plot(df, output_dir, metadata)

def plot_mode_comparison(df, output_dir):
    """Original mode comparison."""
    from analyze_results import plot_mode_comparison as original_plot
    original_plot(df, output_dir)

def plot_overhead_analysis(df, output_dir):
    """Original overhead analysis."""
    from analyze_results import plot_overhead_analysis as original_plot
    original_plot(df, output_dir)

def plot_event_overhead(df, output_dir):
    """Original event overhead."""
    from analyze_results import plot_event_overhead as original_plot
    original_plot(df, output_dir)

def plot_peak_bandwidth_summary(df, output_dir):
    """Original peak bandwidth summary."""
    from analyze_results import plot_peak_bandwidth_summary as original_plot
    original_plot(df, output_dir)

def generate_summary_table(df, output_dir):
    """Original summary table."""
    from analyze_results import generate_summary_table as original_func
    return original_func(df, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Analyze Celerity backend benchmark results with native comparison')
    parser.add_argument('results_dir', nargs='?', default='results',
                       help='Directory containing benchmark CSV files (default: results)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for plots (default: auto-detect from results_dir)')
    
    args = parser.parse_args()
    
    # Auto-detect output directory
    if args.output is None:
        results_path = Path(args.results_dir)
        dir_name = results_path.name
        
        if dir_name.startswith('results_'):
            version_part = dir_name[8:]
            parts = version_part.split('_')
            if parts and parts[-1].isdigit() and len(parts[-1]) >= 8:
                version_part = '_'.join(parts[:-1])
            args.output = f'individual_plots/plots_{version_part}'
        else:
            args.output = 'individual_plots/plots'
    
    # Load data
    print(f"Loading CSV files from: {args.results_dir}")
    df, metadata = load_all_csvs(args.results_dir)
    
    if df is None or df.empty:
        print("No data loaded. Exiting.")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate plots
    print("\n=== Generating SYCL vs Native Comparison Plots ===")
    plot_sycl_vs_native_comparison(df, output_dir, metadata)
    plot_native_overhead_analysis(df, output_dir)
    plot_speedup_analysis(df, output_dir)
    
    print("\n=== Generating Standard Plots ===")
    # Note: These will fail if analyze_results.py is not in the same directory
    # For now, we'll skip them and focus on the comparison plots
    # plot_bandwidth_comparison(df, output_dir, metadata)
    # plot_mode_comparison(df, output_dir)
    # plot_overhead_analysis(df, output_dir)
    # plot_event_overhead(df, output_dir)
    # plot_peak_bandwidth_summary(df, output_dir)
    
    # Generate summary tables
    print("\n=== Generating Comparison Summary ===")
    summary_df = generate_comparison_summary(df, output_dir)
    if summary_df is not None:
        print("\n" + summary_df.to_string(index=False))
    
    # generate_summary_table(df, output_dir)
    
    print(f"\n✅ Analysis complete! Check {output_dir}/ for results.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
