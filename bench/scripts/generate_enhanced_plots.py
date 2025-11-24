#!/usr/bin/env python3
"""
Generate enhanced comparison plots for SYCL vs Native benchmarks.
Creates additional visualizations beyond the standard plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_all_versions(results_dir):
    """Load all version results."""
    results_path = Path(results_dir)
    version_dirs = sorted([d for d in results_path.glob("results_*") if d.is_dir()])
    
    all_data = []
    for version_dir in version_dirs:
        csv_files = list(version_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Detect implementation
                if 'backend' in df.columns and len(df) > 0:
                    backend_val = df['backend'].iloc[0]
                    if 'native' in backend_val.lower():
                        df['implementation'] = 'L0 Native'
                    elif 'level_zero' in backend_val.lower():
                        df['implementation'] = 'L0 Backend'
                    elif 'opencl' in backend_val.lower() or 'generic' in backend_val.lower():
                        df['implementation'] = 'Generic SYCL'
                    else:
                        df['implementation'] = 'L0 Backend'
                
                # Extract version from directory name
                dir_name = version_dir.name
                if 'results_' in dir_name:
                    parts = dir_name.split('_')
                    version_parts = []
                    for i in range(1, len(parts)):
                        if parts[i].isdigit() and len(parts[i]) == 8:
                            break
                        version_parts.append(parts[i])
                    df['version'] = '_'.join(version_parts) if version_parts else 'unknown'
                
                all_data.append(df)
            except Exception as e:
                continue
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined[combined['bench'].str.contains('memcpy', case=False, na=False)].copy()

def plot_speedup_heatmap(df, output_dir):
    """Create heatmap showing L0 Native speedup over L0 Backend for each variant."""
    output_dir = Path(output_dir)
    
    # Determine what implementations we have
    implementations = df['implementation'].unique()
    
    # Decide comparison: prefer L0 Backend vs L0 Native, fallback to Generic SYCL vs L0 Backend
    if 'L0 Backend' in implementations and 'L0 Native' in implementations:
        baseline_impl = 'L0 Backend'
        compare_impl = 'L0 Native'
        title_suffix = 'L0 Native vs L0 Backend'
    elif 'Generic SYCL' in implementations and 'L0 Backend' in implementations:
        baseline_impl = 'Generic SYCL'
        compare_impl = 'L0 Backend'
        title_suffix = 'L0 Backend vs Generic SYCL'
    else:
        print("  Not enough implementations for speedup heatmap")
        return
    
    # Calculate speedup for each version/operation/mode combination
    speedups = []
    
    versions = sorted(df['version'].unique())
    operations = ['D2D', 'H2D', 'D2H']
    modes = [('sync', 'yes'), ('batch', 'yes')]
    
    for version in versions:
        for op in operations:
            for mode, pinned in modes:
                baseline_data = df[
                    (df['version'] == version) &
                    (df['implementation'] == baseline_impl) &
                    (df['op'] == op) &
                    (df['mode'] == mode) &
                    (df['pinned'] == pinned)
                ]
                
                compare_data = df[
                    (df['version'] == version) &
                    (df['implementation'] == compare_impl) &
                    (df['op'] == op) &
                    (df['mode'] == mode) &
                    (df['pinned'] == pinned)
                ]
                
                if not baseline_data.empty and not compare_data.empty:
                    baseline_peak = baseline_data['gib_per_s'].max()
                    compare_peak = compare_data['gib_per_s'].max()
                    speedup = compare_peak / baseline_peak
                    
                    mode_label = f"{mode.capitalize()}+Pin"
                    speedups.append({
                        'version': version,
                        'config': f"{op} {mode_label}",
                        'speedup': speedup
                    })
    
    if not speedups:
        print("No speedup data available")
        return
    
    speedup_df = pd.DataFrame(speedups)
    
    # Pivot for heatmap
    pivot = speedup_df.pivot(index='version', columns='config', values='speedup')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use diverging colormap centered at 1.0
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=1.0,
                vmin=0.9, vmax=1.1, cbar_kws={'label': 'Speedup (Native/SYCL)'},
                linewidths=0.5, ax=ax)
    
    ax.set_title(f'{title_suffix} Speedup Heatmap\n(Green = {compare_impl} Faster, Red = {baseline_impl} Faster)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Backend Version', fontsize=12)
    
    plt.tight_layout()
    output_file = output_dir / 'speedup_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_improvement_bars(df, output_dir):
    """Create bar chart showing percentage improvement."""
    output_dir = Path(output_dir)
    
    # Determine what implementations we have
    implementations = df['implementation'].unique()
    
    # Decide comparison
    if 'L0 Backend' in implementations and 'L0 Native' in implementations:
        baseline_impl = 'L0 Backend'
        compare_impl = 'L0 Native'
        title = 'L0 Native Performance Improvement over L0 Backend (%)'
    elif 'Generic SYCL' in implementations and 'L0 Backend' in implementations:
        baseline_impl = 'Generic SYCL'
        compare_impl = 'L0 Backend'
        title = 'L0 Backend Performance Improvement over Generic SYCL (%)'
    else:
        print("  Not enough implementations for improvement bars")
        return
    
    versions = sorted(df['version'].unique())
    operations = ['D2D', 'H2D', 'D2H']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, op in enumerate(operations):
        ax = axes[idx]
        
        improvements = []
        labels = []
        
        for version in versions:
            # Use batch+pinned as representative
            baseline_data = df[
                (df['version'] == version) &
                (df['implementation'] == baseline_impl) &
                (df['op'] == op) &
                (df['mode'] == 'batch') &
                (df['pinned'] == 'yes')
            ]
            
            compare_data = df[
                (df['version'] == version) &
                (df['implementation'] == compare_impl) &
                (df['op'] == op) &
                (df['mode'] == 'batch') &
                (df['pinned'] == 'yes')
            ]
            
            if not baseline_data.empty and not compare_data.empty:
                baseline_peak = baseline_data['gib_per_s'].max()
                compare_peak = compare_data['gib_per_s'].max()
                improvement = ((compare_peak / baseline_peak) - 1) * 100
                
                improvements.append(improvement)
                # Shorten version names for readability
                short_name = version.replace('_', '\n').replace('v', 'v')
                labels.append(short_name)
        
        # Create bar chart
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top',
                   fontsize=9, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Backend Version', fontsize=10)
        ax.set_ylabel('Improvement (%)', fontsize=10)
        ax.set_title(f'{op} (Batch+Pinned)', fontsize=12)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'native_improvement_bars.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_normalized_performance(df, output_dir):
    """Plot performance normalized to baseline for easy comparison."""
    output_dir = Path(output_dir)
    
    # Get baseline performance
    baseline_version = 'v0_baseline'
    
    # Determine available implementations
    available_impls = df['implementation'].unique()
    impl_map = {}
    if 'L0 Backend' in available_impls:
        impl_map['L0 Backend'] = 'L0 Backend'
    if 'L0 Native' in available_impls:
        impl_map['L0 Native'] = 'L0 Native'
    if 'Generic SYCL' in available_impls:
        impl_map['Generic SYCL'] = 'Generic SYCL'
    
    if len(impl_map) < 2:
        print("  Not enough implementations for normalized performance plots")
        return
    
    operations = ['D2D', 'H2D', 'D2H']
    
    for op in operations:
        fig, axes = plt.subplots(1, len(impl_map), figsize=(8 * len(impl_map), 6))
        if len(impl_map) == 1:
            axes = [axes]
        fig.suptitle(f'{op} Performance Normalized to Baseline', 
                    fontsize=16, fontweight='bold')
        
        for impl_idx, (impl_name, impl_label) in enumerate(impl_map.items()):
            ax = axes[impl_idx]
            
            # Get baseline performance
            baseline_data = df[
                (df['version'] == baseline_version) &
                (df['implementation'] == impl_name) &
                (df['op'] == op) &
                (df['mode'] == 'batch') &
                (df['pinned'] == 'yes')
            ]
            
            if baseline_data.empty:
                ax.text(0.5, 0.5, f'No baseline data for {impl_label}',
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            baseline_peak = baseline_data['gib_per_s'].max()
            
            # Plot normalized performance for each version
            versions = sorted(df['version'].unique())
            normalized_perfs = []
            labels = []
            
            for version in versions:
                version_data = df[
                    (df['version'] == version) &
                    (df['implementation'] == impl_name) &
                    (df['op'] == op) &
                    (df['mode'] == 'batch') &
                    (df['pinned'] == 'yes')
                ]
                
                if not version_data.empty:
                    peak = version_data['gib_per_s'].max()
                    normalized = (peak / baseline_peak) * 100
                    normalized_perfs.append(normalized)
                    labels.append(version.replace('_', '\n'))
            
            if not normalized_perfs:
                ax.text(0.5, 0.5, f'No data for {impl_label}',
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create bar chart
            colors = ['green' if x >= 100 else 'orange' for x in normalized_perfs]
            bars = ax.bar(range(len(normalized_perfs)), normalized_perfs, 
                         color=colors, alpha=0.7)
            
            # Add value labels
            for bar, val in zip(bars, normalized_perfs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
            
            ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, 
                      label='Baseline', alpha=0.7)
            ax.set_xlabel('Backend Version', fontsize=10)
            ax.set_ylabel('Performance (% of Baseline)', fontsize=10)
            ax.set_title(f'{impl_label}', fontsize=12)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
        
        plt.tight_layout()
        output_file = output_dir / f'normalized_performance_{op.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_best_variant_summary(df, output_dir):
    """Create summary showing which variant performs best for each configuration."""
    output_dir = Path(output_dir)
    
    # Get available implementations
    available_impls = sorted(df['implementation'].unique())
    
    if len(available_impls) < 2:
        print("  Not enough implementations for best variant summary")
        return
    
    operations = ['D2D', 'H2D', 'D2H']
    modes = [('sync', 'yes', 'Sync+Pin'), ('batch', 'yes', 'Batch+Pin')]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Best Performing Variant by Configuration', 
                fontsize=16, fontweight='bold')
    
    for mode_idx, (mode, pinned, mode_label) in enumerate(modes):
        for op_idx, op in enumerate(operations):
            ax = axes[mode_idx, op_idx]
            
            # Find best variant for each implementation
            best_results = []
            
            for impl in available_impls:
                versions = df['version'].unique()
                best_perf = 0
                best_version = None
                
                for version in versions:
                    version_data = df[
                        (df['version'] == version) &
                        (df['implementation'] == impl) &
                        (df['op'] == op) &
                        (df['mode'] == mode) &
                        (df['pinned'] == pinned)
                    ]
                    
                    if not version_data.empty:
                        peak = version_data['gib_per_s'].max()
                        if peak > best_perf:
                            best_perf = peak
                            best_version = version
                
                if best_version:
                    best_results.append({
                        'impl': impl,
                        'version': best_version,
                        'perf': best_perf
                    })
            
            # Plot
            if best_results:
                impls = [r['impl'] for r in best_results]
                perfs = [r['perf'] for r in best_results]
                versions = [r['version'].replace('_', '\n') for r in best_results]
                
                # Use different colors for different implementations
                colors = ['steelblue', 'coral', 'lightgreen'][:len(impls)]
                bars = ax.bar(impls, perfs, color=colors, alpha=0.7)
                
                # Add labels
                for bar, perf, version in zip(bars, perfs, versions):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{perf:.2f} GiB/s\n{version}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                ax.set_ylabel('Peak Bandwidth (GiB/s)', fontsize=10)
                ax.set_title(f'{op} - {mode_label}', fontsize=11)
                ax.set_xticklabels(impls, rotation=15, ha='right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes)
    
    plt.tight_layout()
    output_file = output_dir / 'best_variant_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_enhanced_plots.py <results_dir> [output_dir]")
        print("Example: python3 generate_enhanced_plots.py results comparison_all")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'comparison_all'
    
    print(f"Loading data from: {results_dir}")
    df = load_all_versions(results_dir)
    
    if df is None or df.empty:
        print("No data loaded")
        return 1
    
    print(f"Loaded {len(df)} rows")
    print(f"Versions: {', '.join(sorted(df['version'].unique()))}")
    print(f"Implementations: {', '.join(df['implementation'].unique())}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Generating Enhanced Plots ===")
    plot_speedup_heatmap(df, output_path)
    plot_improvement_bars(df, output_path)
    plot_normalized_performance(df, output_path)
    plot_best_variant_summary(df, output_path)
    
    print(f"\n✅ Enhanced plots saved to {output_path}/")
    return 0

if __name__ == '__main__':
    sys.exit(main())
