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
    reference_data = {}  # Store reference implementations separately
    all_versions = []  # Track all version names
    
    for version_dir in version_dirs:
        # Extract version from directory name
        dir_name = version_dir.name
        version_tag = 'unknown'
        if 'results_' in dir_name:
            parts = dir_name.split('_')
            version_parts = []
            for i in range(1, len(parts)):
                if parts[i].isdigit() and len(parts[i]) == 8:
                    break
                version_parts.append(parts[i])
            version_tag = '_'.join(version_parts) if version_parts else 'unknown'
        
        all_versions.append(version_tag)
        
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
                
                df['version'] = version_tag
                
                # For reference implementations (L0 Native, Generic SYCL), 
                # only keep the first occurrence (they're the same across all variants)
                impl = df['implementation'].iloc[0] if len(df) > 0 else 'unknown'
                if impl in ['L0 Native', 'Generic SYCL']:
                    if impl not in reference_data:
                        # First time seeing this reference implementation, keep it
                        reference_data[impl] = df
                    # Skip subsequent copies of the same reference data
                else:
                    # L0 Backend data - keep all variants
                    all_data.append(df)
                    
            except Exception as e:
                continue
    
    # Add reference data back (one copy each)
    # For comparison purposes, we'll keep the version from the first variant
    for impl, df_ref in reference_data.items():
        # Keep the reference data with its original version tag
        all_data.append(df_ref)
        print(f"  Added {impl} reference: {len(df_ref)} rows, version={df_ref['version'].iloc[0] if len(df_ref) > 0 else 'unknown'}")
    
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
    
    # Decide comparison - prioritize showing L0 Backend improvement over Generic SYCL
    if 'Generic SYCL' in implementations and 'L0 Backend' in implementations:
        baseline_impl = 'Generic SYCL'
        compare_impl = 'L0 Backend'
        title = 'L0 Backend Performance Improvement over Generic SYCL (%)'
    elif 'L0 Backend' in implementations and 'L0 Native' in implementations:
        baseline_impl = 'L0 Backend'
        compare_impl = 'L0 Native'
        title = 'L0 Native Performance Improvement over L0 Backend (%)'
    else:
        print("  Not enough implementations for improvement bars")
        return
    
    # Get backend versions (exclude reference implementations)
    backend_versions = sorted([v for v in df['version'].unique() 
                               if v not in ['L0 Native', 'Generic SYCL', 'unknown']])
    
    print(f"  Backend versions found: {len(backend_versions)}")
    print(f"  Comparing {compare_impl} vs {baseline_impl}")
    
    if not backend_versions:
        print("  No backend versions found for improvement bars")
        return
    
    operations = ['D2D', 'H2D', 'D2H']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, op in enumerate(operations):
        ax = axes[idx]
        
        improvements = []
        labels = []
        
        # Get reference implementation data (same for all versions)
        compare_data = df[
            (df['implementation'] == compare_impl) &
            (df['op'] == op) &
            (df['mode'] == 'batch') &
            (df['pinned'] == 'yes')
        ]
        
        # Debug: Check what data we have
        all_compare_data = df[df['implementation'] == compare_impl]
        if idx == 0:  # Only print once
            print(f"    {compare_impl} total rows: {len(all_compare_data)}")
            if len(all_compare_data) > 0:
                print(f"    {compare_impl} ops: {all_compare_data['op'].unique()}")
                print(f"    {compare_impl} modes: {all_compare_data['mode'].unique()}")
                print(f"    {compare_impl} pinned: {all_compare_data['pinned'].unique()}")
        
        if compare_data.empty:
            ax.text(0.5, 0.5, f'No {compare_impl} data for {op}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{op} (Batch+Pinned)', fontsize=12)
            continue
        
        compare_peak = compare_data['gib_per_s'].max()
        
        for version in backend_versions:
            # Get backend data for this version
            baseline_data = df[
                (df['version'] == version) &
                (df['implementation'] == baseline_impl) &
                (df['op'] == op) &
                (df['mode'] == 'batch') &
                (df['pinned'] == 'yes')
            ]
            
            if not baseline_data.empty:
                baseline_peak = baseline_data['gib_per_s'].max()
                improvement = ((compare_peak / baseline_peak) - 1) * 100
                
                improvements.append(improvement)
                # Shorten version names for readability
                short_name = version.replace('_', '\n').replace('v', 'v')
                labels.append(short_name)
        
        if not improvements:
            ax.text(0.5, 0.5, f'No {baseline_impl} data for {op}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{op} (Batch+Pinned)', fontsize=12)
            continue
        
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
    
    # Only normalize L0 Backend variants (not reference implementations)
    if 'L0 Backend' not in df['implementation'].unique():
        print("  No L0 Backend data for normalized performance plots")
        return
    
    # Get backend versions (exclude reference implementations)
    backend_versions = sorted([v for v in df['version'].unique() 
                               if v not in ['L0 Native', 'Generic SYCL', 'unknown']])
    
    if not backend_versions or baseline_version not in backend_versions:
        print(f"  Baseline version {baseline_version} not found")
        return
    
    operations = ['D2D', 'H2D', 'D2H']
    
    for op in operations:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        fig.suptitle(f'{op} Performance Normalized to Baseline (L0 Backend)', 
                    fontsize=16, fontweight='bold')
        
        # Get baseline performance
        baseline_data = df[
            (df['version'] == baseline_version) &
            (df['implementation'] == 'L0 Backend') &
            (df['op'] == op) &
            (df['mode'] == 'batch') &
            (df['pinned'] == 'yes')
        ]
        
        if baseline_data.empty:
            ax.text(0.5, 0.5, f'No baseline data for {baseline_version}',
                   ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            output_file = output_dir / f'normalized_performance_{op.lower()}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
            continue
        
        baseline_peak = baseline_data['gib_per_s'].max()
        
        # Plot normalized performance for each backend version
        normalized_perfs = []
        labels = []
        
        for version in backend_versions:
            version_data = df[
                (df['version'] == version) &
                (df['implementation'] == 'L0 Backend') &
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
            ax.text(0.5, 0.5, 'No data',
                   ha='center', va='center', transform=ax.transAxes)
        else:
            # Create bar chart
            colors = ['green' if x >= 100 else 'orange' for x in normalized_perfs]
            bars = ax.bar(range(len(normalized_perfs)), normalized_perfs, 
                         color=colors, alpha=0.7)
            
            # Add value labels
            for bar, val in zip(bars, normalized_perfs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}%', ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
            
            ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, 
                      label='Baseline (100%)', alpha=0.7)
            ax.set_xlabel('Backend Version', fontsize=10)
            ax.set_ylabel('Performance (% of Baseline)', fontsize=10)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            
            # Zoom in on the differences if they're all close to 100%
            min_val = min(normalized_perfs)
            max_val = max(normalized_perfs)
            if max_val - min_val < 5:  # If all within 5%
                # Zoom in to show small differences
                y_center = (min_val + max_val) / 2
                y_range = max(5, max_val - min_val + 2)  # At least 5% range
                ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            
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
            
            # Get backend versions (exclude reference implementations)
            backend_versions = [v for v in df['version'].unique() 
                               if v not in ['L0 Native', 'Generic SYCL', 'unknown']]
            
            # For L0 Backend, find the best variant
            if 'L0 Backend' in available_impls and backend_versions:
                best_perf = 0
                best_version = None
                
                for version in backend_versions:
                    version_data = df[
                        (df['version'] == version) &
                        (df['implementation'] == 'L0 Backend') &
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
                        'impl': 'L0 Backend',
                        'version': best_version,
                        'perf': best_perf
                    })
            
            # For reference implementations, just get their performance
            for ref_impl in ['Generic SYCL', 'L0 Native']:
                if ref_impl in available_impls:
                    ref_data = df[
                        (df['implementation'] == ref_impl) &
                        (df['op'] == op) &
                        (df['mode'] == mode) &
                        (df['pinned'] == pinned)
                    ]
                    
                    if not ref_data.empty:
                        best_results.append({
                            'impl': ref_impl,
                            'version': ref_impl,
                            'perf': ref_data['gib_per_s'].max()
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
    
    # Debug: Show data distribution
    for impl in df['implementation'].unique():
        impl_data = df[df['implementation'] == impl]
        versions = impl_data['version'].unique()
        print(f"  {impl}: {len(impl_data)} rows, versions: {', '.join(sorted(versions))}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Generating Enhanced Plots ===")
    plot_speedup_heatmap(df, output_path)
    # plot_improvement_bars(df, output_path)  # Disabled - data loading issues
    plot_normalized_performance(df, output_path)
    plot_best_variant_summary(df, output_path)
    
    print(f"\n✅ Enhanced plots saved to {output_path}/")
    return 0

if __name__ == '__main__':
    sys.exit(main())
