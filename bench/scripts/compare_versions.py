#!/usr/bin/env python3
"""
Compare multiple backend versions side-by-side.
Generates comparison plots and tables for different backend implementations.
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

def load_version_data(result_dir):
    """Load data from a single version directory."""
    result_path = Path(result_dir)
    
    # Load metadata
    metadata_file = result_path / "metadata.txt"
    version_tag = "unknown"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.startswith('Backend Tag:'):
                    version_tag = line.split(':', 1)[1].strip()
                    break
    
    # If not in metadata, try to extract from directory name
    if version_tag == "unknown" or version_tag == "none":
        dir_name = result_path.name
        if 'results_' in dir_name:
            parts = dir_name.split('_')
            if len(parts) >= 2 and parts[1].startswith('v'):
                version_parts = []
                for i in range(1, len(parts)):
                    if parts[i].isdigit() and len(parts[i]) == 8:
                        break
                    version_parts.append(parts[i])
                version_tag = '_'.join(version_parts) if version_parts else "unknown"
    
    # Load CSVs
    csv_files = list(result_path.rglob("*.csv"))
    if not csv_files:
        return None, version_tag
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['backend_version'] = version_tag
            
            # Determine implementation type from backend column or filename
            if 'backend' in df.columns:
                backend_val = df['backend'].iloc[0] if len(df) > 0 else ''
                if 'native' in backend_val.lower():
                    df['implementation'] = 'L0 Native'
                elif 'level_zero' in backend_val.lower():
                    df['implementation'] = 'L0 Backend'
                elif 'opencl' in backend_val.lower() or 'generic' in backend_val.lower():
                    df['implementation'] = 'Generic SYCL'
                else:
                    df['implementation'] = 'L0 Backend'
            else:
                # Fallback to filename detection
                filename = csv_file.name
                if 'l0_native_' in filename or 'native_' in filename:
                    df['implementation'] = 'L0 Native'
                elif 'generic_' in filename or 'opencl_' in filename:
                    df['implementation'] = 'Generic SYCL'
                else:
                    df['implementation'] = 'L0 Backend'
            
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        return None, version_tag
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate
    if 'backend' in combined.columns and len(combined['backend'].unique()) == 1:
        combined = combined.drop_duplicates(
            subset=['bytes', 'op', 'mode', 'pinned', 'backend', 'bench'],
            keep='first'
        )
    
    return combined, version_tag

def plot_version_comparison(versions_data, output_dir):
    """Plot bandwidth comparison across versions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Combine all data
    all_data = pd.concat([df for df, _ in versions_data], ignore_index=True)
    memcpy_df = all_data[all_data['bench'] == 'memcpy_linear'].copy()
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    operations = ['D2D', 'H2D', 'D2H']
    modes = [('sync', 'yes'), ('batch', 'yes'), ('sync', 'no'), ('batch', 'no')]  # All four combinations
    
    for op in operations:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{op} Bandwidth: Version Comparison', fontsize=16, fontweight='bold')
        
        for idx, (mode, pinned) in enumerate(modes):
            ax = axes[idx // 2, idx % 2]
            
            for _, version_tag in versions_data:
                data = memcpy_df[
                    (memcpy_df['op'] == op) &
                    (memcpy_df['mode'] == mode) &
                    (memcpy_df['pinned'] == pinned) &
                    (memcpy_df['backend_version'] == version_tag)
                ]
                
                if not data.empty:
                    grouped = data.groupby('size_kib')['gib_per_s'].median().reset_index()
                    ax.plot(grouped['size_kib'], grouped['gib_per_s'],
                           marker='o', linewidth=2, markersize=6,
                           label=version_tag)
            
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Transfer Size (KiB)')
            ax.set_ylabel('Bandwidth (GiB/s)')
            pinned_label = 'Pinned' if pinned == 'yes' else 'Pageable'
            ax.set_title(f'{mode.capitalize()} + {pinned_label}')
            ax.grid(True, alpha=0.3, which='both')
            ax.legend()
        
        plt.tight_layout()
        output_file = output_dir / f'version_comparison_{op.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_mode_comparison(versions_data, output_dir):
    """Plot all modes on single plot per operation (like analyze_results.py)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Combine all data
    all_data = pd.concat([df for df, _ in versions_data], ignore_index=True)
    memcpy_df = all_data[all_data['bench'] == 'memcpy_linear'].copy()
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Version Comparison: All Modes', fontsize=16, fontweight='bold')
    
    operations = ['D2D', 'H2D', 'D2H']
    
    for idx, op in enumerate(operations):
        ax = axes[idx]
        op_data = memcpy_df[memcpy_df['op'] == op]
        
        # Plot all 4 mode combinations for each version
        for _, version_tag in versions_data:
            version_data = op_data[op_data['backend_version'] == version_tag]
            
            for mode in ['sync', 'batch']:
                for pinned in ['yes', 'no']:
                    data = version_data[
                        (version_data['mode'] == mode) &
                        (version_data['pinned'] == pinned)
                    ]
                    
                    if not data.empty:
                        grouped = data.groupby('size_kib')['gib_per_s'].median().reset_index()
                        label = f"{version_tag}: {mode.capitalize()} + {'Pin' if pinned == 'yes' else 'Page'}"
                        linestyle = '-' if mode == 'batch' else '--'
                        linewidth = 2.5 if pinned == 'yes' else 1.5
                        
                        ax.plot(grouped['size_kib'], grouped['gib_per_s'],
                               marker='o', linestyle=linestyle, linewidth=linewidth,
                               markersize=4, label=label, alpha=0.8)
        
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Transfer Size (KiB)')
        ax.set_ylabel('Bandwidth (GiB/s)')
        ax.set_title(f'{op} Performance')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    output_file = output_dir / 'version_comparison_all_modes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_sycl_vs_native_comparison(versions_data, output_dir):
    """Plot implementation comparison across all versions."""
    output_dir = Path(output_dir)
    
    all_data = pd.concat([df for df, _ in versions_data], ignore_index=True)
    memcpy_df = all_data[all_data['bench'].str.contains('memcpy', case=False, na=False)].copy()
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    # Get available implementations
    implementations = memcpy_df['implementation'].unique()
    
    operations = ['D2D', 'H2D', 'D2H']
    
    # Define styles for each implementation
    impl_styles = {
        'Generic SYCL': {'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.6},
        'L0 Backend': {'linestyle': '--', 'linewidth': 1.8, 'alpha': 0.75},
        'L0 Native': {'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.9}
    }
    
    for op in operations:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{op}: Implementation Comparison (All Versions)', fontsize=16, fontweight='bold')
        
        modes = [
            ('sync', 'yes', 'Sync + Pinned'),
            ('sync', 'no', 'Sync + Pageable'),
            ('batch', 'yes', 'Batch + Pinned'),
            ('batch', 'no', 'Batch + Pageable')
        ]
        
        for idx, (mode, pinned, title) in enumerate(modes):
            ax = axes[idx // 2, idx % 2]
            
            # Plot reference implementations first (once each)
            for ref_impl in ['Generic SYCL', 'L0 Native']:
                if ref_impl not in implementations:
                    continue
                
                ref_data = memcpy_df[
                    (memcpy_df['implementation'] == ref_impl) &
                    (memcpy_df['op'] == op) &
                    (memcpy_df['mode'] == mode) &
                    (memcpy_df['pinned'] == pinned)
                ]
                
                if not ref_data.empty:
                    grouped = ref_data.groupby('size_kib')['gib_per_s'].median().reset_index()
                    style = impl_styles.get(ref_impl, {'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.8})
                    ax.plot(grouped['size_kib'], grouped['gib_per_s'],
                           marker='o', markersize=3, label=ref_impl, **style)
            
            # Plot each backend variant
            for _, version_tag in versions_data:
                # Skip reference implementations (already plotted)
                if version_tag in ['L0 Native', 'Generic SYCL']:
                    continue
                
                version_data = memcpy_df[
                    (memcpy_df['backend_version'] == version_tag) &
                    (memcpy_df['implementation'] == 'L0 Backend') &
                    (memcpy_df['op'] == op) &
                    (memcpy_df['mode'] == mode) &
                    (memcpy_df['pinned'] == pinned)
                ]
                
                if not version_data.empty:
                    grouped = version_data.groupby('size_kib')['gib_per_s'].median().reset_index()
                    style = impl_styles.get('L0 Backend', {'linestyle': '--', 'linewidth': 1.8, 'alpha': 0.75})
                    label = version_tag
                    ax.plot(grouped['size_kib'], grouped['gib_per_s'],
                           marker='o', markersize=3, label=label, **style)
            
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Transfer Size (KiB)')
            ax.set_ylabel('Bandwidth (GiB/s)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=6, loc='best', ncol=2)
        
        plt.tight_layout()
        output_file = output_dir / f'implementation_comparison_all_versions_{op.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def generate_comparison_table(versions_data, output_dir):
    """Generate comparison table of peak bandwidths."""
    output_dir = Path(output_dir)
    
    all_data = pd.concat([df for df, _ in versions_data], ignore_index=True)
    memcpy_df = all_data[all_data['bench'] == 'memcpy_linear'].copy()
    
    comparison = []
    
    for _, version_tag in versions_data:
        version_data = memcpy_df[memcpy_df['backend_version'] == version_tag]
        
        for op in ['D2D', 'H2D', 'D2H']:
            for mode in ['sync', 'batch']:
                for pinned in ['yes', 'no']:
                    data = version_data[
                        (version_data['op'] == op) &
                        (version_data['mode'] == mode) &
                        (version_data['pinned'] == pinned)
                    ]
                    
                    if not data.empty:
                        comparison.append({
                            'Version': version_tag,
                            'Operation': op,
                            'Mode': mode.capitalize(),
                            'Pinned': pinned.capitalize(),
                            'Peak (GiB/s)': f"{data['gib_per_s'].max():.2f}",
                            'Median (GiB/s)': f"{data['gib_per_s'].median():.2f}",
                        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Save as CSV
    output_file = output_dir / 'version_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    # Save as Markdown
    output_file = output_dir / 'version_comparison.md'
    try:
        with open(output_file, 'w') as f:
            f.write("# Backend Version Comparison\n\n")
            f.write(comparison_df.to_markdown(index=False))
        print(f"Saved: {output_file}")
    except ImportError:
        # Fallback without tabulate
        with open(output_file, 'w') as f:
            f.write("# Backend Version Comparison\n\n")
            f.write("| " + " | ".join(comparison_df.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(comparison_df.columns)) + " |\n")
            for _, row in comparison_df.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        print(f"Saved: {output_file} (simple format)")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Compare multiple backend versions')
    parser.add_argument('result_dirs', nargs='+',
                       help='Result directories to compare (e.g., results/results_v1_* results/results_v2_*)')
    parser.add_argument('--output', '-o', default='comparison',
                       help='Output directory for comparison plots (default: comparison)')
    
    args = parser.parse_args()
    
    # Load all versions
    print("=== Loading Backend Versions ===")
    versions_data = []
    reference_implementations = {}  # Store reference implementations separately
    
    for result_dir in args.result_dirs:
        print(f"\nLoading: {result_dir}")
        df, version_tag = load_version_data(result_dir)
        if df is not None:
            print(f"  Version: {version_tag}")
            print(f"  Rows: {len(df)}")
            
            # Separate reference implementations from backend implementations
            if 'implementation' in df.columns:
                # Split data by implementation
                backend_data = df[df['implementation'] == 'L0 Backend'].copy()
                native_data = df[df['implementation'] == 'L0 Native'].copy()
                generic_data = df[df['implementation'] == 'Generic SYCL'].copy()
                
                # Store reference implementations only once (they're the same across all variants)
                if not native_data.empty and 'L0 Native' not in reference_implementations:
                    # Update backend_version to match the label we want
                    native_data['backend_version'] = 'L0 Native'
                    reference_implementations['L0 Native'] = (native_data, 'L0 Native')
                    print(f"  Stored L0 Native reference data ({len(native_data)} rows)")
                
                if not generic_data.empty and 'Generic SYCL' not in reference_implementations:
                    # Update backend_version to match the label we want
                    generic_data['backend_version'] = 'Generic SYCL'
                    reference_implementations['Generic SYCL'] = (generic_data, 'Generic SYCL')
                    print(f"  Stored Generic SYCL reference data ({len(generic_data)} rows)")
                
                # Always keep backend data (unique per variant)
                if not backend_data.empty:
                    versions_data.append((backend_data, version_tag))
                    print(f"  Stored L0 Backend data ({len(backend_data)} rows)")
            else:
                # No implementation column, keep all data
                versions_data.append((df, version_tag))
        else:
            print(f"  No data found")
    
    # Add reference implementations back to versions_data
    for impl_name, (ref_df, ref_tag) in reference_implementations.items():
        versions_data.append((ref_df, ref_tag))
        print(f"\nAdded {impl_name} reference data ({len(ref_df)} rows)")
    
    if len(versions_data) < 2:
        print("\nError: Need at least 2 versions to compare")
        return 1
    
    print(f"\n=== Comparing {len(versions_data)} Versions ===")
    for _, version_tag in versions_data:
        print(f"  - {version_tag}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate comparison plots
    print("\n=== Generating Comparison Plots ===")
    plot_version_comparison(versions_data, output_dir)
    plot_mode_comparison(versions_data, output_dir)
    
    # Check if we have multiple implementations to compare
    all_data = pd.concat([df for df, _ in versions_data], ignore_index=True)
    if 'implementation' in all_data.columns:
        implementations = all_data['implementation'].unique()
        if len(implementations) > 1:
            print("\n=== Generating Implementation Comparison ===")
            print(f"Comparing: {', '.join(implementations)}")
            plot_sycl_vs_native_comparison(versions_data, output_dir)
    
    # Generate comparison table
    print("\n=== Generating Comparison Table ===")
    comparison_df = generate_comparison_table(versions_data, output_dir)
    print("\n" + comparison_df.to_string(index=False))
    
    print(f"\n✅ Comparison complete! Check {output_dir}/ for results.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
