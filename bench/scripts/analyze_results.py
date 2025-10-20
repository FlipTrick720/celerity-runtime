#!/usr/bin/env python3
"""
Analyze and plot Celerity backend benchmark results.
Generates publication-quality plots comparing Level Zero vs CUDA performance.
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
            # Parse metadata for later use
            for line in content.split('\n'):
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
        return metadata
    return metadata

def load_all_csvs(results_dir):
    """Load all CSV files from results directory."""
    results_path = Path(results_dir)
    
    # Try to load metadata first
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
            # Add backend version info if available from metadata
            if metadata and 'Backend Tag' in metadata:
                df['backend_version'] = metadata['Backend Tag']
            else:
                # Try to extract from directory name (results_v1_baseline_*)
                dir_name = csv_file.parent.name
                if 'results_' in dir_name:
                    parts = dir_name.split('_')
                    if len(parts) >= 2 and parts[1].startswith('v'):
                        # Extract version tag (e.g., v1_baseline from results_v1_baseline_20251014)
                        version_parts = []
                        for i in range(1, len(parts)):
                            if parts[i].isdigit() and len(parts[i]) == 8:  # Date part
                                break
                            version_parts.append(parts[i])
                        df['backend_version'] = '_'.join(version_parts) if version_parts else 'unknown'
                    else:
                        df['backend_version'] = 'unknown'
                else:
                    df['backend_version'] = 'unknown'
            
            dfs.append(df)
            print(f"  Loaded: {csv_file.name}")
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
    
    if not dfs:
        return None, metadata
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (same backend data in cuda_*.csv files)
    # This happens when CUDA device not found and falls back to available GPU
    print(f"\nTotal rows before dedup: {len(combined)}")
    
    # Check if we have duplicate data (cuda files with level_zero backend)
    if 'backend' in combined.columns:
        backends = combined['backend'].unique()
        versions = combined['backend_version'].unique() if 'backend_version' in combined.columns else ['unknown']
        print(f"Backends found: {', '.join(backends)}")
        print(f"Backend versions found: {', '.join(versions)}")
        
        # If we only have one backend, deduplicate by content
        if len(backends) == 1:
            print("Only one backend found - removing duplicate measurements...")
            # Keep unique combinations of (bytes, op, mode, pinned, backend, backend_version)
            dedup_cols = ['bytes', 'op', 'mode', 'pinned', 'backend', 'bench']
            if 'backend_version' in combined.columns:
                dedup_cols.append('backend_version')
            combined = combined.drop_duplicates(subset=dedup_cols, keep='first')
    
    print(f"Total rows after dedup: {len(combined)}")
    return combined, metadata

def plot_bandwidth_comparison(df, output_dir, metadata=None):
    """Plot bandwidth comparison: Level Zero vs CUDA for all modes."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    
    if memcpy_df.empty:
        print("No memcpy data found")
        return
    
    # Convert bytes to KiB for better readability
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    backends = memcpy_df['backend'].unique()
    backend_versions = memcpy_df['backend_version'].unique() if 'backend_version' in memcpy_df.columns else ['unknown']
    operations = ['D2D', 'H2D', 'D2H']
    
    # Add version info to title if available
    version_str = ""
    if metadata and 'Backend Tag' in metadata:
        version_str = f" ({metadata['Backend Tag']})"
    elif len(backend_versions) == 1 and backend_versions[0] != 'unknown':
        version_str = f" ({backend_versions[0]})"
    
    for op in operations:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{op} Bandwidth Comparison{version_str}', fontsize=16, fontweight='bold')
        
        modes = [
            ('sync', 'yes', 'Sync + Pinned'),
            ('sync', 'no', 'Sync + Pageable'),
            ('batch', 'yes', 'Batch + Pinned'),
            ('batch', 'no', 'Batch + Pageable')
        ]
        
        for idx, (mode, pinned, title) in enumerate(modes):
            ax = axes[idx // 2, idx % 2]
            
            for backend in backends:
                data = memcpy_df[
                    (memcpy_df['op'] == op) &
                    (memcpy_df['backend'] == backend) &
                    (memcpy_df['mode'] == mode) &
                    (memcpy_df['pinned'] == pinned)
                ]
                
                if not data.empty:
                    # Group by size and take median to handle multiple runs
                    grouped = data.groupby('size_kib')['gib_per_s'].median().reset_index()
                    ax.plot(grouped['size_kib'], grouped['gib_per_s'], 
                           marker='o', linewidth=2, markersize=6,
                           label=f'{backend.upper()}')
            
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Transfer Size (KiB)')
            ax.set_ylabel('Bandwidth (GiB/s)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        output_file = output_dir / f'bandwidth_{op.lower()}_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_mode_comparison(df, output_dir):
    """Plot mode comparison: Sync vs Batch, Pinned vs Pageable."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    backends = memcpy_df['backend'].unique()
    
    for backend in backends:
        backend_data = memcpy_df[memcpy_df['backend'] == backend]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{backend.upper()} Backend: Mode Comparison', fontsize=16, fontweight='bold')
        
        operations = ['D2D', 'H2D', 'D2H']
        
        for idx, op in enumerate(operations):
            ax = axes[idx]
            op_data = backend_data[backend_data['op'] == op]
            
            # Plot all 4 combinations
            for mode in ['sync', 'batch']:
                for pinned in ['yes', 'no']:
                    data = op_data[
                        (op_data['mode'] == mode) &
                        (op_data['pinned'] == pinned)
                    ]
                    
                    if not data.empty:
                        grouped = data.groupby('size_kib')['gib_per_s'].median().reset_index()
                        label = f"{mode.capitalize()} + {'Pinned' if pinned == 'yes' else 'Pageable'}"
                        linestyle = '-' if mode == 'batch' else '--'
                        linewidth = 2.5 if pinned == 'yes' else 1.5
                        
                        ax.plot(grouped['size_kib'], grouped['gib_per_s'],
                               marker='o', linestyle=linestyle, linewidth=linewidth,
                               markersize=5, label=label, alpha=0.8)
            
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Transfer Size (KiB)')
            ax.set_ylabel('Bandwidth (GiB/s)')
            ax.set_title(f'{op} Performance')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        output_file = output_dir / f'mode_comparison_{backend}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_overhead_analysis(df, output_dir):
    """Plot small transfer overhead analysis."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    memcpy_df['size_kib'] = memcpy_df['bytes'] / 1024
    
    # Focus on small transfers (< 1 MiB)
    small_df = memcpy_df[memcpy_df['size_kib'] <= 1024].copy()
    
    backends = small_df['backend'].unique()
    
    fig, axes = plt.subplots(len(backends), 3, figsize=(18, 6 * len(backends)))
    if len(backends) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Small Transfer Overhead Analysis (≤1 MiB)', fontsize=16, fontweight='bold')
    
    for backend_idx, backend in enumerate(backends):
        backend_data = small_df[small_df['backend'] == backend]
        
        for op_idx, op in enumerate(['D2D', 'H2D', 'D2H']):
            ax = axes[backend_idx, op_idx]
            op_data = backend_data[backend_data['op'] == op]
            
            # Compare sync vs batch (pinned only for clarity)
            for mode in ['sync', 'batch']:
                data = op_data[
                    (op_data['mode'] == mode) &
                    (op_data['pinned'] == 'yes')
                ]
                
                if not data.empty:
                    grouped = data.groupby('size_kib')['avg_us'].median().reset_index()
                    ax.plot(grouped['size_kib'], grouped['avg_us'],
                           marker='o', linewidth=2, markersize=6,
                           label=f'{mode.capitalize()}')
            
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Transfer Size (KiB)')
            ax.set_ylabel('Latency (μs)')
            ax.set_title(f'{backend.upper()} - {op}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    output_file = output_dir / 'overhead_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_event_overhead(df, output_dir):
    """Plot event submission overhead comparison."""
    event_df = df[df['bench'] == 'event_overhead'].copy()
    
    if event_df.empty:
        print("No event overhead data found")
        return
    
    backends = event_df['backend'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(event_df['kind'].unique()))
    width = 0.35
    
    for idx, backend in enumerate(backends):
        backend_data = event_df[event_df['backend'] == backend]
        grouped = backend_data.groupby('kind')['avg_us'].median()
        
        ax.bar(x_pos + idx * width, grouped.values, width,
               label=backend.upper(), alpha=0.8)
    
    ax.set_xlabel('Operation Type')
    ax.set_ylabel('Average Latency (μs)')
    ax.set_title('Event Submission Overhead: Level Zero vs CUDA', fontweight='bold')
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(grouped.index, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'event_overhead.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_peak_bandwidth_summary(df, output_dir):
    """Plot peak bandwidth summary bar chart."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    
    # Get peak bandwidth for each combination
    summary = []
    
    for backend in memcpy_df['backend'].unique():
        for op in ['D2D', 'H2D', 'D2H']:
            for mode in ['sync', 'batch']:
                for pinned in ['yes', 'no']:
                    data = memcpy_df[
                        (memcpy_df['backend'] == backend) &
                        (memcpy_df['op'] == op) &
                        (memcpy_df['mode'] == mode) &
                        (memcpy_df['pinned'] == pinned)
                    ]
                    
                    if not data.empty:
                        peak = data['gib_per_s'].max()
                        summary.append({
                            'backend': backend,
                            'op': op,
                            'mode': mode,
                            'pinned': pinned,
                            'peak_gib_s': peak
                        })
    
    summary_df = pd.DataFrame(summary)
    
    # Plot for each operation type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Peak Bandwidth Summary', fontsize=16, fontweight='bold')
    
    for idx, op in enumerate(['D2D', 'H2D', 'D2H']):
        ax = axes[idx]
        op_data = summary_df[summary_df['op'] == op]
        
        # Create grouped bar chart
        backends = op_data['backend'].unique()
        x = np.arange(4)  # 4 mode combinations
        width = 0.35
        
        labels = ['Sync+Pin', 'Sync+Page', 'Batch+Pin', 'Batch+Page']
        
        for b_idx, backend in enumerate(backends):
            values = []
            for mode, pinned in [('sync', 'yes'), ('sync', 'no'), ('batch', 'yes'), ('batch', 'no')]:
                val = op_data[
                    (op_data['backend'] == backend) &
                    (op_data['mode'] == mode) &
                    (op_data['pinned'] == pinned)
                ]['peak_gib_s'].values
                values.append(val[0] if len(val) > 0 else 0)
            
            ax.bar(x + b_idx * width, values, width, label=backend.upper(), alpha=0.8)
        
        ax.set_xlabel('Mode')
        ax.set_ylabel('Peak Bandwidth (GiB/s)')
        ax.set_title(f'{op} Peak Performance')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'peak_bandwidth_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""
    memcpy_df = df[df['bench'] == 'memcpy_linear'].copy()
    
    summary = []
    
    for backend in memcpy_df['backend'].unique():
        for op in ['D2D', 'H2D', 'D2H']:
            for mode in ['sync', 'batch']:
                for pinned in ['yes', 'no']:
                    data = memcpy_df[
                        (memcpy_df['backend'] == backend) &
                        (memcpy_df['op'] == op) &
                        (memcpy_df['mode'] == mode) &
                        (memcpy_df['pinned'] == pinned)
                    ]
                    
                    if not data.empty:
                        summary.append({
                            'Backend': backend.upper(),
                            'Operation': op,
                            'Mode': mode.capitalize(),
                            'Pinned': pinned.capitalize(),
                            'Peak (GiB/s)': f"{data['gib_per_s'].max():.2f}",
                            'Median (GiB/s)': f"{data['gib_per_s'].median():.2f}",
                            'Min Latency (μs)': f"{data['avg_us'].min():.2f}",
                        })
    
    summary_df = pd.DataFrame(summary)
    
    # Save as CSV
    output_file = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    # Save as Markdown
    output_file = output_dir / 'summary_statistics.md'
    try:
        with open(output_file, 'w') as f:
            f.write("# Benchmark Summary Statistics\n\n")
            f.write(summary_df.to_markdown(index=False))
        print(f"Saved: {output_file}")
    except ImportError:
        print(f"Warning: 'tabulate' not installed, skipping Markdown table")
        print(f"Install with: pip install tabulate")
        # Create simple markdown table manually
        with open(output_file, 'w') as f:
            f.write("# Benchmark Summary Statistics\n\n")
            f.write("| " + " | ".join(summary_df.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(summary_df.columns)) + " |\n")
            for _, row in summary_df.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
        print(f"Saved: {output_file} (simple format)")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Analyze Celerity backend benchmark results')
    parser.add_argument('results_dir', nargs='?', default='results',
                       help='Directory containing benchmark CSV files (default: results)')
    parser.add_argument('--output', '-o', default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple backend versions (expects multiple result dirs)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading CSV files from: {args.results_dir}")
    df, metadata = load_all_csvs(args.results_dir)
    
    if df is None or df.empty:
        print("No data loaded. Exiting.")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Check if we have multiple backend versions
    if 'backend_version' in df.columns:
        versions = df['backend_version'].unique()
        if len(versions) > 1:
            print(f"\n⚠️  Multiple backend versions detected: {', '.join(versions)}")
            print("Results will be grouped by version in plots.")
            print("Consider analyzing each version separately for clearer comparison:")
            for v in versions:
                print(f"  python3 scripts/analyze_results.py results/results_{v}_* --output plots_{v}")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_bandwidth_comparison(df, output_dir, metadata)
    plot_mode_comparison(df, output_dir)
    plot_overhead_analysis(df, output_dir)
    plot_event_overhead(df, output_dir)
    plot_peak_bandwidth_summary(df, output_dir)
    
    # Generate summary table
    print("\n=== Generating Summary ===")
    summary_df = generate_summary_table(df, output_dir)
    print("\n" + summary_df.to_string(index=False))
    
    # Save version info to summary
    if metadata:
        version_file = output_dir / 'version_info.txt'
        with open(version_file, 'w') as f:
            f.write("=== Backend Version Info ===\n\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved: {version_file}")
    
    print(f"\n✅ Analysis complete! Check {output_dir}/ for results.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
