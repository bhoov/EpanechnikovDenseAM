#%%
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import tyro
from dataclasses import dataclass
from utils import is_interactive

@dataclass
class Args:
    dataf: str = "expresults/LOGL0.jsonl" # Path to the data file

if is_interactive():
    args = Args()
else:
    args = tyro.cli(Args)

dataf = args.dataf
df = pl.read_ndjson(dataf, infer_schema_length=1000).unique(subset=['beta_idx', 'M', 'd', 'seed', 'true_sigma', 'energy_key', 'k'])

#%%
metric_labels = {
    "avg_logl": {
        "title": "Average Log-Likelihood",
        "ylabel": "Avg LogL",
        },
    "num_unique_retrievals": 
        {"title": "Num Unique Samples",
        "ylabel": "Num Unique Samples"},
    "og_mems_recoverable": 
        {"title": "Num OG Mems Recoverable",
        "ylabel": "Num OG Recovered"},
}

def compare_metrics(ax, df, metric_col, d, M, show_legend=False, markersize=8):
    """
    Plot comparison between EPA and LSE for a given metric.
    
    Args:
        df: Polars DataFrame with the data
        metric_col: Column name of the metric to plot
        d: d value to filter for
        M: M value to filter for
        show_legend: Whether to show the legend (only for first subplot)

    """
    # Use mean_ and std_ prefix for columns
    mean_col = f"mean_{metric_col}"
    std_col = f"std_{metric_col}"
    
    # Filter for specific d and M
    dm_df = df.filter((pl.col("d") == d) & (pl.col("M") == M))

    # Show only if there are at least 5 beta values (otherwise still in progress)
    all_betas = sorted(set(dm_df["beta"].to_list()))
    if len(all_betas) < 5:
        ax.text(0.5, 0.5, 'Insufficient data\n(< 5 beta values)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Filter for LSE and EPA separately
    lse_data = dm_df.filter(pl.col("energy_key") == "lse").sort("beta")
    epa_data = dm_df.filter(pl.col("energy_key") == "epa").sort("beta")
    
    # Plot points with error bars, without connecting lines
    if metric_col in ['num_unique_retrievals', 'og_mems_recoverable']:
        # Option 1: Clipped error bars
        yerr_lower = np.minimum(lse_data[mean_col], lse_data[std_col])
        yerr_upper = lse_data[std_col]
        ax.errorbar(lse_data["beta"], lse_data[mean_col], 
                   yerr=[yerr_lower, yerr_upper],
                   fmt='o', label="LSE", linestyle='-', linewidth=0.8, alpha=0.8, 
                   capsize=3, markersize=markersize, zorder=2)
        
        # Option 2: Clipped error bars
        yerr_lower = np.minimum(epa_data[mean_col], epa_data[std_col])
        yerr_upper = epa_data[std_col]
        ax.errorbar(epa_data["beta"], epa_data[mean_col], 
                   yerr=[yerr_lower, yerr_upper],
                   fmt='s', label="LSR", linestyle='-', linewidth=0.8, alpha=1., 
                   capsize=3, markersize=markersize, zorder=2)
    else:
        ax.errorbar(lse_data["beta"], lse_data[mean_col], yerr=lse_data[std_col],
                 label="LSE", marker='o', linestyle='-', linewidth=0.8, alpha=0.8, 
                 capsize=3, markersize=markersize, zorder=2)
        ax.errorbar(epa_data["beta"], epa_data[mean_col], yerr=epa_data[std_col],
                 label="LSR", marker='s', linestyle='-', linewidth=0.8, alpha=1., 
                 capsize=3, markersize=markersize, zorder=2)

    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Get union of all beta values and sort them
    all_betas = sorted(set(lse_data["beta"].to_list() + epa_data["beta"].to_list()))
    
    # Interpolate values for both methods on the common beta grid
    lse_interp = np.interp(all_betas, lse_data["beta"], lse_data[mean_col])
    epa_interp = np.interp(all_betas, epa_data["beta"], epa_data[mean_col])
    
    # Create mask where EPA > LSE using interpolated values
    epa_better = epa_interp > lse_interp

    # Shade regions and annotate
    for i in range(len(all_betas)-1):
        if epa_better[i] and epa_better[i+1]:
            ax.axvspan(all_betas[i], all_betas[i+1], color='orange', alpha=0.13)

            y_pos = 0.01
            # Annotate start if it's the first point or previous point wasn't EPA-better
            if i == 0 or not epa_better[i-1]:
                ax.annotate(f'β={all_betas[i]:.2f}', 
                            xy=(all_betas[i], y_pos),
                            xytext=(all_betas[i], y_pos), 
                            xycoords=('data', 'axes fraction'),
                            ha='center', va='bottom', rotation=90,
                            path_effects=[
                                pe.withStroke(linewidth=4, foreground='white')
                            ],
                            zorder=10)
            # Annotate end if it's the last point or next point isn't EPA-better
            if i == len(all_betas)-2 or not epa_better[i+2]:
                ax.annotate(f'β={all_betas[i+1]:.2f}', 
                            xy=(all_betas[i+1], y_pos),
                            xytext=(all_betas[i+1], y_pos), 
                            xycoords=('data', 'axes fraction'),
                            ha='center', va='bottom', rotation=90,
                            path_effects=[
                                pe.withStroke(linewidth=4, foreground='white')
                            ],
                            zorder=10)
    
    ax.set_xlabel("β", fontsize=22)
    # ax.set_title(f"{metric_labels[metric_col]['title']} Comparison for M={M}, d={d}", fontsize=14)
    
    if show_legend:
        ax.legend(fontsize=22, prop={'weight': 'bold'})  # Bold font in legend

    ax.grid(True)
    ax.margins(y=0.15)

def plot_all_metrics_by_d(df, d, base_title=""):
    """Create a figure for a specific d value, with subplots for each M value and metric."""
    # Get unique M values for this d
    unique_M = [25, 100]
    
    # Create subplot grid - transposed version
    fig, axes = plt.subplots(len(metric_labels), len(unique_M), 
                            figsize=(5*len(unique_M), 4*len(metric_labels)))
    
    # Add suptitle
    # base_title = "" if base_title == "" else (": " + base_title.lstrip())
    fig.suptitle(base_title + f" in $\\mathbf{{d={d}}}$ dims", fontsize=24, y=1.0)

    # Plot each metric for each M value - transposed iteration
    for i, (metric_col, metric_label) in enumerate(metric_labels.items()):
        for j, M in enumerate(unique_M):
            ax = axes[i, j]

            # Only show legend for top-left subplot (i=0, j=0)
            show_legend = (i == 0 and j == 0)
            compare_metrics(ax, df, metric_col, d, M, show_legend, markersize=6.5)
            
            # Only show x-label (β) on bottom row
            if i == len(metric_labels) - 1:
                ax.set_xlabel(r"$\boldsymbol{\beta}$", fontsize=24)
            else:
                ax.set_xlabel('')
            
            # Increase tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Remove title since we have row/column labels
            ax.set_title('')
    
    # Add column labels at the top for M values
    for j, M in enumerate(unique_M):
        axes[0, j].annotate(f'$\\mathbf{{M={M}}}$',
                          xy=(0.5, 1.05),
                          xycoords='axes fraction',
                          ha='center',
                          va='bottom',
                          fontsize=24)
    
    # Add row labels for metrics
    for i, (metric_col, metric_label) in enumerate(metric_labels.items()):
        # axes[i, 0].text(-0.275, 0.5, metric_label['title'], 
        #                transform=axes[i, 0].transAxes,
        #                fontsize=12,
        #                weight='bold',
        #                rotation=90,
        #                horizontalalignment='center',
        #                verticalalignment='center')
        pass
    
    return fig

# Main plotting loop
allow_seeds = [3, 4, 5, 6, 7]

# for k in tqdm([5], desc="k"):
for k in tqdm([5, 10], desc="k"):
    for sigma in tqdm([0.1], desc="sigma"):
        sdf = df.filter(
            (pl.col("seed").is_in(allow_seeds)) 
            & (pl.col("true_sigma") == sigma) 
            & (pl.col("k") == k)
        )

        sdf = sdf.group_by(['beta_idx',  'M', 'd', 'k', 'true_sigma', 'energy_key']).agg(
            pl.col('avg_logl').mean().alias('mean_avg_logl'),
            pl.col('num_unique_retrievals').mean().alias('mean_num_unique_retrievals'),
            pl.col('og_mems_recoverable').mean().alias('mean_og_mems_recoverable'),
            pl.col('avg_logl').std().alias('std_avg_logl'),
            pl.col('num_unique_retrievals').std().alias('std_num_unique_retrievals'),
            pl.col('og_mems_recoverable').std().alias('std_og_mems_recoverable'),
            pl.col('beta').mean().alias('beta'), # Betas match on beta_idx
        )

        base_title = f"Mixture of $\\mathbf{{k={k}}}$ Gaussians"
        unique_d = sorted(sdf["d"].unique().to_list())
        for d in tqdm(unique_d, desc="d"):
            fig = plot_all_metrics_by_d(sdf, d, base_title)
            plt.tight_layout()
            plt.show()

            figdir = Path("figures")
            figdir.mkdir(parents=True, exist_ok=True)
            fname = f"figures/LOGL0__sigma{sigma}_k{k}_multiseed_d{d}.png"
            fig.savefig(fname, bbox_inches='tight', dpi=200)
            plt.close()

# %%
