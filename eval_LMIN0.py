#%%
import polars as pl
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utils import is_interactive
import tyro

@dataclass
class Args:
    dataf: str = "expresults/LMIN0.jsonl" # Path to the data file
    seed: int = 0 # Seed to use for the data

if is_interactive():
    args = Args(dataf="expresults/LMIN0.jsonl")
else:
    args = tyro.cli(Args)

dataf = args.dataf
df = pl.read_ndjson(dataf, infer_schema_length=1000)

#%% Choose single seed
seed = args.seed
sdf = df.filter(pl.col("seed") == seed)

## Choose values for D and M
unique_d = [8, 32]
unique_M = [5, 15, 25]

# Create subplot grid
fig, axes = plt.subplots(len(unique_M), len(unique_d), 
                        figsize=(4*len(unique_d), 3*len(unique_M)),
                        squeeze=False)
# Add suptitle
fig.suptitle("LMIN0", size=24, y=1.00)

# Create subplots
for i, M in enumerate(unique_M):
    for j, d in enumerate(unique_d):
        ax = axes[i,j]
        
        # Filter data for this M,d combination
        subset = sdf.filter((pl.col("M") == M) & (pl.col("d") == d))

        # Plot means without error bars
        ax.plot(subset["beta"], subset["num_old_mems"], 
                label="Stored patterns", marker='o')
        ax.plot(subset["beta"], subset["num_new_mems"], 
                label="New memories", marker='s')
            
        # Add horizontal line at M
        ax.axhline(y=M, color='k', linestyle=':', label='M', linewidth=2.)
        ax.annotate('M', 
            xy=(0.02, M),  # Position of the text (x=2% from left, y=M)
            xycoords=('axes fraction', 'data'),  # Mixed coordinates
            color='k',
            verticalalignment='bottom', fontweight='bold')

        # Plot volume coverage on secondary axis with shading
        ax2 = ax.twinx()
        light_green = '#228B2244'   # Lighter green for shading
        dark_green = '#228B22FF'    # Dark green for text
        ax2.fill_between(subset["beta"], subset["vol_coverage"], 
                        color=light_green, label='Vol. supported')
        ax2.set_ylim(0, 1.02)
        if j == len(unique_d) - 1:
            ax2.set_ylabel('Volume supported', fontsize=14, color=dark_green, fontweight='bold')
        else:
            ax2.set_yticks([])
        
        # Set tick colors for the secondary y-axis
        ax2.tick_params(axis='y', colors=dark_green)

        # Add labels
        if i == len(unique_M) - 1:
            ax.set_xlabel("β", fontsize=18, fontweight='bold')
        if j == 0:
            ax.set_ylabel("Num. memories", fontsize=16)
        # ax.set_title(f"M={M}, d={d}")
        if i == len(unique_M) - 1 and j == len(unique_d) - 1:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2)
        ax.set_yscale("log")

# Add row and column labels after the subplot loop
for ax, d in zip(axes[0], unique_d):
    ax.annotate(f'd={d}', xy=(0.5, 1.05), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='baseline', fontsize=20, weight='bold')

for ax, M in zip(axes[:,0], unique_M):
    ax.annotate(f'M={M}', xy=(-0.2, 0.5), xytext=(-5, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='center', rotation=90, fontsize=20, weight='bold')

fig.tight_layout()
fig.savefig("figures/LMIN0.png", dpi=200)
fig.show()
print("Figure saved to: figures/LMIN0.png")

#%% Aggregate over seed
col_grouping = ["M", "d", "beta_idx"]
df.group_by(col_grouping).agg(
    num_old_mems=pl.col("num_old_mems"),
    num_old_mems_mean=pl.col("num_old_mems").mean(),
    num_old_mems_std=pl.col("num_old_mems").std(),
    num_new_mems=pl.col("num_new_mems"),
    num_new_mems_mean=pl.col("num_new_mems").mean(),
    num_new_mems_std=pl.col("num_new_mems").std(),
)