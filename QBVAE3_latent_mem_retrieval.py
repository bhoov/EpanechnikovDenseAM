#%%
"""
`python QBVAE3_latent_mem_retrieval.py mnist10`

Open 

Flow: 
1. Train the BVAE `QBVAE1_training.py`
2. Check the latent space with `QBVAE2_semantic_eval.py`
3. Test LSR energy (this file)
"""
#%%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
import jax.numpy as jnp
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import jax_utils as ju
from QBVAE_utils import data_transform, load_bvae, load_data, batch_encode_data, MEMS, load_taesd_vae, choose_beta, compute_minima_information, bwr_imshow, get_novel_and_old_minima

import functools as ft
import matplotlib.pyplot as plt
from einops import rearrange
from jax_utils import get_dist_matrix
import jaxlib

from LMIN0_utils import get_all_combinations

from jaxtyping import Float, Array
from typing import *
from tqdm.auto import tqdm, trange
from dataclasses import dataclass
import tyro
from pathlib import Path
import matplotlib.colors as mcolors

print("Is CUDA available?", torch.cuda.is_available())

@dataclass
class Config:
    outpath: Union[Path, str] # Path to save figures
    model_path: str = "expresults/QBVAE--beta-vae-mnist10.pt" # Path to saved VAE model
    nmems:int = 24 # Number of memories to store
    seed: int = 11 # Random seed for picking memories # 6 is good at 36 mems, 3 dim space
    target_K: int = 4 # Target average number of neighbors (including self) for each memory
    show_delta_plots: bool = False # Whether to show differences in basins between emergent LSR and LSE
    grad_tol: float = 1e-4 # Tolerance for LSR gradient norm of candidate minimum to be considered a memory
    mnist_bg_color: str = "#002C4D" # Background color for MNIST images (value 0)
    mnist_fg_color: str = "#FFFFFF" # Foreground color for MNIST images (value 1)

    def __post_init__(self):
        if isinstance(self.outpath, str): self.outpath = Path(self.outpath)
        self.outpath.mkdir(parents=True, exist_ok=True)

    @property
    def dataset(self):
        return (
            "mnist" if "mnist" in self.model_path else
            "tinyimagenet" if "tinyimagenet" in self.model_path else
            "unknown"
        )

def create_custom_colormap(bg_color, fg_color):
    """Create a custom colormap from background color (0) to foreground color (1)"""
    colors = [bg_color, fg_color]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap

def custom_imshow(ax, img, bg_color="#002C4D", fg_color="#FFFFFF", **kwargs):
    """
    Display images with custom colors for MNIST data.
    bg_color: hex color for value 0 (background)
    fg_color: hex color for value 1 (foreground)
    """
    # Create custom colormap
    cmap = create_custom_colormap(bg_color, fg_color)
    
    # Set default values if not provided in kwargs
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap
    
    # For MNIST data, we typically want vmin=0, vmax=1
    if 'vmin' not in kwargs:
        kwargs['vmin'] = 0
    if 'vmax' not in kwargs:
        kwargs['vmax'] = 1
    
    # Display the image
    im = ax.imshow(img, **kwargs)
    return im

default_configs = {
    "mnist10": (
        "MNIST with VAE encoding in 10 dimensions",
        Config(
            outpath="figures/QBVAE--mnist10-mem-retrieval", 
            model_path="expresults/QBVAE--beta-vae-mnist10.pt", 
            nmems=24, 
            seed=8, 
            target_K=5, 
            show_delta_plots=False, 
            grad_tol=1e-4),
    ),
    "tinyimagenet256": (
        "Tiny ImageNet with VAE encoding rasterized to 256 dimensions",
        Config(
            outpath="figures/QBVAE--tinyimagenet256-mem-retrieval",
            model_path="tinyimagenet",
            nmems=40,
            seed=22,
            target_K=3,
            grad_tol=1e-3,
        )
    )

}

if ju.is_interactive():
    config = default_configs["mnist10"][1]
    # config = default_configs["tinyimagenet256"][1]
else:
    config = tyro.extras.overridable_config_cli(default_configs)

key, rng = jr.split(jr.PRNGKey(config.seed))

#%%
# Load saved torch model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = (
    load_bvae(config.model_path) if config.dataset == "mnist" else
    load_taesd_vae() if config.dataset == "tinyimagenet" else
    None
)

Xtrain, Xtest = load_data(config.dataset)

model = model.to(DEVICE)
Xtest = Xtest.to(DEVICE) if Xtest is not None else None

model.eval()
latents, mus, logvars = batch_encode_data(model, Xtrain, do_transform=config.dataset == "mnist")


def decode_latents(model, points):
    """Decode latents to images, accounting for device and instance of points"""
    if isinstance(points, jaxlib.xla_extension.ArrayImpl):
        points = torch.tensor(np.array(points)).to(DEVICE)
    elif isinstance(points, np.ndarray):
        points = torch.tensor(points).to(DEVICE)
    with torch.no_grad():
        decoded = model.decode(points)
        decoded = data_transform.decode(decoded) if config.dataset == "mnist" else decoded
    return decoded

decoded = decode_latents(model, latents[:10])
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
if len(decoded.shape) == 4:
    ax.imshow(rearrange(decoded, "n c h w -> (n h) w c").numpify())
else:
    # mnist
    ax.imshow(rearrange(decoded, "n h w -> (n h) w").numpify())
    
key, rng = jr.split(jr.PRNGKey(config.seed))

# Select memories
latent_idxs = jr.choice(key, jnp.arange(latents.shape[0]), shape=(config.nmems,), replace=False)
Xis = mus[latent_idxs]

# Compute ideal beta for this set of memories
target_avg_K = config.target_K 
chosen_beta, aux = choose_beta(Xis, target_K=target_avg_K, tol=0.)
print(f"Chosen Beta based on target K={target_avg_K}: {chosen_beta}. Resulting in num neighbors per point: {aux['num_neighbors_per_point']}")


lmin_unique_array, lmin_unique_merge_idxs = compute_minima_information(Xis, chosen_beta, grad_tol=1e-4)
mem_info = get_novel_and_old_minima(lmin_unique_array, lmin_unique_merge_idxs)
novel_minima = mem_info["novel_minima"]
novel_minima_memory_idxs = mem_info["novel_minima_memory_idxs"]

print(novel_minima.shape)

lsr_mem = MEMS["lsr"]

queries = mem_info["novel_minima"]
out_points, aux = lsr_mem.vrecall(queries, Xis, depth=1000, beta_schedule=lambda i: chosen_beta, alpha_schedule=0.001, noise_schedule=0.0, grad_tol=0.0)
assert jnp.allclose(out_points, queries, atol=1e-6)

#%% Check the lse retrievals of LSR's emergent memories
lse_mem = MEMS["lse"]
queries = mem_info["old_minima"]
lse_out_points, aux = lse_mem.vrecall(queries, Xis, depth=20_000, beta_schedule=lambda i: chosen_beta, alpha_schedule=0.002, noise_schedule=0.0, grad_tol=0.0)

decoded_lse_out = decode_latents(model, lse_out_points)
fig,ax = plt.subplots(1,1, figsize=(10, 10))
decoded_lse_imgs = decoded_lse_out.numpify()[-20:]
decoded_queries = decode_latents(model, queries).numpify()[-20:]

decoded_queries.shape
decoded_lse_imgs.shape
ax.imshow(rearrange([decoded_queries, decoded_lse_imgs], 's n ... h w -> (n h) (s w) ...'))

ax.set_title("LSE samples\nInit @LSR mems")

#%% Plot eval figure
if config.show_delta_plots:
    key, rng = jr.split(jr.PRNGKey(99))
    Nshow = 5
    idxs = jr.choice(key, jnp.arange(novel_minima.shape[0]), shape=(Nshow,), replace=False)
    true_new_minima_show = novel_minima[idxs]
    novel_lmin_memory_idxs_show = [novel_minima_memory_idxs[i] for i in idxs]
    decoded_lse_out = decode_latents(model, lse_out_points)[idxs.tolist()]

    new_minima_dist_to_Xi = get_dist_matrix(true_new_minima_show, Xis)
    k = 2
    nearest_neighbor_idxs = jnp.argsort(new_minima_dist_to_Xi, axis=-1)[:, :k] # Just show nearest 1 neighbor

    # Index Xis by nearest neighbor idxs
    Xi_show = []
    for nni in nearest_neighbor_idxs:
        Xi_show.append(Xis[nni.tolist()])
    Xi_show = np.stack(Xi_show)

    model = model.eval()
    decoded_new_minima = decode_latents(model, true_new_minima_show).numpify()
    decoded_nearest_neighbors = np.stack([decode_latents(model, xs).numpify() for xs in Xi_show])

    decoded_minima_show = rearrange(decoded_new_minima, 'n h w -> (n h) w')
    decoded_neighbors_show = rearrange(decoded_nearest_neighbors, 'n k h w -> (n h) (k w)')
    decoded_lse_out_show = rearrange(decoded_lse_out.numpify(), 'n h w -> (n h) w')

    # Determine the symmetric color range
    max_abs_val = np.max([
        np.abs(decoded_minima_show).max(),
        np.abs(decoded_neighbors_show).max(),
        np.abs(decoded_lse_out_show).max()
    ])
    vmin = -max_abs_val
    vmax = max_abs_val

    # Compute delta between decoded_new_minima
    delta_nns = []
    delta_lses = []
    for inn in range(decoded_nearest_neighbors.shape[0]):
        dnm = decoded_new_minima[inn]
        dnn = decoded_nearest_neighbors[inn]
        dlse = decoded_lse_out[inn].numpify()

        delta_nn = dnm - dnn
        delta_lse = dlse - dnm
        delta_nns.append(delta_nn)
        delta_lses.append(delta_lse)

    delta_nns_show = rearrange(np.stack(delta_nns), "n k h w -> (n h) (k w)")
    delta_lses_show = rearrange(np.stack(delta_lses), "n h w -> (n h) w")

    decoded_Xis = decode_latents(model, Xis).numpify()
    decoded_Xis_show = rearrange(decoded_Xis, 'n h w -> h (n w)')
    import matplotlib.gridspec as gridspec

    # Define relative widths: plot_width, main_gap_width, tight_gap_width
    mgw, tgw = 2, 0.1 
    pw1, pw2 = 2, 4
    width_ratios = [pw1, mgw, 
                    pw2, tgw, pw2, mgw, 
                    pw1, tgw, pw1]

    # Adjust figure size to accommodate second row
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 9, height_ratios=[3, 1], width_ratios=width_ratios)

    # Create subplots using GridSpec for top row
    axs = [None] * 6  # Increased to 6 to include the new subplot
    axs[0] = fig.add_subplot(gs[0, 0])
    axs[1] = fig.add_subplot(gs[0, 2])
    axs[2] = fig.add_subplot(gs[0, 4])
    axs[3] = fig.add_subplot(gs[0, 6])
    axs[4] = fig.add_subplot(gs[0, 8])

    # Create a new subplot that spans the entire second row
    axs[5] = fig.add_subplot(gs[1, :])

    # Plotting code for top row
    axs[0].imshow(decoded_minima_show, cmap='bwr', vmin=vmin, vmax=vmax)
    axs[0].set_xticks([])

    # Create y-tick labels from novel_lmin_memory_idxs_show
    pattern_height = decoded_new_minima.shape[1]  # Height of each pattern (likely 28)
    tick_positions = [pattern_height//2 + i * pattern_height for i in range(Nshow)]

    # Create labels in format (A,D) based on novel_lmin_memory_idxs_show
    y_tick_labels = []
    [y_tick_labels.append(f"({','.join([f'{idx}' for idx in memory_idxs])})") for memory_idxs in novel_lmin_memory_idxs_show]

    axs[0].set_yticks(tick_positions)
    axs[0].set_yticklabels(y_tick_labels, fontsize=12)
    axs[0].set_title("Emergent LSR memories")

    axs[1].imshow(decoded_neighbors_show, cmap='bwr', vmin=vmin, vmax=vmax)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title(f"Top {k} nearest Xi")

    axs[2].imshow(delta_nns_show, cmap='bwr', vmin=vmin, vmax=vmax)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_title("Delta")

    axs[3].imshow(decoded_lse_out_show, cmap='bwr', vmin=vmin, vmax=vmax)
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    axs[3].set_title("LSE samples\nInit @LSR mems")

    axs[4].imshow(delta_lses_show, cmap='bwr', vmin=vmin, vmax=vmax)
    axs[4].set_xticks([])
    axs[4].set_yticks([])
    axs[4].set_title("Delta")

    # Add the bottom row with all stored patterns
    axs[5].imshow(decoded_Xis_show, cmap='bwr', vmin=vmin, vmax=vmax)
    axs[5].set_yticks([])
    axs[5].set_title(f"All {config.nmems} stored patterns")

    # Add alphabetical labels centered on each pattern (assuming 28 pixel width)
    pattern_width = Xtrain.shape[-1]
    tick_positions = [pattern_width//2 + i * pattern_width for i in range(config.nmems)]
    tick_labels = [f"{i+1}" for i in range(config.nmems)]
    axs[5].set_xticks(tick_positions)
    axs[5].set_xticklabels(tick_labels, fontsize=12)

    plt.tight_layout()

# %% New figures for LSR vs LSE, showing all minima
# Select memories
key, rng = jr.split(jr.PRNGKey(config.seed)) # Start seed over for reproducibility
latent_idxs = jr.choice(key, jnp.arange(latents.shape[0]), shape=(config.nmems,), replace=False)
Xis = mus[latent_idxs]

target_avg_K = config.target_K
chosen_beta, aux = choose_beta(Xis, target_K=target_avg_K, tol=0.)
lmin_unique_array, lmin_unique_merge_idxs = compute_minima_information(Xis, chosen_beta, grad_tol=config.grad_tol)
lsr_mem_info = get_novel_and_old_minima(lmin_unique_array, lmin_unique_merge_idxs)
novel_lsr_mems = lsr_mem_info["novel_minima"]
old_lsr_mems = lsr_mem_info["old_minima"]


# Compute all LSE minima by initializing at each of the memories
queries = Xis
all_lse_mems, aux = lse_mem.vrecall(queries, Xis, depth=8000, beta_schedule=lambda i: chosen_beta, alpha_schedule=0.02, noise_schedule=0.0, grad_tol=0.0)
unique_lse_mems, unique_lse_mems_idxs = np.unique(all_lse_mems.round(decimals=1), return_index=True, axis=0)
unique_lse_mems_sorted = unique_lse_mems[np.argsort(unique_lse_mems_idxs)]
unique_lse_mems_decoded = decode_latents(model, unique_lse_mems_sorted).numpify()


#%%

def format_imgs(imgs: Float[Array, "N h w"], nw: int, max_height=int(2**16-1), max_width=int(2**16-1)):
    """Auto rearrange imgs into a grid of width nw, padding with white images if necessary"""
    n = imgs.shape[0]
    h, w = imgs.shape[-2:]
    # n, h, w = imgs.shape  # Get number of images, height, and width
    nh = int(np.ceil(n / nw)) # Number of rows needed
    n_total = nh * nw # Total number of images in the grid
    # Pad images?
    n_pad = n_total - n
    if n_pad > 0:
        img_shape = (h, w) if config.dataset == "mnist" else imgs.shape[-3:]
        pad_imgs = np.zeros((n_pad, *img_shape), dtype=imgs.dtype) if config.dataset == "mnist" else np.ones((n_pad, *img_shape), dtype=imgs.dtype)
        imgs = np.concatenate([imgs, pad_imgs], axis=0)
    
    grid = rearrange(imgs, '(nh nw) ... h w -> (nh h) (nw w) ...', nh=nh, nw=nw)
    # grid = grid[:max_height, :max_width]
    return grid

# Pad images to the same width
nimgs_wide = 8
decoded_Xis = decode_latents(model, Xis).numpify()
decoded_Xis_show = format_imgs(decoded_Xis, nimgs_wide)

# Sort the old LSR memories to be the same order as the stored memories
oldidx_arr = np.stack(lsr_mem_info['old_minima_memory_idxs']).squeeze()
old_lsr_mems_sorted = old_lsr_mems[np.argsort(oldidx_arr)]

decoded_novel_lsr_mems = decode_latents(model, novel_lsr_mems).numpify()
decoded_old_lsr_mems = decode_latents(model, old_lsr_mems_sorted).numpify()
decoded_unique_lse_mems = decode_latents(model, unique_lse_mems).numpify()

decoded_unique_lse_mems_show = format_imgs(decoded_unique_lse_mems, nw=8)
decoded_novel_lsr_mems_show = format_imgs(decoded_novel_lsr_mems, nw=12)
decoded_old_lsr_mems_show = format_imgs(decoded_old_lsr_mems, nw=8)

# Calculate the height for each figure based on content
fig_width = 7  # Same width for all figures
img_height_to_width_ratio = lambda img: img.shape[0] / nimgs_wide

# Figure 1 - All stored memories
height = fig_width * img_height_to_width_ratio(decoded_Xis_show)
fig, ax = plt.subplots(1,1, figsize=(fig_width, height), dpi=150)
custom_imshow(ax, decoded_Xis_show, bg_color=config.mnist_bg_color, fg_color=config.mnist_fg_color)
ax.set_title("Stored patterns")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout()
try:
    fig.savefig(config.outpath / "stored_patterns.png", dpi=150, bbox_inches="tight")
except Exception as e:
    print(f"Error saving stored_patterns.png: {e}")

# Figure 2 - Novel LSR memories
height = fig_width * img_height_to_width_ratio(decoded_novel_lsr_mems_show)
fig, ax = plt.subplots(1,1, figsize=(fig_width, height), dpi=150)
custom_imshow(ax, decoded_novel_lsr_mems_show, bg_color=config.mnist_bg_color, fg_color=config.mnist_fg_color)
ax.set_title("Novel LSR mems")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout()
try:
    fig.savefig(config.outpath / "novel_lsr_mems.png", dpi=150, bbox_inches="tight")
except Exception as e:
    print(f"Error saving novel_lsr_mems.png: {e}")

# Figure 3 - Old LSR memories
height = fig_width * img_height_to_width_ratio(decoded_old_lsr_mems_show)
fig, ax = plt.subplots(1,1, figsize=(fig_width, height), dpi=150)
custom_imshow(ax, decoded_old_lsr_mems_show, bg_color=config.mnist_bg_color, fg_color=config.mnist_fg_color)
ax.set_title("Preserved patterns as memories")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout()
try:
    fig.savefig(config.outpath / "old_lsr_mems.png", dpi=150, bbox_inches="tight")
except Exception as e:
    print(f"Error saving old_lsr_mems.png: {e}")

# Figure 4 - All LSE memories
height = fig_width * img_height_to_width_ratio(decoded_unique_lse_mems_show)
fig, ax = plt.subplots(1,1, figsize=(fig_width, height), dpi=150)
custom_imshow(ax, decoded_unique_lse_mems_show, bg_color=config.mnist_bg_color, fg_color=config.mnist_fg_color)
ax.set_title("All LSE mems")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout()
try:
    fig.savefig(config.outpath / "lse_mems.png", dpi=150, bbox_inches="tight")
except Exception as e:
    print(f"Error saving lse_mems.png: {e}")

print("Figures saved to: ", config.outpath)