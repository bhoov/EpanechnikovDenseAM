"""
Recreate plots for Fig 1 in the paper
"""
#%%
import jax.numpy as jnp
import numpy as np
from memories import EpaMemory, LseMemory
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils import detect_peaks
import matplotlib.patheffects as pe

memLSR = EpaMemory(eps=1e-6, lmda=0.0)
memLSE = LseMemory()

#%% 1D case
mems = np.array([[-1.], [1.]])  # Shape: (2, 1)
xs = np.linspace(-2.1, 2.1, 200)[:, None]  # Shape: (200, 1)

# Calculate energies for both memories
beta = 0.9
Es_lsr = memLSR.venergy(xs, mems, beta=beta)
Es_lse = memLSE.venergy(xs, mems, beta=beta)

def plot_1D_energyfn(ax, xs, energies, mems, memory, beta, tol=0.1, add_legend=False, show_xlabels=False):
    """Plot energy landscape and detect/label local minima.
    
    Args:
        ax: matplotlib axis to plot on
        xs: input points (N, D)
        energies: energy values at xs (N,)
        mems: memory points (M, D)
        memory: memory object with energy method
        beta: temperature parameter
        tol: tolerance for considering a point "close" to a memory
        add_legend: whether to add the full legend to this plot
        show_xlabels: whether to show x-axis labels (for bottom row)
    """
    # Plot energy landscape
    ax.plot(xs.flatten(), energies, label=memory.__class__.__name__.replace('Memory', ''), alpha=0.6, linewidth=2)
    
    # Find local minima using scipy's find_peaks on negative energies
    min_indices, _ = find_peaks(-energies)
    
    # Plot and label each local minimum
    preserved_mem = None
    novel_mem = None
    for idx in min_indices:
        x_val = xs[idx, 0]
        e_val = energies[idx]
        
        # Check if this minimum is close to any memory
        is_memory = any(np.abs(x_val - mem[0]) < tol for mem in mems)
        
        outstroke_width = 10 
        if is_memory:
            preserved_mem = ax.plot(x_val, e_val, 'b', marker='*', linestyle="None", markersize=15, label='Preserved Memories' if add_legend else None, path_effects=[pe.withStroke(linewidth=outstroke_width, foreground='white')])[0]
        else:
            # novel_mem = ax.plot(x_val, e_val, '#F87400', marker='*', linestyle="None", markersize=15, label='Novel Memories' if add_legend else None)[0]
            novel_mem = ax.plot(x_val, e_val, 'r', marker='*', linestyle="None", markersize=15, label='Novel Memories' if add_legend else None, path_effects=[pe.withStroke(linewidth=outstroke_width, foreground='white')])[0]
    
    # Add whitespace to bottom of plot
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax)

    # Plot memory points on x-axis and vertical lines
    stored_pattern = None
    for mem in mems:
        stored_pattern = ax.plot(mem[0], 0, 'ko', transform=ax.get_xaxis_transform(), markersize=12, label='Stored Patterns' if add_legend else None)[0]
        ax.axvline(x=mem[0], color='black', linestyle=':', alpha=0.6)
    
    # Add legend only if requested
    if add_legend:
        handles = []
        labels = []
        if preserved_mem is not None:
            handles.append(preserved_mem)
            labels.append('Preserved Memories')
        if novel_mem is not None:
            handles.append(novel_mem)
            labels.append('Novel Memories')
        if stored_pattern is not None:
            handles.append(stored_pattern)
            labels.append('Stored Patterns')
        ax.legend(handles, labels, loc='upper right', fontsize=12)
    
    # Remove all ticks and labels except x=-1,1 on bottom row
    if show_xlabels:
        ax.set_xticks([-1, 1])
        ax.set_xticklabels(['-1', '1'])
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    return ax

betas_epa_lse = [
    {"epa": 0.4, "lse": 0.4},
    {"epa": 0.9, "lse": 1.00001},
    {"epa": 2.1, "lse": 4.11},
]
    
# Create a figure with subplots for each beta value, but transpose the layout
fig, axes = plt.subplots(2, len(betas_epa_lse), figsize=(4*len(betas_epa_lse), 5))

# Loop through beta values and create plots
for i, beta_pair in enumerate(betas_epa_lse):
    # Calculate energies for both memories with current beta values
    Es_lsr = memLSR.venergy(xs, mems, beta=beta_pair["epa"])
    Es_lse = memLSE.venergy(xs, mems, beta=beta_pair["lse"])
    
    # Plot both energy landscapes
    ax1, ax2 = axes[:, i]  # Now indexing columns instead of rows
    plot_1D_energyfn(ax1, xs, Es_lsr, mems, memLSR, beta_pair["epa"], 
                     add_legend=(i==1), 
                     show_xlabels=False)
    plot_1D_energyfn(ax2, xs, Es_lse, mems, memLSE, beta_pair["lse"],
                     show_xlabels=True)

plt.tight_layout()
plt.show()

fig.savefig("figures/fig1_1d.png", dpi=200)
#%% 2D case

negy = 2 * np.cos(np.pi/6)

mems = np.array([
    [-1., 0],
    [1., 0],
    [0, -negy],
])

def plot_2D_energyfn(ax, beta, mems, memLSR, tol=0.1, show_legend=True, n_contours=15):
    """Plot 2D energy landscape with contours and minima.
    
    Args:
        ax: matplotlib axis to plot on
        beta: temperature parameter
        mems: memory points (M, 2)
        memLSR: memory object with energy method
        tol: tolerance for considering a point "close" to a memory
    """
    # Create grid of points for energy landscape
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3.5, 2, 200)
    X, Y = np.meshgrid(x, y)
    points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    # Calculate energies for all points
    energies = memLSR.venergy(points, mems, beta=beta)
    Z = energies.reshape(X.shape)

    # Create a mask for the meaningful energy region
    energy_threshold = Z.max() * 0.95
    meaningful_region = Z < energy_threshold
    unsupported_region = Z >= energy_threshold

    # Detect local minima
    minima_mask = detect_peaks(-Z)
    isolated_minima = minima_mask & meaningful_region

    # Get coordinates of filtered minima
    minima_coords = np.where(isolated_minima)
    minima_x = X[isolated_minima]
    minima_y = Y[isolated_minima]
    minima_z = Z[isolated_minima]

    # Determine which minima are close to stored memories
    is_preserved = np.array([
        any(np.sqrt(np.sum((np.array([x, y]) - mem)**2)) < tol for mem in mems)
        for x, y in zip(minima_x, minima_y)
    ])

    # Rescale energies for visualization
    Z = Z - Z.min() + 1e-3

    # Plot contours of the energy landscape
    min_energy = Z.min()
    max_energy = Z.max()
    levels = np.logspace(np.log10(min_energy + 1e-3), np.log10(max_energy), n_contours)
    ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.75)
    
    # Overlay mask for unsupported regions
    mask_rgba = np.zeros((*Z.shape, 4))  # RGBA array
    mask_rgba[unsupported_region] = [0.8, 0.8, 0.8, 0.6]  # Light gray with 0.5 alpha
    ax.imshow(mask_rgba, extent=[x.min(), x.max(), y.min(), y.max()], 
             interpolation='nearest', origin='lower')

    # Overlay the memory points
    ax.scatter(mems[:, 0], mems[:, 1], c='k', s=200, label='Stored patterns', alpha=0.8)

    # Plot preserved and novel minima separately with white edges
    outstroke_width = 6
    preserved_minima = ~is_preserved  # Invert because we want to show novel ones in red
    ax.scatter(minima_x[preserved_minima], minima_y[preserved_minima], 
              c='r', s=800, marker='*', zorder=100, label='Novel memories', 
              path_effects=[pe.withStroke(linewidth=outstroke_width, foreground='white')])
    ax.scatter(minima_x[~preserved_minima], minima_y[~preserved_minima], 
              c='blue', s=800, marker='*', zorder=100, label='Preserved memories',
              path_effects=[pe.withStroke(linewidth=outstroke_width, foreground='white')])

    if show_legend:
        ax.legend(fontsize=24, markerscale=1, frameon=True, 
                 fancybox=True, shadow=True, loc='lower left')

    ax.set_xlim(x.min(), x.max())  # Set x limits to match meshgrid
    ax.set_ylim(y.min(), y.max())  # Set y limits to match meshgrid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'$\\beta={beta}$', fontsize=50, fontweight='bold')

    return ax


configs = [
    {"beta": 0.3, "show_legend": True, "n_contours": 30},
    {"beta": 0.6, "show_legend": False, "n_contours": 25},
    {"beta": 1.1, "show_legend": False, "n_contours": 15},
    {"beta": 2.1, "show_legend": False, "n_contours": 15},
]
fig, axes = plt.subplots(1, len(configs), figsize=(8*len(configs), 8))

for iax, (ax, config) in enumerate(zip(axes, configs)):
    plot_2D_energyfn(ax, beta=config["beta"], mems=mems, memLSR=memLSR, show_legend=config['show_legend'], n_contours=config["n_contours"])

plt.tight_layout()
plt.show()

fig.savefig("figures/fig1_2d.png", dpi=200)
# %%
