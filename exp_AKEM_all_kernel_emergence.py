"""
Test different kernels to see if they exhibit emergent behavior
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from scipy.signal import find_peaks
from utils import detect_local_minima_or_plateaus_1d

jax.config.update("jax_enable_x64", True)

# Kernel function implementations
def gaussian_kernel(u, bandwidth=1.0):
    """Gaussian kernel: exp(-β*u²/2) where β=1.0"""
    scaled_u = u / bandwidth
    return jnp.exp(-0.5 * scaled_u**2) / (bandwidth * jnp.sqrt(2 * jnp.pi))

def triangle_kernel(u, bandwidth=1.0):
    """Triangle kernel: ReLU(1 - β|x - μ|) where β=1.0"""
    scaled_u = u / bandwidth
    return jnp.clip(1 - jnp.abs(scaled_u), 0) / bandwidth

def uniform_kernel(u, bandwidth=1.0):
    """Uniform kernel: rect(β=1.0) - constant within |u| ≤ 1/β"""
    scaled_u = u / bandwidth
    return jnp.where(jnp.abs(scaled_u) <= 1, 1.0, 0.0) / bandwidth

def cosine_kernel(u, bandwidth=1.0):
    """Cosine kernel: cos(π/2 * min(β|x - μ|, 1)) where β=1.0"""
    scaled_u = u / bandwidth
    return jnp.cos((jnp.pi / 2) * jnp.minimum(jnp.abs(scaled_u), 1)) / bandwidth

def quartic_kernel(u, bandwidth=1.0):
    """Quartic kernel: ReLU(1 - β|x - μ|²)² where β=1.0"""
    scaled_u = u / bandwidth
    return jnp.clip(1 - scaled_u**2, 0)**2 / bandwidth

def triweight_kernel(u, bandwidth=1.0):
    """Triweight kernel: ReLU(1 - β|x - μ|²)³ where β=1.0"""
    scaled_u = u / bandwidth
    return jnp.clip(1 - scaled_u**2, 0)**3 / bandwidth

def tricube_kernel(u, bandwidth=1.0):
    """Tricube kernel: ReLU(1 - β|x - μ|³)³ where β=1.0"""
    scaled_u = u / bandwidth
    return jnp.clip(1 - jnp.abs(scaled_u)**3, 0)**3 / bandwidth

def epanechnikov_kernel(u, bandwidth=1.0):
    """Epanechnikov kernel: ReLU(1 - β|x - μ|²) where β=1.0"""
    scaled_u = u / bandwidth
    return jnp.clip(1 - scaled_u**2, 0) / bandwidth

def compute_kernel_sum(x_grid, centers, kernel_func, bandwidth):
    """
    Compute sum of kernels for given centers
    
    Args:
        x_grid: Points to evaluate at
        centers: Kernel center points (e.g., [-1, 1])
        kernel_func: Kernel function to use
        bandwidth: Kernel bandwidth
    
    Returns:
        kernel_sum: Sum of kernel values at x_grid points
    """
    kernel_sum = jnp.zeros_like(x_grid)
    for center in centers:
        u = x_grid - center
        kernel_sum += kernel_func(u, bandwidth)
    return kernel_sum

def detect_local_minima(x_grid, neg_log_sum, energy_threshold=None, flat_tolerance=1e-10):
    """
    Detect local minima in the negative log sum, including flat regions
    
    Args:
        x_grid: x-axis values
        neg_log_sum: negative log sum values
        energy_threshold_percentile: Percentile threshold to filter out very high energy regions
        flat_tolerance: Tolerance for detecting flat regions
    
    Returns:
        minima_x: x-coordinates of local minima
        minima_y: y-coordinates of local minima
        is_isolated: boolean array indicating if each minimum is isolated (True) or part of flat region (False)
    """
    neg_log_sum = np.array(neg_log_sum)
    x_grid = np.array(x_grid)
    
    # Set energy threshold to filter out very high energy regions
    if energy_threshold is None:
        energy_threshold = np.percentile(neg_log_sum[np.isfinite(neg_log_sum)], 95)
    
    # Detect flat minima using the custom function
    minima_mask = detect_local_minima_or_plateaus_1d(neg_log_sum, energy_threshold)
    
    if np.any(minima_mask):
        minima_indices = np.where(minima_mask)[0]
        minima_x = x_grid[minima_mask]
        minima_y = neg_log_sum[minima_mask]
        
        # Check if each minimum is part of a flat region by examining the energy landscape
        is_isolated = []
        for i, idx in enumerate(minima_indices):
            current_energy = neg_log_sum[idx]
            
            # Check for flat region by looking at nearby points in the energy landscape
            is_flat_region = False
            search_radius = 10  # Check within 10 grid points
            
            # Count how many nearby points have approximately the same energy
            flat_count = 0
            for offset in range(-search_radius, search_radius + 1):
                check_idx = idx + offset
                if (check_idx >= 0 and check_idx < len(neg_log_sum) and 
                    abs(neg_log_sum[check_idx] - current_energy) <= flat_tolerance):
                    flat_count += 1
            
            # Calculate what fraction of available points have similar energy
            available_points = min(idx + search_radius + 1, len(neg_log_sum)) - max(idx - search_radius, 0)
            flat_fraction = flat_count / available_points
            is_flat_region = flat_fraction > 0.3  # e.g., 30% of available points
            is_isolated.append(not is_flat_region)
        
        return minima_x, minima_y, np.array(is_isolated)
    else:
        return np.array([]), np.array([]), np.array([])

def plot_basin_merging_analysis():
    """
    Create grid plot showing basin merging behavior under different kernels and bandwidths
    """
    # Define kernels
    kernels = {
        'Gaussian': gaussian_kernel,
        'Triangle': triangle_kernel, 
        'Uniform': uniform_kernel,
        'Triweight': triweight_kernel,
        'Quartic': quartic_kernel,
        'Tricube': tricube_kernel,
        'Cosine': cosine_kernel,
        'Epanechnikov': epanechnikov_kernel,
    }
    
    # Define bandwidths to test
    # bandwidths = [0.2, 0.5, 1.0, 1.5, 2.0]
    # bandwidths = [0.2, 0.5, 1.0, 1.5, 1.9, 1.95, 2.0, 2.05, 3.]
    bandwidths = [0.2, 0.5, 1.5, 1.9, 2.0, 2.3, 3.]
    
    # Kernel centers at +/- 1
    centers = jnp.array([-1.0, 1.0])
    
    # Grid for evaluation (changed to [-2, 2])
    x_grid = jnp.linspace(-2, 2, 1000)
    
    # Create figure
    fig, axes = plt.subplots(len(bandwidths), len(kernels), 
                             figsize=(20,12), sharex=True, sharey='row', squeeze=False)
    
    # Store minima information for analysis
    minima_data = {}
    eps = 1e-10
    threshold = -jnp.log(eps)

    for i, (kernel_name, kernel_func) in enumerate(kernels.items()):
        minima_data[kernel_name] = {}
        for j, bandwidth in enumerate(bandwidths):
            ax = axes[j, i]
            
            # Compute kernel sum
            kernel_sum = compute_kernel_sum(x_grid, centers, kernel_func, bandwidth)
            
            # Convert to negative log sum
            neg_log_sum = -jnp.log(kernel_sum + eps)  # Add small epsilon to avoid log(0)
            
            # Detect local minima with isolation info
            minima_x, minima_y, is_isolated = detect_local_minima(np.array(x_grid), np.array(neg_log_sum), energy_threshold=0.99*threshold)
            
            # Store minima data
            minima_data[kernel_name][bandwidth] = {
                'x': minima_x,
                'y': minima_y,
                'is_isolated': is_isolated,
                'count': len(minima_x)
            }
            
            # Plot energy landscape
            ax.plot(x_grid, neg_log_sum, 'b-', linewidth=1.5)
            
            # Plot detected minima with different sizes
            color = "#DF0F0F"
            if len(minima_x) > 0:
                # Plot isolated minima (large dots)
                if np.any(is_isolated):
                    isolated_x = minima_x[is_isolated]
                    isolated_y = minima_y[is_isolated]
                    ax.plot(isolated_x, isolated_y, 'ro', markersize=8, markerfacecolor=color, 
                           markeredgecolor=color, markeredgewidth=1.5, label='Isolated minima')
                
                # Plot flat region minima (small dots) - changed to darkred
                if np.any(~is_isolated):
                    flat_x = minima_x[~is_isolated]
                    flat_y = minima_y[~is_isolated]
                    ax.plot(flat_x, flat_y, 'ro', markersize=3, markerfacecolor=color, 
                           markeredgecolor=color, markeredgewidth=0., label='Flat minima')
            
            # Reference lines
            ax.axvline(-1, color=color, linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(1, color=color, linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            # Labels
            if j == 0:
                # Move title higher up
                ax.set_title(f'{kernel_name}', fontsize=16, fontweight='bold', pad=30)
                # Add equation underneath kernel name but above the plot area
                equations = {
                    'Gaussian': r'$\exp(-\beta/2 \|x\|^2)$',
                    'Triangle': r'$\text{ReLU}(1 - \beta||x||)$',
                    'Uniform': r'$I(\beta||x|| \leq 1)$',
                    'Epanechnikov': r'$\text{ReLU}(1 - \beta ||x||^2)$',
                    'Cosine': r'$\cos(\frac{\pi}{2}\min(\beta||x||, 1))$',
                    'Quartic': r'$\text{ReLU}(1 - \beta ||x||^2)^2$',
                    'Triweight': r'$\text{ReLU}(1 - \beta ||x||^2)^3$',
                    'Tricube': r'$\text{ReLU}(1 - \beta ||x||^3)^3$'
                }
                # Position equation between title and plot area
                ax.text(0.5, 1.05, equations[kernel_name], transform=ax.transAxes, 
                       fontsize=15, ha='center', va='bottom')
            if i == 0:
                ax.set_ylabel(f'β = {1/bandwidth**2:0.02f}', fontsize=14)
            # Remove the xlabel condition entirely - no xlabel needed
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_grid.min(), x_grid.max())
            # Set specific x-ticks
            ax.set_xticks([-1,0, 1])
            # Make x-tick labels bigger
            ax.tick_params(axis='x', labelsize=14)
    
    plt.tight_layout()
    
    # Print summary of minima counts
    print("\nSummary of Local Minima Counts:")
    print("=" * 70)
    print(f"{'Kernel':<12} | ", end="")
    for bw in bandwidths:
        print(f"h={bw:<4} | ", end="")
    print()
    print("-" * 70)
    
    for kernel_name in kernels.keys():
        print(f"{kernel_name:<12} | ", end="")
        for bandwidth in bandwidths:
            data = minima_data[kernel_name][bandwidth]
            total = data['count']
            isolated = np.sum(data['is_isolated']) if len(data['is_isolated']) > 0 else 0
            flat = total - isolated
            print(f"{isolated}+{flat}={total:<2} | ", end="")
        print()
    
    return fig, minima_data

fig, minima_data = plot_basin_merging_analysis()
plt.show()
fig.savefig('figures/AKEM_all_kernel_emergence.png', dpi=300, bbox_inches='tight')
# %%
