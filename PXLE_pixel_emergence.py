"""
To satisfy this, we design an experiment to store all 60k rasterized MNIST training images into the LSR DenseAM. In this large data regime, we show multiple examples of novel images forming between two or more stored patterns that are still retrievable.
"""

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import jax.random as jr
from memories import EpaMemory
from data_utils import get_mnist_traindata
import jax_utils as ju
from loguru import logger
from memories import EpaMemory
from QBVAE_utils import bwr_imshow

Xtrain, Ytrain = get_mnist_traindata()

# Flatten images to vectors
Xtrain_flat = Xtrain.reshape(Xtrain.shape[0], -1)  # Shape: (60000, 784)
logger.info(f"Loaded MNIST training data: {Xtrain_flat.shape}")
Xis = jnp.array(Xtrain_flat)

mem = EpaMemory(beta=0.05)


x = Xis[:1]
x = x + jr.normal(jr.key(42), x.shape) * 0.2
mem.venergy(x, Xis)

#%%
import matplotlib.pyplot as plt
import numpy as np
from PXLE_utils import visualize_mnist_images

#%%
"""

1) Select 8 diverse images from the dataset. 
2) Choose a beta s.t. their basins overlap a bit. Tune this beta based on following results
3) Find all emergent minima that arise because of the interactions of these stored patterns.
Use 
4) Visualize these emergent minima
"""

from QBVAE_utils import compute_minima_information, get_novel_and_old_minima
# compute_minima_information(Xis, beta, grad_tol)

N = 8
ixs = jr.choice(jr.key(2), jnp.arange(Xis.shape[0]), (N,))
Xi0 = Xis[ixs]
dists = ju.get_dist_matrix(Xi0, Xi0)
dists = dists.at[jnp.arange(N), jnp.arange(N)].set(dists.max())
r_min_max = jnp.array([dists.min(), dists.max()]) / 2 # non-overlapping basin distance
beta_max, beta_min = ju.beta_from_r(r_min_max)

beta = 0.4 * beta_max

lmin_unique_array, lmin_unique_merge_idxs = compute_minima_information(Xis[ixs], beta)
info = get_novel_and_old_minima(lmin_unique_array, lmin_unique_merge_idxs)

novel_minima = info['novel_minima']
old_minima = info['old_minima']
visualize_mnist_images(novel_minima[:8])
visualize_mnist_images(old_minima)

"""
Time to go back to the story that I want to write. Specifically, one reviewer request was to store 60k MNIST images. Another reviewer request was to see what pixel superposition looks like.

The above seems to be a good pixel superposition
"""
# %%

def visualize_novel_and_stored_memories(info, figsize=(16, 8)):
    """
    Visualize novel emergent memories alongside original stored patterns.
    
    Args:
        info: Dictionary containing novel_minima, novel_minima_memory_idxs, 
              old_minima, and old_minima_memory_idxs
        figsize: Figure size tuple
    
    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    novel_minima = info['novel_minima']
    novel_memory_idxs = info['novel_minima_memory_idxs']
    old_minima = info['old_minima']
    old_memory_idxs = info['old_minima_memory_idxs']
    
    # Take first 8 novel memories if more exist
    n_novel = min(8, len(novel_minima))
    novel_to_show = novel_minima[:n_novel]
    novel_idxs_to_show = novel_memory_idxs[:n_novel]
    
    # Sort stored patterns by their original indices for A-H order
    stored_pairs = list(zip(old_minima, old_memory_idxs))
    stored_pairs_sorted = sorted(stored_pairs, key=lambda x: x[1][0])  # Sort by original index
    old_minima_sorted = [pair[0] for pair in stored_pairs_sorted]
    old_memory_idxs_sorted = [pair[1] for pair in stored_pairs_sorted]
    
    # Create figure with two rows
    fig, (ax_novel, ax_stored) = plt.subplots(2, 8, figsize=figsize)
    
    # Plot novel memories (top row)
    for i in range(8):
        ax = ax_novel[i]
        if i < n_novel:
            # Reshape to 28x28 if needed
            img = novel_to_show[i]
            if len(img.shape) == 1:
                img = img.reshape(28, 28)
            
            # Use bwr_imshow for novel memories
            bwr_imshow(ax, img)
            
            # Create label from constituent memory indices
            constituent_idxs = novel_idxs_to_show[i]
            # Convert to alphabetical labels (A, B, C, ...)
            labels = [chr(65 + idx) for idx in constituent_idxs]  # 65 is ASCII for 'A'
            title = '+'.join(labels)
            ax.set_title(f"Novel: {title}", fontsize=10, fontweight='bold')
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Plot stored patterns (bottom row) - now in A-H order
    for i in range(8):
        ax = ax_stored[i]
        if i < len(old_minima_sorted):
            # Reshape to 28x28 if needed
            img = old_minima_sorted[i]
            if len(img.shape) == 1:
                img = img.reshape(28, 28)
            
            # Use bwr_imshow for stored patterns
            bwr_imshow(ax, img)
            
            # Get the original memory index
            original_idx = old_memory_idxs_sorted[i][0]  # Should be single element list
            # Convert to alphabetical label
            label = chr(65 + original_idx)
            ax.set_title(f"Stored: {label}", fontsize=10, fontweight='bold')
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add row labels
    ax_novel[0].set_ylabel('Novel Memories', fontsize=12, fontweight='bold')
    ax_stored[0].set_ylabel('Stored Patterns', fontsize=12, fontweight='bold')
    
    # Add main title
    fig.suptitle('Emergent Novel Memories vs Original Stored Patterns', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, (ax_novel, ax_stored)

fig, axes = visualize_novel_and_stored_memories(info, figsize=(16, 8))
fig.savefig('figures/PXLE_pixel_emergence.png', dpi=150, bbox_inches='tight')
plt.show()
