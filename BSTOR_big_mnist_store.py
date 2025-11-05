#%%
"""
Store all 60k MNIST images into the LSR DenseAM. Find emergent memories
"""

from data_utils import get_mnist_traindata
from memories import EpaMemory
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax_utils import r_from_beta, get_dist_matrix
from PXLE_utils import visualize_mnist_images
from QBVAE_utils import compute_minima_information, get_novel_and_old_minima
import matplotlib.pyplot as plt
    
#%% (slow)
Xtrain, Ytrain = get_mnist_traindata()
Xis = Xtrain.reshape(Xtrain.shape[0], -1)

mem = EpaMemory(beta=0.11)
n_samples = Xis.shape[0]
Es, dEs = mem.venergy_and_grad_batched(Xis, Xis, batch_size=1000)

#%%
dE_norms = jnp.linalg.norm(dEs, axis=-1)
tol = 0.
is_preserved = dE_norms <= tol
n_zero = (is_preserved).sum()
print(f"{n_zero} out of {Xis.shape[0]} gradients are zero")
print(f"Final shapes: Es {Es.shape}, dEs {dEs.shape}")

preserved_idxs = set(map(int, np.where(is_preserved)[0]))

#%% Optimize
N = 15
idx = 38
assert idx in preserved_idxs
seed_img = Xis[idx][None]
fig, ax = visualize_mnist_images(seed_img)

#%%
# Compute overlapping basins
r = r_from_beta(mem.beta)
dists = get_dist_matrix(seed_img, Xis)
is_nearby = (dists[0] < (2*r))
print("Number nearby: ", is_nearby.sum())

is_near = jnp.argwhere(is_nearby)
stored_idxs = jr.choice(jr.key(10), is_near, (N,))[...,0]
stored_idxs = jnp.hstack([jnp.array(idx), stored_idxs])
stored_Xis = Xis[stored_idxs]
fig, ax = visualize_mnist_images(stored_Xis)

#%% SLOW

lmin_unique_array, lmin_unique_merge_idxs = compute_minima_information(stored_Xis, mem.beta)
info = get_novel_and_old_minima(lmin_unique_array, lmin_unique_merge_idxs)

#%%
print("novel")
novel_mem_fig, ax = visualize_mnist_images(info['novel_minima']); novel_mem_fig.show()
plt.show()
print("Old")
old_mem_fig, ax = visualize_mnist_images(info['old_minima']); old_mem_fig.show()
plt.show()
print("Seed")
seed_fig, ax = visualize_mnist_images(seed_img); seed_fig.show()
plt.show()

seed_fig.savefig("figures/BSTOR_seed.png")
novel_mem_fig.savefig("figures/BSTOR_novel.png")
old_mem_fig.savefig("figures/BSTOR_old.png")