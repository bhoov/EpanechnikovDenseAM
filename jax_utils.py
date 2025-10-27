#%%
import jax.numpy as jnp
import jax
import jax.tree_util as jtu
import jax.random as jr
from jaxtyping import Float, Array, PRNGKeyArray

def count_parameters(pytree):
    """Count the number of parameters in a pytree where the leaves are arrays."""
    def count_leaves(leaf): return jnp.size(leaf)
    return jtu.tree_reduce(lambda x, y: x+y, jtu.tree_map(count_leaves, pytree), initializer=0.)

def allclose(xguess:jax.Array, xtrue:jax.Array, atol=1e-8, rtol=0.):
    """A more intuitive 'allclose' function for jax arrays
    Tests for the "straight line" fit of the two arrays when normalized between 0 and 1 according to xtrue

    This function is not symmetric for `xguess` and `xtrue`.
    Requires two identical length jax arrays
    """
    xmin, xmax = xtrue.min(), xtrue.max()
    x = (xtrue - xmin) / (xmax - xmin)
    xhat = (xguess - xmin) / (xmax - xmin)
    return jnp.allclose(xhat, x, atol=atol, rtol=rtol)

#%%
from typing import *
from dataclasses import dataclass
import equinox as eqx
import functools as ft

@dataclass
class _AtWrapper:
    _tree_at: Callable
    def set(self, replace): return self._tree_at(replace=replace)
    def set_fn(self, replace_fn:Callable): return self._tree_at(replace_fn=replace_fn)

def _at(pytree, where, is_leaf=None) -> _AtWrapper:
    """
    Lightweight wrapper around an eqx.Module to make it easier to do model surgery

    Example:
    ```
    module = module.at(lambda module: module.layer.weight).set(0)
    module = module.at(lambda module: module.layer.weight).set_fn(lambda w: w+5)
    ```

    You can even chain these operations:
    ```
    module = (
        module.at(lambda module: module.layer.weight).set(0)
        .at(lambda module: module.layer.bias).set(1)
        .at(lambda module: module.step).set_fn(lambda s: s+1)
        )
    ```

    Todo:
    - Accept "keystr" to index, instead of callable
    """
    return _AtWrapper(ft.partial(eqx.tree_at, where=where, pytree=pytree, is_leaf=is_leaf))

# MONKEY PATCH it in
eqx.Module.at = _at

# %% Check if we are running in interactive mode or not
def is_interactive() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


#%% Encode and decode keys
import jax.numpy as jnp

def encode_key(key: jnp.ndarray) -> str: 
    """
    Example:
    ```
        @ft.lru_cache
        def cache_func(key: str, d, m, beta):
            key = decode_key(key)
            ...

        key = jr.PRNGKey(5)
        cache_func(encode_key(key), ...)
    ```
    """
    return jnp.array_str(key)

def decode_key(key: str) -> jnp.ndarray:
    """
    Example:
    ```
        @ft.lru_cache
        def cache_func(key: str, ...):
            key = decode_key(key)
            ...

        key = jr.PRNGKey(5)
        cache_func(encode_key(key), ...)
    ```
    """
    return jnp.fromstring(key[1:-1], sep=" ", dtype=jnp.uint32)

#%% Analytically compute the volume of a d-dimensional ball of radius r
import numpy as np
def nball_volume(d: int, # Dimensionality of the ball
                 r: float, # Radius of the ball
                 ) -> float:
    """Compute the volume of a d-dimensional ball of radius r."""
    return (np.pi**(d/2) * r**d) / jax.scipy.special.gamma(d/2 + 1)

def get_dist_matrix(Xs: Float[Array, "M d"], Ys: Float[Array, "N d"]=None) -> Float[Array, "M N"]:
    """Compute the distance matrix of a set of points"""
    Ys = Xs if Ys is None else Ys
    return jax.vmap(lambda x, y: jnp.sqrt(jnp.sum((x - y)**2, axis=-1)), in_axes=(0, None))(Xs, Ys)

def compute_beta_r_ranges(Xis: Float[Array, "M d"]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Given a set of memories, compute the interesting ranges beta and r on the unit hypercube
    
    Returns:
        beta_range = (beta_min, beta_max)
        r_range = (r_min, r_max)
    """
    M, d = Xis.shape
    # Compute beta_range
    # dists = jax.vmap(lambda x, y: jnp.sqrt(jnp.sum((x - y)**2, axis=-1)), in_axes=(0, None))(Xis, Xis)
    dists = get_dist_matrix(Xis)
    dists = jnp.where(dists == 0, jnp.inf, dists)
    Rmin = 0.5 * dists.min()
    Rmax = np.sqrt(d)
    beta_range = (2/Rmax**2, (2/Rmin**2).item())
    r_range = (Rmin.item(), Rmax)
    return beta_range, r_range

def r_from_beta(beta: float) -> float:
    return np.sqrt(2 / beta)

def beta_from_r(r: float) -> float:
    """Assume r is the squared radius around each memory"""
    return 2 / r**2

def nchoosek(n: int, k: int) -> int:
    """Compute n choose k"""
    return np.prod(np.arange(n, n-k, -1)) // np.prod(np.arange(1, k+1))

def sample_nchoosek(M: int, # Total number of items
                    k: int, # Number of items to choose
                    nsamples: int, # (maximum) Number of unique samples to return
                    key: PRNGKeyArray, # JAX random key
                    ensure_unique: bool = True, # Whether to ensure unique combinations
                    n_oversample_factor: int = 5 # Factor to oversample by to ensure unique combinations
                    ) -> Float[Array, "nsamples M"]:
    """Getting all nchoosek combinations is infeasible at high M, so we sample
    
    To ensure unique combinations, we oversample by a factor of n_oversample_factor
    """
    if not ensure_unique:
        keys = jr.split(key, nsamples)
        return jax.vmap(lambda key: jr.choice(key, jnp.arange(M), shape=(k,), replace=False))(keys)

   # For unique sampling
    max_possible = nchoosek(M, k)
    if nsamples >= max_possible:
        logger.debug(f"Using analytical enumeration for {nsamples} samples of {M} choose {k}")
        # Use analytical enumeration like in LMIN0
        masks = jnp.arange(2**M, dtype=jnp.uint32)
        masks = jnp.expand_dims(masks, -1) >> jnp.arange(M)
        masks = masks & 1
        # Filter to only k-sized combinations
        sizes = jnp.sum(masks, axis=1)
        valid_mask = sizes == k
        return jnp.where(masks[valid_mask])[1].reshape(-1, k)
    
    # Otherwise do stochastic sampling
    unique_masks = set()
    rng = key
    while len(unique_masks) < nsamples:
        rng, sample_key = jr.split(rng)
        # Generate a batch of samples
        batch_size = min((nsamples - len(unique_masks)) * n_oversample_factor, 10000)
        keys = jr.split(sample_key, batch_size)
        masks = jax.vmap(lambda key: jr.choice(key, jnp.arange(M), shape=(k,), replace=False))(keys)
        
        # Convert to tuples for set operations
        for mask in masks:
            unique_masks.add(tuple(sorted(mask.tolist())))
            if len(unique_masks) >= nsamples:
                break
    return jnp.array(list(unique_masks)[:nsamples])

def sample_from_unit_ball(key, 
                          N:int,  # Number of samples
                          d:int,  # Dimension of unit ball
                          max_radius:float, # Radius of the ball
                          ):
    """Uniformly samples N points from the unit ball in d dimensions"""
    rand_directions = jr.uniform(key, shape=(N, d), minval=-1, maxval=1)
    rand_directions = rand_directions / jnp.linalg.norm(rand_directions, axis=1, keepdims=True)
    radii = jr.uniform(key, shape=(N, 1), minval=0, maxval=max_radius)
    return rand_directions * radii

def sample_from_surface_of_ball(key, 
                          N:int,  # Number of samples
                          d:int,  # Dimension of unit ball
                          radius:float, # Radius of the ball
                          ):
    """Uniformly samples N points from the surface of the unit ball in d dimensions"""
    rand_directions = jr.uniform(key, shape=(N, d), minval=-1, maxval=1)
    rand_directions = rand_directions / jnp.linalg.norm(rand_directions, axis=1, keepdims=True)
    radii = jr.uniform(key, shape=(N, 1), minval=0.98*radius, maxval=0.9999*radius)
    return rand_directions * radii

## Using sample from surface of ball
# d = 8
# nmems = 1000
# rng = jr.PRNGKey(0); key, rng = jr.split(rng)
# assume_radius = 0.21
# memories = jr.normal(key, shape=(nmems, d))
# samples_per_memory = 100
# supported_v0s = []
# for m in range(nmems):
#     key, rng = jr.split(rng)
#     perturbations = sample_from_surface_of_ball(key, samples_per_memory, d, assume_radius) #perturbations have radius assume_radius
#     supported_v0_on_surface_of_m = memories[m] + perturbations # Shape (100, 8)
#     supported_v0s.append(supported_v0_on_surface_of_m)
# supported_v0s = jnp.concatenate(supported_v0s, axis=0) # Shape (100_000, 8)

#%% JAX device configuration
import os
from jax import config as jconfig

def configure_jax_devices(device:str):
    """Switch between CPU and GPU for JAX"""
    if device == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        jconfig.update('jax_platforms', 'cpu')
    elif device == "gpu":
        jconfig.update('jax_platforms', 'cuda')
    
#%%
import numpy as np
def spectral_clustering(retrieved_mems: Array, beta: float, eigenval_tol: float = 1e-5) -> int:
    retrieval_dists = jax.vmap(lambda x: jnp.sum((x - retrieved_mems)**2, axis=-1))(retrieved_mems)
    # Consider two points connected if their distSquared <= 2 / beta 
    adjacency_mat = retrieval_dists <= 2 / beta 

    # Compute the normalized Laplacian
    degree_mat = jnp.sum(adjacency_mat, axis=1)
    degree_mat_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(degree_mat + 1e-8))
    laplacian = jnp.eye(len(retrieved_mems)) - degree_mat_inv_sqrt @ adjacency_mat @ degree_mat_inv_sqrt

    # Get eigenvalues and eigenvectors
    # eigenvals, eigenvecs = jnp.linalg.eigh(laplacian)
    eigenvals, eigenvecs = np.linalg.eigh(laplacian)

    # Count number of connected components (clusters) by counting eigenvalues close to zero
    num_clusters = np.sum(eigenvals < eigenval_tol)
    return int(num_clusters)

def spectral_clustering_batched(retrieved_mems: Array, beta: float, batch_size: int = 1000, eigenval_tol: float = 1e-5) -> int:
    """Memory-efficient spectral clustering using batched computation.
    
    Args:
        retrieved_mems: Array of shape (N, d) containing the retrieved memories
        beta: The beta parameter determining connectivity threshold
        batch_size: Size of batches for processing
    
    Returns:
        Number of connected components (clusters)
    """
    N = len(retrieved_mems)
    
    # Compute degree matrix in batches
    degree_vec = jnp.zeros(N)
    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        batch_mems = retrieved_mems[i:batch_end]
        
        # Compute distances and connections for this batch
        batch_dists = jax.vmap(lambda x: jnp.sum((x - retrieved_mems)**2, axis=-1))(batch_mems)
        batch_adj = (batch_dists <= 2 / beta).astype(jnp.float32)
        
        # Update degree counts
        degree_vec = degree_vec.at[i:batch_end].set(jnp.sum(batch_adj, axis=1))
    
    # Compute D^(-1/2)
    degree_vec_inv_sqrt = 1.0 / jnp.sqrt(degree_vec + 1e-8)
    
    # Initialize array to store largest eigenvalues
    k = min(25, N)  # Only compute top k eigenvalues
    eigenvals = jnp.zeros(k)
    
    # Power iteration to find largest eigenvalues
    def matrix_vector_prod(v):
        result = jnp.zeros_like(v)
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            batch_mems = retrieved_mems[i:batch_end]
            
            # Compute batch of adjacency matrix
            batch_dists = jax.vmap(lambda x: jnp.sum((x - retrieved_mems)**2, axis=-1))(batch_mems)
            batch_adj = (batch_dists <= 2 / beta).astype(jnp.float32)
            
            # Apply D^(-1/2) A D^(-1/2) v for this batch
            batch_result = batch_adj @ (degree_vec_inv_sqrt * v)
            result = result.at[i:batch_end].set(degree_vec_inv_sqrt[i:batch_end] * batch_result)
        
        return v - result  # Return I - D^(-1/2) A D^(-1/2) v
    
    # Use power iteration to estimate smallest eigenvalues
    v = jr.normal(jr.PRNGKey(0), (N,))
    v = v / jnp.linalg.norm(v)
    
    for i in range(k):
        for _ in range(20):  # Number of power iterations
            v = matrix_vector_prod(v)
            v = v / jnp.linalg.norm(v)
        eigenvals = eigenvals.at[i].set(v @ matrix_vector_prod(v))
        
        # Deflate to find next eigenvalue
        v = v - jnp.sum(v) * jnp.ones(N) / jnp.sqrt(N)
        v = v / jnp.linalg.norm(v)
    
    # Count eigenvalues close to zero
    num_clusters = int(jnp.sum(eigenvals < eigenval_tol))
    return num_clusters
