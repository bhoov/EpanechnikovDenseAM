import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray    
import jax_utils as ju
import functools as ft

from typing import *

def generate_rand_memories(
    key: PRNGKeyArray, 
    dim: int, # Dimension of each memory
    M: int, # Number of memories
    ) -> Float[Array, "M d"]:
    """Generate unique random memories"""
    mems = jr.uniform(key, (3*M, dim), jnp.float32, minval=0, maxval=1)
    mems = jnp.unique(mems, axis=0)
    return mems[:M]

@ft.lru_cache
def cache_generate_rand_memories(
    keystr: str, 
    dim: int, # Dimension of each memory
    M: int, # Number of memories
    ) -> Float[Array, "M d"]:
    key = ju.decode_key(keystr)
    return generate_rand_memories(key, dim, M)

def compute_beta_r_ranges(Xis: Float[Array, "M d"]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Given a set of memories, compute the interesting ranges beta and r
    
    Returns:
        beta_range = (beta_min, beta_max)
        r_range = (r_min, r_max)
    """
    # Compute beta_range
    d = Xis.shape[-1]
    dists = jax.vmap(lambda x, y: jnp.sqrt(jnp.sum((x - y)**2, axis=-1)), in_axes=(0, None))(Xis, Xis)
    dists = jnp.where(dists == 0, jnp.inf, dists)
    Rmin = 0.5 * dists.min()
    Rmax = np.sqrt(d)
    beta_range = (2/Rmax**2, (2/Rmin**2).item())
    r_range = (Rmin.item(), Rmax)
    return beta_range, r_range

def get_all_combinations(n: int, max_size: Optional[int] = None) -> Float[Array, "N M"]:
    """Generate all possible combinations of n items, from (n choose 1) up until max_size (or n if None)

    Total number of combinations is N=2^n at no max_size
    """
    if max_size is None: max_size = n
    # Create a mask for all possible combinations
    masks = jnp.arange(2**n, dtype=jnp.uint32)
    masks = jnp.expand_dims(masks, -1) >> jnp.arange(n)
    masks = masks & 1
    
    # Filter by combination size
    sizes = jnp.sum(masks, axis=1)
    valid_mask = (sizes > 0) & (sizes <= max_size)
    masks = masks[valid_mask]
    
    return masks
