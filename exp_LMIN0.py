#%%
"""
Pack as many memories as possible into 16 dims

- How do I quantify new memory?
- How do I compare against gaussian kernel?
- Both require that I am at the fixed point. How do i ensure that?
"""

#%%
import os
from pathlib import Path
import tyro
from jax import config
from tyro.conf import Positional
from typing import *
from dataclasses import dataclass
from utils import is_interactive
import functools as ft
from loguru import logger
import json
import numpy as np
from LMIN0_utils import cache_generate_rand_memories, compute_beta_r_ranges, get_all_combinations

@dataclass
class Args:
    outf: Positional[str] # File to save output to
    nbetas: int = 50 # Number of betas to use
    beta_idx: int = 0 # Index of beta to process in the computed range(beta_min, beta_max, nbetas) for the randomly specified memories.
    M: int = 8 # Number of memories
    d: int = 16 # Dimension of each memory
    num_vol_samples: int = 1_000_000 # Number of volume samples to use for estimating volume coverage
    device: Optional[str] = None # GPU device to use, if None, use default GPU.
    zero_grad_thresh: float = 5e-6 # Threshold to consider a memory to be at a local minimum
    seed: int = 0 # Seed for random number generator (for generating memories)

    def __post_init__(self):
        assert self.beta_idx >= 0 and self.beta_idx < self.nbetas, "beta_idx must be in the range [0, nbetas)"
        assert self.outf.endswith(".jsonl"), "outf must end with .jsonl"
        self.outf = Path(self.outf)
        self.outf.parent.mkdir(parents=True, exist_ok=True)

        if self.device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            config.update('jax_platforms', 'cpu')
        elif self.device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.device

if is_interactive():
    args = Args("expresults/tst-LMIN0.jsonl", beta_idx=30, device='cpu', M=25, d=32)
else:
    args = tyro.cli(Args)

#%%
## Following imports require the JAX device to be set
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray    
from memories import EpaMemory
import jax_utils as ju

logger.info(f"Initializing memory patterns and beta ranges for {args.nbetas} betas")
mem = EpaMemory(beta=1., eps=0., lmda=0.) # Beta is placeholder, overridden in functions
d = args.d
M = args.M
rng = jr.PRNGKey(args.seed)
zero_grad_thresh = args.zero_grad_thresh
key, rng = jr.split(rng)
Xis = cache_generate_rand_memories(ju.encode_key(key), d, M)
beta_range, r_range = compute_beta_r_ranges(Xis)
betas = np.linspace(beta_range[0], beta_range[1], args.nbetas)
beta = betas[args.beta_idx]

def r_from_beta(beta: float) -> float:
    return np.sqrt(2 / beta)

def beta_from_r(r: float) -> float:
    """Assume r is the squared radius around each memory"""
    return 2 / r**2

# %% Computing the number of local minima
# Generate all possible combinations using JAX
@jax.jit
def process_combinations(masks: Float[Array, "N M"], Xis: Float[Array, "M d"], beta: float):
    """Compute the energy, gradient norm, and size of each combination of memories """
    def compute_combo(mask):
        # Get memories for this combination
        curr_mems = Xis * jnp.expand_dims(mask, -1)
        # Compute centroid (accounting for mask)
        centroid = curr_mems.sum(0) / mask.sum()
        # Compute energy and gradient
        E, grad = mem.energy_and_grad(centroid, Xis, beta=beta)
        gradnorm = jnp.linalg.norm(grad)
        return E, gradnorm, jnp.sum(mask)
    return jax.vmap(compute_combo)(masks)

def estimate_volume_coverage(Xis: Array, beta: float, num_samples: int = 500_000, key: PRNGKeyArray = jr.PRNGKey(0)):
    """Estimate how much of the hypersphere volume is covered by Xis"""
    # Sample points uniformly from the unit hypercube
    samples = jr.uniform(key, (num_samples, Xis.shape[1]))
    
    # For each sample, check if its energy is inf
    energies = mem.venergy(samples, Xis, beta=beta)
    return jnp.mean(~jnp.isinf(energies))

def compute_local_minima_info(Xis: Float[Array, "M d"], beta: float, num_vol_samples: int):
    """Put the above functions together into a single pipeline"""
    # Compute hypercube local minima
    M, d = Xis.shape
    masks = get_all_combinations(M)
    print(masks.shape)
    energies, gradnorms, sizes = process_combinations(masks, Xis, beta)

    # Filter invalid results
    valid_mask = ~jnp.isinf(energies)
    gradnorms = gradnorms[valid_mask]
    sizes = sizes[valid_mask]

    # Count memories
    gradnorm_mask = gradnorms < zero_grad_thresh
    num_old_mems = jnp.sum((sizes == 1) & gradnorm_mask)
    num_new_mems = jnp.sum((sizes > 1) & gradnorm_mask)
    total_mems = num_new_mems + num_old_mems

    # Estimate volume coverage
    vol_coverage = estimate_volume_coverage(Xis, beta, num_vol_samples)

    info = {
        "beta": beta,
        "beta_min": beta_range[0],
        "beta_max": beta_range[1],
        "r_min": r_range[0],
        "r_max": r_range[1],
        "num_old_mems": num_old_mems.item(),
        "num_new_mems": num_new_mems.item(),
        "num_mems": total_mems.item(),
        "vol_coverage": vol_coverage.item(),

        "num_vol_samples": num_vol_samples,
        "zero_grad_thresh": zero_grad_thresh,
        "nbetas": args.nbetas,
        "beta_idx": args.beta_idx,
        "M": M,
        "d": d,
        "seed": args.seed,
    }

    return info

if __name__ == "__main__":
    logger.info(f"args: {args}")
    logger.info(f"Beginning computation for {args.M} {args.d}-dimensional memories")
    info = compute_local_minima_info(Xis, beta, args.num_vol_samples)
    logger.info(f"info: {info}")

    logger.info(f"Writing to {args.outf}")
    with open(args.outf, "a") as f:
        f.write(json.dumps(info) + "\n")