#%%
import numpy as np
import os

import jax
from jax import config
import jax.numpy as jnp
import jax.random as jr
from distributions import GMMDistribution
import jax_utils as ju
from memories import EpaMemory, LseMemory
import optax
import tyro
from dataclasses import dataclass
from tyro.conf import Positional
from typing import *
from tqdm.auto import tqdm
from pathlib import Path
from jaxtyping import Float, Array, PRNGKeyArray
from loguru import logger
import json

@dataclass
class Args:
    outf: Positional[str] # File to save output to
    d: int = 2 # Data dimension
    k: int = 5 # Number of mixtures
    true_sigma: float = 0.1 # True stdev around each mixture
    beta_idx: Optional[int] = 0 # Index of beta to process in the computed geomspace(beta_min, beta_max, nbetas) for the randomly specified memories. If None, loop through all betas.
    M: int = 10 # Number of samples from true distribution
    N: int = 50 # Number of points to sample in the ball around each memory
    seed: int = 8 # Random seed
    nbetas: int = 10 # Number of betas to test
    energy_key: str = "epa" # Energy function to use
    depth: int = 5000 # Number of descent iterations
    epa_grad_tol: float = 1e-3 # Gradient tolerance to consider something at local minimum
    lse_grad_tol: float = 1e-1 # Gradient tolerance to consider something at local minimum for lse kernel
    lr_start: float = 0.01 # Initial learning rate for energy descent. Cosine decayed to lr_end
    lr_end: float = 0.001 # Final learning rate for energy descent
    orthogonal_init: Optional[bool] = False # Whether to initialize the true gmm means orthogonally
    do_spectral_clustering: Optional[bool] = True # Whether to do spectral clustering to check uniqueness of retrieved memories
    sample_from_surface_of_ball: Optional[bool] = False # Whether to sample from the surface of the ball
    epa_do_normal_retrieval: Optional[bool] = True # Whether to do normal retrieval for epa

    device: Optional[str] = None # GPU device to use, if None, use default GPU.

    def __post_init__(self):
        assert self.energy_key in ["epa", "lse"], "Energy function must be either epa or lse"
        assert self.outf.endswith(".jsonl"), "outf must end with .jsonl"
        self.outf = Path(self.outf)
        self.outf.parent.mkdir(parents=True, exist_ok=True)

        if self.device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            config.update('jax_platforms', 'cpu')
        elif self.device == "gpu":
            print(f"Available devices: {jax.devices()}")
            config.update('jax_platforms', 'cuda')
            # print(f"New CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
        elif self.device is not None:
            # Set the visible devices to the device string
            os.environ['CUDA_VISIBLE_DEVICES'] = self.device
            print(f"New CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")

if ju.is_interactive():
    args = Args("expresults/tst-LOGL0.jsonl", d=2, k=5, true_sigma=0.1, M=10, seed=1, nbetas=10, beta_idx=9, energy_key="lse", depth=10000, lr_start=0.03, lr_end=0.0003, device='cpu')
else:
    args = tyro.cli(Args)

if args.do_spectral_clustering:
    raise NotImplementedError("Spectral clustering does not scale for LSE kernel")

ENERGY_FNS = {
    "epa": EpaMemory(eps=0., lmda=0.),
    "lse": LseMemory(),
}

rng = jr.PRNGKey(args.seed)
key, rng = jr.split(rng)
gmm = GMMDistribution.k_mixtures_in_d(args.k, args.d, args.true_sigma**2, key)

key, rng = jr.split(rng)
Xis = jnp.array(gmm.sample(key, args.M)) # These serve as the memories for my energies

# Valid beta range
beta_range, r_range = ju.compute_beta_r_ranges(Xis)
betas = np.geomspace(beta_range[0], beta_range[1], args.nbetas)

# Initialize memory
mem = ENERGY_FNS[args.energy_key]

# Define metric fns
compute_novelty = jax.jit(jax.vmap(lambda x: jnp.linalg.norm(x - Xis, axis=-1).min()))
compute_logl = jax.jit(jax.vmap(gmm.log_pdf))

def count_unique_retrievals_grid(retrieved_mems, beta):
    # Define grid cell size based on beta
    cell_size = np.sqrt(2/beta)
    
    # Normalize points to grid coordinates
    grid_coords = (retrieved_mems / cell_size).astype(int)
    
    # Count unique grid cells occupied
    return len(np.unique(grid_coords, axis=0))

def compute_metrics(Xis: Array, beta: float, rng: PRNGKeyArray, beta_idx: Optional[int] = None):
    ## M1: Check how many original memories are recoverable
    _, dEdmem = mem.venergy_and_grad(Xis, Xis, beta=beta)
    dEnorms = jnp.linalg.norm(dEdmem, axis=-1)
    og_mems_avg_gradnorm = dEnorms.mean()

    if args.energy_key == "epa":
        og_mems_recoverable = jnp.sum(dEnorms < args.epa_grad_tol)
    elif args.energy_key == "lse":
        # Use more tolerance for lse, since you will never *exactly* recover the original memories
        dEnorms = dEnorms / np.sqrt(beta)
        og_mems_avg_scaled_gradnorm = dEnorms.mean()
        print(f"M1c: {og_mems_avg_scaled_gradnorm:.2f} average scaled gradient norm for original memories")
        og_mems_recoverable = jnp.sum(dEnorms < args.lse_grad_tol) # THIS IS A VERY ROUGH METRIC

    print(f"M1a: {og_mems_avg_gradnorm:.2f} average gradient norm for original memories")
    print(f"M1b: {og_mems_recoverable} original memories are recoverable")

    # retrieved_mems = np.zeros((args.M * args.N, args.d))
    retrieved_mems_list = []

    ## Sample N points from each memory
    pbarM = tqdm(enumerate(range(args.M)), total=args.M)
    for mu, m in pbarM:
        pbarM.set_description(f"Analyzing memory idx {m}/{args.M - 1} [beta={beta:.4f}]")
        key, rng = jr.split(rng)
        if args.sample_from_surface_of_ball:
            perturbations = ju.sample_from_surface_of_ball(key, args.N, args.d, jnp.sqrt(2/beta))
        else:
            perturbations = ju.sample_from_unit_ball(key, args.N, args.d, jnp.sqrt(2/beta))

        rand_points = Xis[m] + perturbations

        # Save Xis for debugging
        if mu == 0:  # Only save once
            os.makedirs('debug', exist_ok=True)
            np.save('debug/Xis.npy', Xis)

        # All sampled points should be in the ball around each memory
        Es = mem.venergy(rand_points, Xis, beta=beta)
        if any(jnp.isinf(Es)):
            print("Error at beta = ", beta)
            print(Es)
            break

        # Do gradient descent from these rand_points to find the energy minima. Consider these points for log-likelihood computation
        alpha_schedule = optax.cosine_decay_schedule(args.lr_start, args.depth, alpha=args.lr_end)

        if args.energy_key == "epa":
            if args.epa_do_normal_retrieval:
                out_points, aux = mem.vrecall(rand_points, Xis, depth=args.depth, beta_schedule=beta, alpha_schedule=alpha_schedule, noise_schedule=0.0, grad_tol=1e-7)
            else:
                out_points = rand_points
            out_points, outbasins = mem.vcustom_retrieval(out_points, Xis, beta=beta)
        else:
            out_points, aux = mem.vrecall(rand_points, Xis, depth=args.depth, beta_schedule=beta, alpha_schedule=alpha_schedule, noise_schedule=0.0, grad_tol=1e-7)

        # Inside the loop, replace the ni_start/ni_end indexing with:
        if jnp.any(jnp.isnan(out_points)):
            num_nans = jnp.sum(jnp.isnan(out_points))
            msg = f"Error in gradient descent: encountered {num_nans} NaNs for memory idx {m} and beta {beta}, energy key {args.energy_key}"
            print(msg)
            
            # Filter out NaN points and only keep valid ones
            valid_mask = ~jnp.any(jnp.isnan(out_points), axis=1)
            valid_points = out_points[valid_mask]
            retrieved_mems_list.append(valid_points)
        else:
            retrieved_mems_list.append(out_points)

    retrieved_mems = jnp.concatenate(retrieved_mems_list, axis=0)
    novelty_distances = compute_novelty(retrieved_mems)
    avg_novelty_distance = novelty_distances.mean()
    is_novel = novelty_distances > 1e-1
    num_novel = jnp.sum(is_novel)

    print(f"M2: {avg_novelty_distance:.2f} average distance to nearest memory")

    if args.energy_key == "epa":
        num_unique_retrievals = len(np.unique(retrieved_mems, axis=0))
        print(f"M3: {num_unique_retrievals} unique memories retrieved")
    elif args.energy_key == "lse":
        if args.do_spectral_clustering:
            print("Spectral Clustering does not scale for for LSE kernel")
        else:
            num_unique_retrievals = count_unique_retrievals_grid(retrieved_mems, beta)
            print(f"M3: {num_unique_retrievals} unique memories retrieved via Grid Counting")

    # Compute the log-likelihood of the retrieved memories
    if jnp.any(jnp.isnan(retrieved_mems)):
        print("NaN observed in retrieved memories!!")

    log_likelihood = compute_logl(retrieved_mems)
    avg_logl = log_likelihood.mean()

    beta_idx = args.beta_idx if beta_idx is None else beta_idx
    sample_logl = [round(x, 4) for x in np.array(log_likelihood).tolist()]
    sample_is_novel = np.array(is_novel, dtype=np.int8).tolist()
    assert len(sample_logl) == len(sample_is_novel)

    info = {
        "beta": beta,
        "beta_idx": beta_idx,
        "beta_min": beta_range[0],
        "beta_max": beta_range[1],
        "og_mems_avg_gradnorm": og_mems_avg_gradnorm.item(),
        "og_mems_recoverable": og_mems_recoverable.item(),
        "avg_novelty_distance": avg_novelty_distance.item(),
        "num_unique_retrievals": num_unique_retrievals,
        "num_novel": num_novel.item(),
        "avg_logl": avg_logl.item(),

        # Sample-wise metrics
        "sample_logl": sample_logl,
        "sample_is_novel": sample_is_novel,

        # Following are the arguments used to run the experiment
        "d": args.d,
        "k": args.k,
        "true_sigma": args.true_sigma,
        "M": args.M,
        "N": args.N,
        "seed": args.seed,
        "nbetas": args.nbetas,
        "energy_key": args.energy_key,
        "depth": args.depth,
        "epa_grad_tol": args.epa_grad_tol,
        "lse_grad_tol": args.lse_grad_tol,
        "lr_start": args.lr_start,
        "lr_end": args.lr_end,
        "device": args.device,
        "sample_from_surface_of_ball": args.sample_from_surface_of_ball,
    }
    return info


if __name__ == "__main__":
    logger.info(f"args: {args}")
    logger.info(f"Beginning computation for {args.M} {args.d}-dimensional memories")
    # Loop through all betas if beta_idx is None
    if args.beta_idx is None:
        for beta_idx, beta in enumerate(betas):
            info = compute_metrics(Xis, beta, rng, beta_idx)

            logger.info(f"Writing to {args.outf}")
            with open(args.outf, "a") as f:
                f.write(json.dumps(info) + "\n")
    else:
        beta = betas[args.beta_idx]
        info = compute_metrics(Xis, beta, rng, args.beta_idx)
        # logger.info(f"info: {info}")
        logger.info(f"Writing to {args.outf}")
        with open(args.outf, "a") as f:
            f.write(json.dumps(info) + "\n")