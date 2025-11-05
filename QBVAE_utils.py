#%%
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from fasttransform import Transform
from typing import Union
from pathlib import Path
import functools as ft
import numpy as np
from tqdm.auto import tqdm
from memories import EpaMemory, LseMemory
from fastcore.foundation import patch
import jax
import jaxlib
import functools as ft
from diffusers import DiffusionPipeline, AutoencoderTiny
from typing import *
from jaxtyping import Float, Array
from jax_utils import get_dist_matrix
import jax.numpy as jnp
from tqdm.auto import trange
from LMIN0_utils import get_all_combinations

class TAESD_Wrapper(nn.Module):
    def __init__(self, vae: AutoencoderTiny):
        super().__init__()
        self.vae = vae
        self.preprocess = Transform(lambda x: x, lambda x: x)
        self.latent_process = Transform(
            lambda z: z.reshape(z.shape[0], -1),
            lambda z: z.reshape(z.shape[0], 4, 8, 8) # Hardcoded for tiny imagenet
        )

    def encode(self, x):
        mu = self.vae.encode(x)
        mu = mu.latents
        mu = self.latent_process(mu)
        logvar = torch.zeros_like(mu) # Logvar isn't used in TAESD
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu

    def decode(self, z):
        z = self.latent_process.decode(z)
        return self.vae.decode(z).sample

    def encode_x(self, x):
        """From img to latent with preprocessing"""
        with torch.no_grad():
            x_normalized = self.preprocess(x)
            z, _ = self.encode(x_normalized)
        return z

    def decode_z(self, z):
        """From latent to img with postprocessing"""
        with torch.no_grad():
            xhat = self.decode(z).sample
            xhat = self.preprocess.decode(xhat)
        return xhat

    def to(self, device):
        self.vae.to(device)
        return self

    def eval(self):
        self.vae.eval()
        return self

    @property
    def device(self):
        return self.vae.device

    

@ft.lru_cache(maxsize=None)
def load_taesd_vae():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
    vae = pipe.vae 
    vae.eval()

    return TAESD_Wrapper(vae)

class BetaVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=32, beta=4.0):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar):
        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Total loss with beta weighting
        return recon_loss + self.beta * kl_loss

    @property
    def device(self):
        return next(self.parameters()).device

data_transform = Transform(
    lambda x: rearrange(x, "... h w -> ... (h w)"), 
    lambda xenc: (
        h := int(np.sqrt(xenc.shape[-1])),
        rearrange(xenc, "... (h w) -> ... h w", h=h, w=h)
    )[-1]
)

@ft.lru_cache
def load_bvae(
    path:Union[str, Path], # path to the model checkpoint
    beta=4.0, # beta value used to train the model. Doesn't affect inference
    ):
    """
    Example usage:

        model = load_bvae("beta_vae_mnist.pt")
    """
    # Load the model checkpoint
    state_dict = torch.load(path)

    # Get the dimensions of the model from checkpoint
    W1 = state_dict["encoder.0.weight"]
    Wf = state_dict["fc_mu.weight"]
    input_dim = W1.shape[1]
    hidden_dim = W1.shape[0]
    latent_dim = Wf.shape[0]

    # Initialize the model
    model = BetaVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, beta=beta)
    model.load_state_dict(state_dict)
    return model

@ft.lru_cache
def load_data(dataset="mnist"):
    if dataset == "mnist":
        Xtrain = torch.tensor(np.load("data/mnist/Xtrain.npy"), dtype=torch.float32)
        Xtest = torch.tensor(np.load("data/mnist/Xtest.npy"), dtype=torch.float32)
    elif dataset == "cifar":
        Xtrain = torch.tensor(np.load("data/cifar10/Xtrain.npy"), dtype=torch.float32)
        Xtest = torch.tensor(np.load("data/cifar10/Xtest.npy"), dtype=torch.float32)
        Xtrain = rearrange(Xtrain, "b h w c -> b c h w")
        Xtest = rearrange(Xtest, "b h w c -> b c h w")
    elif dataset == "tinyimagenet":
        Xtrain = torch.tensor(np.load("data/tiny-imgnet/Xtrain.npy"), dtype=torch.float32)
        Xtest = torch.tensor(np.load("data/tiny-imgnet/Xtest.npy"), dtype=torch.float32)
        Xtrain = rearrange(Xtrain, "b h w c -> b c h w")
        Xtest = rearrange(Xtest, "b h w c -> b c h w")
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    return Xtrain, Xtest

def batch_encode_data(model, data, batch_size=256, do_transform=True):
    latents = []
    mus = []
    logvars = []

    print(f"Encoding {len(data)} images...")
    model = model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            batch = batch.to(model.device)
            batch_transformed = data_transform(batch) if do_transform else batch
            mu, logvar = model.encode(batch_transformed)
            z = model.reparameterize(mu, logvar)
            
            latents.append(z.cpu().numpy())
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

    return np.concatenate(latents), np.concatenate(mus), np.concatenate(logvars)

MEMS = {
    "epa": EpaMemory(eps=0., lmda=0.),
    "lsr": EpaMemory(eps=0., lmda=0.),
    "lse": LseMemory(),
}

@patch
def numpify(self: torch.Tensor) -> np.ndarray:
    return self.detach().cpu().numpy()

@patch
def numpify(self: jax.Array) -> np.ndarray:
    return np.array(self)

@patch
def numpify(self: jaxlib.xla_extension.ArrayImpl) -> np.ndarray:
    return np.array(self)

def choose_beta(
    Xis: Float[Array, "N d"], # Array of memory vectors (N, D)
    target_K: int = 4, # Target average number of neighbors for each memory
    tol: float = 1e-5, # Tolerance for radius convergence in binary search
    max_iters: int = 800, # Maximum iterations for binary search
) -> Tuple[float, float]: # Returns beta and the chosen radius r
    """
    Binary search to find beta s.t. the avg number of neighbors within 2 ball radii is close to target_K.
    
    Based on pairwise distances.
    """
    num_mems = Xis.shape[0]
    if num_mems <= 1:
        print(f"Warning: Cannot compute neighbors for {num_mems} memory/memories. Returning default beta=0.1.")
        return 0.1 # Default beta for edge case

    # Compute pairwise distance matrix. Assumes get_dist_matrix sets D_ii = inf or handles it.
    dists = get_dist_matrix(Xis)

    # Initialize binary search bounds for radius r
    r_min = 0.0
    # Find max finite distance for r_max
    finite_dists = jnp.where(jnp.isinf(dists), -jnp.inf, dists)
    r_max = 2*jnp.max(finite_dists)

    if r_max <= 1e-9: # Use a small threshold instead of 0 for float comparison
         print("Warning: All memories seem identical or very close. Returning default beta=0.1.")
         return 0.1 # Default beta for edge case

    # Initial guess for r
    r = (r_min + r_max) / 2

    for i in range(max_iters):
        if r <= 1e-9: # Avoid radius becoming too small or zero
             print(f"Warning: Radius became too small ({r}) during search. Using current value.")
             break

        diameter = 2 * r
        # Count neighbors within the diameter (ball of radius r)
        # Assumes D_ii = inf, so a point is not counted as its own neighbor
        neighbors_mask = dists < diameter
        num_neighbors_per_point = jnp.sum(neighbors_mask, axis=1) # Sum over axis 1 (columns) for each row (point i)
        current_K = jnp.mean(num_neighbors_per_point)
        r_old = r
        if current_K < target_K: r_min = r
        else: r_max = r

        # Update r
        r = (r_min + r_max) / 2
        delta_r = jnp.abs(r - r_old)

        # Check for convergence
        if delta_r < tol:
            print(f"Converged in {i+1} iterations. Final r={r}, avg K={current_K}")
            break
    else: # Loop finished without break (max_iters reached)
        print(f"Warning: Binary search for radius did not converge within {max_iters} iterations.")
        print(f"Final r={r:.4f}, avg K={current_K:.4f}. The goal was K={target_K}.")

    chosen_r = r
    beta = 2 / (r**2)

    aux = {
        "beta": beta,
        "r": chosen_r,
        "current_K": current_K,
        "num_neighbors_per_point": num_neighbors_per_point,
        "dists": dists,
        "neighbors_mask": neighbors_mask,
    }

    return beta, aux

def compute_minima_information(
    Xis: Float[Array, "N d"], # Array of memory vectors (N, D)
    beta: float, # Beta to use for the ball radius
    grad_tol: float = 1e-5, # Tolerance for LSR gradient norm of candidate minimum to be considered a memory
):
    """
    I need to update the `return_true_minima` function to return:
    
    - `unique_minima` :: Float[Array]. Array of all unique local minima
    - `merge_idxs` :: List[Tuple[int, ...]]. Annotation of unique_minima, where each idx is from the original Xis
    - `minima_is_original` :: Bool[Array]. Annotate whether each idx is original memory or not. True whenever len(merge_idxs[i]) == 1
    """
    r = jnp.sqrt(2/beta)
    dists = get_dist_matrix(Xis)
    local_minima = []
    local_minima_idxs = []

    pbar = trange(Xis.shape[0])

    for m in pbar:
        # How many neighbors have balls interacting with current memory m?
        neighbors_mask = dists[m] < 2 * r # Always includes self
        n_neighbors = (neighbors_mask).sum() # All memories in the ball
        pbar.set_postfix(n_neighbors=n_neighbors)
        neighbors_m = Xis[neighbors_mask]

        if n_neighbors == 1:
            print(f"1 neighbor for memory {m}. Using it as centroid.")
            possible_centroids = neighbors_m
        else:
            m_col = neighbors_mask[:m].sum() # Current memory m's column index
            print(f"{n_neighbors} neighbors '{np.where(neighbors_mask)[0]}' for memory {m} (neighbor idx {m_col}). Using all combinations of neighbors as centroids.")

            # Only include combos that include the current memory m
            if n_neighbors > 15:
                # Skip this memory too many neighbors
                print(f"Skipping memory {m} with {n_neighbors} neighbors because it has too many neighbors.")
                continue
            neighbor_combos = get_all_combinations(n_neighbors).astype(bool) # Shape (2^n_neighbors - 1, n_neighbors), no all zero binary vector
            neighbor_possible_minima = neighbor_combos[neighbor_combos[:, m_col]]
            possible_centroids = np.stack([neighbors_m[npm].mean(0) for npm in neighbor_possible_minima])

        for ic, c in enumerate(possible_centroids):
            # Check if centroid is a local minimum
            centroid_dists = get_dist_matrix(c[None], Xis)
            neighbor_idxs = np.where(centroid_dists[0] < r)[0]
            if len(neighbor_idxs) == 0: pass # this centroid was not formed from interacting basins but from neighbors that are >2r apart
            else:
                # print(f"Found {len(neighbor_idxs)} neighbors for memory {m} and centroid {c}")
                local_minima.append(c) # Not a local minimum if the energy gradient is not close to zero
                local_minima_idxs.append(neighbor_idxs)

    local_minima = np.stack(local_minima)
    lmin_array, lmin_unique_idxs, unique_counts = np.unique(local_minima.round(decimals=6), axis=0, return_index=True, return_counts=True)
    local_minima_memory_idxs = [local_minima_idxs[i] for i in lmin_unique_idxs]

    lsr_mem = EpaMemory(eps=0., lmda=0.)
    E, dEdx = lsr_mem.venergy_and_grad(lmin_array, Xis, beta=beta)
    grad_norms = np.linalg.norm(dEdx, axis=-1)
    minima_idxs = np.where(grad_norms < grad_tol)[0]

    lmin_unique_array = lmin_array[minima_idxs]
    lmin_unique_merge_idxs = [local_minima_memory_idxs[m] for m in minima_idxs]

    return lmin_unique_array, lmin_unique_merge_idxs

def get_novel_and_old_minima(lmin_unique_array, lmin_unique_merge_idxs):
    novel_mem_idxs = [i for i, ui in enumerate(lmin_unique_merge_idxs) if len(ui) > 1]
    old_mem_idxs = [i for i, ui in enumerate(lmin_unique_merge_idxs) if len(ui) == 1]

    novel_minima = lmin_unique_array[novel_mem_idxs]
    novel_minima_memory_idxs = [lmin_unique_merge_idxs[i] for i in novel_mem_idxs]
    old_minima = lmin_unique_array[old_mem_idxs]
    old_minima_memory_idxs = [lmin_unique_merge_idxs[i] for i in old_mem_idxs]

    return {
        "novel_minima": novel_minima,
        "novel_minima_memory_idxs": novel_minima_memory_idxs,
        "old_minima": old_minima,
        "old_minima_memory_idxs": old_minima_memory_idxs,
    }

def bwr_imshow(ax, img, **kwargs):
    """
    Helper function for displaying images with a blue-white-red colormap centered at 0.
    """
    # Set default values if not provided in kwargs
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'bwr'
    
    # Determine vmin and vmax to center at 0 if not provided
    if 'vmin' not in kwargs or 'vmax' not in kwargs:
        abs_max = np.max(np.abs(img))
        if 'vmin' not in kwargs:
            kwargs['vmin'] = -abs_max
        if 'vmax' not in kwargs:
            kwargs['vmax'] = abs_max
    
    # Display the image
    im = ax.imshow(img, **kwargs)
    return im
