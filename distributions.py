#%%
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal
from typing import *
from jaxtyping import Float, Array, Int
import equinox as eqx
import jax_utils as ju

class GMMDistribution(eqx.Module):
    means: Float[Array, "k d"]
    covs: Float[Array, "k d d"]
    weights: Float[Array, "k"]
    d: int # Dimension of the data
    k: int # Number of mixtures

    def __init__(self, means, covs, weights):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.d = means.shape[1]
        self.k = means.shape[0]
        assert self.d == self.covs.shape[1] == self.covs.shape[2] == self.means.shape[1]
        assert self.k == self.weights.shape[0] == self.covs.shape[0] == self.means.shape[0]

    @classmethod
    def k_mixtures_in_d(cls, 
                        k:int, # Number of mixtures
                        d:int, # Dimension of the data
                        cvars: Union[float, Float[Array, "d"], Float[Array, "d d"], Float[Array, "k d d"]], # Covariance matrix. Scalar multiplies I(dxd). Vector multiplies diagonal of I(dxd). Matrix is Covariance
                        key:jr.PRNGKey,
                        orthogonal_init: bool = False, # If true, ensure that the means are orthogonal
                        ):
        """Initialize a GMM distribution with k mixtures in d dimensions, where each mode is uniformly sampled from the unit hypercube
        
        Default assumes equal mixing

        Orthogonal constraint applied assuming an origin at the center of the hypercube s.t.
            (gmm.means - 0.5) @ (gmm.means - 0.5).T ~= I
        """
        if orthogonal_init:
            assert k <= d, "Orthogonal initialization requires k <= d"
            # Orthogonal initialization means take eigenvectors of random gaussian matrix, ensuring the eigenvectors are in the hypercube [0,1] for all dims
            means = jr.uniform(key, shape=(d, d))
            means, _ = jnp.linalg.qr(means)
            means = means[:k]

            # Scale the orthogonal vectors to fit in the unit hypercube
            # First scale down to ensure max magnitude in any dimension is ≤ 0.5
            scale = 0.5 / jnp.max(jnp.abs(means))
            means = means * scale
            # Then center in the unit hypercube by adding 0.5
            means = means + 0.5
        else:
            means = jr.uniform(key, shape=(k, d), minval=0, maxval=1)

        if isinstance(cvars, float) or isinstance(cvars, Float[Array, ""]):
            covs = jnp.stack([cvars * jnp.eye(d) for _ in range(k)])
        elif isinstance(cvars, Float[Array, "d"]):
            covs = jnp.stack([jnp.diag(cvars) for _ in range(k)])
        elif isinstance(cvars, Float[Array, "d d"]):
            covs = jnp.stack([cvars for _ in range(k)])
        elif isinstance(cvars, Float[Array, "k d d"]):
            covs = cvars
        else:
            raise ValueError(f"Invalid cvars type")

        weights = jnp.ones(k) / k
        return cls(means, covs, weights)

    def fix_beta(self, beta):
        """Assign beta (1/sigma^2) as the covariance for all gaussians"""
        self = self.at(lambda s: s.covs).set(jnp.stack([1/beta * jnp.eye(self.d) for _ in range(self.k)]))
        return self
    
    def log_pdf(self, x):
        """
        The formula for GMM probability is:

        p(x) = Σᵢ wᵢ * N(x|μᵢ,Σᵢ)

        In log space, this becomes:

        log p(x) = log(Σᵢ wᵢ * N(x|μᵢ,Σᵢ)) = logsumexp(log(wᵢ) + log(N(x|μᵢ,Σᵢ)))
        """
        # Calculate log prob for each component
        component_log_probs = jnp.array([
            multivariate_normal.logpdf(x, mean, cov) 
            for mean, cov in zip(self.means, self.covs)
        ])
        # Add log weights and use logsumexp for numerical stability
        return jax.scipy.special.logsumexp(jnp.log(self.weights) + component_log_probs)
    
    def sample(self, key, n_samples):
        """
        Sample from the GMM distribution.
        """
        component_key, rng = jr.split(key)
        # Sample component index corresponding to the gaussian that we should sample from, according to the weights
        components = jr.categorical(component_key, jnp.log(self.weights), shape=(n_samples,))
        
        # Sample from the selected components
        samples = np.zeros((n_samples, self.d))
        for i in range(len(self.weights)):
            mask = components == i
            n_i = np.sum(mask)
            if n_i > 0:
                key, rng = jr.split(rng)
                component_samples = jr.multivariate_normal(
                    key, 
                    self.means[i], 
                    self.covs[i], 
                    shape=(n_i,)
                )
                samples[mask] = component_samples
        return samples



#%% Test distributions
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # means = jnp.array([[0, 0], [6, 6]])
    # covs =  0.4 * jnp.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    # weights = jnp.array([0.1, 0.9])
    # weights = weights / jnp.sum(weights)
    # gmm = GMMDistribution(means, covs, weights)

    key = jr.PRNGKey(0)
    gmm = GMMDistribution.k_mixtures_in_d(2, 2, 0.02**2, jr.PRNGKey(8))

    gmm = gmm.fix_beta(1/(0.05**2))

    x = gmm.sample(key, 1000)

    # Create a grid of points
    x_range = np.linspace(-0.5, 1.5, 500)
    y_range = np.linspace(-0.5, 1.5, 500)
    X, Y = np.meshgrid(x_range, y_range)
    xy_grid = np.stack([X.flatten(), Y.flatten()], axis=1)

    # Evaluate log_pdf at each grid point
    log_probs = jax.vmap(gmm.log_pdf)(xy_grid)
    Z = log_probs.reshape(X.shape)

    # Create contour plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Log Probability Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('GMM Log Probability Density')

    # Plot the sampled points
    key = jr.PRNGKey(1)
    x = gmm.sample(key, 5000)
    plt.scatter(x[:, 0], x[:, 1], c='red', alpha=0.1, s=1)

    # Plot the means as purple stars
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], 
            c='purple', marker='*', s=200, 
            label='Component Means', zorder=3)
    plt.legend()
    plt.show()

    # %% Test orthogonal init
    key = jr.PRNGKey(0)
    gmm = GMMDistribution.k_mixtures_in_d(2, 4, 0.02**2, jr.PRNGKey(0), orthogonal_init=True)
    gmm = gmm.fix_beta(1/(0.05**2))
    x = gmm.sample(key, 1000)
    plt.scatter(x[:, 0], x[:, 1], c='red', alpha=0.1, s=1)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.show()

    (gmm.means - 0.5) @ (gmm.means - 0.5).T

    # assert jnp.allclose(gmm.means @ gmm.means.T, jnp.eye(2))


    # %%
    k = 3
    d = 4
    key = jr.PRNGKey(0)

    means = jr.normal(key, shape=(d, d))

    Q, R = jnp.linalg.qr(means)

    means = Q[:k]
    (means.T @ means) # Column vectors?

    means.max()
    Q.shape
    R.shape
