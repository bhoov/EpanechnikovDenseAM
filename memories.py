#%%
"""
Coming up with unified helper functions for my Associative Memories
"""
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import *
from jaxtyping import Float, Array, UInt, Bool
import equinox as eqx
import jax.random as jr
from fastcore.meta import delegates
import numpy as np
import functools as ft
from pydantic import BaseModel
from tqdm.auto import trange

#%%
def epa_energy(q: Float[Array, "D"], # query
               Xi: Float[Array, "M D"], # memories
               beta: Float = 0.5, # beta scaling param
               eps: Float = 0., # numerical stability in the log
               lmda: Float = 0. # L2 regularization on the query
               ):
    return (
        -(1 / beta * jnp.log(jnp.sum(jax.nn.relu(1 - 0.5 * beta * jnp.sum((q - Xi)**2, axis=-1))) + eps)) + (lmda * jnp.sum(q**2))
    )

def lse_energy(q: Float[Array, "D"], Xi: Float[Array, "M D"], beta: Float = 0.5):
    return (
        -1 / beta * jax.nn.logsumexp(
            -0.5 * beta * jnp.sum((q - Xi)**2, axis=-1)
        )
    )

def epa_or_lse_energy(q: Float[Array, "D"], Xi: Float[Array, "M D"], beta: Float = 0.5, eps: Float = 0.0):
    scaled_dists = 0.5 * beta * jnp.sum((q - Xi)**2, axis=-1)
    E_epa = -1 / beta * jnp.log(jnp.sum(jax.nn.relu(1 - beta * scaled_dists)+eps))
    E_gauss = -1 / beta * jax.nn.logsumexp(-beta * scaled_dists)
    E = jax.lax.select(jnp.isinf(E_epa), E_gauss, E_epa)
    return E

class AssociativeMemory(eqx.Module, ABC):
    """Defines the interface and basic methods for using Associative Memories"""
    default_energy_kwargs: Dict[str, Any] = eqx.field(default_factory=dict)

    def __init__(self, **default_energy_kwargs):
        self.default_energy_kwargs = default_energy_kwargs

    @abstractmethod
    def energy(self, x: Float[Array, "D"], memories: Float[Array, "M D"], **kwargs) -> Float[Array, ""]: ...

    @delegates(energy)
    def venergy(self, x: Float[Array, "B D"], memories: Float[Array, "M D"], **kwargs) -> Float[Array, "b"]: 
        kwargs = {**self.default_energy_kwargs, **kwargs}
        energy_fn = ft.partial(self.energy, **kwargs)
        return jax.vmap(energy_fn, in_axes=(0, None))(x, memories)

    @delegates(energy)
    def energy_and_grad(self, x: Float[Array, "D"], memories: Float[Array, "M D"], **kwargs) -> Tuple[Float[Array, ""], Float[Array, "D"]]:
        kwargs = {**self.default_energy_kwargs, **kwargs}
        energy_fn = ft.partial(self.energy, **kwargs)
        return jax.value_and_grad(energy_fn)(x, memories)

    @delegates(energy_and_grad)
    def venergy_and_grad(self, x: Float[Array, "B D"], memories: Float[Array, "M D"], **kwargs) -> Tuple[Float[Array, "B"], Float[Array, "B D"]]:
        energy_fn = ft.partial(self.energy, **kwargs)
        return jax.vmap(jax.value_and_grad(energy_fn), in_axes=(0, None))(x, memories)

    @delegates(venergy_and_grad)
    def venergy_and_grad_batched(self, x: Float[Array, "B D"], memories: Float[Array, "M D"], batch_size=128, **kwargs) -> Tuple[Float[Array, "B"], Float[Array, "B D"]]:
        n_samples = x.shape[0]
        Es = np.empty((n_samples,))
        dEs = np.empty((n_samples, x.shape[-1]))
        energy_fn = jax.jit(ft.partial(self.venergy_and_grad, **kwargs))

        for i in trange(0, n_samples, batch_size):
            start_idx = i
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = x[start_idx:end_idx]
            Es_batch, dEs_batch = energy_fn(x_batch, memories)
            Es[start_idx:end_idx] = Es_batch
            dEs[start_idx:end_idx] = dEs_batch

        return Es, dEs

    @property
    def beta(self): return self.default_energy_kwargs.get('beta', None)

    @delegates(energy)
    def recall(self, 
               q: Float[Array, "D"], # Initial query
               memories: Float[Array, "M D"], # Arg to the `self.energy` function
               depth: int=1000, # Number of steps to run
               return_grads: bool = False, # Whether to return gradients
               clamp_idxs: Optional[Bool[Array, "D"]]=None, # Whether to clamp the gradients
               beta_schedule: Optional[Union[float, Callable[[int], float]]] = 0.5, # Beta schedule, defaults to constant 0.5
               alpha_schedule: Optional[Union[float, Callable[[int], float]]] = 0.1, # Alpha schedule, defaults to constant 0.1
               noise_schedule: Optional[Union[float, Callable[[int], float]]] = 0.0, # Noise schedule, defaults to deterministic 0.0
               grad_tol: float = 1e-6, # Tolerance for gradient descent to be considered converged
               collect_states: bool = False, # Whether to collect states throughout trajectory
               key: Optional[jr.PRNGKey] = None, # RNG key, defaults to new key
               **kwargs) -> Tuple[Float[Array, "D"], Dict[str, Any]]:
        """Run energy dynamics using `jax.lax.scan`"""
        # Preprocess schedules
        if isinstance(beta_schedule, float):
            beta_sched = lambda i: beta_schedule
        else:
            beta_sched = beta_schedule

        if isinstance(alpha_schedule, float):
            alpha_sched = lambda i: float(alpha_schedule)
        else:
            alpha_sched = alpha_schedule

        if isinstance(noise_schedule, float):
            noise_sched = lambda i: float(noise_schedule)
        else:
            noise_sched = noise_schedule

        dEdxf = jax.jit(jax.value_and_grad(ft.partial(self.energy, **kwargs)))
        def step(state, i):
            x, rng = state["x"], state["rng"]
            key, rng = jr.split(rng)
            beta = beta_sched(i) 
            alpha = alpha_sched(i)
            noise = noise_sched(i)
            E, dEdx = dEdxf(x, memories, beta=beta)
            if clamp_idxs is not None:
                dEdx = jnp.where(clamp_idxs, 0, dEdx)
            x = x - alpha * (dEdx + noise * jr.normal(key, shape=dEdx.shape))
            is_converged = jnp.max(jnp.abs(dEdx)) < grad_tol
            aux = {"E": E, "beta": beta, "alpha": alpha, "noise": noise, "is_converged": is_converged}
            if return_grads:
                aux["dEdx"] = dEdx
            if collect_states:
                aux["x"] = x
            next_state = {"x": x, "rng": rng}
            return next_state, aux

        rng = jr.PRNGKey(0) if key is None else key
        state = {
            "x": q,
            "rng": rng,
        }
        state, aux = jax.lax.scan(step, state, jnp.arange(depth))
        x = state["x"]
        return x, aux

    @delegates(recall)
    def vrecall( self, q: Float[Array, "B D"], memories: Float[Array, "M D"], **kwargs) -> Float[Array, "D"]: 
        """Run energy dynamics with simple gradient descent on a batch of queries """
        recallf = ft.partial(self.recall, **kwargs)
        return jax.vmap(recallf, in_axes=(0, None))(q, memories)

class EpaMemory(AssociativeMemory):
    """A basic Epanechnikov Associative Memory"""

    def energy(self, x: Float[Array, "D"], memories: Float[Array, "M D"], **kwargs) -> Float[Array, ""]:
        kwargs = {**self.default_energy_kwargs, **kwargs}
        return epa_energy(x, memories, **kwargs)

    def custom_retrieval(self, 
                        x: Float[Array, "D"], 
                        memories: Float[Array, "M D"], 
                        # grad_thresh: float = 1e-9, # Consider the point a local minima if its gradient is below this threshold
                        **kwargs) -> Float[Array, ""]:
        """Custom retrieval function for EPA memories, led by the intution that energy minima 
        occur at the centroid of memories whose basins overlap

        Algorithm:
        - Starting with x:
        - Compute \grad_z E(x) and find S(z) = { xi_m :  I( beta/2 || z - xi_m ||^2 <= 1) } --- the set of memory basin that the point is in
        - Set z <- mean(S(z))
        - Repeat until || \grad_z E(z) || is close to zero or the set S(z) does not change.

        Will return `nan` if the point is not in any basin
        """
        beta = kwargs.get("beta", self.default_energy_kwargs.get("beta", 0.5)) # Tied to default of energy function above...
        init_val = {
            "x": x,
            "niter": 0,
            "S": -jnp.ones(memories.shape[0], dtype=jnp.int8),
            "S_prev": -jnp.ones(memories.shape[0], dtype=jnp.int8),
        }
    
        def cond_fun(state):
            # Continue if S is all negative (first iteration) or if S has changed
            return jnp.logical_or(
                jnp.all(state["S"] == -1),
                jnp.any(state["S"] != state["S_prev"])
            )

        def body_fun(state):
            x = state["x"]
            dists = jnp.sqrt(jnp.sum((x - memories)**2, axis=-1))
            S = (beta/2 * dists**2 <= 1).astype(jnp.int8)
            # jax.debug.print("Iteration {}: S= {}, dists2= {}, dists= {}", state["niter"], S, dists**2, dists)

            nS = jnp.sum(S)
            S2 = jnp.tile(S[..., None], (1, memories.shape[-1]))
            Smems = jnp.where(S2, memories, 0.)
            new_x = jnp.sum(Smems, axis=0) / nS
            return {"x": new_x, "S": S, "S_prev": state["S"], "niter": state["niter"] + 1}

        final_state = jax.lax.while_loop(cond_fun, body_fun, init_val)
        aux = {
            "out_basins": final_state["S"],
            "niter": final_state["niter"]
        }
        return final_state["x"], aux

    @delegates(custom_retrieval)
    def vcustom_retrieval(self, x: Float[Array, "B D"], memories: Float[Array, "M D"], **kwargs) -> Float[Array, "B D"]:
        custom_retrievalf = ft.partial(self.custom_retrieval, **kwargs)
        return jax.vmap(custom_retrievalf, in_axes=(0, None))(x, memories)

class LseMemory(AssociativeMemory):
    """A basic Gaussian Associative Memory"""

    def energy(self, x: Float[Array, "D"], memories: Float[Array, "M D"], **kwargs) -> Float[Array, ""]:
        kwargs = {**self.default_energy_kwargs, **kwargs}
        return lse_energy(x, memories, **kwargs)
    
class EpaOrLseMemory(AssociativeMemory):
    """An Epanechnikov or Gaussian Associative Memory, Epanechnikov for all regions near the memory, gaussian elsewhere"""
    def energy(self, x: Float[Array, "D"], memories: Float[Array, "M D"], **kwargs) -> Float[Array, ""]:
        kwargs = {**self.default_energy_kwargs, **kwargs}
        return epa_or_lse_energy(x, memories, **kwargs)