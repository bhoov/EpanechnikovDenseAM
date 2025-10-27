"""
Experiment to test the effect of beta on the retrieval of memories

python launch_LOGL0v2.py --launch_cpu

# In separate terminal:
python launch_LOGL0v2.py --launch_gpu

(Kill running processes for my user with `pkill python`)

Separate jobs into those needing CPU and those needing GPU
"""
#%%
import os
from itertools import product
from fastcore.parallel import parallel
from typing import *
import pandas as pd
import tyro
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Args:
    """
    Launch CPU or GPU experiments. If launching both, GPU will wait until CPU completes
    """
    test: bool = False # Run a small test run

args = tyro.cli(Args)

if args.test:
    outf = "expresults/LOGL0_test.jsonl"
    N = 3
    nbetas = 3
    depth = 13000 # Still want to get near the local min

    ks = [5]
    true_sigmas = [0.1]
    energy_keys = ["epa", "lse"]
    ds = [8]
    Ms = [5]
    seeds = [3]
else:
    outf = "expresults/LOGL0.jsonl"
    N = 500
    nbetas = 20
    depth = 13000

    ks = [5, 10]
    true_sigmas = [0.1]
    ds = [8, 16]
    Ms = [25, 100]
    seeds = [3,4,5,6,7]
    energy_keys = ["epa", "lse"]

epa_grad_tol = 1e-3
lse_grad_tol = 1e-1

DEFAULT_DEVICE = "cpu"
N_CPU_CORES = 45
sample_from_surface_of_ball = True
epa_do_normal_retrieval = False
do_spectral_clustering = False

try:
    df = pd.read_json(outf, lines=True)
except:
    df = None


def launch_cmd(kwargs, do_system_launch=True):
    beta_idx, M, d, seed, true_sigma, energy_key, k = kwargs['beta_idx'], kwargs['M'], kwargs['d'], kwargs['seed'], kwargs['true_sigma'], kwargs['energy_key'], kwargs['k']
    device = kwargs.get('device', DEFAULT_DEVICE)
    orthogonal_init = kwargs.get('orthogonal_init', False)
    if df is not None:
        subdf = df[
            (df['beta_idx'] == beta_idx) &
            (df['M'] == M) &
            (df['d'] == d) &
            (df['k'] == k) &
            (df['seed'] == seed) &
            (df['true_sigma'] == true_sigma) &
            (df['energy_key'] == energy_key) 
        ]
        
        if len(subdf) > 0:
            print("\tAlready done")
            return None

    cmd_name = f"python exp_LOGL0.py {outf} --beta_idx {beta_idx} --M {M} --d {d} --seed {seed} --true_sigma {true_sigma} --energy_key {energy_key} --device {device} --N {N} --nbetas {nbetas} --depth {depth} --k {k} --orthogonal_init {orthogonal_init} --sample_from_surface_of_ball {sample_from_surface_of_ball} --epa_do_normal_retrieval {epa_do_normal_retrieval} --do_spectral_clustering {do_spectral_clustering} --epa_grad_tol {epa_grad_tol} --lse_grad_tol {lse_grad_tol}"
    print(f"\tLaunching run [{kwargs['_idx']}/{kwargs['_total_exps']}]...")
    if do_system_launch:
        os.system(cmd_name)
    else:
        return cmd_name

def dict_product(*args, **kwargs) -> Iterable[Dict[str, Any]]:
    """Where kwargs are name: iterable pairs. Create an outer product across all iterables"""
    assert len(args) == 0, "Only works on key-value pairs"

    ks, vs = kwargs.keys(), kwargs.values()
    for v in product(*vs):
        yield dict(zip(ks, v))

params = list(dict_product(
        beta_idx=range(nbetas), 
        M=Ms, 
        d=ds, 
        seed=seeds, 
        true_sigma=true_sigmas, 
        energy_key=energy_keys, 
        k=ks
    ))

## Launch experiments
for i, p in enumerate(params):
    p['_idx'] = i
    p['_total_exps'] = len(params)

print(f"Launching {len(params)} experiments")
parallel(launch_cmd, params, n_workers=N_CPU_CORES)
