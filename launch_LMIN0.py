""" Experiment to test the effect of beta, data dimension, and number of stored patterns vs
the number of memories in LSR energy.

(Kill parallel running processes for my user with `pkill python`)
"""
#%%
import os
from itertools import product
from fastcore.parallel import parallel
from typing import *
import pandas as pd
import tyro
from dataclasses import dataclass

@dataclass
class Config:
    test: bool = False

args = tyro.cli(Config)

# Sort from hardest to easiest
if args.test:
    nbetas = 3
    Ms = [5]
    ds = [4]
    seeds = [0]
    outf = "expresults/LMIN0_test.jsonl"
else:
    nbetas = 50
    Ms = [25,15,5]
    ds = [32, 8]
    seeds = [0, 42, 88]
    outf = "expresults/LMIN0.jsonl"

DEFAULT_DEVICE = "cpu"
N_CPU_CORES = 16

try:
    df = pd.read_json(outf, lines=True)
except:
    df = None

def launch_cmd(kwargs):
    beta_idx, M, d, seed = kwargs['beta_idx'], kwargs['M'], kwargs['d'], kwargs['seed']
    if df is not None:
        subdf = df[(df['beta_idx'] == beta_idx) & (df['M'] == M) & (df['d'] == d) & (df['seed'] == seed)]
        print(len(subdf))
        if len(subdf) > 0:
            print("\tAlready done")
            return

    cmd_name = f"python exp_LMIN0.py {outf} --beta_idx {beta_idx} --M {M} --d {d} --seed {seed} --device {DEFAULT_DEVICE}"
    print(f"\tLaunching run [{kwargs['idx']}/{kwargs['N']}]...")
    os.system(cmd_name)

def dict_product(*args, **kwargs) -> Iterable[Dict[str, Any]]:
    """Where kwargs are name: iterable pairs. Create an outer product across all iterables"""
    assert len(args) == 0, "Only works on key-value pairs"

    ks, vs = kwargs.keys(), kwargs.values()
    for v in product(*vs):
        yield dict(zip(ks, v))

params = list(dict_product(beta_idx=range(nbetas), M=Ms, d=ds, seed=seeds))
for i, p in enumerate(params):
    p['idx'] = i
    p['N'] = len(params)

print(f"Launching {len(params)} experiments")
parallel(launch_cmd, params, n_workers=N_CPU_CORES)