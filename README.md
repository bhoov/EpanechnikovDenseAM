# Dense Associative Memory with Epanechnikov Energy
> Code and experiments accompanying the paper of the same name, accepted as a spotlight🥇 to NeurIPS 2025

## Getting Started

([install uv](https://docs.astral.sh/uv/getting-started/installation/), if needed)

```
uv sync
```

Test the devices on the environment to ensure that GPU is recognized and working properly:

```
uv run python test_jax_torch.py 
```

## Experiment glossary

### Recreate Fig 1 of the paper

```
uv run python eval_1d2d.py
```

Outputs are saved to `figures/fig1_1d.png` and `figures/fig1_2d.png`.

### Recreate "All Kernel Emergence" tests in appendix [AKEM]

```
uv run python eval_1d2d.py
```

Output saved to `figures/AKEM_all_kernel_emergence.png`.

