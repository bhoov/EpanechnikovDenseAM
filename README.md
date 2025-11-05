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

### Enumerating **L**ocal **MIN**ima (LMIN0)

> How does the number of memories in the LSR energy scale with the number of memories and data dimension?

```bash
uv run python launch_LMIN0.py # Output is saved in `expresults/LMIN0.jsonl`
# uv run python launch_LMIN0.py --test # for small test run

uv run python eval_LMIN0.py # Output figure is saved in `figures/LMIN0.png`
# uv run python eval_LMIN0.py --dataf expresults/LMIN0_test.jsonl # For small test run
```

Output is saved in `expresults/LMIN0.jsonl`

### Evaluating LogLikelihood Quality (LOGL0)

> How well do memories in LSR energy model the log-likelihood of some true distribution (vs. LSE memories, which are theoretically optimal but not diverse)?

```bash
uv run python launch_LOGL0.py # Output data is saved in `expresults/LOGL0.jsonl`
# uv run python launch_LOGL0.py --test # for small test run

uv run python eval_LOGL0.py # Output figures are saved in `figures/LOGL0__*.png`
# uv run python eval_LOGL0.py --dataf expresults/LOGL0_test.jsonl # for small test run

uv run python eval_LOGL0_poster.py # Output figures saved in `figures/LOGL0poster*.svg`
```

### Characterize **A**ll **K**ernel **EM**ergence behavior (AKEM)

```
uv run python exp_AKEM_all_kernel_emergence.py
```

Output saved to `figures/AKEM_all_kernel_emergence.png`.

### **B**ig **STO**rage and Emergence in MNIST (BSTOR)

```
uv run python BSTOR_big_mnist_store.py
```

Output figures saved in `figures/BSTOR*.png`

### **P**i**X**e**L** **E**mergence (PXLE)

```
uv run python PXLE_pixel_emergence.py # Outputs saved to `figures/PXLE*.png`
```