# Dense Associative Memory with Epanechnikov Energy

> Code and experiments accompanying the paper of the same name, accepted as a **spotlight🥇** to NeurIPS 2025

<div align="center">
  <img src="https://raw.githubusercontent.com/bhoov/EpanechnikovDenseAM/refs/heads/main/assets/big-epa-thumbnail.png"
  alt="Dense Associative Memory with Epanechnikov Energy" width="800"/>

We propose a novel energy function for Dense Associative Memory (DenseAM) networks, the log-sum-ReLU (LSR), inspired by optimal kernel density estimation. Unlike the common log-sum-exponential (LSE) function, LSR is based on the Epanechnikov kernel and enables exact memory retrieval with exponential capacity without requiring exponential separation functions. Moreover, it introduces abundant additional emergent local minima while preserving perfect pattern recovery — a characteristic previously unseen in DenseAM literature. Empirical results show that LSR energy has significantly more local minima (memories) that have comparable log-likelihood to LSE-based models. Analysis of LSR’s emergent memories on image datasets reveals a degree of creativity and novelty, hinting at this method’s potential for both large-scale memory storage and generative tasks.
</div>

## Getting Starte

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

### Qualitative latent emergence with VAEs (QBVAE)

We use two different techniques for two datasets:

1. MNIST -- we train our own VAE 
2. Tiny Imagenet -- we use a pretrained VAE that is used as the encoder/decoder of standard diffusion models.


**MNIST**

Train and evaluate the model. Pretty fast process

```
uv run accelerate launch --mixed_precision="fp16" QBVAE1_training.py mnist10
uv run python QBVAE3_latent_mem_retrieval.py mnist10
```

The trained model is default saved and loaded to folder `expresults/QBVAE--beta-vae-mnist10.pt`.
Output figures are saved to `figures/QBVAE--mnist10-mem-retrieval/*.png"

**Tiny Imagenet**

```
uv run python QBVAE3_latent_mem_retrieval.py tinyimagenet256
```

## Cite our work

```
@inproceedings{
hoover2025dense,
title={Dense Associative Memory with Epanechnikov Energy},
author={Benjamin Hoover and Zhaoyang Shi and Krishna Balasubramanian and Dmitry Krotov and Parikshit Ram},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=ZbQ5Zq3zA3}
}
```