# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of [ControlNet](https://github.com/lllyasviel/ControlNet) adapted for image-to-image interpolation using diffusion models, as described in ["Interpolating between Images with Diffusion Models"](https://arxiv.org/abs/2307.12560). The core contribution is a hierarchical bisection algorithm that interpolates in latent space using partial DDIM denoising steps.

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate control
```

Python 3.8.5, PyTorch 1.12.1, CUDA 11.3. No test suite or linting config exists — this is a research codebase.

## Running Interpolation

The `ContextManager` class in `cm.py` is the main entry point. Sample scripts in `sample_scripts/` show typical usage:

```python
from controlnet import cm  # this repo is imported as a package named 'controlnet'

CM = cm.ContextManager(version='2.1')  # or '1.5'
img1 = Image.open('img1.png').convert('RGB').resize((512, 512))
img2 = Image.open('img2.png').convert('RGB').resize((512, 512))

# Primary method: hierarchical bisection interpolation
CM.interpolate(img1, img2, prompt='...', n_prompt='...', num_frames=17, out_dir='output')

# num_frames must satisfy: (num_frames - 1) is a power of 2 (e.g. 5, 9, 17, 33, 65)
```

Sample scripts expect model checkpoints at `./controlnet/models/` (set via `sys.path` to `$NFS/code/controlnet/controlnet`). Adjust paths for your environment.

## Architecture

### Core Interpolation (`cm.py`)

`ContextManager` exposes four interpolation methods:
- `interpolate()` — **main method**: hierarchical bisection over noise levels, processes frames level-by-level (coarse → fine). `num_frames - 1` must be a power of 2.
- `interpolate_then_diffuse()` — simpler sequential pass, no power-of-2 constraint on frame count.
- `interpolate_qc()` — like `interpolate()` but generates `n_choices` candidates per frame and selects by CLIP score or manual input.
- `interpolate_naive()` — baseline: spherical interpolation of clean latents without any denoising.

The key mechanism: given two endpoint images, `get_latent_stack()` builds a sequence of progressively noisier latents for each image by adding noise incrementally (not independently — `share_noise=True` uses the same noise realization). At each refinement level, intermediate frame latents are interpolated between these pre-noised endpoints, then denoised for just the timestep range of that level using `ddim_sampler.p_sample_ddim()`.

`learn_conditioning()` optionally optimizes per-image text embeddings via gradient descent to better reconstruct each endpoint image, improving identity preservation.

### Modified DDIM Sampler (`cldm/ddim_hacked.py`)

Extends the upstream `DDIMSampler` with:
- `sample(..., timesteps=K)` — starts denoising from step `K` instead of from pure noise.
- `p_sample_ddim()` — exposed as a public method for single-step denoising, used directly in the main interpolation loop.
- `pred_eps()` — predicts the noise given a noisy latent, used during conditioning optimization.

### ControlNet Model (`cldm/cldm.py`)

- `ControlledUnetModel` — the frozen SD U-Net decoder that receives residuals from the ControlNet encoder.
- `ControlLDM` — wraps both SD and the ControlNet encoder; `control_scales` (list of 13 floats) modulates the strength of each injected feature map. Set to `[1]*13` for full control strength.

### Annotators (`annotator/`)

Image preprocessors for extracting control signals: `canny`, `hed`, `midas` (depth), `mlsd` (lines), `openpose` (pose), `uniformer` (segmentation). `ContextManager.change_mode()` lazy-loads the relevant annotator and swaps model weights accordingly.

### Model Configs (`models/`)

`cldm_v15.yaml` and `cldm_v21.yaml` define the full ControlNet+SD architecture via OmegaConf. Models are instantiated with `cldm.model.create_model(config_path)` then weights loaded separately.

## Training a New ControlNet

1. Initialize weights from an SD checkpoint:
   ```bash
   python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
   # or for SD2:
   python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt
   ```

2. Prepare dataset in `tutorial_dataset.py` format: each item is `{jpg: [-1,1] target, hint: [0,1] control, txt: prompt}`.

3. Train:
   ```bash
   python tutorial_train.py      # SD1.5
   python tutorial_train_sd21.py # SD2.1
   ```
   Key training flags: `sd_locked=True` (freeze SD decoder), `only_mid_control=False` (use all ControlNet outputs).

## Memory

Set `save_memory = True` in `config.py` to enable sliced attention (reduces VRAM at cost of speed). This is read by `share.py` which is imported at the top of most scripts.
