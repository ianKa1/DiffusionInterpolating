# `ddim_sampler.sample()` — Argument Reference

Call site: `cm.py:565` inside `interpolate_qc()`.

```python
samples, _ = self.ddim_sampler.sample(
    ddim_steps, 1,
    shape, cond, verbose=False, eta=ddim_eta,
    x_T=noisy_latent, timesteps=cur_step,
    unconditional_guidance_scale=guide_scale,
    unconditional_conditioning=un_cond
)
```

## Arguments

### `S` (`ddim_steps`)
Total number of DDIM steps to schedule. Controls how finely the [0, T] noise interval is discretized. More steps = slower sampling but smoother denoising trajectory. Note: the actual number of steps *executed* is controlled by `timesteps`, not `S` — `S` just sets up the schedule.

### `batch_size = 1`
Number of latents to denoise in parallel. Fixed at 1 here since frames are processed one at a time.

### `shape`
Latent spatial dimensions as `(C, H, W)` — e.g. `(4, 32, 32)` for a 256×256 output image. The VAE downsamples by 8×, so a 256×256 image lives in a 32×32 latent grid with 4 channels.

### `conditioning` (`cond`)
Positive conditioning dict passed to the U-Net at every denoising step. Contains:
- `c_crossattn`: CLIP text embeddings — steers content toward the prompt.
- `c_concat`: Control signal (e.g. pose map, canny edges) — steers structure via ControlNet.

### `verbose = False`
Suppresses per-step progress printing inside the sampler loop.

### `eta` (`ddim_eta`)
Stochasticity injected at each denoising step.
- `0` = fully deterministic DDIM: given the same `x_T`, always produces the same output.
- `1` = fully stochastic DDPM.
Interpolation uses `0` so that results are reproducible and vary smoothly with the interpolated `x_T`.

### `x_T` (`noisy_latent`)
The starting latent to denoise from — the core of the interpolation mechanism. Rather than pure Gaussian noise, this is a spherical/linear interpolation of the two endpoint image latents, each pre-noised to noise level `cur_step`. The fraction of the mix corresponds to the frame's position between the two endpoints.

### `timesteps` (`cur_step`)
Which noise level to begin denoising from. The `ddim_hacked.py` modification enables **partial denoising**: the sampler starts at step `cur_step` instead of at the end of the full schedule. Coarse refinement levels use a high `cur_step` (many denoising steps applied); fine levels use a low `cur_step` (only a few steps, preserving detail already established at coarser levels).

### `unconditional_guidance_scale` (`guide_scale`)
Classifier-free guidance (CFG) scale. At each step the update is:

```
output = uncond + guide_scale × (cond − uncond)
```

Typical value: `7.5`. Higher values push the output harder toward the prompt at the cost of diversity and potential artifacts.

### `unconditional_conditioning` (`un_cond`)
Negative conditioning dict, same structure as `cond`. Usually encodes an empty or negative prompt. Used as the baseline in CFG — the model steers *away* from this and *toward* `cond` proportionally to `guide_scale`.
