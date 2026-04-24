# Implementation Plan: Barycentric 3-Image Interpolation

## Goal

Extend `interpolate_qc` to interpolate between **3 images** instead of 2, using recursive triangle subdivision. Each intermediate frame is generated at the centroid (masspoint) of a subtriangle, with its barycentric coordinate used as the interpolation fraction.
Write it in a seperate file for clairty, though it should be inside cm.py logically.

---

## Reference: How 1D Bisection Works (interpolate_qc)

| Concept | 1D |
|---|---|
| Input | 2 endpoint images |
| Structure | Binary tree of intervals |
| New point per level | Midpoint of interval |
| Local fraction | 0.5 (always midpoint) |
| Global fraction (`frac`) | `frame_ix / (num_frames - 1)` |
| Conditioning | `slerp(cond1, cond2, frac)` |
| Noisy latent | Shared noise added to both neighbors, interpolated at `frac=0.5` |
| Level ordering | Coarse → fine (large `cur_step` → small) |

---

## 2D Extension: Barycentric Subdivision

### Input
- 3 images: `img1`, `img2`, `img3` — the triangle vertices
- 3 optimised conditionings: `cond1`, `cond2`, `cond3`

### Subdivision Scheme: Star (Barycentric) Subdivision

Replace each triangle with 3 children by inserting the centroid as a new vertex.

```
Level 0:  1 triangle  (v1, v2, v3)          — 3 known endpoint frames
Level 1:  generate centroid g0              — 1 new frame, 3 subtriangles
Level 2:  generate 3 centroids             — 3 new frames, 9 subtriangles
Level L:  3^(L-1) new frames,  3^L triangles
```

Total interior frames after L levels: `(3^L − 1) / 2`

**Why star subdivision over 4-way midpoint subdivision?**
In 4-way subdivision, edge midpoints must exist before triangle centroids can be computed — a circular dependency within the same level. Star subdivision avoids this: the centroid only uses the 3 parent vertices, which are always already computed.

### Per-Frame Generation (at centroid `g = (λ1, λ2, λ3)`)

**1. Conditioning** — global barycentric position in semantic triangle:
```
cond_g = λ1·cond1 + λ2·cond2 + λ3·cond3
```
Analogous to `slerp(cond1, cond2, frac)` in 1D.

**2. Noisy latent** — shared noise added to all 3 parent vertex latents, then averaged (local weight = 1/3 each, since g is the centroid):
```
noise = randn_like(L1)                             # shared across all 3
noisy_a = sqrt_alphas[t] * La + sqrt_one_minus[t] * noise
noisy_b = sqrt_alphas[t] * Lb + sqrt_one_minus[t] * noise
noisy_c = sqrt_alphas[t] * Lc + sqrt_one_minus[t] * noise
noisy_latent = (noisy_a + noisy_b + noisy_c) / 3
```
Analogous to `interp(noisy_l1, noisy_l2, 0.5)` in 1D.

**3. Partial DDIM denoising** — same as 1D:
```python
samples, _ = ddim_sampler.sample(
    ddim_steps, 1, shape, cond_g,
    x_T=noisy_latent, timesteps=cur_step, ...
)
```

**4. Star subdivision** — replace `(a, b, c)` with 3 children:
```
(a, b, g),  (b, c, g),  (a, c, g)
```

---

## Step Schedule

Reuse `get_step_schedule(min_steps, max_steps, num_levels)` unchanged.
- Level 1 (coarsest): largest `cur_step` → most denoising
- Level L (finest): smallest `cur_step` → least denoising

---

## Conditioning Optimisation: `learn_conditioning_triple`

Extend `learn_conditioning` from 2 images to 3. Same structure:
- 3 free parameters: `cond1`, `cond2`, `cond3` (all initialised from prompt embedding)
- 1 shared `uncond` (also optimised)
- Per iteration: one Adam step per `(image, cond)` pair (3 steps total), same diffusion denoising loss
- Same augmentation: `ColorJitter + RandomResizedCrop`

---

## Data Structures

```
latent_map : dict  (λ1, λ2, λ3) → latent tensor
             initialised with v1→L1, v2→L2, v3→L3

triangles  : list of (a, b, c) barycentric coord tuples
             starts as [(v1, v2, v3)], grows 3× per level
```

---

## Output

Frames saved as `L{level}_{λ1:.3f}_{λ2:.3f}_{λ3:.3f}.png`.
Endpoint frames saved as `1.000_0.000_0.000.png`, etc.

---

## File Structure

- `cm_barycentric.py` — new file containing:
  - `learn_conditioning_triple()`
  - `interpolate_barycentric()`
- Imports `get_step_schedule`, `slerp`, `interpolate_linear` from `cm.py`

---

## Open Questions

1. **Slerp vs linear for conditioning mix** — 1D uses slerp; weighted average of 3 tensors generalises naturally but slerp doesn't extend to 3 points without a defined mixing path.
2. **n_choices / CLIP selection** — the 1D version generates multiple candidates and picks by CLIP score. Could be applied per centroid independently.
3. **ControlNet control signal** — 1D interpolates pose maps between 2 endpoints. For 3 images this requires barycentric blending of 3 pose maps, which needs a defined triangle interpolation for pose keypoints.
