# Interpolating between Images with Diffusion Models

**Authors**: Clinton J. Wang, Polina Golland (MIT CSAIL)
**arXiv**: 2307.12560v1 [cs.CV] 24 Jul 2023

---

## Area & Scope

This is a **generative modeling / image synthesis** paper that tackles the creative editing frontier of interpolating between arbitrary real images. It sits within the broader context of controllable image generation with diffusion models, specifically addressing a gap in existing text-to-image pipelines: no deployed system can smoothly transition between two real input images that differ significantly in style, content, or pose.

## Problem

Existing interpolation techniques are severely limited:
- **GAN-based methods** (StyleGAN, etc.) have rich latent spaces with nice interpolation properties but fail at **GAN inversion** — they can't faithfully reconstruct arbitrary real images, only generated ones from narrow domains (e.g., aligned faces).
- **Video interpolation** methods assume temporal continuity and similar styles, not drastic semantic or stylistic shifts.
- **Style transfer** techniques don't handle gradual content transformations across many frames.
- **Latent diffusion interpolation** (e.g., Stable Diffusion latent blending) has been demonstrated only for *generated* images, not real ones.

The paper aims to enable high-quality interpolations between real images with **diverse styles, layouts, subjects, and poses** — something no prior method achieves.

## Contribution

The authors claim the following contributions:

1. **A zero-shot interpolation pipeline** for latent diffusion models that operates on real images.
2. **Textual inversion** to adapt text embeddings to each input image, enabling better conditioning during interpolation.
3. **Pose conditioning** via ControlNet to handle subjects in different poses, preventing anatomical artifacts.
4. **CLIP-based candidate ranking** to select the highest-quality interpolation among multiple random seeds.
5. **Branching noise schedule** that couples noise locally (between parent-child frames) but decouples siblings, yielding smooth yet creative transitions.

**Assessment**: The novelty is primarily in the *combination and adaptation* of existing techniques (textual inversion, ControlNet, CLIP ranking) for the specific task of real image interpolation. The branching noise schedule is a clever engineering choice. The problem itself is underexplored and the results are impressive, but most individual components are borrowed from prior work. The contribution is **incremental but well-motivated** — it fills a real gap in creative workflows.

## General Pipeline

1. **Initialization**: Encode both input images into latent space: z₀⁰ = E(x₀), z₀ᴺ = E(xₙ).
2. **Textual inversion**: Fine-tune text embeddings for each input image by optimizing them to reconstruct the image at random noise levels (100–500 gradient steps).
3. **Pose extraction** (optional): Use OpenPose to get keypoints for each input. For stylized images, first translate to photorealistic style before pose estimation.
4. **Iterative interpolation** (branching structure):
   - Start with the two inputs at high noise level tₖ.
   - Interpolate their noisy latents using slerp to generate the midpoint.
   - Denoise the midpoint to a lower noise level.
   - Recursively repeat for each pair of neighbors, using progressively lower noise levels as frames get closer together.
5. **Conditioning during denoising**: Interpolate text embeddings and poses between parent frames. Feed these to the denoising U-Net via ControlNet.
6. **CLIP ranking** (optional): Generate multiple candidates with different random seeds, rank by CLIP similarity to quality prompts (e.g., "high quality, detailed" vs. "blurry, distorted").
7. **Decode** final latents to pixel space.

The branching structure is key: noise is shared between a frame and its two parents, but *not* between siblings. This keeps adjacent frames smooth while allowing creative variation.

## Advantages over Prior Work

- **GAN interpolation** (StyleGAN, etc.): Breaks down on real images due to poor inversion. This method uses diffusion models with much better reconstruction fidelity.
- **Naive latent interpolation** (interpolate-only): Completely fails — the LDM latent space is not well-structured without denoising.
- **Simple denoise-interpolate** (no branching): Produces alpha-blending artifacts rather than semantic transformations.
- **Latent blending (Lunarring)**: Only shown for generated images; this work extends it to real images via textual inversion and pose conditioning.

**Empirical advantage**: Qualitatively, the method produces convincing semantic transitions (e.g., human → mountain, cartoon → photo) rather than ghostly overlays. Quantitatively, FID and PPL metrics *do not favor this method* — they prefer boring alpha blends! The authors correctly argue that **standard metrics fail to capture interpolation quality**, prioritizing low deviation from inputs over creative plausibility.

## Limitations

1. **Large style/layout gaps**: Fails when input images are too different (e.g., very abstract vs. photorealistic, or completely different object categories).
2. **Pose estimation errors**: OpenPose fails on stylized images (mitigated by style transfer preprocessing, but not always successful).
3. **Semantic mismatches**: The model sometimes can't figure out which objects correspond between frames (e.g., mismatched body parts).
4. **Text hallucination**: Occasionally inserts spurious text into frames.
5. **Computational cost**: Requires 200+ DDIM steps per frame for quality, plus optional CLIP ranking over multiple candidates.
6. **No quantitative metric**: FID/PPL don't work; evaluation is purely qualitative. This is an open problem.

**Assumptions**:
- Inputs must be reasonably reconstructible by the LDM encoder-decoder.
- A shared text prompt can describe both inputs (or captions are provided).
- For pose conditioning, subjects must be detectable by OpenPose (or stylistically translatable first).

## Implementation Details

**Model**: Stable Diffusion v2.1 with ControlNet for pose conditioning.

**Textual inversion**:
- Loss: L(c_text) = ||ε̂ - ε||², where ε̂ is the predicted noise and ε is ground truth.
- Optimize the entire text embedding (not just a custom token).
- 100–500 iterations, learning rate 10⁻⁴.
- Shared negative prompt for both images.

**Noise schedule**:
- DDIM sampling with 200+ timesteps for quality.
- Noise range: 25%–65% of the full schedule.
  - <25%: images look like alpha composites.
  - >65%: images deviate too much from parents.
- Linear schedule within this range.

**Branching structure**:
- Inputs x₀, xₙ → midpoint xₙ/₂ at noise level t_K.
- Recursively split intervals, using t_{K-1}, t_{K-2}, … for finer subdivisions.
- Each frame's noise level decreases as it gets closer to an input.

**Pose conditioning**:
- OpenPose for keypoint detection.
- For stylized images: first apply image-to-image translation to photorealistic style, then extract pose.
- Linearly interpolate shared keypoints between input poses.
- Feed interpolated pose to ControlNet.

**CLIP ranking**:
- Positive prompts: "high quality, detailed, 2D"
- Negative prompts: "blurry, distorted, 3D render"
- Score = CLIP_similarity(positive) - CLIP_similarity(negative).
- Generate multiple candidates per frame with different random seeds; keep highest scorer.

**Interpolation type**: Spherical linear interpolation (slerp) for latents and text embeddings; linear for poses. Empirically, slerp vs. linear makes little difference.

**Dataset**: 26 curated image pairs across diverse domains (photos, cartoons, logos, posters, video games, artwork).

---

## Key Insights

1. **The branching noise schedule is critical**: It balances smoothness (via shared noise between parent-child) and creativity (via independent noise between siblings). Without it, interpolations either look like alpha blends or have jarring discontinuities.

2. **Standard metrics fail for interpolation quality**: FID and PPL favor low-variance, conservative interpolations (alpha blending) over semantically meaningful transformations. The field needs better evaluation protocols.

3. **Textual inversion + pose conditioning are essential for real images**: Unlike generated images, real images don't come with known text prompts or poses — these must be inferred and fine-tuned.

4. **CLIP ranking helps, but manual selection is better**: For creative applications, users should be able to select among candidates or specify custom prompts/poses mid-sequence.

5. **Pose conditioning works even when wrong**: Surprisingly, even incorrect pose estimates improve results by preventing abrupt pose changes between frames.
