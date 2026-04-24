"""
Barycentric 3-image interpolation using diffusion models.
See BARYCENTRIC_PLAN.md for the full design rationale.

Usage:
    import sys
    sys.path.insert(0, '/root')
    sys.path.insert(0, '/root/DiffusionInterpolating')
    from cm import ContextManager
    from cm_barycentric import interpolate_barycentric

    CM = ContextManager(version='1.5')
    interpolate_barycentric(CM, img1, img2, img3, num_levels=3, out_dir='bary_out', ...)
"""

import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from PIL import Image

from cm import get_step_schedule, interpolate_linear
from annotator.openpose import util


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(img):
    return torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).cuda()


def _encode(ldm, img_t):
    return ldm.get_first_stage_encoding(
        ldm.encode_first_stage(img_t.float() / 127.5 - 1.0))


def _save_frame(ldm, latent, path):
    img = ldm.decode_first_stage(latent)
    img = ((img.clamp(-1, 1) + 1) / 2 * 255)
    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(img).save(path)


def interp_poses_bary(pose_md1, pose_md2, pose_md3, g, shape):
    """
    Barycentric interpolation of 3 pose metadata dicts at weights g=(λ1,λ2,λ3).

    Each keypoint coordinate is a weighted sum of the 3 corresponding
    candidates.  A keypoint is skipped if it is missing (-1) in any source.
    Extends interp_poses() from cm.py to 3 images.
    """
    candidate = []
    subsets   = []
    cand_ix   = 0
    λ1, λ2, λ3 = g

    for person in range(len(pose_md1['subset'])):
        subset = [-1] * 20
        for i in range(18):
            j1 = int(pose_md1['subset'][person][i])
            j2 = int(pose_md2['subset'][person][i])
            j3 = int(pose_md3['subset'][person][i])
            if j1 == -1 or j2 == -1 or j3 == -1:
                subset[i] = -1
                continue
            x = (λ1 * pose_md1['candidate'][j1][0]
                 + λ2 * pose_md2['candidate'][j2][0]
                 + λ3 * pose_md3['candidate'][j3][0])
            y = (λ1 * pose_md1['candidate'][j1][1]
                 + λ2 * pose_md2['candidate'][j2][1]
                 + λ3 * pose_md3['candidate'][j3][1])
            candidate.append([x, y, 0, cand_ix])
            subset[i] = cand_ix
            cand_ix += 1
        subsets.append(subset)

    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, np.array(candidate), np.array(subsets))
    return canvas


# ---------------------------------------------------------------------------
# Conditioning optimisation for 3 endpoints
# ---------------------------------------------------------------------------

def learn_conditioning_triple(CM, img1_t, img2_t, img3_t,
                               cond_base, uncond_base,
                               ddim_steps, guide_scale,
                               num_iters=200, cond_lr=1e-4):
    """
    Extend learn_conditioning() to 3 images.

    Optimises cond1, cond2, cond3 and shared uncond via diffusion denoising
    loss on each endpoint image. One Adam step per (image, cond) pair per
    iteration, matching the 2-image version's update pattern.

    Returns: cond1, cond2, cond3, uncond  (all detached, no grad)
    """
    ldm = CM.model
    augment = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.7, 1.0)),
    ])

    cond1  = cond_base.clone().requires_grad_(True)
    cond2  = cond_base.clone().requires_grad_(True)
    cond3  = cond_base.clone().requires_grad_(True)
    uncond = uncond_base.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([cond1, cond2, cond3, uncond], lr=cond_lr)
    CM.ddim_sampler.make_schedule(ddim_steps, verbose=False)
    T = ddim_steps

    un_cond_dict = {"c_crossattn": [uncond], "c_concat": None}

    for _ in range(num_iters):
        u   = np.random.randint(T // 3, 2 * T // 3)
        t_u = CM.ddim_sampler.ddim_timesteps[u]
        tu  = torch.tensor([t_u], device="cuda", dtype=torch.long)

        with torch.autocast("cuda"):
            for img_t, cond in [(img1_t, cond1), (img2_t, cond2), (img3_t, cond3)]:
                L = _encode(ldm, augment(img_t))
                noise = torch.randn_like(L)
                x_t = (ldm.sqrt_alphas_cumprod[t_u] * L
                       + ldm.sqrt_one_minus_alphas_cumprod[t_u] * noise)
                eps = CM.ddim_sampler.pred_eps(
                    x_t,
                    {"c_crossattn": [cond], "c_concat": None},
                    tu,
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond_dict,
                )
                loss = (eps - noise).pow(2).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    for c in (cond1, cond2, cond3, uncond):
        c.requires_grad_(False)

    return cond1, cond2, cond3, uncond


# ---------------------------------------------------------------------------
# Main interpolation function
# ---------------------------------------------------------------------------

def interpolate_barycentric(CM, img1, img2, img3,
                             num_levels=3,
                             out_dir="bary_out",
                             prompt="", n_prompt="",
                             ddim_steps=200,
                             guide_scale=7.5,
                             min_steps=0.3,
                             max_steps=0.55,
                             ddim_eta=0.0,
                             optimize_cond=200,
                             cond_lr=1e-4,
                             schedule_type="linear",
                             cond_path=None,
                             controls=None,
                             n_choices=1,
                             qc_prompts=None):
    """
    Interpolate between 3 images using recursive barycentric subdivision.

    Frames are generated at the centroid of each subtriangle, coarse-to-fine,
    using partial DDIM denoising — generalising interpolate_qc to 2D.

    Produces (3^num_levels - 1) / 2 interior frames plus 3 endpoint frames.

    Args:
        CM           : ContextManager instance
        img1/2/3     : PIL Images, same size (e.g. 512x512 RGB)
        num_levels   : recursion depth
        out_dir      : output directory (recreated each call)
        prompt       : positive text prompt (init for conditioning optimisation)
        n_prompt     : negative text prompt
        ddim_steps   : total DDIM step schedule length
        guide_scale  : classifier-free guidance scale
        min_steps    : finest denoising start, as fraction of ddim_steps
        max_steps    : coarsest denoising start, as fraction of ddim_steps
        ddim_eta     : DDIM stochasticity (0 = deterministic)
        optimize_cond: conditioning optimisation iterations (0 to skip)
        cond_lr      : Adam lr for conditioning optimisation
        schedule_type: 'linear', 'convex', or 'concave'
        cond_path    : optional path to cache/restore optimised conditionings
        controls     : tuple of 3 pose metadata dicts (pose_md1, pose_md2, pose_md3)
                       if provided, activates ControlNet pose conditioning
        n_choices    : candidates generated per frame; best selected by CLIP score
                       (requires qc_prompts)
        qc_prompts   : (pos_prompt, neg_prompt) strings for CLIP ranking;
                       if None and n_choices > 1, falls back to n_choices=1
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    # Convert fractions → step counts, matching interpolate_qc
    if min_steps < 1:
        min_steps = int(ddim_steps * min_steps)
    if max_steps < 1:
        max_steps = int(ddim_steps * max_steps)

    ldm = CM.model

    # --- ControlNet mode ---
    if controls is not None:
        CM.change_mode('pose')
        pose_md1, pose_md2, pose_md3 = controls
    else:
        CM.init_mode()
        ldm.control_scales = [1] * 13

    # --- CLIP model for n_choices ranking ---
    clip_model = clip_preprocess = None
    pos_embedding = neg_embedding = None
    if n_choices > 1 and qc_prompts is not None:
        import clip
        clip_model, clip_preprocess = clip.load('ViT-L/14', device='cuda')
        qc_prompt, qc_neg_prompt = qc_prompts
        with torch.no_grad():
            pos_embedding = clip_model.encode_text(clip.tokenize([qc_prompt]).to('cuda'))
            neg_embedding = clip_model.encode_text(clip.tokenize([qc_neg_prompt]).to('cuda'))
    else:
        n_choices = 1

    # --- Encode endpoint images to latent space ---
    img1_t, img2_t, img3_t = _to_tensor(img1), _to_tensor(img2), _to_tensor(img3)
    L1 = _encode(ldm, img1_t)
    L2 = _encode(ldm, img2_t)
    L3 = _encode(ldm, img3_t)
    shape = L1.shape[-3:]  # (C, H/8, W/8)
    img_shape = img1_t.shape[-2:]  # (H, W) for pose canvas

    # --- Conditionings ---
    if cond_path and os.path.exists(cond_path):
        cond1, cond2, cond3, uncond_base = torch.load(cond_path)
    else:
        with torch.no_grad():
            cond_base   = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])

        if optimize_cond:
            cond1, cond2, cond3, uncond_base = learn_conditioning_triple(
                CM, img1_t, img2_t, img3_t,
                cond_base, uncond_base,
                ddim_steps, guide_scale,
                num_iters=optimize_cond, cond_lr=cond_lr,
            )
        else:
            cond1 = cond2 = cond3 = cond_base

        if cond_path:
            torch.save((cond1, cond2, cond3, uncond_base), cond_path)

    # --- DDIM schedule ---
    CM.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)
    timesteps     = CM.ddim_sampler.ddim_timesteps
    step_schedule = get_step_schedule(min_steps, max_steps, num_levels, schedule_type)
    print(f"step_schedule: {step_schedule}")

    yaml.dump(
        dict(prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps,
             guide_scale=guide_scale, step_schedule=step_schedule,
             ddim_eta=ddim_eta, num_levels=num_levels,
             optimize_cond=optimize_cond, schedule_type=schedule_type),
        open(f"{out_dir}/args.yaml", "w"),
    )

    # --- Initialise triangular grid ---
    # Points keyed by (λ1, λ2, λ3) global barycentric coords.
    v1, v2, v3 = (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)
    latent_map  = {v1: L1, v2: L2, v3: L3}

    img1.save(f"{out_dir}/1.000_0.000_0.000.png")
    img2.save(f"{out_dir}/0.000_1.000_0.000.png")
    img3.save(f"{out_dir}/0.000_0.000_1.000.png")

    triangles = [(v1, v2, v3)]  # grows 3x each level

    # --- Hierarchical star subdivision, coarse → fine ---
    for level in range(1, num_levels + 1):
        cur_step = step_schedule[-level]
        t        = timesteps[cur_step]
        print(f"Level {level}  timestep={t}  cur_step={cur_step}  triangles={len(triangles)}")

        un_cond        = {"c_crossattn": [uncond_base], "c_concat": None}
        next_triangles = []

        for (a, b, c) in triangles:

            # Global barycentric coordinate of centroid (masspoint)
            g = (
                (a[0] + b[0] + c[0]) / 3,
                (a[1] + b[1] + c[1]) / 3,
                (a[2] + b[2] + c[2]) / 3,
            )
            print(f"\t g=({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})")

            # Conditioning: global bary-weighted mix of 3 endpoint embeddings
            cond_g = g[0] * cond1 + g[1] * cond2 + g[2] * cond3
            cond   = {"c_crossattn": [cond_g], "c_concat": None}

            # Pose control signal: barycentric blend of 3 pose maps
            if controls is not None:
                pose_img = interp_poses_bary(
                    pose_md1, pose_md2, pose_md3, g, img_shape)
                control = (torch.from_numpy(pose_img.transpose(2, 0, 1))
                           .float().cuda().unsqueeze(0) / 255.0)
                cond["c_concat"]    = [control]
                un_cond["c_concat"] = [control]

            # Noisy latent: shared noise added to all 3 parent vertex latents,
            # then averaged — local bary weight 1/3 each (centroid of triangle)
            La, Lb, Lc = latent_map[a], latent_map[b], latent_map[c]

            # Generate n_choices candidates, pick best by CLIP score
            candidates   = []
            clip_scores  = []

            for _ in range(n_choices):
                noise    = torch.randn_like(La)
                noisy_a  = ldm.sqrt_alphas_cumprod[t] * La + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                noisy_b  = ldm.sqrt_alphas_cumprod[t] * Lb + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                noisy_c  = ldm.sqrt_alphas_cumprod[t] * Lc + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                noisy_latent = (noisy_a + noisy_b + noisy_c) / 3

                samples, _ = CM.ddim_sampler.sample(
                    ddim_steps, 1, shape, cond,
                    verbose=False, eta=ddim_eta,
                    x_T=noisy_latent, timesteps=cur_step,
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond,
                )
                candidates.append(samples)

                if clip_model is not None:
                    image = ldm.decode_first_stage(samples)
                    with torch.no_grad():
                        image = clip_preprocess.transforms[0](image)
                        if shape[-1] != shape[-2]:
                            image = clip_preprocess.transforms[1](image)
                        image_features = clip_model.encode_image(image)
                    score = (F.cosine_similarity(image_features, pos_embedding)
                             - F.cosine_similarity(image_features, neg_embedding))
                    clip_scores.append(score.item())

            # Pick best candidate
            best = int(np.argmax(clip_scores)) if clip_scores else 0
            latent_map[g] = candidates[best]
            _save_frame(ldm, candidates[best],
                        f"{out_dir}/L{level}_{g[0]:.3f}_{g[1]:.3f}_{g[2]:.3f}.png")

            # Star subdivision: (a,b,c) → 3 children through g
            next_triangles += [(a, b, g), (b, c, g), (a, c, g)]

        triangles = next_triangles

    return latent_map
