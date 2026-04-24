import pdb
import shutil
from share import *

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
F = torch.nn.functional

import yaml
from tqdm import trange
from annotator.util import HWC3
from annotator.openpose import OpenposeDetector
from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.openpose import util

def get_step_schedule(min_steps, max_steps, num_levels, schedule_type='convex'):
    diff = max_steps - min_steps
    if schedule_type == 'concave':
        return [0]+[int(diff * x**.5)+min_steps for x in np.linspace(0, 1, num_levels)]
    elif schedule_type == 'convex':
        return [0]+[int(diff * x**2)+min_steps for x in np.linspace(0, 1, num_levels)]
    elif schedule_type == 'linear':
        return [0]+[int(x) for x in np.linspace(min_steps, max_steps, num_levels)]

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def interpolate_linear(p0,p1, frac):
    return p0 + (p1 - p0) * frac

@torch.no_grad()
def slerp(p0, p1, fract_mixing: float):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """ 
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
        
    return interp

def interp_poses(pose_md1, pose_md2, alpha, shape):
    candidate = []
    subsets = []
    cand_ix = 0
    for person in range(len(pose_md1['subset'])):
        subset = [-1] * 20
        for i in range(18):
            j = int(pose_md1['subset'][person][i])
            k = int(pose_md2['subset'][person][i])
            if j == -1 or k == -1:
                subset[i] = -1
                continue
            candidate.append([interpolate_linear(pose_md1['candidate'][j][0], pose_md2['candidate'][k][0], alpha),
                interpolate_linear(pose_md1['candidate'][j][1], pose_md2['candidate'][k][1], alpha),
                0,cand_ix])
            subset[i] = cand_ix
            cand_ix += 1
        subsets.append(subset)
    # candidate.append([-1,-1,0,i])
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, np.array(candidate), np.array(subsets))
    # candidate = pose_md1['candidate']
    # subsets = pose_md1['subset']
    # Image.fromarray(canvas).save('rick_poses/test.png')
    return canvas

class ContextManager:
    def __init__(self, version='1.5'):
        self.filters = {}
        self.mode = None
        self.version = version
        if version == '2.1':
            self.model = create_model('./models/cldm_v21.yaml').cuda()
        else:
            self.model = create_model('./models/cldm_v15.yaml').cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def init_mode(self):
        if self.mode is None:
            self.change_mode('pose')

    def change_mode(self, mode):
        if self.mode == mode:
            return
        
        if mode not in self.filters:
            if mode == 'pose':
                self.filters[mode] = OpenposeDetector()
            elif mode == 'canny':
                self.filters[mode] = CannyDetector()
            elif mode == 'seg':
                self.filters[mode] = UniformerDetector()
        
        if mode == 'pose':
            if self.version == '2.1':
                self.model.load_state_dict(load_state_dict('./controlnet/models/openpose-sd21.ckpt', location='cuda'))
            else:
                self.model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
        elif mode == 'canny':
            if self.version == '2.1':
                self.model.load_state_dict(load_state_dict('./controlnet/models/canny-sd21.ckpt', location='cuda'))
            else:
                self.model.load_state_dict(load_state_dict('./controlnet/models/control_sd15_canny.pth', location='cuda'))
        elif mode == 'seg':
            self.model.load_state_dict(load_state_dict('./controlnet/models/control_sd15_seg.pth', location='cuda'))
        self.mode = mode

    def get_canny(self, image, lower_bound=220, upper_bound=255):
        self.change_mode('canny')
        # with torch.autocast('cuda'):
        canny = self.filters['canny'](HWC3(np.array(image)), lower_bound, upper_bound)
        return canny
        
    def get_pose(self, image, return_metadata=False, filter_largest=True):
        self.change_mode('pose')
        pred_pose, metadata = self.filters['pose'](HWC3(np.array(image)))
        if len(metadata['subset']) > 1:
            if filter_largest:
                sizes = []
                for ss in metadata['subset']:
                    min_x = min_y = 1000
                    max_x = max_y = 0
                    for i in range(18):
                        if ss[i] != -1:
                            x, y = metadata['candidate'][int(ss[i])][:2]
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)
                    sizes.append((max_x-min_x) * (max_y-min_y))
                ix = np.argmax(sizes)
                metadata['subset'] = [metadata['subset'][ix]]
                pred_pose = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
                pred_pose = util.draw_bodypose(pred_pose, np.array(metadata['candidate']), np.array(metadata['subset']))
            else: # order left to right
                minX = []
                for ss in metadata['subset']:
                    min_x = 1000
                    for i in range(18):
                        if ss[i] != -1:
                            min_x = min(min_x, metadata['candidate'][int(ss[i])][0])
                    minX.append(min_x)
                indices = np.argsort(minX)
                metadata['subset'] = [metadata['subset'][ix] for ix in indices]

        if return_metadata:
            return pred_pose, metadata
        return pred_pose
        
    def learn_conditioning(self, img1, img2, cond_base, uncond_base, ddim_steps, guide_scale, num_iters=200, cond_lr=1e-4):
        # augment = transforms.TrivialAugmentWide(num_magnitude_bins=20)
        augment = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=(512,512), scale=(0.7,1.0)),
        ])

        cond = {"c_crossattn": [cond_base], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        ldm = self.model
        uncond_base.requires_grad_(True)
        cond1 = cond_base
        cond2 = cond_base.clone()
        cond1.requires_grad_(True)
        cond2.requires_grad_(True)
        optimizer = torch.optim.Adam([cond1, cond2, uncond_base], lr=cond_lr) #
        T = ddim_steps
        self.ddim_sampler.make_schedule(T, verbose=False)
        for cur_iter in range(num_iters):
            L1 = ldm.get_first_stage_encoding(ldm.encode_first_stage(augment(img1).float() / 127.5 - 1.0))
            L2 = ldm.get_first_stage_encoding(ldm.encode_first_stage(augment(img2).float() / 127.5 - 1.0))
            with torch.autocast('cuda'):
                u = np.random.randint(T//3, 2*T//3)
                t_u = self.ddim_sampler.ddim_timesteps[u]
                tu = torch.tensor([t_u], device='cuda', dtype=torch.long)

                cond["c_crossattn"] = [cond1]
                noise = torch.randn_like(L1)
                x_t_u = ldm.sqrt_alphas_cumprod[t_u] * L1 + \
                    ldm.sqrt_one_minus_alphas_cumprod[t_u] * noise
                eps = self.ddim_sampler.pred_eps(x_t_u, cond, tu, unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
                loss1 = (eps - noise).pow(2).mean()
                loss1.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                cond["c_crossattn"] = [cond2]
                noise = torch.randn_like(L2)
                x_t_u = ldm.sqrt_alphas_cumprod[t_u] * L2 + \
                    ldm.sqrt_one_minus_alphas_cumprod[t_u] * noise
                eps = self.ddim_sampler.pred_eps(x_t_u, cond, tu, unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
                loss2 = (eps - noise).pow(2).mean()
                loss2.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # if cur_iter % 50 == 0:
                #     print(f'iter {cur_iter}: {loss1.item()}, {loss2.item()}')

        cond1.requires_grad_(False)
        cond2.requires_grad_(False)
        uncond_base.requires_grad_(False)
        return cond1, cond2, uncond_base

    def interpolate_naive(self, img1, img2, num_frames, out_dir='blend'):
        if isinstance(img1, Image.Image):
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()
        ldm = self.model
        L1 = ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
        L2 = ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        for frame_ix in trange(1,num_frames-1):
            frac = frame_ix/(num_frames-1)
            latent = slerp(L1, L2, frac)
            x_samples = ldm.decode_first_stage(latent).permute(0, 2, 3, 1)
            x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(x_samples[0]).save(f'{out_dir}/{frame_ix:03d}.png')

    def interpolate_then_diffuse(self, img1, img2, num_frames, controls=None, control_type='pose', min_steps=.25, max_steps=.5, prompt=None, n_prompt=None, ddim_steps=250, guide_scale=7.5, schedule_type='linear', optimize_cond=0, cond_path=None, cond_lr=1e-4, out_dir='blend'): #steps_per_frame=10, 
        """
        each successive frame has more noise than the previous
        """
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)

        if isinstance(img1, Image.Image):
            img1.save(f'{out_dir}/{0:03d}.png')
            img2.save(f'{out_dir}/{num_frames-1:03d}.png')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()
        
        if controls is None:
            self.init_mode()
        else:
            self.change_mode(control_type)
            if control_type == 'pose':
                pose_md1, pose_md2 = controls
            else:
                raise NotImplementedError
        ldm = self.model
        ldm.control_scales = [1] * 13

        if cond_path and os.path.exists(cond_path):
            assert optimize_cond > 0
            cond1, cond2, uncond_base = torch.load(cond_path)
        else:
            cond1 = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])

            if optimize_cond:
                cond1, cond2, uncond_base = self.learn_conditioning(img1, img2, cond1, uncond_base, ddim_steps, guide_scale=guide_scale, num_iters=optimize_cond, cond_lr=cond_lr)
                if cond_path:
                    torch.save((cond1, cond2, uncond_base), cond_path)

        cond = {"c_crossattn": [cond1], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

        img1 = img1.float() / 127.5 - 1.0
        img2 = img2.float() / 127.5 - 1.0
        # schedules include endpoints
        self.ddim_sampler.make_schedule(ddim_steps, verbose=False)
        step_schedule = get_step_schedule(min_steps, max_steps, (num_frames+1)//2, schedule_type=schedule_type)
        timestep_schedule = [self.ddim_sampler.ddim_timesteps[s] for s in step_schedule]
        latents1, latents2 = self.get_latent_stack(img1, img2, timestep_schedule)
        latents = [None] * num_frames
        latents[0] = latents1[0]
        latents[-1] = latents2[0]
        shape = latents[0].shape[-3:]
        
        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, step_schedule=step_schedule)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))

        for frame_ix in trange(1,num_frames-1):
            frac = frame_ix/(num_frames-1)
            f = min(frame_ix, num_frames - frame_ix - 1)
            latents[frame_ix] = slerp(latents1[f], latents2[f], frac)
            if controls is not None:
                pose_img = interp_poses(pose_md1, pose_md2, alpha=frac, shape=img1.shape[-2:]).transpose(2,0,1)
                control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0
                cond["c_concat"] = un_cond["c_concat"] = [control]

            if optimize_cond:
                cond["c_crossattn"] = [interpolate_linear(cond1, cond2, frac)]

            samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
                shape, cond, verbose=False,
                x_T=latents[frame_ix], timesteps=step_schedule[f],
                unconditional_guidance_scale=guide_scale,
                unconditional_conditioning=un_cond)

            x_samples = ldm.decode_first_stage(samples).permute(0, 2, 3, 1)
            x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(x_samples[0]).save(f'{out_dir}/{frame_ix:03d}.png')

    def interpolate(self, img1, img2, controls=None, control_type='pose', prompt=None, n_prompt=None, min_steps=.25, max_steps=.5, ddim_steps=250, num_frames=17, guide_scale=7.5, schedule_type='linear', optimize_cond=0, latent_interp='spherical', cond_interp='spherical', cond_path=None, cond_lr=1e-4, bias=0, retroactive_interp=True, share_noise=True, out_dir='blend'):
        """
        ddim_steps: number of steps in DDIM sampling
        num_frames: includes endpoints (both original images)
        """
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
    
        if isinstance(img1, Image.Image):
            img1.save(f'{out_dir}/{0:03d}.png')
            img2.save(f'{out_dir}/{num_frames-1:03d}.png')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()

        if controls is None:
            self.init_mode()
        else:
            self.change_mode(control_type)
            if control_type == 'pose':
                pose_md1, pose_md2 = controls
            else:
                raise NotImplementedError
        ldm = self.model
        ldm.control_scales = [1] * 13

        if cond_path and os.path.exists(cond_path):
            assert optimize_cond > 0
            cond1, cond2, uncond_base = torch.load(cond_path)
        else:
            cond1 = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])

            if optimize_cond:
                cond1, cond2, uncond_base = self.learn_conditioning(img1, img2, cond1, uncond_base, ddim_steps, guide_scale=guide_scale, num_iters=optimize_cond, cond_lr=cond_lr)
                if cond_path:
                    torch.save((cond1, cond2, uncond_base), cond_path)

        cond = {"c_crossattn": [cond1], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

        img1 = img1.float() / 127.5 - 1.0
        img2 = img2.float() / 127.5 - 1.0
        # schedules include endpoints
        num_levels = int(np.log2(num_frames-1)) # does not include endpoints
        assert np.log2(num_frames-1) % 1 < 1e-5
        self.ddim_sampler.make_schedule(ddim_steps, verbose=False)
        step_schedule = get_step_schedule(min_steps, max_steps, num_levels, schedule_type=schedule_type)
        timesteps = self.ddim_sampler.ddim_timesteps
        timestep_schedule = [timesteps[s] for s in step_schedule]
        latents1, latents2 = self.get_latent_stack(img1, img2, timestep_schedule, share_noise=share_noise)
        latents = [None] * num_frames
        latents[0] = latents1[0]
        latents[-1] = latents2[0]
        
        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, step_schedule=step_schedule, bias=bias, retroactive_interp=retroactive_interp, share_noise=share_noise)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
        
        if latent_interp == 'linear':
            interpolate_latents = interpolate_linear
        else:
            interpolate_latents = slerp

        for level in trange(1,num_levels+1):
            cur_ix = step_schedule[-level]
            prev_ix = step_schedule[-level-1]
            latents[0] = latents1[-level]
            latents[-1] = latents2[-level]
            df = 2**(num_levels-level)

            for frame_ix in range(df, num_frames-1, df*2):
                frac = .5
                if frame_ix-df == 0:
                    frac -= bias
                if frame_ix+df == num_frames-1:
                    frac += bias
                latents[frame_ix] = interpolate_latents(latents[frame_ix-df], latents[frame_ix+df], frac)

            if retroactive_interp:
                if level == 2:
                    latents[num_frames//2] = interpolate_latents(latents[num_frames//4], latents[3*num_frames//4], .5)
                
                if level == 3:
                    latents[num_frames//4] = interpolate_latents(latents[num_frames//8], latents[3*num_frames//8], .5)
                    latents[num_frames//2] = interpolate_latents(latents[3*num_frames//8], latents[5*num_frames//8], .5)
                    latents[3*num_frames//4] = interpolate_latents(latents[5*num_frames//8], latents[7*num_frames//8], .5)
            
            for frame_ix in range(df, num_frames-1, df): # exclude endpoints
                frac = frame_ix/(num_frames-1)

                if controls is not None:
                    pose_img = interp_poses(pose_md1, pose_md2, alpha=frac, shape=img1.shape[-2:]).transpose(2,0,1)
                    control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0
                    cond["c_concat"] = un_cond["c_concat"] = [control]

                if optimize_cond:
                    if cond_interp == 'linear':
                        cond["c_crossattn"] = [interpolate_linear(cond1, cond2, frac)]
                    else:
                        cond["c_crossattn"] = [slerp(cond1, cond2, frac)]
                
                for i, t in enumerate(np.flip(timesteps[prev_ix:cur_ix])):
                    index = cur_ix - i - 1
                    ts = torch.tensor([t], device='cuda', dtype=torch.long)

                    latents[frame_ix] = self.ddim_sampler.p_sample_ddim(latents[frame_ix], cond, ts, index=index, unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond)[0]

        for frame_ix in range(1,num_frames-1):
            x_samples = ldm.decode_first_stage(latents[frame_ix]).permute(0, 2, 3, 1)
            x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(x_samples[0]).save(f'{out_dir}/{frame_ix:03d}.png')
        
    def interpolate_qc(self, img1, img2, n_choices=8, qc_prompts=None, controls=None, control_type='pose', scale_control=1.5, prompt=None, n_prompt=None, min_steps=.3, max_steps=.55, ddim_steps=250, num_frames=17, guide_scale=7.5, schedule_type='linear', optimize_cond=0, latent_interp='spherical', cond_interp='spherical', cond_path=None, cond_lr=1e-4, bias=0, ddim_eta=0, out_dir='blend'):
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
    
        if isinstance(img1, Image.Image):
            img1.save(f'{out_dir}/{0:03d}.png')
            img2.save(f'{out_dir}/{num_frames-1:03d}.png')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()

        if controls is None:
            self.init_mode()
        else:
            self.change_mode(control_type)
            if control_type == 'pose':
                pose_md1, pose_md2 = controls
            else:
                raise NotImplementedError
        ldm = self.model
        if not scale_control:
            ldm.control_scales = [1] * 13

        if cond_path and os.path.exists(cond_path):
            assert optimize_cond > 0
            cond1, cond2, uncond_base = torch.load(cond_path)
        else:
            cond1 = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])

            if optimize_cond:
                cond1, cond2, uncond_base = self.learn_conditioning(img1, img2, cond1, uncond_base, ddim_steps, guide_scale=guide_scale, num_iters=optimize_cond, cond_lr=cond_lr)
                if cond_path:
                    print(f"cond1 {cond1.shape} cond2 {cond2.shape} uncond_base {uncond_base.shape}")
                    torch.save((cond1, cond2, uncond_base), cond_path)

        cond = {"c_crossattn": [cond1], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

        if qc_prompts is not None:
            qc_prompt, qc_neg_prompt = qc_prompts
            import clip
            clip_model, preprocess = clip.load('ViT-L/14', device='cuda')
            with torch.no_grad():
                pos_embedding = clip_model.encode_text(clip.tokenize([qc_prompt]).to('cuda'))
                neg_embedding = clip_model.encode_text(clip.tokenize([qc_neg_prompt]).to('cuda'))


        # schedules include endpoints
        num_levels = int(np.log2(num_frames-1)) # does not include endpoints
        assert np.log2(num_frames-1) % 1 < 1e-5
        self.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)
        step_schedule = get_step_schedule(min_steps, max_steps, num_levels, schedule_type=schedule_type)
        print(f"step_schedule {step_schedule}")
        timesteps = self.ddim_sampler.ddim_timesteps
        final_latents = [None] * num_frames
        final_latents[0] = ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
        final_latents[-1] = ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))
        print(f"final_latents' shape {final_latents[0].shape}")
        shape = final_latents[0].shape[-3:]
        print(f"shape {shape}")
        
        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, step_schedule=step_schedule, bias=bias, ddim_eta=ddim_eta, scale_control=scale_control)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
        
        if latent_interp == 'linear':
            interpolate_latents = interpolate_linear
        else:
            interpolate_latents = slerp

        for level in range(1,num_levels+1):
            cur_step = step_schedule[-level]
            t = timesteps[cur_step]
            df = 2**(num_levels-level)
            print(f"Level {level} with timestep {t} and frame step {df} cur_step {cur_step}")
                
            for frame_ix in range(df, num_frames-1, df*2):
                print(f"\t frame_ix {frame_ix}")
                frac = frame_ix/(num_frames-1)
                if scale_control:
                    ldm.control_scales = [scale_control - 2*abs(frac-.5) * (scale_control-1)] * 13 # range from 1 to scale_control
                    print(f"\t\t control scale {ldm.control_scales[0]}")

                if controls is not None:
                    pose_img = interp_poses(pose_md1, pose_md2, alpha=frac, shape=img1.shape[-2:]).transpose(2,0,1)
                    control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0
                    cond["c_concat"] = un_cond["c_concat"] = [control]
                if optimize_cond:
                    if cond_interp == 'linear':
                        cond["c_crossattn"] = [interpolate_linear(cond1, cond2, frac)]
                    else:
                        cond["c_crossattn"] = [slerp(cond1, cond2, frac)]

                latent_frac = .5
                if frame_ix-df == 0:
                    latent_frac -= bias
                if frame_ix+df == num_frames-1:
                    latent_frac += bias

                candidates = []
                clip_scores = []
                for choice_ix in range(n_choices):
                    noise = torch.randn_like(final_latents[0])
                    l1 = ldm.sqrt_alphas_cumprod[t] * final_latents[frame_ix-df] + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                    l2 = ldm.sqrt_alphas_cumprod[t] * final_latents[frame_ix+df] + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                    noisy_latent = interpolate_latents(l1, l2, latent_frac)

                    print(f"ddim_sampler.sample args: ddim_steps={ddim_steps}, batch_size=1, shape={shape}, eta={ddim_eta}, timesteps={cur_step}, guide_scale={guide_scale}")
                    print(f"  cond keys={list(cond.keys()) if isinstance(cond, dict) else type(cond)}")
                    print(f"  x_T shape={noisy_latent.shape}, dtype={noisy_latent.dtype}")
                    print(f"  un_cond keys={list(un_cond.keys()) if isinstance(un_cond, dict) else type(un_cond)}")
                    samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
                        shape, cond, verbose=False, eta=ddim_eta,
                        x_T=noisy_latent, timesteps=cur_step,
                        unconditional_guidance_scale=guide_scale,
                        unconditional_conditioning=un_cond)
                    candidates.append(samples)

                    image = ldm.decode_first_stage(samples)
                    if qc_prompts is None: #manual
                        image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                        Image.fromarray(image[0]).save(f'{out_dir}/{frame_ix:03d}_{choice_ix}.png')
                    else:
                        with torch.no_grad():
                            image = preprocess.transforms[0](image)
                            if shape[-1] != shape[-2]:
                                image = preprocess.transforms[1](image)
                            image_features = clip_model.encode_image(image)
                        # clip_scores.append((image_features @ clip_text_embedding.T).item())
                        clip_scores.append(F.cosine_similarity(image_features, pos_embedding).item() - F.cosine_similarity(image_features, neg_embedding).item())

                if qc_prompts is None: #manual
                    print(f'Enter choice (0-{n_choices}):')
                    choice = input()
                    for choice_ix in range(n_choices):
                        if choice_ix != int(choice):
                            os.remove(f'{out_dir}/{frame_ix:03d}_{choice_ix}.png')
                        else:
                            os.rename(f'{out_dir}/{frame_ix:03d}_{choice_ix}.png', f'{out_dir}/{frame_ix:03d}.png')
                else:
                    choice = np.argmax(clip_scores)
                    image = ldm.decode_first_stage(candidates[int(choice)])
                    image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    Image.fromarray(image[0]).save(f'{out_dir}/{frame_ix:03d}.png')

                
                final_latents[frame_ix] = candidates[int(choice)]

            n_choices = max(n_choices-1, 3) # reduce choices at fine-grained levels
    
    """def revise_frames(self, frame_range, out_dir, num_steps=.4, n_choices=6, qc_prompts=None, controls=None, control_type='pose', scale_control=False, prompt=None, n_prompt=None, ddim_steps=250, num_frames=17, guide_scale=7.5, optimize_cond=0, latent_interp='spherical', cond_interp='spherical', cond_path=None, cond_lr=1e-4, bias=0, ddim_eta=0):
        if num_steps < 1:
            num_steps = int(ddim_steps * num_steps)

        img1 = Image.open(f'{out_dir}/{frame_range[0]-1:03d}.png')
        img2 = Image.open(f'{out_dir}/{frame_range[1]:03d}.png')
        if isinstance(img1, Image.Image):
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()

        if controls is None:
            self.init_mode()
        else:
            self.change_mode(control_type)
            if control_type == 'pose':
                pose_md1, pose_md2 = controls
            else:
                raise NotImplementedError
        ldm = self.model
        if not scale_control:
            ldm.control_scales = [1] * 13

        if cond_path and os.path.exists(cond_path):
            assert optimize_cond > 0
            cond1, cond2, uncond_base = torch.load(cond_path)
        else:
            cond1 = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])

            if optimize_cond:
                cond1, cond2, uncond_base = self.learn_conditioning(img1, img2, cond1, uncond_base, ddim_steps, guide_scale=guide_scale, num_iters=optimize_cond, cond_lr=cond_lr)
                if cond_path:
                    torch.save((cond1, cond2, uncond_base), cond_path)

        cond = {"c_crossattn": [cond1], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}

        if qc_prompts is not None:
            qc_prompt, qc_neg_prompt = qc_prompts
            import clip
            clip_model, preprocess = clip.load('ViT-L/14', device='cuda')
            with torch.no_grad():
                pos_embedding = clip_model.encode_text(clip.tokenize([qc_prompt]).to('cuda'))
                neg_embedding = clip_model.encode_text(clip.tokenize([qc_neg_prompt]).to('cuda'))

        # schedules include endpoints
        self.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)
        t = self.ddim_sampler.ddim_timesteps[num_steps]
        L1 = ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
        L2 = ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))
        shape = L1.shape[-3:]
        
        if latent_interp == 'linear':
            interpolate_latents = interpolate_linear
        else:
            interpolate_latents = slerp

        for frame_ix in range(frame_range[0], frame_range[1]):
            frac = frame_ix/(num_frames-1)
            latent_frac = (frame_ix - frame_range[0]) / (frame_range[1] - frame_range[0])
            if scale_control:
                ldm.control_scales = [1.5-abs(frac-.5)] * 13 # range from 1.5 to 1

            if controls is not None:
                pose_img = interp_poses(pose_md1, pose_md2, alpha=frac, shape=img1.shape[-2:]).transpose(2,0,1)
                control = torch.from_numpy(pose_img).float().cuda().unsqueeze(0) / 255.0
                cond["c_concat"] = un_cond["c_concat"] = [control]
            if optimize_cond:
                if cond_interp == 'linear':
                    cond["c_crossattn"] = [interpolate_linear(cond1, cond2, frac)]
                else:
                    cond["c_crossattn"] = [slerp(cond1, cond2, frac)]

            candidates = []
            clip_scores = []
            for choice_ix in range(n_choices):
                noise = torch.randn_like(L1)
                l1 = ldm.sqrt_alphas_cumprod[t] * L1 + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                l2 = ldm.sqrt_alphas_cumprod[t] * L2 + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                noisy_latent = interpolate_latents(l1, l2, latent_frac)

                samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
                    shape, cond, verbose=False, eta=ddim_eta,
                    x_T=noisy_latent, timesteps=cur_step,
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond)
                candidates.append(samples)

                image = ldm.decode_first_stage(samples)
                if qc_prompts is None: #manual
                    image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    Image.fromarray(image[0]).save(f'{out_dir}/{frame_ix:03d}_{choice_ix}.png')
                else:
                    with torch.no_grad():
                        image = preprocess.transforms[0](image)
                        if shape[-1] != shape[-2]:
                            image = preprocess.transforms[1](image)
                        image_features = clip_model.encode_image(image)
                    # clip_scores.append((image_features @ clip_text_embedding.T).item())
                    clip_scores.append(F.cosine_similarity(image_features, pos_embedding).item() - F.cosine_similarity(image_features, neg_embedding).item())

            if qc_prompts is None: #manual
                print(f'Enter choice (0-{n_choices}):')
                choice = input()
                for choice_ix in range(n_choices):
                    if choice_ix != int(choice):
                        os.remove(f'{out_dir}/{frame_ix:03d}_{choice_ix}.png')
                    else:
                        os.rename(f'{out_dir}/{frame_ix:03d}_{choice_ix}.png', f'{out_dir}/{frame_ix:03d}.png')
            else:
                choice = np.argmax(clip_scores)
                image = ldm.decode_first_stage(candidates[int(choice)])
                image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                Image.fromarray(image[0]).save(f'{out_dir}/{frame_ix:03d}.png')

            
            final_latents[frame_ix] = candidates[int(choice)]
    """
    def visualize_poses(self, poses, num_frames, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for frame_ix in range(num_frames):
            frac = frame_ix/(num_frames-1)
            pose_img = interp_poses(*poses, alpha=frac, shape=(768,768))
            Image.fromarray(pose_img).save(f'{out_dir}/{frame_ix:03d}.png')
        
    def get_latent_stack(self, img1, img2, timesteps, share_noise=True):
        ldm = self.model
        latents1 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img1))]
        latents2 = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img2))]
        
        t_prev = None
        for t_now in timesteps[1:]:
            noise = torch.randn_like(latents1[-1])
            latents1.append(self.add_more_noise(latents1[-1], noise, t_now, t_prev))
            if not share_noise:
                noise = torch.randn_like(latents2[-1])
            latents2.append(self.add_more_noise(latents2[-1], noise, t_now, t_prev))
            t_prev = t_now
        return latents1, latents2
    
    def add_more_noise(self, latents, noise, t2, t1=None):
        ldm = self.model
        if t1 is None:
            return ldm.sqrt_alphas_cumprod[t2] * latents + \
                ldm.sqrt_one_minus_alphas_cumprod[t2] * noise

        a1 = ldm.sqrt_alphas_cumprod[t1]
        sig1 = ldm.sqrt_one_minus_alphas_cumprod[t1]
        a2 = ldm.sqrt_alphas_cumprod[t2]
        sig2 = ldm.sqrt_one_minus_alphas_cumprod[t2]

        scale = a2/a1
        sigma = (sig2**2 - (scale * sig1)**2).sqrt()
        return scale * latents + sigma * noise
    
    @torch.no_grad()
    def img2img(self, prompt, n_prompt, init_img=None, control=None, noise=None, mode=None, time_frac=0.3, ddim_steps=50, ctrl_scale=1, guide_scale=7.5, eta=0):
        if mode is not None:
            self.change_mode(mode)
        elif self.mode is None:
            print('no mode set')
            return
        
        ldm = self.model
        cond = {"c_concat": None, "c_crossattn": [ldm.get_learned_conditioning([prompt])]}
        un_cond = {"c_concat": None, "c_crossattn": [ldm.get_learned_conditioning([n_prompt])]}

        if control is not None:
            if not isinstance(control, torch.Tensor):
                control = torch.from_numpy(control).float().cuda().unsqueeze(0) / 255.0
                if len(control.shape) == 3:
                    control = control.tile(1, 3, 1, 1)

            cond["c_concat"] = un_cond["c_concat"] = [control]

        if init_img is not None:
            if isinstance(init_img, Image.Image):
                init_img = torch.tensor(np.array(init_img)).float().cuda() / 127.5 - 1.0
            latents = ldm.get_first_stage_encoding(ldm.encode_first_stage(init_img.permute(2,0,1).unsqueeze(0)))
        
        T = int(time_frac * ldm.num_timesteps)
        t = torch.tensor([T], dtype=torch.long, device='cuda')
        noise = torch.randn_like(latents)
        noisy_latents = (extract_into_tensor(ldm.sqrt_alphas_cumprod, t, latents.shape) * latents +
            extract_into_tensor(ldm.sqrt_one_minus_alphas_cumprod, t, latents.shape) * noise)
            
        shape = noisy_latents[0].shape[-3:]

        ldm.control_scales = [ctrl_scale] * 13
        samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
            shape, cond, verbose=False, eta=eta, x_T=noisy_latents, timesteps=int(time_frac * ddim_steps),
            unconditional_guidance_scale=guide_scale,
            unconditional_conditioning=un_cond)

        x_samples = ldm.decode_first_stage(samples).permute(0, 2, 3, 1)
        x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        return x_samples[0]

    @torch.no_grad()
    def generate(self, control, prompt, n_prompt, mode=None, ddim_steps=50, ctrl_scale=1, guide_scale=7.5, eta=0):
        if mode is not None:
            self.change_mode(mode)
        elif self.mode is None:
            print('no mode set')
            return
        
        if not isinstance(control, torch.Tensor):
            control = torch.from_numpy(control).float().cuda().unsqueeze(0) / 255.0
            if len(control.shape) == 3:
                control = control.tile(1, 3, 1, 1)
        ldm = self.model
        cond = {"c_concat": [control], "c_crossattn": [ldm.get_learned_conditioning([prompt])]}
        un_cond = {"c_concat": [control], "c_crossattn": [ldm.get_learned_conditioning([n_prompt])]}

        shape = (4, control.size(-2)//8, control.size(-1)//8)
        self.model.control_scales = [ctrl_scale] * 13
        samples, _ = self.ddim_sampler.sample(ddim_steps, 1,
                                shape, cond, verbose=False, eta=eta,
                                unconditional_guidance_scale=guide_scale,
                                unconditional_conditioning=un_cond)

        x_samples = self.model.decode_first_stage(samples).permute(0, 2, 3, 1)
        x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        return x_samples[0]
