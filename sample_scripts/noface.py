import sys
from PIL import Image
import os
osp = os.path
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, '/root')                                                       
sys.path.insert(0, '/root/DiffusionInterpolating') 
from DiffusionInterpolating import cm

CM = cm.ContextManager()
img1 = Image.open('sample_imgs/noface1.png').resize((768, 512))
img2 = Image.open('sample_imgs/noface2.jpeg').resize((768, 512))

prompt = 'portrait, cartoon, mask, ghost, high resolution, highly detailed, ultra HD, 4k, simple, elegant'
n_prompt = 'lowres, messy, lopsided, disfigured, low quality, photo'

qc_prompt = 'portrait, cartoon, mask, detailed, high quality, simple, elegant'
qc_neg_prompt = 'lowres, distorted, ugly, blurry, photo, low quality, multiple faces'
CM.interpolate_qc(img1, img2, qc_prompts=None, cond_path='data/noface200.pt', n_choices=2
                  , prompt=prompt, n_prompt=n_prompt, min_steps=.3, max_steps=.55, optimize_cond=200
                  , ddim_steps=200, num_frames=65, guide_scale=7.5, ddim_eta=0, schedule_type='linear', out_dir='noface_clip')
