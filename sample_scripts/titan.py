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
img1 = Image.open('sample_imgs/titan1.png').resize((512, 512))
img2 = Image.open('sample_imgs/titan3.png').resize((512, 512))

prompt = 'anime, giant humanoid titan, muscular body, dramatic lighting, dark atmosphere, detailed anatomy, epic scene, high quality, sharp linework'
n_prompt = 'lowres, blurry, low quality, disfigured, extra limbs, photo, realistic, western cartoon, chibi'

qc_prompt = 'titan, muscular, dramatic, high quality, detailed, sharp'
qc_neg_prompt = 'lowres, blurry, photo, realistic, disfigured, low quality, ugly, extra limbs'
CM.interpolate_qc(img1, img2, qc_prompts=(qc_prompt, qc_neg_prompt), cond_path='data/titan200_2.pt', n_choices=4
                  , prompt=prompt, n_prompt=n_prompt, min_steps=.3, max_steps=.55, optimize_cond=200
                  , ddim_steps=200, num_frames=33, guide_scale=7.5, ddim_eta=0, schedule_type='linear', out_dir='titan_clip')
