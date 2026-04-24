import sys
import os
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/DiffusionInterpolating')

from PIL import Image
from cm import ContextManager
from cm_barycentric import interpolate_barycentric

CM = ContextManager()

img1 = Image.open('sample_imgs/president1.png').resize((512, 512))
img2 = Image.open('sample_imgs/president2.png').resize((512, 512))
img3 = Image.open('sample_imgs/president3.png').resize((512, 512))

prompt   = 'official portrait, male politician, suit and tie, white background, sharp focus, professional photography, high resolution'
n_prompt = 'lowres, blurry, disfigured, cartoon, painting, drawing, low quality, ugly, watermark, multiple faces'

qc_prompt     = 'official portrait, politician, suit, sharp focus, professional, high quality'
qc_neg_prompt = 'lowres, blurry, disfigured, cartoon, low quality, ugly, multiple faces'

interpolate_barycentric(
    CM, img1, img2, img3,
    num_levels=3,
    out_dir='president_bary',
    prompt=prompt,
    n_prompt=n_prompt,
    ddim_steps=200,
    guide_scale=7.5,
    min_steps=0.3,
    max_steps=0.55,
    ddim_eta=0.0,
    optimize_cond=200,
    cond_lr=1e-4,
    schedule_type='linear',
    cond_path='data/president_bary200.pt',
    controls=None,
    n_choices=5,
    qc_prompts=(qc_prompt, qc_neg_prompt),
)
