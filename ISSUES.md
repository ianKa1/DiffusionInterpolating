# Setup Issues Log

## 1. conda not in PATH
**Problem:** After installing miniconda, `conda` command not found.  
**Cause:** `conda init bash` was never run, so no shell integration was added to `~/.bashrc`.  
**Fix:** `~/miniconda3/bin/conda init bash && source ~/.bashrc`

---

## 2. Hardcoded NFS path in sample scripts
**Problem:** All scripts in `sample_scripts/` have:
```python
sys.path.append(osp.expandvars('$NFS/code/controlnet/controlnet'))
```
`$NFS` is unset in this environment, so imports fail.  
**Fix:** Changed to `sys.path.append('/workspace')` in `alien.py`, and created symlink:
```bash
ln -s /workspace/DiffusionInterpolating /workspace/controlnet
```
So `from controlnet import cm` resolves to `/workspace/DiffusionInterpolating/cm.py`.

---

## 3. `share` module not found
**Problem:** `cm.py` does `from share import *`. Running the script from `sample_scripts/` adds that directory to `sys.path`, not the repo root where `share.py` lives.  
**Fix:** Set `PYTHONPATH=/workspace/DiffusionInterpolating` when running.

---

## 4. Wrong working directory for model paths
**Problem:** `cm.py` loads models with relative paths like `./controlnet/models/cldm_v15.yaml`. This assumes CWD is the *parent* of the repo (i.e. `/workspace`), not inside it.  
**Fix:** Run all scripts from `/workspace`:
```bash
cd /workspace && PYTHONPATH=/workspace/DiffusionInterpolating \
  /workspace/conda-envs/control/bin/python DiffusionInterpolating/sample_scripts/alien.py
```

---

## 5. Missing `data/` directory
**Problem:** Scripts open images from `data/alienX.png` (relative to CWD `/workspace`). No such directory exists.  
**Fix:** Created symlink pointing to the sample images:
```bash
ln -s /workspace/DiffusionInterpolating/sample_imgs /workspace/data
```

---

## 6. Missing system library: `libSM.so.6`
**Problem:** OpenCV fails to import:
```
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
```
**Fix:**
```bash
apt-get update && apt-get install -y libxrender1 libxext6 libsm6
```

---

## 7. No model checkpoints
**Problem:** No `.ckpt` or `.pth` weight files present anywhere. `cm.py` expects them at `./controlnet/models/`.  
**Fix:** Downloaded from `lllyasviel/ControlNet` on HuggingFace:
- `models/control_sd15_openpose.pth` (1.9 GB)
- `annotator/ckpts/body_pose_model.pth` (200 MB)
- `annotator/ckpts/hand_pose_model.pth` (141 MB)

---

## 8. No SD 2.1 models available
**Problem:** `alien.py` uses `ContextManager()` which defaults to `version='2.1'`, requiring `openpose-sd21.ckpt`. The only public source found (`lllyasviel/ControlNet`) only has SD 1.5 models.  
**Fix:** Changed `alien.py` to `ContextManager(version='1.5')`.  
**Note:** `thibaud/controlnet-sd21-openpose-diffusers` exists but is in diffusers format (different state dict keys — incompatible without a conversion script).

---

## 9. CLIP tokenizer/model download failing
**Problem:** SD 1.5 model init tries to load `openai/clip-vit-large-patch14` via transformers:
```
OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'
```
**Root cause:** Version sandwich between `transformers==4.19.2` and `huggingface_hub`:
- `huggingface_hub==0.36.2` (installed by conda): API incompatible with `transformers==4.19.2` — calls to `cached_download` fail silently.
- `huggingface_hub==0.9.1` (downgrade attempt): API compatible, but HuggingFace CDN now returns 307 redirects with no `Content-Length` header. Old `huggingface_hub` interprets missing `Content-Length` as empty response → all downloaded files are 0 bytes.
- Middle versions (e.g. `0.12.0`) have the same 0-byte problem.

The underlying `requests` library follows 307 redirects and downloads correctly — only `huggingface_hub`'s download layer is broken.

**Fix:**
1. Downloaded tokenizer and model files directly using `requests`:
   - `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `config.json`, `pytorch_model.bin`
   - Saved to `/workspace/clip-vit-large-patch14/`
2. Patched default version in `ldm/modules/encoders/modules.py` line 95:
   ```python
   # Before:
   def __init__(self, version="openai/clip-vit-large-patch14", ...):
   # After:
   def __init__(self, version="/workspace/clip-vit-large-patch14", ...):
   ```
3. Cleared stale `.pyc` cache: `find ldm -name "*.pyc" -delete`
