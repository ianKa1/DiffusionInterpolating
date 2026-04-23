## STEP 1 install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda config --add envs_dirs /workspace/conda-envs
conda config --add pkgs_dirs /workspace/conda-pkgs

## STEP 2 redirect huggingface cache
export HF_HOME=/workspace/hf
export TRANSFORMERS_CACHE=/workspace/hf/transformers
export HF_DATASETS_CACHE=/workspace/hf/datasets
export HUGGINGFACE_HUB_CACHE=/workspace/hf/hub
source ~/.bashrc


## STEP 3 Conda env
conda env create -f environment.yaml
conda activate control