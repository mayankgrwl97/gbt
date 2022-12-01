# Environment Setup

## Instructions

Run the following commands to create a conda environment with required dependencies.

```bash
# create conda env
conda create -n gbt python=3.8
conda activate gbt

# install pytorch
conda install -c conda-forge ninja
conda install -c conda-forge cxx-compiler=1.3.0
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"

# install co3d
pip install "git+https://github.com/facebookresearch/co3d.git"

# misc installations
conda install -c anaconda ipython
pip install omegaconf hydra-core accelerate matplotlib plotly opencv-python lpips scikit-image
```
