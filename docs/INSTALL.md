## Installation

### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- Other requirements:
    - ninja yacs cython matplotlib tqdm shapely
    - opencv-python
    - sklearn
    - scikit-image
    - timm
    - setuptools==59.5.0
- For using aistron as a third-party library: 
    - Current development: `pip install git+https://github.com/trqminh/aistron`
    - Release version: `pip install git+https://github.com/trqminh/aistron@v0.1.1`

### Example conda environment setup
```bash
conda create -n aistron python=3.8 -y
conda activate aistron
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch

# coco api
pip install pycocotools

#  detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# other dependencies
pip install ninja yacs cython matplotlib tqdm shapely
pip install opencv-python
pip install sklearn
pip install scikit-image
pip install timm
pip install setuptools==59.5.0

# aistron
pip install git+https://github.com/trqminh/aistron
```