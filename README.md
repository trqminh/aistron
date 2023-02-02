<img src="./assets/aistron_logo.svg" width = "320" height = "110" alt="logo" />

## 1. Introduction

AIStron is an amodal instance segmentation (AIS) library that provides current AIS methods. The library is built as a detectron2 (version 0.6) project. This repository works with Pytorch 1.10+ or higher.

Major features:
aistron provides the following features, note that the features are mostly collected from other AIS repositories (we add the credit to the specific repository). We simply put all this together for better maintaining and developing new methods.
- Data pipeline: 

- Amodal Instance Segmentation Evalutor:

- Methods:
    - [x] (VRSP-Net)[]
    - [x] BCNet
    - [x] AISFormer


## 2. todos 


## 3. Installation
Follow the below commands to create a python environment for aistron
```
conda create -n aistron python=3.8 -y
conda activate aistron
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch
pip install ninja yacs cython matplotlib tqdm shapely
pip install opencv-python==4.4.0.40
pip install sklearn
pip install scikit-image
pip install timm==0.4.12

# coco api
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# clone detectron2 v0.6
cd detectron2/
python setup.py build develop

# optional, just in case
pip install setuptools==59.5.0
```
