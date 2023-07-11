## Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers [BCNet, CVPR 2021]

Lei Ke, Yu-Wing Tai, Chi-Keung Tang

[[`Paper`](https://openaccess.thecvf.com/content/CVPR2021/papers/Ke_Deep_Occlusion-Aware_Instance_Segmentation_With_Overlapping_BiLayers_CVPR_2021_paper.pdf)] [[`Original Github`](https://github.com/lkeab/BCNet)]

![image](https://github.com/trqminh/aistron/assets/30286786/3bccf402-5b62-4507-83cc-45db5db5065b)

In this repository, we aim to replicate BCNet in aistron, 
using the original BCNet [code](https://github.com/lkeab/BCNet) as our reference. 
We have implemented BCNet with the Faster R-CNN meta-architecture
and are also planning to explore its implementation with the FCOS meta-architecture. 
We encourage and welcome pull requests from the community.


## Training
All configs can be trained with:
```bash
export AISTRON_DATASETS=/path/to/datasets/
python projects/BCNet/train_net.py --config-file projects/BCNet/path/to/config.yaml --num-gpus 1
```
As this repository primarily deals with amodal segmentation, we don't have the training configuration for training BCNet on COCO dataset. However, it can be easily achieved by making modifications to the forward pass of the mask head.

## Evaluation
Model evaluation can be done as follows:
```bash
export AISTRON_DATASETS=/path/to/datasets/
python projects/BCNet/train_net.py --config-file projects/BCNet/path/to/config.yaml \
    --eval-only MODEL.WEIGHTS /path/to/the/model/weight.pth
```

## Pretrained Models
Coming soon

