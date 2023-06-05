## Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers [BCNet, CVPR 2021]

Lei Ke, Yu-Wing Tai, Chi-Keung Tang

[[`Paper`](https://openaccess.thecvf.com/content/CVPR2021/papers/Ke_Deep_Occlusion-Aware_Instance_Segmentation_With_Overlapping_BiLayers_CVPR_2021_paper.pdf)] [[`Original Github`](https://github.com/lkeab/BCNet)]

![image](https://github.com/trqminh/aistron/assets/30286786/7bd8c889-4ce6-4017-81d8-6e49b944a574)

In this repository, we aim to replicate BCNet in aistron, 
using the original BCNet [code](https://github.com/lkeab/BCNet) as our reference. 
We have implemented BCNet with the Faster R-CNN meta-architecture
and are also planning to explore its implementation with the FCOS meta-architecture. 
We encourage and welcome pull requests from the community.


## Training
All configs can be trained with:
```bash
python projects/BCNet/train_net.py --config-file projects/BCNet/path/to/config.yaml --num-gpus 1
```

## Evaluation
Model evaluation can be done as follows:
```bash
python projects/BCNet/train_net.py --config-file projects/BCNet/path/to/config.yaml \
    --eval-only MODEL.WEIGHTS /path/to/the/model/weight.pth
```

## Pretrained Models

### KINS2020 Dataset
| Name | Backbone | epochs |AP|AR|Visible AP| Trained model |
|-------|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[BCNet](configs/KINS2020/bcnet_R50_FPN_kins2020_6ep_bs1.yaml)|Resnet-50|~6|...|...|-|[model]()|
|[BCNet](configs/KINS2020/bcnet_R101_FPN_kins2020_6ep_bs1.yaml)|Resnet-101|~6|...|...|-|[model]()|


### D2SA Dataset
| Name | Backbone | epochs |AP|AR|Visible AP| Trained model |
|-------|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[BCNet](configs/D2SA/bcnet_R50_FPN_d2sa_18ep_bs2.yaml)|Resnet-50|~18|...|...|-|[model]()|

### COCOA Dataset
COCOA No stuff, no class dataset

| Name | Backbone | epochs |AP|AR|Visible AP| Trained model |
|-------|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[BCNet](configs/COCOA/bcnet_R101_FPN_cocoa_8ep_bs2.yaml)|Resnet-101|~8|...|...|-|[model]()|

### COCOA-cls Dataset
COCOA with classes

| Name | Backbone | epochs |AP|AR|Visible AP| Trained model |
|-------|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[BCNet](configs/COCOA-cls/bcnet_R50_FPN_cocoa_cls_8ep_bs2.yaml)|Resnet-50|~8|...|...|-|[model]()|
