## Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers [BCNet, CVPR 2021]

Lei Ke, Yu-Wing Tai, Chi-Keung Tang

[[`Paper`](https://openaccess.thecvf.com/content/CVPR2021/papers/Ke_Deep_Occlusion-Aware_Instance_Segmentation_With_Overlapping_BiLayers_CVPR_2021_paper.pdf)] [[`Github`](https://github.com/lkeab/BCNet)]

TODO: Insert Figure here

In this repository, we reproduce BCNet in aistron based on the original code [BCNet](https://github.com/lkeab/BCNet).
We implement BCNet with Faster R-CNN as meta-arch. We will attempt to implement BCNet with FCOS meta-arch as well. 
Pull requests are welcome.


## Training
All configs can be trained with:
```bash
cd projects/BCNet/
python train_net.py --config-file /path/to/config.yaml --num-gpus 1
```

## Evaluation
Model evaluation can be done as follows:
```bash
cd projects/BCNet/
python train_net.py --config-file projects/BCNet/configs/path/to/config.yaml \
    --eval-only MODEL.WEIGHTS /path/to/the/model/weight.pth
```

## Pretrained Models

### KINS2020 Dataset
| Name | Backbone | epochs |AP|AR|Visible AP| Trained model |
|-------|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[BCNet]()|Resnet-101|~6|...|...|-|[model]()|


### D2SA Dataset
| Name | Backbone | epochs |AP|AR|Visible AP| Trained model |
|-------|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[BCNet]()|Resnet-50|~18|...|...|-|[model]()|

### COCOA Dataset
COCOA No stuff, no class dataset

| Name | Backbone | epochs |AP|AR|Visible AP| Trained model |
|-------|:---:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[BCNet]()|Resnet-101|~8|...|...|-|[model]()|
