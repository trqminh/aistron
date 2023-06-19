## AISFormer: Amodal Instance Segmentation with Transformer  [AISFormer, BMVC 2022]

Minh Tran, Khoa Vo, Kashu Yamazaki, Arthur Fernandes, Michael Kidd, Ngan Le

[[`Arxiv`](https://arxiv.org/pdf/2210.06323.pdf)] [[`Original Github`](https://github.com/UARK-AICV/AISFormer)]

![image](https://github.com/trqminh/aistron/assets/30286786/7f84b50d-12bc-4f8d-8507-2b0eb14519e9)

In this repository, we aim to replicate AISFormer in aistron, 
using the original AISFormer [code](https://github.com/UARK-AICV/AISFormer) as our reference. 


## Training
All configs can be trained with:
```bash
python projects/AISFormer/train_net.py --config-file projects/AISFormer/path/to/config.yaml --num-gpus 1
```

## Evaluation
Model evaluation can be done as follows:
```bash
python projects/AISFormer/train_net.py --config-file projects/AISFormer/path/to/config.yaml \
    --eval-only MODEL.WEIGHTS /path/to/the/model/weight.pth
```

## Pretrained Models
Coming soon.
