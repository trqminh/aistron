## Getting started with aistron

This document provides a brief intro of the usage of aistron

### Inference Demo with Pre-trained Models
1. Pick a model and its config file from
  [model zoo](../docs/MODEL_ZOO.md),
  for example, `../configs/KINS2020/maskrcnn_R50_FPN_kins.yaml`.
2. We provide `demo.py` that is able to demo builtin configs. Run it with:
```bash
python demo/demo.py --config-file configs/KINS2020/maskrcnn_R50_FPN_kins.yaml \
  --input input1.jpg input2.jpg \
  --output demo_output.jpg
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and output the result to output.jpg.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

We provide some demo images from datasets to run the demo on, [../assets/demo_examples/](../assets/demo_examples/). 
For example:
```bash
python demo/demo.py --config-file configs/KINS2020/maskrcnn_R50_FPN_kins.yaml \
        --input assets/demo_examples/kins_example.png \
        --output demo_output.png \
        --opts MODEL.WEIGHTS ../data/train_outputs/aistron/maskrcnn/maskrcnn_R50_FPN_kins/model_final.pth \
```
