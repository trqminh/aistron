## Getting started with aistron

This document provides a brief intro of the usage of aistron

### 1. Inference Demo with Pre-trained Models
1. Pick a model and its config file from
  [model zoo](../docs/MODEL_ZOO.md),
  for example, [`configs/KINS2020/maskrcnn_R50_FPN_kins2020_6ep_bs1.yaml`](../configs/KINS2020/maskrcnn_R50_FPN_kins2020_6ep_bs1.yaml).
2. We provide `demo.py` that is able to demo builtin configs. Run it with:
```bash
python demo/demo.py --config-file configs/KINS2020/maskrcnn_R50_FPN_kins2020_6ep_bs1.yaml \
  --input input1.jpg input2.jpg \
  --output demo_output.jpg \
  --segm_type amodal \ # visible or amodal
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and output the result to output.jpg.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`. We suggest using model weight trained on COCOA datasets for better experience.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

We provide some demo images from datasets to run the demo on, [assets/demo_examples/](../assets/demo_examples/). 
For example:
```bash
python demo/demo.py --config-file configs/KINS2020/maskrcnn_R50_FPN_kins2020_6ep_bs1.yaml \
        --input assets/demo_examples/kins_example.png \
        --output demo_output.png \
        --opts MODEL.WEIGHTS /path/to/the/weight.pth \
```
### 2. Train and Evaluation
We provide a script `tools/train_net.py`, that is made to train all the configs provided in aistron. It also can be used
as an example for training your project that uses aistron as a library.

To train a model with `tools/train_net.py`, first setup the corresponding datasets following [`datasets/README.md`](../datasets/README.md), then run the following command to train with a specific config file:
```bash
export AISTRON_DATASETS=../data/datasets/
config_file=configs/KINS2020/maskrcnn_R50_FPN_kins_6ep_bs1.yaml
python tools/train_net.py --config-file ${config_file} --num-gpus 1 \
```
During training, the model configs, checkpoints and logs will be saved to the directory specified in the `OUTPUT_DIR` variable in the corresponding config file.

To evaluate a model's performance, use:
```bash
model_output='../data/train_outputs/aistron/maskrcnn/maskrcnn_R50_FPN_kins2020_6ep_bs1'
python3 tools/train_net.py --num-gpus 1 \
        --config-file ${model_output}/config.yaml \
        --eval-only MODEL.WEIGHTS ${model_output}/model_final.pth \
        2>&1 | tee ${model_output}/test_log.txt
```