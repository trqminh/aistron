# Prepare Datasets for AIStron
This document explains how to setup the builtin datasets for AIStron. 
AIStron currently support amodal segmentation methods
as instance segmentation level only (not supported semantic and panoptic)   

AIStron has builtin support for a few datasets. The datasets are assumed to exist in a directory specified by the environment variable `AISTRON_DATASETS`.
```
$AISTRON_DATASETS/
  KINS/
  D2SA/
  COCOA/
```

You can set the location for builtin datasets by `export AISTRON_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

In this document, we provide the python scripts that turn the about-to-be-registered datasets' annotations
to one standard coco-like JSON with shared the same convention of annotation keys.
The annotation keys are either annotated or derived and are typical ground truths for almost AIS methods.
Those annotations keys are as follows:
- `'amodal_bbox'`:
- `'visible_bbox'`:
- `'amodal_seg'`:
- `'visible_seg'`:
- `'background_objs_segm'`:
- `'occluder_segm'`:
- `'segmentation'`:
- `'bbox'`:

TODO: describe these above

## Prepare KINS dataset
After downloading [KINS dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset)
Let say we organize dataset as follow:
```
$AISTRON_DATASETS/
  KINS/
    annotations/
      update_train_2020.json
      update_test_2020.json
    train_imgs/
    test_imgs/
```
Running the following to get the aistron universal format json file for KINS annotation:
```
bash scripts/prepare_kins.sh
```
After this, there will be two generated annotations: `update_train_2020_aistron.json` and `update_test_2020_aistron.json`

Finally, the expected dataset structure for KINS:
```
$AISTRON_DATASETS/
  KINS/
    annotations/
      update_train_2020.json
      update_train_2020_aistron.json
      update_test_2020.json
      update_test_2020_aistron.json
    train_imgs/
    test_imgs/
```
