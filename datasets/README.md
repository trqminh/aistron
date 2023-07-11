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

In this document, we provide the python scripts that turn the orginal annotations (from datasets, e.g. KINS, D2SA, COCOA,...)
to one standard coco-like JSON with shared the same convention of annotation keys for amodal instance segmentation.
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

## Prepare KINS and KINS2020 dataset
The dataset can be downloaded from here: [KINS dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset).
There are two version of annotation available so we decide to split it to KINS (`instance_{train/val}.json`) and KINS2020 (`update_{train/test}_2020.json`).

For the builtin dataset registry to work, you should organize the dataset as follow:
```
$AISTRON_DATASETS/
  KINS/
    annotations/
      update_train_2020.json
      update_test_2020.json
      instances_train.json
      instances_val.json
    training/
    testing/
```
Then, running the following commands to get the aistron universal format json file for KINS annotation:
```bash
export AISTRON_DATASETS=path/to/where/you/put/your/datasets
python datasets/prepare_kins.py \
        $AISTRON_DATASETS/KINS/annotations/instances_train.json \
python datasets/prepare_kins.py \
        $AISTRON_DATASETS/KINS/annotations/instances_val.json \
python datasets/prepare_kins2020.py \
        $AISTRON_DATASETS/KINS/annotations/update_train_2020.json \
python datasets/prepare_kins2020.py \
        $AISTRON_DATASETS/KINS/annotations/update_test_2020.json \
```

After this, there will be generated annotations for each one as follow:
```
$AISTRON_DATASETS/
  KINS/
    annotations/
      update_train_2020.json
      update_train_2020_aistron.json
      update_test_2020.json
      update_test_2020_aistron.json
      instances_train.json
      instances_train_aistron.json
      instances_val.json
      instances_val_aistron.json
    training/
    testing/
```

With this expected dataset structure, when aistron is imported (`import aistron`), it will register 
the four kins train and test datasets under the name of `kins_train`, `kins_test`, `kins2020_train` and `kins2020_test`, respectively.
These names can be used by specifying under `DATASETS.TRAIN` and `DATASETS.TEST` in a config file.
Take a look at [`aistron/data/datasets/register_kins.py`](../aistron/data/datasets/register_kins.py) for
more details.

## Prepare D2SA dataset
TODO: describe more on the dataset preparation and structure

Running the following commands to get the aistron universal format json file for D2SA annotations:
```bash
export AISTRON_DATASETS=../data/datasets/

python datasets/prepare_d2sa.py \
        $AISTRON_DATASETS/D2SA/d2s_amodal_annotations_v1/D2S_amodal_training_rot0.json \

python datasets/prepare_d2sa.py \
        $AISTRON_DATASETS/D2SA/d2s_amodal_annotations_v1/D2S_amodal_augmented.json \

python datasets/prepare_d2sa.py \
        $AISTRON_DATASETS/D2SA/d2s_amodal_annotations_v1/D2S_amodal_validation.json \
```

## Prepare COCOA and COCOA-cls dataset
TODO: describe more on the dataset preparation and structure

Running the following commands to get the aistron universal format json file for COCOA annotations:
```bash
export AISTRON_DATASETS=../data/datasets/

# COCOA
python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_train2014_detectron_no_stuff.json \

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_val2014_detectron_no_stuff.json \

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_test2014_detectron_no_stuff.json \

# COCOA-cls
python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_train2014_with_classes.json \

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_val2014_with_classes.json \
```

## Visualize datasets
Run the following command to overlay instance groundtruths on top of the images for specific registered dataset.

```bash
export AISTRON_DATASETS=../data/datasets/
output_dir=../data/viz/aistron_viz_gt_test_kins/ # directory to output the visualize images 
dataset_name=kins_train # kins2020_train, d2sa_train, or your_custom_datasets_name
segm_type=amodal # or visible (the mode of the visualized masks)

python tools/visualize_data.py \
    --output-dir ${output_dir} \
    --dataset-name ${dataset_name} \
    --segm_type ${segm_type} \
    --source "annotation" \

```