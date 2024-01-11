export AISTRON_DATASETS=../data/datasets/

python datasets/prepare_coco.py \
        $AISTRON_DATASETS/coco/annotations/instances_val2017.json \

python datasets/prepare_coco.py \
        $AISTRON_DATASETS/coco/annotations/instances_train2017.json \
