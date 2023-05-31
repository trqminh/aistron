export AISTRON_DATASETS=../data/datasets/

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_train2014_detectron_no_stuff.json \

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_val2014_detectron_no_stuff.json \

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_test2014_detectron_no_stuff.json \

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_train2014_with_classes.json \

python datasets/prepare_cocoa.py \
        $AISTRON_DATASETS/COCOA/annotations/COCO_amodal_val2014_with_classes.json \
