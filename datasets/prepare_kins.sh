export AISTRON_DATASETS=../data/datasets/

python datasets/prepare_kins2020.py \
        $AISTRON_DATASETS/KINS/annotations/update_train_2020.json \

python datasets/prepare_kins2020.py \
        $AISTRON_DATASETS/KINS/annotations/update_test_2020.json \

python datasets/prepare_kins.py \
        $AISTRON_DATASETS/KINS/annotations/instances_train.json \

python datasets/prepare_kins.py \
        $AISTRON_DATASETS/KINS/annotations/instances_val.json \
