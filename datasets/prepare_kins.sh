export AISTRON_DATASETS=../data/datasets/

python datasets/prepare_kins.py \
        $AISTRON_DATASETS/KINS/annotations/update_train_2020.json \
        $AISTRON_DATASETS/KINS/train_imgs/ \ 

python datasets/prepare_kins.py \
        $AISTRON_DATASETS/KINS/annotations/update_test_2020.json \
        $AISTRON_DATASETS/KINS/test_imgs/ \ 
