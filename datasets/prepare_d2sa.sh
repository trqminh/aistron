export AISTRON_DATASETS=../data/datasets/

python datasets/prepare_d2sa.py \
        $AISTRON_DATASETS/D2SA/d2s_amodal_annotations_v1/D2S_amodal_training_rot0.json \

python datasets/prepare_d2sa.py \
        $AISTRON_DATASETS/D2SA/d2s_amodal_annotations_v1/D2S_amodal_augmented.json \

python datasets/prepare_d2sa.py \
        $AISTRON_DATASETS/D2SA/d2s_amodal_annotations_v1/D2S_amodal_validation.json \