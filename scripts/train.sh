export CUDA_VISIBLE_DEVICES=0
export AISTRON_DATASETS=../data/datasets/

config_file=configs/Base-RCNN-FPN-KINS2020.yaml
python train_net.py --config-file ${config_file} --num-gpus 1 \
    OUTPUT_DIR ../data/train_outputs/test/ \
    DATALOADER.NUM_WORKERS 0
