export CUDA_VISBLE_DEVICES=0
export AISTRON_DATASETS=../data/datasets/

python train_net.py --config-file configs/KINS2020/aisformer_R50_FPN_kins2020_6ep_bs1.yaml --num-gpus 1 \
