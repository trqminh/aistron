export CUDA_VISIBLE_DEVICES=0

model_output='../data/train_outputs/test_aistron'

python3 train_net.py --num-gpus 1 \
        --config-file ${model_output}/config.yaml \
        --eval-only MODEL.WEIGHTS ${model_output}/model_final.pth \
        2>&1 | tee ${model_output}/test_log.txt

