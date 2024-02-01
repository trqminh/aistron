export CUDA_VISIBLE_DEVICES=0,1
#export AISTRON_DATASETS=../data/datasets/

#python projects/BCNet/train_net.py --config-file projects/BCNet/configs/KINS2020/bcnet_R50_FPN_kins2020_6ep_bs1.yaml --num-gpus 1 --resume \

#python projects/BCNet/train_net.py --config-file projects/BCNet/configs/KINS2020/bcnet_R101_FPN_kins2020_6ep_bs1.yaml --num-gpus 1 --resume \

#python projects/BCNet/train_net.py --config-file projects/BCNet/configs/COCOA/bcnet_R101_FPN_cocoa_8ep_bs2.yaml --num-gpus 1 --resume \

#python projects/BCNet/train_net.py --config-file projects/BCNet/configs/COCOA-cls/bcnet_R50_FPN_cocoa_cls_8ep_bs2.yaml --num-gpus 1 --resume \

#python projects/BCNet/train_net.py --config-file projects/BCNet/configs/D2SA/bcnet_R50_FPN_d2sa_18ep_bs2.yaml --num-gpus 1 --resume \

# the training command below could lead to the unused parameter error
# change line 74 in detectron2/engine/defaults.py to:
# ddp = DistributedDataParallel(model, find_unused_parameters=True, **kwargs)
python projects/BCNet/train_net.py --config-file projects/BCNet/configs/coco/bcnet_R50_FPN_coco_1x.yaml --num-gpus 2 --resume \