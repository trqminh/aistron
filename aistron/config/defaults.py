from detectron2.config import CfgNode as CN

def add_aistron_config(cfg):
    cfg.AISTRON = CN()

    # this config is used to check the upper bound performance
    # of the mask head, not depending on the detector performance
    cfg.AISTRON.INFERENCE_WITH_GT_BOXES = False
