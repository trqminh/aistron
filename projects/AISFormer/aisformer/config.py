from detectron2.config import CfgNode as CN

def add_aisformer_config(cfg):
    cfg.AISFORMER = CN()
    cfg.AISFORMER.INVISIBLE_MASK_LOSS = True
    cfg.AISFORMER.N_HEADS = 2
    cfg.AISFORMER.N_LAYERS = 1
    cfg.AISFORMER.EMB_DIM = 256
