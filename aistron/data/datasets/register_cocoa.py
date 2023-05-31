'''
COCOA and COCOA-cls dataset
register cocoa and cocoa-cls dataset
'''

import os
import random
from os.path import join
from .coco_amodal import register_aistron_cocolike_instances

RANDOM_COLOR = [random.randint(0, 255) for _ in range(3)]
COCOA_CATEGORIES = [
    {'color': RANDOM_COLOR, 'isthing': 1, 'supercategory': 'superobject', 'id': 1, 'name': 'object'}
]


COCOA_cls_CATEGORIES = [
    {'isthing': 1, 'supercategory': 'person', 'id': 1, 'name': 'person', 'color': [33, 28, 99]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle', 'color': [237, 38, 21]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 3, 'name': 'car', 'color': [80, 174, 39]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle', 'color': [184, 177, 32]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 5, 'name': 'airplane', 'color': [152, 171, 106]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 6, 'name': 'bus', 'color': [5, 203, 201]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 7, 'name': 'train', 'color': [67, 223, 96]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 8, 'name': 'truck', 'color': [18, 130, 26]},
    {'isthing': 1, 'supercategory': 'vehicle', 'id': 9, 'name': 'boat', 'color': [173, 4, 161]},
    {'isthing': 1, 'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light', 'color': [165, 90, 48]},
    {'isthing': 1, 'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant', 'color': [35, 89, 54]},
    {'isthing': 1, 'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign', 'color': [105, 167, 170]},
    {'isthing': 1, 'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter', 'color': [7, 150, 62]},
    {'isthing': 1, 'supercategory': 'outdoor', 'id': 15, 'name': 'bench', 'color': [163, 138, 133]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 16, 'name': 'bird', 'color': [139, 70, 143]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 17, 'name': 'cat', 'color': [83, 12, 66]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 18, 'name': 'dog', 'color': [225, 88, 118]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 19, 'name': 'horse', 'color': [169, 65, 107]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 20, 'name': 'sheep', 'color': [166, 8, 40]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 21, 'name': 'cow', 'color': [71, 73, 148]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 22, 'name': 'elephant', 'color': [178, 218, 172]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 23, 'name': 'bear', 'color': [144, 159, 148]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 24, 'name': 'zebra', 'color': [23, 178, 231]},
    {'isthing': 1, 'supercategory': 'animal', 'id': 25, 'name': 'giraffe', 'color': [211, 251, 111]},
    {'isthing': 1, 'supercategory': 'accessory', 'id': 27, 'name': 'backpack', 'color': [175, 21, 200]},
    {'isthing': 1, 'supercategory': 'accessory', 'id': 28, 'name': 'umbrella', 'color': [82, 196, 164]},
    {'isthing': 1, 'supercategory': 'accessory', 'id': 31, 'name': 'handbag', 'color': [30, 21, 166]},
    {'isthing': 1, 'supercategory': 'accessory', 'id': 32, 'name': 'tie', 'color': [166, 136, 55]},
    {'isthing': 1, 'supercategory': 'accessory', 'id': 33, 'name': 'suitcase', 'color': [23, 223, 248]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 34, 'name': 'frisbee', 'color': [100, 23, 93]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 35, 'name': 'skis', 'color': [74, 211, 233]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 36, 'name': 'snowboard', 'color': [205, 50, 213]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 37, 'name': 'sports ball', 'color': [57, 183, 174]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 38, 'name': 'kite', 'color': [122, 61, 156]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 39, 'name': 'baseball bat', 'color': [97, 183, 216]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 40, 'name': 'baseball glove', 'color': [15, 2, 92]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 41, 'name': 'skateboard', 'color': [49, 11, 236]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 42, 'name': 'surfboard', 'color': [65, 171, 154]},
    {'isthing': 1, 'supercategory': 'sports', 'id': 43, 'name': 'tennis racket', 'color': [101, 222, 32]},
    {'isthing': 1, 'supercategory': 'kitchen', 'id': 44, 'name': 'bottle', 'color': [254, 76, 150]},
    {'isthing': 1, 'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass', 'color': [133, 252, 214]},
    {'isthing': 1, 'supercategory': 'kitchen', 'id': 47, 'name': 'cup', 'color': [41, 105, 139]},
    {'isthing': 1, 'supercategory': 'kitchen', 'id': 48, 'name': 'fork', 'color': [15, 67, 225]},
    {'isthing': 1, 'supercategory': 'kitchen', 'id': 49, 'name': 'knife', 'color': [74, 207, 136]},
    {'isthing': 1, 'supercategory': 'kitchen', 'id': 50, 'name': 'spoon', 'color': [91, 171, 91]},
    {'isthing': 1, 'supercategory': 'kitchen', 'id': 51, 'name': 'bowl', 'color': [32, 218, 157]},
    {'isthing': 1, 'supercategory': 'food', 'id': 52, 'name': 'banana', 'color': [107, 217, 115]},
    {'isthing': 1, 'supercategory': 'food', 'id': 53, 'name': 'apple', 'color': [142, 42, 101]},
    {'isthing': 1, 'supercategory': 'food', 'id': 54, 'name': 'sandwich', 'color': [7, 154, 101]},
    {'isthing': 1, 'supercategory': 'food', 'id': 55, 'name': 'orange', 'color': [123, 133, 18]},
    {'isthing': 1, 'supercategory': 'food', 'id': 56, 'name': 'broccoli', 'color': [110, 190, 106]},
    {'isthing': 1, 'supercategory': 'food', 'id': 57, 'name': 'carrot', 'color': [46, 131, 64]},
    {'isthing': 1, 'supercategory': 'food', 'id': 58, 'name': 'hot dog', 'color': [30, 207, 205]},
    {'isthing': 1, 'supercategory': 'food', 'id': 59, 'name': 'pizza', 'color': [170, 112, 109]},
    {'isthing': 1, 'supercategory': 'food', 'id': 60, 'name': 'donut', 'color': [87, 203, 181]},
    {'isthing': 1, 'supercategory': 'food', 'id': 61, 'name': 'cake', 'color': [243, 147, 162]},
    {'isthing': 1, 'supercategory': 'furniture', 'id': 62, 'name': 'chair', 'color': [204, 109, 61]},
    {'isthing': 1, 'supercategory': 'furniture', 'id': 63, 'name': 'couch', 'color': [211, 182, 94]},
    {'isthing': 1, 'supercategory': 'furniture', 'id': 64, 'name': 'potted plant', 'color': [99, 100, 148]},
    {'isthing': 1, 'supercategory': 'furniture', 'id': 65, 'name': 'bed', 'color': [228, 78, 220]},
    {'isthing': 1, 'supercategory': 'furniture', 'id': 67, 'name': 'dining table', 'color': [56, 151, 239]},
    {'isthing': 1, 'supercategory': 'furniture', 'id': 70, 'name': 'toilet', 'color': [109, 204, 186]},
    {'isthing': 1, 'supercategory': 'electronic', 'id': 72, 'name': 'tv', 'color': [231, 44, 242]},
    {'isthing': 1, 'supercategory': 'electronic', 'id': 73, 'name': 'laptop', 'color': [229, 51, 153]},
    {'isthing': 1, 'supercategory': 'electronic', 'id': 74, 'name': 'mouse', 'color': [124, 122, 22]},
    {'isthing': 1, 'supercategory': 'electronic', 'id': 75, 'name': 'remote', 'color': [247, 197, 216]},
    {'isthing': 1, 'supercategory': 'electronic', 'id': 76, 'name': 'keyboard', 'color': [138, 228, 75]},
    {'isthing': 1, 'supercategory': 'electronic', 'id': 77, 'name': 'cell phone', 'color': [195, 159, 122]},
    {'isthing': 1, 'supercategory': 'appliance', 'id': 78, 'name': 'microwave', 'color': [167, 89, 44]},
    {'isthing': 1, 'supercategory': 'appliance', 'id': 79, 'name': 'oven', 'color': [141, 39, 75]},
    {'isthing': 1, 'supercategory': 'appliance', 'id': 80, 'name': 'toaster', 'color': [5, 147, 148]},
    {'isthing': 1, 'supercategory': 'appliance', 'id': 81, 'name': 'sink', 'color': [231, 122, 131]},
    {'isthing': 1, 'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator', 'color': [226, 24, 135]},
    {'isthing': 1, 'supercategory': 'indoor', 'id': 84, 'name': 'book', 'color': [234, 57, 111]},
    {'isthing': 1, 'supercategory': 'indoor', 'id': 85, 'name': 'clock', 'color': [142, 183, 19]},
    {'isthing': 1, 'supercategory': 'indoor', 'id': 86, 'name': 'vase', 'color': [24, 67, 186]},
    {'isthing': 1, 'supercategory': 'indoor', 'id': 87, 'name': 'scissors', 'color': [47, 123, 141]},
    {'isthing': 1, 'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear', 'color': [235, 158, 75]},
    {'isthing': 1, 'supercategory': 'indoor', 'id': 89, 'name': 'hair drier', 'color': [84, 201, 56]},
    {'isthing': 1, 'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush', 'color': [160, 30, 87]},
]



def _get_cocoa_instances_meta(cat_list):
    thing_ids = [k["id"] for k in cat_list if k["isthing"] == 1]
    thing_colors = [k["color"] for k in cat_list if k["isthing"] == 1]
    # assert len(thing_ids) == 7, len(thing_ids)
    # Mapping from the incontiguous category id to an id in [0, 6]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in cat_list if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def register_cocoa(root):
    register_aistron_cocolike_instances("cocoa_train", _get_cocoa_instances_meta(COCOA_CATEGORIES),
        join(root, "COCOA/annotations/COCO_amodal_train2014_detectron_no_stuff_aistron.json"),
        join(root, "COCOA/train2014/")
    )

    register_aistron_cocolike_instances("cocoa_val", _get_cocoa_instances_meta(COCOA_CATEGORIES),
        join(root, "COCOA/annotations/COCO_amodal_val2014_detectron_no_stuff_aistron.json"),
        join(root, "COCOA/val2014/")
    )

    register_aistron_cocolike_instances("cocoa_test", _get_cocoa_instances_meta(COCOA_CATEGORIES),
        join(root, "COCOA/annotations/COCO_amodal_test2014_detectron_no_stuff_aistron.json"),
        join(root, "COCOA/test2014/")
    )

def register_cocoa_cls(root):
    register_aistron_cocolike_instances("cocoa_cls_train", _get_cocoa_instances_meta(COCOA_cls_CATEGORIES),
        join(root, "COCOA/annotations/COCO_amodal_train2014_with_classes_aistron.json"),
        join(root, "COCOA/train2014/")
    )

    register_aistron_cocolike_instances("cocoa_cls_val", _get_cocoa_instances_meta(COCOA_cls_CATEGORIES),
        join(root, "COCOA/annotations/COCO_amodal_val2014_with_classes_aistron.json"),
        join(root, "COCOA/val2014/")
    )

_root = os.getenv("AISTRON_DATASETS", "datasets")
register_cocoa(_root)
register_cocoa_cls(_root)
