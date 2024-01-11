import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np
import cv2
import pycocotools
from PIL import Image
from skimage import measure

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes, polygons_to_bitmask
from fvcore.common.file_io import PathManager, file_lock


from detectron2.data import MetadataCatalog, DatasetCatalog
import sys

from PIL import Image, ImageDraw, ImageFilter
import imantics
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
from copy import deepcopy

from detectron2.utils.logger import setup_logger
import sys
import pycocotools.mask as mask_util

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def process_occluder_gt_and_misc(json_file, dataset_name=None, extra_annotation_keys=None):
    """
    To generate the background object mask and occluder gts given the converted aistron json (from a amodal dataset)
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
        if 'tmp' in json_file:
            os.remove(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        print('meta:', meta)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    sum_box = 0
    sum_co_box = 0
    intersect_rate = 0.0
    intersect_num = 0

    num_instances_without_valid_segmentation = 0
    index_c = 0
    n_i_seg_fail = 0

    
    for jdex, (img_dict, anno_dict_list) in enumerate(imgs_anns):
        record = {}
        #record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["file_name"] = img_dict["file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        print('file name:', jdex, ':', record["file_name"])
        # if jdex == 437:
        #     import pdb;pdb.set_trace()
        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj['amodal_bbox'] = anno['amodal_bbox']
            obj['visible_bbox'] = anno['visible_bbox']

            segm = anno.get("amodal_segm", None)
            assert segm != None
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    if not all(isinstance(elem, list) for elem in segm):
                        # this is the exceptional case of cocoa json where
                        # maybe they assume there is just one mask blob inside of the region of interest
                        segm = [segm]
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["amodal_segm"] = segm

            i_segm = anno.get("visible_segm", None) 
            if i_segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(i_segm, dict):
                    # filter out invalid polygons (< 3 points)
                    if not all(isinstance(elem, list) for elem in i_segm):
                        # this is the exceptional case of cocoa json where
                        # maybe they assume there is just one mask blob inside of the region of interest
                        i_segm = [i_segm]
                    i_segm = [poly for poly in i_segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(i_segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["visible_segm"] = i_segm


            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        # record["annotations"] = objs
        seg_list = []
        i_seg_list = []

        tmp_objs = deepcopy(objs)
        objs = []
        for obj in tmp_objs:
            if 'visible_segm' in obj: 
                seg_list.append(obj['amodal_segm'])
                i_seg_list.append(obj['visible_segm'])
                objs.append(obj)
            else:
                n_i_seg_fail += 1

        record["annotations"] = objs

        #print('seg list:', seg_list)

        #dirname = "mask-vis"
        #os.makedirs(dirname, exist_ok=True)

        bitmask_list = []
        if len(seg_list) > 0:
            for index, seg in enumerate(seg_list):
                #print('seg len:', len(seg))
                if isinstance(seg, dict):
                    try:
                        bitmask = mask_util.decode(seg)
                    except:
                        seg = pycocotools.mask.frPyObjects(seg, seg['size'][0], seg['size'][1])
                        bitmask = mask_util.decode(seg)
                else:
                    invalid = False
                    for sub_seg in seg:
                        #print('seg len:', len(sub_seg))
                        if len(sub_seg) < 6:
                            invalid = True
                    if not invalid:
                        bitmask = polygons_to_bitmask(seg, img_dict["height"], img_dict["width"])
                    else:
                        bitmask = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=bool)

                bitmask_list.append(bitmask.astype('int'))

        i_bitmask_list = []
        if len(i_seg_list) > 0:
            for index, seg in enumerate(i_seg_list):
                #print('seg len:', len(seg))
                if isinstance(seg, dict):
                    try:
                        bitmask = mask_util.decode(seg)
                    except:
                        seg = pycocotools.mask.frPyObjects(seg, seg['size'][0], seg['size'][1])
                        bitmask = mask_util.decode(seg)
                else:
                    invalid = False
                    for sub_seg in seg:
                        #print('seg len:', len(sub_seg))
                        if len(sub_seg) < 6:
                            invalid = True
                    if not invalid:
                        bitmask = polygons_to_bitmask(seg, img_dict["height"], img_dict["width"])
                    else:
                        bitmask = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=bool)

                i_bitmask_list.append(bitmask.astype('int'))


        box_list = []
        for obj in objs:
            box_list.append([obj['amodal_bbox'][0],obj['amodal_bbox'][1],obj['amodal_bbox'][0]+obj['amodal_bbox'][2],obj['amodal_bbox'][1]+obj['amodal_bbox'][3]])

        i_box_list = []
        for obj in objs:
            i_box_list.append([obj['visible_bbox'][0],obj['visible_bbox'][1],obj['visible_bbox'][0]+obj['visible_bbox'][2],obj['visible_bbox'][1]+obj['visible_bbox'][3]])

        box_mask_list = []
        for index, obj in enumerate(objs):
            box_mask = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=int)
            box_mask[int(box_list[index][1]):int(box_list[index][3]),int(box_list[index][0]):int(box_list[index][2])] = 1
            box_mask_list.append(box_mask)

        i_box_mask_list = []
        for index, obj in enumerate(objs):
            i_box_mask = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=int)
            i_box_mask[int(i_box_list[index][1]):int(i_box_list[index][3]),int(i_box_list[index][0]):int(i_box_list[index][2])] = 1
            i_box_mask_list.append(i_box_mask)

        sum_box += len(box_list)

        for index1, a_box in enumerate(box_list):
            union_mask_whole = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=int)
            for index2, b_box in enumerate(i_box_list):
                if index1 != index2:
                    iou = bb_intersection_over_union(a_box, b_box)
                    if iou > 0.05:
                        # union_mask = np.multiply(box_mask_list[index1], bitmask_list[index2])
                        union_mask = np.multiply(box_mask_list[index1], i_bitmask_list[index2])
                        union_mask_whole += union_mask
            
            print("===========================================")
            print('bit mask area:', bitmask_list[index1].sum())
            union_mask_whole[union_mask_whole > 1.0] = 1.0
            print('cropped union mask area:', union_mask_whole.sum())
            intersect_mask = union_mask_whole * bitmask_list[index1]
            print('intersect mask area:', intersect_mask.sum()) 
            print('intersect rate:', intersect_mask.sum()/float(bitmask_list[index1].sum()))
            print("===========================================")
            
            if intersect_mask.sum() >= 1.0:
                intersect_num += 1

            if float(bitmask_list[index1].sum()) > 1.0:
                intersect_rate += intersect_mask.sum()/float(bitmask_list[index1].sum())
            
            union_mask_non_zero_num = np.count_nonzero(union_mask_whole.astype(int))
            record["annotations"][index1]['background_objs_segm'] = []
            record["annotations"][index1]['occluder_segm'] = []
            if union_mask_non_zero_num > 20:
                sum_co_box += 1
                contours = measure.find_contours(union_mask_whole.astype(int), 0)
                for contour in contours:
                    if contour.shape[0] > 500: 
                        contour = np.flip(contour, axis=1)[::10,:]
                    elif contour.shape[0] > 200: 
                        contour = np.flip(contour, axis=1)[::5,:]
                    elif contour.shape[0] > 100: 
                        contour = np.flip(contour, axis=1)[::3,:]
                    elif contour.shape[0] > 50: 
                        contour = np.flip(contour, axis=1)[::2,:]
                    else:
                        contour = np.flip(contour, axis=1)

                    segmentation = contour.ravel().tolist()
                    record["annotations"][index1]['background_objs_segm'].append(segmentation)
                    record["annotations"][index1]['occluder_segm'].append(segmentation)
        
        dataset_dicts.append(record)
        #if jdex > 10000:
        #    break

    #print('sum intersect rate:', intersect_rate)
    #print('sum box:', sum_box)

    avg_intersect_rate = intersect_rate/float(sum_box)
    avg_intersect_rate_over_inter = intersect_rate/float(intersect_num)
    #print('avg rate:', avg_intersect_rate)
    #print('avg rate over intersect:', avg_intersect_rate_over_inter)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
 
    return dataset_dicts



def convert_to_aistron_cocolike_dict(dataset_dicts, categories):
    """
    given: dataset instance segmentation dictionary and categories
    return: a coco-like dictionary annotation 
    """
    print('categories:', categories)
    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            amodal_bbox = annotation["amodal_bbox"]
            bbox_mode = annotation["bbox_mode"]
            amodal_bbox = BoxMode.convert(amodal_bbox, bbox_mode, BoxMode.XYWH_ABS)
            bbox_xy = BoxMode.convert(amodal_bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            amodal_area = Boxes([bbox_xy]).area()[0].item()
            
            visible_bbox = annotation["visible_bbox"]
            bbox_mode = annotation["bbox_mode"]
            visible_bbox = BoxMode.convert(visible_bbox, bbox_mode, BoxMode.XYWH_ABS) 
            
            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["amodal_bbox"] = [round(float(x), 3) for x in amodal_bbox]
            coco_annotation["visible_bbox"] = [round(float(x), 3) for x in visible_bbox]
            coco_annotation["amodal_area"] = amodal_area
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)
            coco_annotation["category_id"] = annotation["category_id"]

            # keep the annotation still works as standard COCO
            coco_annotation["bbox"] = coco_annotation["amodal_bbox"]
            coco_annotation["area"] = coco_annotation["amodal_area"]

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "amodal_segm" in annotation:
                coco_annotation["amodal_segm"] = annotation["amodal_segm"]
                # keep the annotation still works as standard COCO
                coco_annotation["segmentation"] = annotation["amodal_segm"]
            if "background_objs_segm" in annotation:
                coco_annotation["background_objs_segm"] = annotation["background_objs_segm"]
            if "visible_segm" in annotation:
                coco_annotation["visible_segm"] = annotation["visible_segm"]
            if "occluder_segm" in annotation:
                coco_annotation["occluder_segm"] = annotation["occluder_segm"]


            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict
