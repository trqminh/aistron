# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
import pycocotools.mask as mask_util
from detectron2.data.detection_utils import transform_keypoint_annotations
import detectron2.data.transforms as T
from detectron2.data.transforms import TransformGen

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)


"""
Most code inherit form Detectron2 DatasetMapper
"""

__all__ = ["AmodalDatasetMapper"]


class AmodalDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[AmodalDatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret
    

    def _transform_segm(self, segm, transforms, image_size):
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            return [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            return mask
        else:
            raise ValueError(
                "Cannot transform this segm of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )
    
    def _segms_to_polygon_masks(self, segms, mask_format, image_size):
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )

        return masks

    def _amodal_transform_instance_annotations(self, annotation, transforms, image_size, *, keypoint_hflip_indices=None):
        """
        Apply transforms to box, segmentation and keypoints annotations of a single instance.
        """
        if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)
        # bbox is 1d (per-instance bounding box)
        amodal_bbox = BoxMode.convert(annotation["amodal_bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # clip transformed amodal_bbox to image size
        amodal_bbox = transforms.apply_box(np.array([amodal_bbox]))[0].clip(min=0)
        annotation["amodal_bbox"] = np.minimum(amodal_bbox, list(image_size + image_size)[::-1])
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        visible_bbox = BoxMode.convert(annotation["visible_bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # clip transformed visible_bbox to image size
        visible_bbox = transforms.apply_box(np.array([visible_bbox]))[0].clip(min=0)
        annotation["visible_bbox"] = np.minimum(visible_bbox, list(image_size + image_size)[::-1])

        if "amodal_segm" in annotation:
            annotation["amodal_segm"] = self._transform_segm(annotation["amodal_segm"], transforms, image_size)
        if "visible_segm" in annotation:
            annotation["visible_segm"] = self._transform_segm(annotation["visible_segm"], transforms, image_size)
        if "background_objs_segm" in annotation:
            annotation["background_objs_segm"] = self._transform_segm(annotation["background_objs_segm"], transforms, image_size)
        if "occluder_segm" in annotation:
            annotation["occluder_segm"] = self._transform_segm(annotation["occluder_segm"], transforms, image_size)

        if "keypoints" in annotation:
            keypoints = transform_keypoint_annotations(
                annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
            )
            annotation["keypoints"] = keypoints

        return annotation

    def _amodal_annotations_to_instances(self, annos, image_size, mask_format="polygon"):
        """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.
        Args:
            annos (list[dict]): a list of instance annotations in one image, each
                element for one instance.
            image_size (tuple): height, width
        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes",
                "gt_masks", "gt_keypoints", "gt_amodal_boxes", "
                "gt_amodal_masks" 
                "gt_visible_masks"
                "gt_background_objs"
                "gt_occluder_masks"              
                if they can be obtained from `annos`.
                This is the format that builtin models expect.
        """
        amodal_boxes = (
            np.stack(
                [BoxMode.convert(obj["amodal_bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
            )
            if len(annos)
            else np.zeros((0, 4))
        )
        visible_boxes = (
            np.stack(
                [BoxMode.convert(obj["visible_bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
            )
            if len(annos)
            else np.zeros((0, 4))
        )
        target = Instances(image_size)
        target.gt_amodal_boxes = Boxes(amodal_boxes)
        target.gt_visible_boxes = Boxes(visible_boxes)
        
        target.gt_boxes = Boxes(amodal_boxes) # for using the default box heads in detectron2 (maskrcnn)

        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        if len(annos) and "amodal_segm" in annos[0]:
            amodal_segms = [obj["amodal_segm"] for obj in annos]
            visible_segms = [obj["visible_segm"] for obj in annos]
            background_objs_segms = [obj["background_objs_segm"] for obj in annos]
            occluder_segms = [obj["occluder_segm"] for obj in annos]

            target.gt_amodal_masks = self._segms_to_polygon_masks(amodal_segms, mask_format, image_size)
            target.gt_visible_masks = self._segms_to_polygon_masks(visible_segms, mask_format, image_size)
            target.gt_background_objs_masks = self._segms_to_polygon_masks(background_objs_segms, mask_format, image_size)
            target.gt_occluder_masks = self._segms_to_polygon_masks(occluder_segms, mask_format, image_size)

            # for using default mask heads in detectron2 (maskrcnn)
            target.gt_masks = target.gt_amodal_masks

        if len(annos) and "keypoints" in annos[0]:
            kpts = [obj.get("keypoints", []) for obj in annos]
            target.gt_keypoints = Keypoints(kpts)

        return target
        

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            self._amodal_transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = self._amodal_annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        '''
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        '''

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
