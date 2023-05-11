# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn import functional as F

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances



def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_amodal_masks"):
        results.pred_amodal_masks = paste_masks_in_image(
            results.pred_amodal_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )
        if not results.has("pred_masks"):
            results.pred_masks = results.pred_amodal_masks # this is for coco evaluation to work
            # since it needs a pred_masks in prediction for coco api to create index
            # bbox can create index too, but if iou_type segm, they pop it out
            # it's in the loadRes method in coco.py

    if results.has("pred_visible_masks"):
        results.pred_visible_masks = paste_masks_in_image(
            results.pred_visible_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )


    if results.has("pred_masks") and not results.has('pred_amodal_masks'): # maskrcnn cases
        results.pred_masks = paste_masks_in_image(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )
        if not results.has("pred_amodal_masks"):
            results.pred_amodal_masks = results.pred_masks
        if not results.has("pred_visible_masks"):
            results.pred_visible_masks = results.pred_masks


    return results
