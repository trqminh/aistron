import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels, GenericMask
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes


class AmodalVisualizer(Visualizer):
    """
    """
    def draw_dataset_dict(self, dic, segm_type='amodal'):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.
        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.
            segm_type options could be: ['amodal', 'visible']
        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if segm_type == 'amodal':
                if "amodal_segm" in annos[0]:
                    masks = [x["amodal_segm"] for x in annos]
                else:
                    masks = None
                boxes = [
                    BoxMode.convert(x["amodal_bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                    if len(x["amodal_bbox"]) == 4
                    else x["amodal_bbox"]
                    for x in annos
                ]

            elif segm_type == 'visible':
                if "visible_segm" in annos[0]:
                    masks = [x["visible_segm"] for x in annos]
                else:
                    masks = None
                boxes = [
                    BoxMode.convert(x["visible_bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                    if len(x["visible_bbox"]) == 4
                    else x["visible_bbox"]
                    for x in annos
                ]

            elif segm_type == 'background_objs' or 'occluding':
                if "background_objs_segm" in annos[0]:
                    masks = [x["background_objs_segm"] for x in annos]
                else:
                    masks = None
                boxes = [
                    BoxMode.convert(x["visible_bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                    if len(x["visible_bbox"]) == 4
                    else x["visible_bbox"]
                    for x in annos
                ]

            else:
                if "segmentation" in annos[0]:
                    masks = [x["segmentation"] for x in annos]
                else:
                    masks = None
                boxes = [
                    BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                    if len(x["bbox"]) == 4
                    else x["bbox"]
                    for x in annos
                ]

            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None


            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                    for c in category_ids
                ]
            names = self.metadata.get("thing_classes", None)
            labels = _create_text_labels(
                category_ids,
                scores=None,
                class_names=names,
                is_crowd=[x.get("iscrowd", 0) for x in annos],
            )
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )

        return self.output

    def draw_instance_predictions(self, predictions, segm_type='amodal'):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if segm_type == 'amodal':
            if predictions.has("pred_amodal_masks"):
                masks = np.asarray(predictions.pred_amodal_masks)
                masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
            elif predictions.has("pred_masks"):
                masks = np.asarray(predictions.pred_masks)
                masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
            else:
                masks = None
        elif segm_type == 'visible':
            assert predictions.has("pred_visible_masks"), "No visible masks!"
            masks = np.asarray(predictions.pred_visible_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        elif segm_type == 'occluding':
            assert predictions.has("pred_occluding_masks"), "No occluding masks!"
            masks = np.asarray(predictions.pred_occluding_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        elif segm_type == 'occluded':
            assert predictions.has("pred_occluded_masks"), "No occluded masks!"
            masks = np.asarray(predictions.pred_occluded_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]


        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output