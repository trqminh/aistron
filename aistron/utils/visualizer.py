from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes


class AmodalVisualizer(Visualizer):
    """
    """
    def draw_dataset_dict(self, dic, option='amodal'):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.
        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.
            options could be: ['amodal', 'visible']
        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if option == 'amodal':
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

            if option == 'visible':
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
