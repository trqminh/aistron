import logging
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from ..postprocessing import detector_postprocess


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNAmodal(GeneralizedRCNN):
    @configurable
    def __init__(self, *, init_value=None, inference_with_gt_boxes=False, **kwargs):
        super().__init__(**kwargs)
        self.inference_with_gt_boxes = inference_with_gt_boxes

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["inference_with_gt_boxes"] = cfg.AISTRON.INFERENCE_WITH_GT_BOXES
        return ret

    @staticmethod
    def convert_gt_to_detected(gt_instances):
        detected_instances = []
        for gt_instance in gt_instances:
            detected_instance = Instances(gt_instance.image_size)
            detected_instance.pred_boxes = gt_instance.gt_boxes
            detected_instance.pred_classes = gt_instance.gt_classes
            detected_instance.scores = torch.ones_like(gt_instance.gt_classes, dtype=torch.float32)
            '''
            ## add gt for verifying evaluators
            detected_instance.gt_background_objs_masks = gt_instance.gt_background_objs_masks
            detected_instance.gt_amodal_masks = gt_instance.gt_amodal_masks
            detected_instance.gt_visible_masks = gt_instance.gt_visible_masks
            '''
            detected_instances.append(detected_instance)

        return detected_instances


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        inherit from GeneralizedRCNN.forward
        https://github.com/facebookresearch/detectron2/blob/3ff5dd1cff4417af07097064813c9f28d7461d3c/detectron2/modeling/meta_arch/rcnn.py#L126C1-L176C22

        """
        if not self.training:
            if not self.inference_with_gt_boxes:
                return self.inference(batched_inputs)

            detected_instances = None
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                detected_instances = self.convert_gt_to_detected(gt_instances)
            else:
                print("WARNING: no gt_instances found in batched_inputs to inference_with_gt_boxes")

            return self.inference(batched_inputs, detected_instances=detected_instances)

        return super().forward(batched_inputs)



    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self._postprocess(results, batched_inputs, images.image_sizes)
        return results

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results