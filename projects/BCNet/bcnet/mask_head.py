'''
reimplement based on https://github.com/lkeab/BCNet/blob/main/detectron2/modeling/roi_heads/mask_head.py
'''


import fvcore.nn.weight_init as weight_init
import torch
from typing import List

from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .boundary import get_instances_contour_interior

from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY

__all__= ["BCNet"]

@ROI_MASK_HEAD_REGISTRY.register()
class BCNet(nn.Module):
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, 
                        vis_period=0, 
                        num_classes=None,
                        conv_dims=None,
                        norm=None,
                        num_conv=None,
                        input_channels=None,
                        cls_agnostic_mask=None,
                        **kwargs):
        super().__init__()
        # fmt: off
        self.vis_period = vis_period
        self.norm = norm
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.boundary_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.boundary_deconv_bo = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.boundary_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.bo_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.query_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.query_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)


        self.scale = 1.0 / (input_channels ** 0.5)
        self.blocker_bound_bo = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized
        self.blocker_bound = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.predictor_bo = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.boundary_predictor_bo = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.boundary_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        self.num_mask_classes = num_mask_classes

        for layer in self.conv_norm_relus + self.boundary_conv_norm_relus + [self.deconv, self.bo_deconv, self.boundary_deconv, self.boundary_deconv_bo, self.query_transform_bound_bo, self.key_transform_bound_bo, self.value_transform_bound_bo, self.output_transform_bound_bo, self.query_transform_bound, self.key_transform_bound, self.value_transform_bound, self.output_transform_bound]:
            weight_init.c2_msra_fill(layer)
            layer = layer.to('cuda')
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

        nn.init.normal_(self.predictor_bo.weight, std=0.001)
        if self.predictor_bo.bias is not None:
            nn.init.constant_(self.predictor_bo.bias, 0)

        nn.init.normal_(self.boundary_predictor_bo.weight, std=0.001)
        if self.boundary_predictor_bo.bias is not None:
            nn.init.constant_(self.boundary_predictor_bo.bias, 0)

        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        vis_period        = cfg.VIS_PERIOD
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        norm              = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK

        ret.update(
            input_shape=input_shape,
            vis_period=vis_period,
            num_classes=num_classes,
            conv_dims=conv_dims,
            norm=norm,
            num_conv=num_conv,
            input_channels=input_channels,
            cls_agnostic_mask=cls_agnostic_mask,
        )
        
        return ret 

    @staticmethod
    def prepare_gt(pred_mask_logits, instances: List[Instances], mask_type=None):
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = pred_mask_logits.size(2)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            if mask_type == 'invisible':
                gt_amodal_masks_per_image = instances_per_image.get('gt_amodal_masks').crop_and_resize(
                    instances_per_image.proposal_boxes.tensor, mask_side_len
                ).to(device=pred_mask_logits.device)
                gt_visible_masks_per_image = instances_per_image.get('gt_visible_masks').crop_and_resize(
                    instances_per_image.proposal_boxes.tensor, mask_side_len
                ).to(device=pred_mask_logits.device)
                gt_masks_per_image = gt_amodal_masks_per_image ^ gt_visible_masks_per_image
            else:
                boundary_mask = False
                if 'boundary' in mask_type:
                    boundary_mask = True
                    mask_type = mask_type.replace('_boundary', '')
                
                gt_masks_per_image = instances_per_image.get(mask_type).crop_and_resize(
                    instances_per_image.proposal_boxes.tensor, mask_side_len
                ).to(device=pred_mask_logits.device)

                if boundary_mask:
                    boundary_ls = []
                    for mask in gt_masks_per_image:
                        mask_b = mask.data.cpu().numpy()
                        boundary, inside_mask, weight = get_instances_contour_interior(mask_b)
                        boundary = torch.from_numpy(boundary).to(device=mask.device).unsqueeze(0)

                        boundary_ls.append(boundary)

                    gt_masks_per_image = cat(boundary_ls, dim=0)

            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)

        if not cls_agnostic_mask:
            gt_classes = cat(gt_classes, dim=0)

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        return gt_masks, gt_masks_bool, gt_classes


    @torch.jit.unused
    def mask_loss(self, pred_mask_logits: torch.Tensor, 
                  instances: List[Instances], vis_period: int = 0,
                  mask_type=None):
        """
        Inherit mask loss from Mask R-CNN with mask_type param for amodal segmentation gt
        https://github.com/facebookresearch/detectron2/blob/0df924ce6066fb97d5413244614b12fbabaf65c8/detectron2/modeling/roi_heads/mask_head.py#L33
        """
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
        gt_masks, gt_masks_bool, gt_classes = self.prepare_gt(pred_mask_logits, instances, mask_type)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        # Log the training accuracy (using gt classes and 0.5 threshold)
        mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
            gt_masks_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

        storage = get_event_storage()
        storage.put_scalar("shapeformer/accuracy_{}".format(mask_type), mask_accuracy)
        storage.put_scalar("shapeformer/false_positive_{}".format(mask_type), false_positive)
        storage.put_scalar("shapeformer/false_negative_{}".format(mask_type), false_negative)
        if vis_period > 0 and storage.iter % vis_period == 0:
            pred_masks = pred_mask_logits.sigmoid()
            vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
            name = "Left: mask prediction;   Right: mask GT"
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name + f" ({idx})", vis_mask)

        mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
        return mask_loss

    @staticmethod
    def mask_inference(pred_mask_logits: torch.Tensor, 
                       pred_instances: List[Instances],
                       pred_mask_type=None):
        """
        Inherit from detectron2 mask_rcnn_inference with prediction mask type
        https://github.com/facebookresearch/detectron2/blob/0df924ce6066fb97d5413244614b12fbabaf65c8/detectron2/modeling/roi_heads/mask_head.py#L33
        """
        cls_agnostic_mask = pred_mask_logits.size(1) == 1

        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits.sigmoid()
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            device = (
                class_pred.device
                if torch.jit.is_scripting()
                else ("cpu" if torch.jit.is_tracing() else class_pred.device)
            )
            @torch.jit.script_if_tracing
            def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
                """
                Tracing friendly way to cast tensor to another tensor's device. Device will be treated
                as constant during tracing, scripting the casting process as whole can workaround this issue.
                """
                return src.to(dst.device)
            indices = move_device_like(torch.arange(num_masks, device=device), class_pred)
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

        for prob, instances in zip(mask_probs_pred, pred_instances):
            instances.set(pred_mask_type, prob)  # (1, Hmask, Wmask)   

    def layers(self,x):
        B, C, H, W = x.size()
        x_ori = x.clone()

        for cnt, layer in enumerate(self.boundary_conv_norm_relus):
            x = layer(x)

            if cnt == 1 and len(x) != 0:
                # x: B,C,H,W
                # x_query: B,C,HW
                #x_input = AddCoords()(x)
                x_query_bound_bo = self.query_transform_bound_bo(x).view(B, C, -1)
                # x_query: B,HW,C
                x_query_bound_bo = torch.transpose(x_query_bound_bo, 1, 2)
                # x_key: B,C,HW
                x_key_bound_bo = self.key_transform_bound_bo(x).view(B, C, -1)
                # x_value: B,C,HW
                x_value_bound_bo = self.value_transform_bound_bo(x).view(B, C, -1)
                # x_value: B,HW,C
                x_value_bound_bo = torch.transpose(x_value_bound_bo, 1, 2)
                # W = Q^T K: B,HW,HW
                x_w_bound_bo = torch.matmul(x_query_bound_bo, x_key_bound_bo) * self.scale
                x_w_bound_bo = F.softmax(x_w_bound_bo, dim=-1)
                # x_relation = WV: B,HW,C
                x_relation_bound_bo = torch.matmul(x_w_bound_bo, x_value_bound_bo)
                # x_relation = B,C,HW
                x_relation_bound_bo = torch.transpose(x_relation_bound_bo, 1, 2)
                # x_relation = B,C,H,W
                x_relation_bound_bo = x_relation_bound_bo.view(B,C,H,W)

                x_relation_bound_bo = self.output_transform_bound_bo(x_relation_bound_bo)
                x_relation_bound_bo = self.blocker_bound_bo(x_relation_bound_bo)

                x = x + x_relation_bound_bo

        x_bound_bo = x.clone()

        x_bo = x.clone()
        
        x = x_ori + x

        for cnt, layer in enumerate(self.conv_norm_relus):
            x = layer(x)
            if cnt == 1 and len(x) != 0:
                # x: B,C,H,W
                # x_query: B,C,HW
                #x_input = AddCoords()(x)
                x_query_bound = self.query_transform_bound(x).view(B, C, -1)
                # x_query: B,HW,C
                x_query_bound = torch.transpose(x_query_bound, 1, 2)
                # x_key: B,C,HW
                x_key_bound = self.key_transform_bound(x).view(B, C, -1)
                # x_value: B,C,HW
                x_value_bound = self.value_transform_bound(x).view(B, C, -1)
                # x_value: B,HW,C
                x_value_bound = torch.transpose(x_value_bound, 1, 2)
                # W = Q^T K: B,HW,HW
                x_w_bound = torch.matmul(x_query_bound, x_key_bound) * self.scale
                x_w_bound = F.softmax(x_w_bound, dim=-1)
                # x_relation = WV: B,HW,C
                x_relation_bound = torch.matmul(x_w_bound, x_value_bound)
                # x_relation = B,C,HW
                x_relation_bound = torch.transpose(x_relation_bound, 1, 2)
                # x_relation = B,C,H,W
                x_relation_bound = x_relation_bound.view(B,C,H,W)

                x_relation_bound = self.output_transform_bound(x_relation_bound)
                x_relation_bound = self.blocker_bound(x_relation_bound)

                x = x + x_relation_bound

        x_bound = x.clone()

        x = F.relu(self.deconv(x))
        mask_head_features = x.clone()
        mask = self.predictor(x) 

        x_bo = F.relu(self.bo_deconv(x_bo))
        mask_bo = self.predictor_bo(x_bo) 

        x_bound_bo = F.relu(self.boundary_deconv_bo(x_bound_bo))
        boundary_bo = self.boundary_predictor_bo(x_bound_bo) 

        x_bound = F.relu(self.boundary_deconv(x_bound))
        boundary = self.boundary_predictor(x_bound) 

        return mask, boundary, mask_bo, boundary_bo

    def forward(self, x, instances: List[Instances]):
        if self.training:
            amodal_masks, amodal_boundaries, bo_masks, bo_boundaries = self.layers(x)

            loss_a_mask = self.mask_loss(amodal_masks, instances, mask_type="gt_amodal_masks", vis_period =self.vis_period)
            loss_bo_mask = self.mask_loss(bo_masks, instances, mask_type="gt_background_objs_masks", vis_period =self.vis_period)
            loss_a_boundary = self.mask_loss(amodal_boundaries, instances, mask_type="gt_amodal_masks_boundary", vis_period =self.vis_period)
            loss_bo_boundary = self.mask_loss(amodal_boundaries, instances, mask_type="gt_background_objs_masks_boundary", vis_period =self.vis_period)

            return {
                "loss_a_mask": loss_a_mask,
                "loss_bo_mask": loss_bo_mask,
                "loss_a_boundary": loss_a_boundary,
                "loss_bo_boundary": loss_bo_boundary,
            }
        else:
            # Inference forward
            amodal_masks, amodal_bound, bo_masks, bo_bound = self.layers(x)
            self.mask_inference(amodal_masks, instances, 'pred_amodal_masks')
            self.mask_inference(bo_masks, instances, 'pred_occluding_masks')
            self.mask_inference(amodal_bound, instances, 'pred_amodal_boundary')
            self.mask_inference(bo_bound, instances, 'pred_occluding_boundary')
            return instances

