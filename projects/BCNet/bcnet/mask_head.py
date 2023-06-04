'''
reimplement based on https://github.com/lkeab/BCNet/blob/main/detectron2/modeling/roi_heads/mask_head.py
'''


import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.layers import interpolate, get_instances_contour_interior
from pytorch_toolbelt import losses as L

from pytorch_toolbelt.modules import AddCoords


__all__= ["BCNet"]

@ROI_MASK_HEAD_REGISTRY.register()
class BCNet(nn.Module):
    def __init__(self,cfg,input_shape: ShapeSpec):
        super(BCNet, self).__init__()
        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
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

    def forward(self,x):
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
