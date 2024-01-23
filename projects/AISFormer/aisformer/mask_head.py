
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry



from .position_encoding import PositionEmbeddingLearned
from .mlp import MLP
from .transformer import (
    TransformerDecoder, TransformerEncoder, 
    TransformerDecoderLayer, TransformerEncoderLayer
)
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY


__all__= ["AISFormer"]

@ROI_MASK_HEAD_REGISTRY.register()
class AISFormer(nn.Module):
    @staticmethod
    def _init_roi_feature_learner(conv_dim, n_layers, upsample=True):
        '''
        this module enrich the CxHxW roi feature with convolutional layers
        the obtained feature will be Cx2Hx2W
        this is quite the same as the mask rcnn mask head in detectron2
        '''
        modules = []
        for i in range(n_layers):
            modules.append(
                Conv2d(conv_dim, conv_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.ReLU())
            )

        if upsample:
            modules.extend([
                ConvTranspose2d(conv_dim, conv_dim, kernel_size=(2, 2), stride=(2, 2)),
                nn.ReLU(),
            ])

        modules.append(
            Conv2d(conv_dim, conv_dim, kernel_size=(1, 1), stride=(1, 1))
        )

        module_seq = nn.Sequential(*modules)

        # init weights
        for i in range(len(module_seq)):
            if i < n_layers:
                weight_init.c2_msra_fill(module_seq[i])

        if upsample:
            weight_init.c2_msra_fill(module_seq[n_layers])

        nn.init.normal_(module_seq[-1].weight, std=0.001)
        if module_seq[-1].bias is not None:
            nn.init.constant_(module_seq[-1].bias, 0)

        return module_seq

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, vis_period=0, aisformer=None, **kwargs):
        super().__init__()
        conv_dim = input_shape.channels
        self.aisformer = aisformer
        self.vis_period = vis_period

        # deconv short range learning
        self.mask_feat_learner_TR = self._init_roi_feature_learner(conv_dim, 8, upsample=True)

        # pixel embedding
        self.pixel_embed = self._init_roi_feature_learner(conv_dim, 8, upsample=False)

        # mask embedding
        emb_dim = self.aisformer.EMB_DIM
        self.mask_embed = MLP(emb_dim, emb_dim, emb_dim, 3)
        for layer in self.mask_embed.layers:
            torch.nn.init.xavier_uniform_(layer.weight)

        # subtract modeling
        self.subtract_model = MLP(emb_dim*2, emb_dim, emb_dim, 2)
        for layer in self.subtract_model.layers:
            torch.nn.init.xavier_uniform_(layer.weight)

        # norm rois
        self.norm_rois = nn.LayerNorm(emb_dim)


        # transformer layers
        self.positional_encoding = PositionEmbeddingLearned(emb_dim//2)

        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=self.aisformer.N_HEADS, normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.aisformer.N_LAYERS)

        decoder_layer = TransformerDecoderLayer(d_model=emb_dim, nhead=self.aisformer.N_HEADS, normalize_before=False)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=self.aisformer.N_LAYERS) # 6 is the default of detr

        n_output_masks = 4 # 4 embeddings, vi_mask, occluder, a_mask, invisible_mask
        self.query_embed = nn.Embedding(num_embeddings=n_output_masks, embedding_dim=emb_dim)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        aisformer = cfg.AISFORMER
        vis_period = cfg.VIS_PERIOD

        ret.update(
            input_shape=input_shape,
            aisformer=aisformer,
            vis_period=vis_period
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        return ret 

    @torch.jit.unused
    @staticmethod
    def mask_loss(pred_mask_logits: torch.Tensor, 
                  instances: List[Instances], vis_period: int = 0,
                  mask_type=None):
        """
        Inherit mask loss from Mask R-CNN.
        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.
        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """
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
                gt_masks_per_image = instances_per_image.get(mask_type).crop_and_resize(
                    instances_per_image.proposal_boxes.tensor, mask_side_len
                ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and 0.5 threshold)
        mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
            gt_masks_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

        storage = get_event_storage()
        storage.put_scalar("aisformer/accuracy_{}".format(mask_type), mask_accuracy)
        storage.put_scalar("aisformer/false_positive_{}".format(mask_type), false_positive)
        storage.put_scalar("aisformer/false_negative_{}".format(mask_type), false_negative)
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
    @torch.jit.script_if_tracing
    def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        Tracing friendly way to cast tensor to another tensor's device. Device will be treated
        as constant during tracing, scripting the casting process as whole can workaround this issue.
        """
        return src.to(dst.device)

    @staticmethod
    def mask_inference(pred_mask_logits: torch.Tensor, 
                       pred_instances: List[Instances],
                       pred_mask_type=None):
        """
        Inherit from detectron2 mask_rcnn_inference
        Convert pred_mask_logits to estimated foreground probability masks while also
        extracting only the masks for the predicted classes in pred_instances. For each
        predicted box, the mask of the same class is attached to the instance by adding a
        new "pred_masks" field to pred_instances.
        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. Each Instances must have field "pred_classes".
        Returns:
            None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
                Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
                masks the resolution predicted by the network; post-processing steps, such as resizing
                the predicted masks to the original image resolution and/or binarizing them, is left
                to the caller.
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
            indices = self.move_device_like(torch.arange(num_masks, device=device), class_pred)
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

        for prob, instances in zip(mask_probs_pred, pred_instances):
            instances.set(pred_mask_type, prob)  # (1, Hmask, Wmask)


    def layers(self, x):
        x_ori = x.clone()
        bs = x_ori.shape[0]

        # short range learning
        x = self.mask_feat_learner_TR(x)
        x_short = x.clone()

        # position emb
        pos_embed = self.positional_encoding.forward_tensor(x)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # encode
        feat_embs = x.flatten(2).permute(2, 0, 1)
        encoded_feat_embs = self.transformer_encoder(feat_embs, 
                                                    pos=pos_embed)

        # decode
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed) 
        decoder_output = self.transformer_decoder(tgt, encoded_feat_embs, 
                                        pos=pos_embed, 
                                        query_pos=query_embed) # (1, n_masks, bs, dim)

        decoder_output = decoder_output.squeeze(0).moveaxis(1,0)

        # predict mask
        roi_embeding =  encoded_feat_embs.permute(1,2,0).unflatten(-1, (28,28))
        roi_embeding = roi_embeding + x_short # long range + short range
        roi_embeding = self.norm_rois(roi_embeding.permute(0,2,3,1)).permute(0,3,1,2)
        roi_embeding = self.pixel_embed(roi_embeding)

        mask_embs = self.mask_embed(decoder_output)
        if self.aisformer.INVISIBLE_MASK_LOSS:
            combined_feat = torch.cat([mask_embs[:,2,:],mask_embs[:,0,:]], axis=1)
            invisible_embs = self.subtract_model(combined_feat)
            invisible_embs = invisible_embs.unsqueeze(1)
            mask_embs = torch.concat([mask_embs, invisible_embs], axis=1)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embs, roi_embeding)

        vi_masks        = outputs_mask[:,0,:,:].unsqueeze(1) #visible mask
        bo_masks        = outputs_mask[:,1,:,:].unsqueeze(1) #occluder (bo - background objects) mask
        a_masks         = outputs_mask[:,2,:,:].unsqueeze(1) #amodal mask
        invisible_masks = outputs_mask[:,-1,:,:].unsqueeze(1) #invisible mask

        return vi_masks, bo_masks, a_masks, invisible_masks

    def forward(self, x, instances: List[Instances]):
        if self.training: # nn.Module attributes 
            vi_masks, bo_masks, a_masks, invisible_masks = self.layers(x)

            loss_a_mask = self.mask_loss(a_masks, instances, mask_type='gt_amodal_masks', vis_period =self.vis_period)
            loss_vi_mask = self.mask_loss(vi_masks, instances, mask_type="gt_visible_masks", vis_period =self.vis_period)
            loss_bo_mask = self.mask_loss(bo_masks, instances, mask_type="gt_background_objs_masks", vis_period =self.vis_period)
            loss_invisible_mask = self.mask_loss(invisible_masks, instances, mask_type='invisible', vis_period =self.vis_period)

            return {
                "loss_vi_mask": loss_vi_mask,
                "loss_bo_mask": loss_bo_mask,
                "loss_a_mask": loss_a_mask,
                "loss_invisible_mask": loss_invisible_mask
            }
        else:
            ## Inference forward
            vi_masks, bo_masks, a_masks, invisible_masks = self.layers(x)
            self.mask_inference(vi_masks, instances, 'pred_visible_masks')
            self.mask_inference(a_masks, instances, 'pred_amodal_masks')
            self.mask_inference(bo_masks, instances, 'pred_occluding_masks')
            self.mask_inference(invisible_masks, instances, 'pred_occluded_masks')
            return instances
