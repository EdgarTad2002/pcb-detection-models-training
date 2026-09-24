"""
End-to-End VMamba Object Detector (Non-YOLO)
==============================================
Assembles the hierarchical VMamba backbone, MambaFPN neck, and AnchorFreeHead
into a unified object detector with zero external framework dependencies.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision

from .backbone import VMambaBackbone
from .fpn import MambaFPN
from .head import AnchorFreeHead
from .loss import DetectionLoss


class VMambaDetector(nn.Module):
    """
    Standalone VMamba Object Detector.
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone_dims: List[int] = [96, 192, 384, 768],
        backbone_depths: List[int] = [2, 2, 2, 2],
        stage_types: Optional[List[str]] = None,
        fpn_channels: int = 128,
        strides: List[int] = [4, 8, 16, 32],
        pretrained_backbone: Optional[str] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # 1. Hierarchical VMamba Backbone (MambaVision Hybrid: Conv early, Mamba deep)
        self.backbone = VMambaBackbone(
            in_chans=3,
            dims=backbone_dims,
            depths=backbone_depths,
            stage_types=stage_types,
            out_indices=[0, 1, 2, 3],
            use_checkpoint=use_checkpoint,
        )
        if pretrained_backbone:
            self.backbone.load_pretrained(pretrained_backbone)

        # 2. Multi-Scale Feature Pyramid Network (including P2 for micro-capacitors)
        self.neck = MambaFPN(
            in_channels_list=backbone_dims,
            out_channels=fpn_channels,
            include_p2=True,
        )

        # 3. Decoupled Anchor-Free Detection Head
        self.head = AnchorFreeHead(
            num_classes=num_classes,
            in_channels=fpn_channels,
            feat_channels=fpn_channels,
            strides=strides,
        )

        # 4. Training Loss Criterion
        self.criterion = DetectionLoss(
            num_classes=num_classes,
            strides=strides,
        )

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes_list: Optional[List[torch.Tensor]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Args:
            images: (B, 3, H, W) normalized image tensor
            gt_boxes_list: Optional list of (N_i, 5) ground truth tensors [cls, x1, y1, x2, y2]
        Returns:
            If gt_boxes_list is provided: loss dict {"loss", "loss_cls", "loss_box", "loss_ctr"}
            If gt_boxes_list is None: list of detection dicts per image
        """
        # 1. Backbone forward -> [C2, C3, C4, C5]
        features = self.backbone(images)

        # 2. Neck forward -> [P2, P3, P4, P5]
        pyramid_feats = self.neck(features)

        # 3. Head forward -> classification, box regression, centerness
        cls_scores, bbox_preds, centernesses = self.head(pyramid_feats)

        # Training phase: compute multi-task loss
        if gt_boxes_list is not None:
            return self.criterion(cls_scores, bbox_preds, centernesses, gt_boxes_list)

        # Inference phase: decode bounding boxes
        return self._post_process(cls_scores, bbox_preds, centernesses, images.shape[2:])

    def _post_process(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        centernesses: List[torch.Tensor],
        img_shape: Tuple[int, int],
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.50,
        max_detections: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decodes multi-level predictions into filtered bounding boxes via NMS."""
        B = cls_scores[0].shape[0]
        device = cls_scores[0].device
        H_img, W_img = img_shape

        all_boxes_b = [[] for _ in range(B)]
        all_scores_b = [[] for _ in range(B)]
        all_labels_b = [[] for _ in range(B)]

        for cls_lvl, reg_lvl, ctr_lvl, stride in zip(cls_scores, bbox_preds, centernesses, self.strides):
            _, _, H_feat, W_feat = cls_lvl.shape

            # Spatial grid anchor points
            shifts_x = (torch.arange(0, W_feat, device=device) + 0.5) * stride
            shifts_y = (torch.arange(0, H_feat, device=device) + 0.5) * stride
            sy, sx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            anchor_x = sx.reshape(-1) # (L,)
            anchor_y = sy.reshape(-1) # (L,)

            # (B, num_classes, H, W) -> (B, L, num_classes)
            prob_cls = torch.sigmoid(cls_lvl.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            # (B, 1, H, W) -> (B, L)
            prob_ctr = torch.sigmoid(ctr_lvl.permute(0, 2, 3, 1).reshape(B, -1))
            # Combined score: sqrt(cls * ctr)
            scores = torch.sqrt(prob_cls * prob_ctr.unsqueeze(-1)) # (B, L, num_classes)

            # (B, 4, H, W) -> (B, L, 4) [l, t, r, b]
            ltrb = reg_lvl.permute(0, 2, 3, 1).reshape(B, -1, 4)

            # Decode to [x1, y1, x2, y2]
            x1 = (anchor_x.unsqueeze(0) - ltrb[:, :, 0]).clamp(min=0, max=W_img)
            y1 = (anchor_y.unsqueeze(0) - ltrb[:, :, 1]).clamp(min=0, max=H_img)
            x2 = (anchor_x.unsqueeze(0) + ltrb[:, :, 2]).clamp(min=0, max=W_img)
            y2 = (anchor_y.unsqueeze(0) + ltrb[:, :, 3]).clamp(min=0, max=H_img)
            boxes = torch.stack([x1, y1, x2, y2], dim=-1) # (B, L, 4)

            for b in range(B):
                max_score, max_cls = scores[b].max(dim=-1)
                valid = max_score >= conf_threshold
                if valid.any():
                    all_boxes_b[b].append(boxes[b, valid])
                    all_scores_b[b].append(max_score[valid])
                    all_labels_b[b].append(max_cls[valid])

        # Run per-image batched NMS
        results = []
        for b in range(B):
            if len(all_boxes_b[b]) == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                })
                continue

            boxes_cat = torch.cat(all_boxes_b[b], dim=0)
            scores_cat = torch.cat(all_scores_b[b], dim=0)
            labels_cat = torch.cat(all_labels_b[b], dim=0)

            keep = torchvision.ops.batched_nms(boxes_cat, scores_cat, labels_cat, iou_threshold)
            keep = keep[:max_detections]

            results.append({
                "boxes": boxes_cat[keep],
                "scores": scores_cat[keep],
                "labels": labels_cat[keep],
            })

        return results
