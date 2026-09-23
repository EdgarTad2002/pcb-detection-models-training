"""
Decoupled Anchor-Free Detection Head
======================================
Predicts per-pixel classification logits, bounding box distances (l, t, r, b),
and centerness quality scores across multi-scale FPN levels.
"""

from typing import Dict, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn import ConvModule


class Scale(nn.Module):
    """Learnable scale layer for multi-level regression output."""

    def __init__(self, init_val: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class AnchorFreeHead(nn.Module):
    """
    Decoupled Anchor-Free Detection Head.
    Operates on multi-scale FPN feature maps [P2, P3, P4, P5].
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 128,
        feat_channels: int = 128,
        num_convs: int = 3,
        strides: List[int] = [4, 8, 16, 32],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # Classification tower
        cls_tower = []
        for i in range(num_convs):
            ch = in_channels if i == 0 else feat_channels
            cls_tower.append(ConvModule(ch, feat_channels, kernel_size=3, padding=1))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Regression tower
        reg_tower = []
        for i in range(num_convs):
            ch = in_channels if i == 0 else feat_channels
            reg_tower.append(ConvModule(ch, feat_channels, kernel_size=3, padding=1))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Predictor heads
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(feat_channels, 4, kernel_size=3, padding=1)
        self.centerness_pred = nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1)

        # Learnable scale parameter per pyramid level
        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])

        self._init_weights()

    def _init_weights(self):
        # Normal initialization for convs
        for m in [self.cls_pred, self.reg_pred, self.centerness_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

        # Prior probability for focal loss: -log((1 - p) / p) with p=0.01
        bias_init = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_pred.bias, bias_init)

    def forward(
        self, feats: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            feats: List of FPN feature maps [P2, P3, P4, P5]
        Returns:
            cls_scores: List of (B, num_classes, H_i, W_i)
            bbox_preds: List of (B, 4, H_i, W_i) in distance format [l, t, r, b]
            centernesses: List of (B, 1, H_i, W_i)
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []

        for feat, stride, scale in zip(feats, self.strides, self.scales):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            # Class logits
            cls_score = self.cls_pred(cls_feat)
            # Centerness
            centerness = self.centerness_pred(reg_feat)
            # Regression distances (l, t, r, b) - enforce positive distances via exp/relu
            reg_out = scale(self.reg_pred(reg_feat))
            bbox_pred = F.relu(reg_out) * stride

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)

        return cls_scores, bbox_preds, centernesses
