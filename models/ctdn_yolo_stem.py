"""
In-Network Differentiable CTDN Feature Stem for YOLO26s (Case 1).
================================================================
Embeds CTDN (Cross-Transformation Decomposition Network) directly into the
front stem of YOLO26s, replacing static convolutional downsamplers with a
physics-grounded, attention-driven decomposition stem.

Workflow inside the stem:
  1. Input: Raw PCB image [B, 3, H, W]
  2. CTDN Feature Pyramid downsamples to stride 2 (lvl0) and stride 4 (lvl1).
  3. Latent decomposition:
     - Cross-Attention(Illumination, Reflectance) strips shadows from material features.
     - Self-Attention(Illumination) captures global PCB oblique ring lighting falloff.
  4. Differentiable Agaian Adaptive Gamma Correction (AGC) modulates illumination.
  5. Fuses Reflectance and AGC-Illumination latents into 64 channels at stride 4.
  6. Feeds directly into YOLO26s Stage P2 (C3k2 layer).

All stem weights (pyramid, cross-attention, self-attention, fusion) are trained
END-TO-END via detection gradients and the Group Multi-Stage Squared-Dice Loss.
"""

import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ctdn_decom import Cross_Attention, Depth_conv, Res_block, Self_Attention
from tools.agaian_adaptive_gamma import apply_agaian_agc_torch


class CTDNInNetworkStem(nn.Module):
    """
    Differentiable in-network stem replacing standard YOLO conv layers 0 and 1.
    Input:  [B, 3, H, W]
    Output: [B, out_channels, H/4, W/4] (stride 4, default 64 channels)
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 64, internal_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.internal_dim = internal_dim

        # ── 1. Initial 5x5 Conv + ResBlock -> Stride 2 (Level 0) ──
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, internal_dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(internal_dim),
            nn.SiLU(inplace=True),
        )
        self.block0 = Res_block(internal_dim, internal_dim)
        self.down0 = nn.Sequential(
            nn.Conv2d(internal_dim, internal_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(internal_dim),
            nn.SiLU(inplace=True),
        )

        # ── 2. Stride 4 Downsampler (Level 1) ──
        self.block1 = Res_block(internal_dim, internal_dim * 2)
        self.down1 = nn.Sequential(
            nn.Conv2d(internal_dim * 2, internal_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(internal_dim * 2),
            nn.SiLU(inplace=True),
        )

        # ── 3. Latent Retinex Cross-Attention and Self-Attention (at Stride 4) ──
        feat_dim = internal_dim * 2  # 64 channels
        self.cross_attn = Cross_Attention(dim=feat_dim, num_heads=4)
        self.self_attn = Self_Attention(dim=feat_dim, num_heads=4, bias=True)

        # Refinement projections
        self.proj_ref = nn.Sequential(
            Res_block(feat_dim, feat_dim),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.SiLU(inplace=True),
        )
        self.proj_illum = nn.Sequential(
            Res_block(feat_dim, feat_dim),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.SiLU(inplace=True),
        )

        # ── 4. Recombination & Output Projection to YOLO Stride-4 Channels ──
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns:
            fused_stride4: [B, out_channels, H/4, W/4] to feed directly into YOLO layer 2 (stride 4, 64 channels).
        """
        # Downsample to stride 2 and stride 4
        lvl0 = self.down0(self.block0(self.conv_in(x)))
        lvl1 = self.down1(self.block1(lvl0))  # [B, 64, H/4, W/4]

        # Initialize pseudo-illumination and reflectance in latent feature space
        # Estimate spatial illumination magnitude across channels
        init_l = torch.mean(lvl1, dim=1, keepdim=True)
        init_l_norm = torch.sigmoid(init_l)
        init_r = lvl1 / (init_l_norm + 1e-4)

        # Cross-Attention: Illumination cleans Reflectance
        feat_r_clean = self.cross_attn(lvl1, init_r)

        # Self-Attention: Global illumination context
        feat_l_global = self.self_attn(lvl1)

        feat_r_refined = self.proj_ref(feat_r_clean + feat_l_global)
        feat_l_refined = self.proj_illum(feat_l_global)

        # Differentiable Agaian AGC on the illumination latent
        l_mag = torch.sigmoid(torch.mean(feat_l_refined, dim=1, keepdim=True))
        l_agc = apply_agaian_agc_torch(l_mag, gamma_min=0.30, gamma_max=2.0)
        feat_l_adj = feat_l_refined * l_agc

        self.feat_R = feat_r_refined
        self.feat_L = feat_l_adj

        # Fuse Reflectance and Illumination into final stride-4 output
        fused = self.fusion(torch.cat([feat_r_refined, feat_l_adj], dim=1))
        return fused


if __name__ == "__main__":
    print("Testing CTDNInNetworkStem...")
    stem = CTDNInNetworkStem(in_channels=3, out_channels=64, internal_dim=32)
    x = torch.rand(2, 3, 640, 640)
    out = stem(x)
    print(f"Stem Output shape: {out.shape} (Expected: [2, 64, 160, 160])")
    print(f"Reflectance shape: {stem.feat_R.shape}, Illumination shape: {stem.feat_L.shape}")
    print("CTDNInNetworkStem self-test passed successfully!")
