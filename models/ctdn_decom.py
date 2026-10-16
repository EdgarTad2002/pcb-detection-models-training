"""
CTDN (Cross-Transformation Decomposition Network) for PCB AOI.
Adapted from LightenDiffusion (Jiang et al., ECCV 2024 / Agaian et al.).

Decomposes a PCB image into:
  - Material Reflectance (R): Intrinsic material colors, solder pads, silkscreen, traces (invariant to shadows).
  - Illumination (L): Spatial lighting distribution, shadows under tall components, ring glare.

Architecture:
  - Feature Pyramid Encoder (down2, down4, down8)
  - Retinex_decom with Cross_Attention(Illumination, Reflectance) and Self_Attention(Illumination)
  - Dynamic Reconstruction Decoder
"""

import math
import os
from typing import Dict, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore", category=UserWarning)


class Depth_conv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_ch: int, out_ch: int):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


class Res_block(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_channels: int, out_channels: int):
        super(Res_block, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + self.conv(x)


class upsampling(nn.Module):
    """Bilinear / Transpose upsampling."""
    def __init__(self, in_channels: int, out_channels: int):
        super(upsampling, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class channel_down(nn.Module):
    def __init__(self, channels: int):
        super(channel_down, self).__init__()
        self.conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv2(self.relu(self.conv1(self.relu(self.conv0(x))))))


class channel_up(nn.Module):
    def __init__(self, channels: int):
        super(channel_up, self).__init__()
        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(self.relu(self.conv0(x)))))


class feature_pyramid(nn.Module):
    """Multi-scale feature pyramid encoder downsampling /2, /4, /8."""
    def __init__(self, channels: int = 64):
        super(feature_pyramid, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(5, 5), stride=(1, 1), padding=2),
        )
        self.block0 = Res_block(channels, channels)
        self.down0 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = Res_block(channels, channels * 2)
        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block2 = Res_block(channels * 2, channels * 4)
        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.convs(x)
        level0 = self.down0(self.block0(f0))      # stride 2
        level1 = self.down1(self.block1(level0))  # stride 4
        level2 = self.down2(self.block2(level1))  # stride 8
        return level0, level1, level2


class Self_Attention(nn.Module):
    """Global self-attention over illumination representations."""
    def __init__(self, dim: int, num_heads: int = 8, bias: bool = True):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class Cross_Attention(nn.Module):
    """Cross-attention transferring illumination guidance to reflectance."""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        mixed_query = self.query(hidden_states)
        mixed_key = self.key(ctx)
        mixed_value = self.value(ctx)

        q = self.transpose_for_scores(mixed_query)
        k = self.transpose_for_scores(mixed_key)
        v = self.transpose_for_scores(mixed_value)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        return out


class Retinex_decom(nn.Module):
    """Decomposes feature maps into Reflectance (R) and Illumination (L)."""
    def __init__(self, channels: int = 64):
        super(Retinex_decom, self).__init__()
        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(
            Res_block(channels, channels),
            Res_block(channels, channels),
        )

        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks1 = nn.Sequential(
            Res_block(channels, channels),
            Res_block(channels, channels),
        )

        self.cross_attention = Cross_Attention(dim=channels, num_heads=8)
        self.self_attention = Self_Attention(dim=channels, num_heads=8, bias=True)

        self.conv0_1 = nn.Sequential(
            Res_block(channels, channels),
            nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.conv1_1 = nn.Sequential(
            Res_block(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        init_illumination = torch.max(x, dim=1, keepdim=True)[0]
        init_reflectance = x / torch.clamp(init_illumination, min=1e-4)

        feat_R = self.blocks0(self.conv0(init_reflectance))
        feat_L = self.blocks1(self.conv1(init_illumination))

        feat_R_cross = self.cross_attention(feat_L, feat_R)
        feat_L_content = self.self_attention(feat_L)

        R_final = self.conv0_1(feat_R_cross + feat_L_content)
        L_final = self.conv1_1(feat_L - feat_L_content)

        R = torch.sigmoid(R_final)
        L = torch.sigmoid(L_final)
        L_3ch = L.expand(-1, 3, -1, -1)
        return R, L_3ch


class CTDN(nn.Module):
    """
    Complete Cross-Transformation Decomposition Network.
    Provides forward decomposition for images into Reflectance (R) and Illumination (L).
    """
    def __init__(self, channels: int = 64):
        super(CTDN, self).__init__()
        self.pyramid = feature_pyramid(channels)
        self.channel_down = channel_down(channels)
        self.retinex = Retinex_decom(channels)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decomposes input images [B, 3, H, W] into R, L and hierarchical features.
        """
        lvl0, lvl1, lvl2 = self.pyramid(images)
        bottleneck = self.channel_down(lvl2)
        R_bottleneck, L_bottleneck = self.retinex(bottleneck)

        # Upsample decomposed maps back to input resolution if needed
        h, w = images.shape[-2:]
        R = F.interpolate(R_bottleneck, size=(h, w), mode="bilinear", align_corners=False)
        L = F.interpolate(L_bottleneck, size=(h, w), mode="bilinear", align_corners=False)

        return {
            "R": R,
            "L": L,
            "lvl0": lvl0,
            "lvl1": lvl1,
            "lvl2": lvl2,
            "bottleneck": bottleneck,
        }

    @torch.no_grad()
    def decompose(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method returning (R, L) normalized in [0, 1]."""
        self.eval()
        out = self.forward(img_tensor)
        return out["R"], out["L"]
