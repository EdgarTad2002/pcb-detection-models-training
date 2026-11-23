#!/usr/bin/env python3
"""
weber_stem.py
=============
Extended Weber Contrast Law-Driven Edge Detection Stem for YOLO26s (Eq. 5).

Implements the Professor's pixel-wise 3x3 local neighborhood Weber Contrast
formula as an in-network GPU convolutional/sliding-window stem injected into
Layer 0 of YOLO26s:

    f(L) = 300 * ((L_max - L_min) / (L_min + c))^{0.01} * log((L_max - L_min) / (L_min + c))   [Eq. 5]

where:
    - L_max and L_min denote the maximum and minimum pixel intensities in a
      local 3x3 region around each pixel of image L (in [0, 255] intensity scale),
    - c = 1.0 is the Weber background stabilization constant.

Mathematical & Numerical Design:
    1. For every pixel (i, j), L_max and L_min are extracted over a 3x3 sliding
       window with replicate border padding (avoiding artificial boundary edges)
       using GPU max/min pooling (`F.max_pool2d` with kernel_size=3, stride=1).
    2. Let W = (L_max - L_min) / (L_min + c) be the local 3x3 Weber contrast.
       Analytically, lim_{W -> 0+} W^{0.01} * log(W) = 0. To avoid IEEE fp32
       log(0) = -inf on uniform 3x3 patches while preserving exact Eq. 5 values
       for all active edges (W > 0), we evaluate:
           W_safe = clamp(W, min=1e-5)
           gate   = W / (W + 1e-4)
           f(L)   = 300.0 * gate * (W_safe ** 0.01) * log(W_safe)
    3. We compute f(L) across both the 3 RGB channels and the luminance channel
       L_gray = 0.299*R + 0.587*G + 0.114*B, scaled by 1/300.0 into O(1) network
       range, together with the non-negative log-Weber edge magnitude log(1 + W_gray).
    4. A lightweight 3x3 convolutional adapter fuses [x, f(L_rgb)/300, f(L_gray)/300,
       log(1 + W_gray)] (8 channels -> 16 -> 3) with a zero-initialized output
       projection so that at step 0 the wrapped YOLO26s model is 100% identical
       to the pretrained checkpoint and learns the optimal edge enhancement end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtendedWeberContrastEdge(nn.Module):
    """
    Pixel-wise 3x3 Extended Weber Contrast Law (Eq. 5) edge extractor + 3x3 conv adapter.
    """

    def __init__(
        self,
        ksize: int = 3,
        c: float = 1.0,
        alpha: float = 0.01,
        scale: float = 300.0,
        hidden: int = 16,
    ):
        super().__init__()
        self.ksize = ksize
        self.c = float(c)
        self.alpha = float(alpha)
        self.scale = float(scale)

        # Luminance weights [R, G, B]
        luma = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("luma_weights", luma)

        # 8 input channels:
        #   - 3 ch: original RGB x in [0, 1]
        #   - 3 ch: Eq. 5 per-channel Weber response f(L_rgb) / 300.0
        #   - 1 ch: Eq. 5 luminance Weber response f(L_gray) / 300.0
        #   - 1 ch: positive log-Weber contrast log(1 + W_gray)
        self.adapter = nn.Sequential(
            nn.Conv2d(8, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=False),
        )
        # Zero-init final 1x1 conv so wrapped YOLO26s starts as exact identity at step 0
        nn.init.zeros_(self.adapter[-1].weight)

    def _weber_eq5(self, L: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates Eq. (5) for every pixel in a local 3x3 neighborhood of L in [0, 255].
        Returns (f_norm, W) where f_norm = f(L) / 300.0.
        """
        p = self.ksize // 2
        L_pad = F.pad(L, (p, p, p, p), mode="replicate")
        L_max = F.max_pool2d(L_pad, kernel_size=self.ksize, stride=1)
        L_min = -F.max_pool2d(-L_pad, kernel_size=self.ksize, stride=1)

        diff = (L_max - L_min).clamp(min=0.0)
        W = diff / (L_min + self.c)

        # Exact Eq. 5: f(L) = 300 * W^{0.01} * log(W), with smooth 0 limit as W -> 0+
        W_safe = W.clamp(min=1e-5)
        gate = W / (W + 1e-4)
        f_L = self.scale * gate * (W_safe ** self.alpha) * torch.log(W_safe)
        f_norm = (f_L / self.scale).clamp(-6.0, 6.0)
        return f_norm, W

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, H, W) in [0, 1].
        Returns (f_rgb_norm, f_gray_norm, log1p_W_gray) in float32.
        """
        x32 = x.float().clamp(0.0, 1.0)
        L_rgb = x32 * 255.0
        L_gray = (L_rgb * self.luma_weights.float()).sum(dim=1, keepdim=True)

        f_rgb_norm, _ = self._weber_eq5(L_rgb)
        f_gray_norm, W_gray = self._weber_eq5(L_gray)
        log1p_W_gray = torch.log1p(W_gray).clamp(0.0, 6.0)
        return f_rgb_norm, f_gray_norm, log1p_W_gray

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        dev_type = "cuda" if x.is_cuda else "cpu"
        # Evaluate power/log in fp32 to prevent fp16 underflow/overflow
        with torch.autocast(device_type=dev_type, enabled=False):
            x32 = x.float()
            f_rgb_norm, f_gray_norm, log1p_W_gray = self.features(x32)
            feats = torch.cat([x32, f_rgb_norm, f_gray_norm, log1p_W_gray], dim=1)
            w_dtype = self.adapter[0].weight.dtype
            residual = self.adapter(feats.to(w_dtype))
        return residual.to(in_dtype)


class WeberContrastStemInputBlock(nn.Module):
    """Wraps YOLO26s Layer 0: x -> x + ExtendedWeberContrastEdge(x) -> orig_conv."""

    def __init__(self, orig_conv: nn.Module, ksize: int = 3, c: float = 1.0, hidden: int = 16):
        super().__init__()
        self.orig_conv = orig_conv
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "Conv")
        self.np = getattr(orig_conv, "np", sum(p.numel() for p in orig_conv.parameters()))
        self.weber = ExtendedWeberContrastEdge(ksize=ksize, c=c, hidden=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orig_conv(x + self.weber(x))


def wrap_model_with_weber_stem(det_model: nn.Module) -> nn.Module:
    """Idempotently wraps det_model.model[0] with WeberContrastStemInputBlock."""
    if not isinstance(det_model.model[0], WeberContrastStemInputBlock):
        det_model.model[0] = WeberContrastStemInputBlock(det_model.model[0])
    return det_model
