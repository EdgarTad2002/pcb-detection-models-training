#!/usr/bin/env python3
"""
Convolutional Grand Unified Learnable Stem for YOLO26s
======================================================
Replaces CTDN transformer/attention with pure, differentiable 2D convolutions:
  1. Illumination Estimation:
     - 15x15 Learnable Gaussian Convolution over max(R,G,B).
  2. Agaian Adaptive Gamma Correction (AGC):
     - Differentiable IEEE Access 10365161 Eq. 23 applied to L.
  3. Reflectance Extraction:
     - logR = log(I + eps) - log(L + eps).
  4. Sargis Weber Contrast Operator on Reflectance:
     - Differentiable 3x3 local min/max pooling contrast with alpha=0.20 exponent.
  5. Convolutional Residual Adapter:
     - Concat[I, logR, W, L_AGC] (8 channels) -> 1x1 Conv -> 3x3 DWConv -> 1x1 Conv.
     - Final layer is ZERO-initialized: starts as exact pretrained YOLO26s (guaranteed stability).
  6. Pure 2D convolutions throughout: zero attention overhead, sub-millisecond latency.
"""

import math
from pathlib import Path
import sys

# Ensure root in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from tools.agaian_adaptive_gamma import apply_agaian_agc_torch


def _gaussian_kernel(ksize: int = 15, sigma: float = 4.0) -> torch.Tensor:
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g2 = torch.outer(g1, g1)
    return g2 / g2.sum()


class LearnableConvGrandUnifiedEdge(nn.Module):
    """
    Pure Convolutional Grand Unified Stem:
    Gaussian Conv (Illumination) + Agaian AGC + Weber Contrast on Reflectance + Residual Adapter.
    """

    def __init__(self, ksize: int = 15, sigma: float = 4.0, eps: float = 1e-3, hidden: int = 16):
        super().__init__()
        self.eps = eps
        self.ksize = ksize

        # 1. Learnable Illumination Estimator (Gaussian initialized)
        self.illum = nn.Conv2d(1, 1, ksize, padding=ksize // 2, bias=False, padding_mode="replicate")
        with torch.no_grad():
            self.illum.weight.copy_(_gaussian_kernel(ksize, sigma)[None, None])

        # 2. Adapter: 8 channels [x (3), log_r (3), W (1), L_agc (1)] -> 3 channel residual
        self.adapter = nn.Sequential(
            nn.Conv2d(8, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=False),
        )
        # Zero-init last conv so initial residual is zero (model starts at exact baseline)
        nn.init.zeros_(self.adapter[-1].weight)

    def features(self, x: torch.Tensor):
        """x: (B, 3, H, W) in [0, 1]. Returns L_agc, log_r, W in float32."""
        x = x.float()
        p = self.ksize // 2
        maxc = x.max(dim=1, keepdim=True)[0]

        # 1. Illumination via Gaussian Convolution
        L = F.conv2d(F.pad(maxc, (p, p, p, p), mode="replicate"), self.illum.weight.float())
        L = L.clamp(min=self.eps, max=1.0)

        # 2. Agaian Adaptive Gamma Correction on Illumination (IEEE Access Eq. 23)
        L_agc = apply_agaian_agc_torch(L, gamma_min=0.30, gamma_max=2.0)

        # 3. Reflectance in log domain
        log_r = torch.log(x + self.eps) - torch.log(L + self.eps)
        log_r = log_r.clamp(-4.0, 2.0)

        # 4. Sargis Weber Contrast on Reflectance via Differentiable Local Min/Max Pooling
        gray_r = log_r.mean(dim=1, keepdim=True)
        # Normalize gray_r into positive range [0, 1] for stable contrast ratio
        gray_norm = torch.sigmoid(gray_r)
        l_max = F.max_pool2d(gray_norm, kernel_size=3, stride=1, padding=1)
        l_min = -F.max_pool2d(-gray_norm, kernel_size=3, stride=1, padding=1)
        # Weber contrast formula: ((Lmax - Lmin) / (Lmin + 1e-3))^0.20
        weber = ((l_max - l_min) / (l_min + 1e-3)).clamp(min=0.0, max=10.0) ** 0.20

        return L_agc, log_r, weber

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        dev_type = "cuda" if x.is_cuda else "cpu"
        # Run Retinex math in FP32 for numerical stability under AMP
        with torch.autocast(device_type=dev_type, enabled=False):
            x32 = x.float()
            L_agc, log_r, weber = self.features(x32)
            # Concatenate [raw RGB (3), log Reflectance (3), Weber edge (1), AGC Illumination (1)] = 8 ch
            feats = torch.cat([x32, log_r, weber, L_agc], dim=1)
            w_dtype = self.adapter[0].weight.dtype
            residual = self.adapter(feats.to(w_dtype))
        return residual.to(in_dtype)


class ConvGrandUnifiedStemInputBlock(nn.Module):
    """Wraps YOLO Layer 0 Conv with the Learnable Convolutional Grand Unified Stem."""

    def __init__(self, orig_conv: nn.Module, ksize: int = 15, sigma: float = 4.0, hidden: int = 16):
        super().__init__()
        self.orig_conv = orig_conv
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "Conv")
        self.np = getattr(orig_conv, "np", sum(p.numel() for p in orig_conv.parameters()))
        self.stem = LearnableConvGrandUnifiedEdge(ksize=ksize, sigma=sigma, hidden=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orig_conv(x + self.stem(x))


def wrap_model_with_conv_gu_stem(det_model: nn.Module) -> nn.Module:
    """Wraps det_model.model[0] with the Convolutional Grand Unified Stem."""
    if not isinstance(det_model.model[0], ConvGrandUnifiedStemInputBlock):
        det_model.model[0] = ConvGrandUnifiedStemInputBlock(det_model.model[0])
    return det_model


class ConvGrandUnifiedTrainer(DetectionTrainer):
    """DetectionTrainer injecting the Convolutional Grand Unified Stem into Layer 0."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        wrap_model_with_conv_gu_stem(model)
        if weights:
            model.load(weights)
        return model


if __name__ == "__main__":
    print("Testing LearnableConvGrandUnifiedEdge...")
    block = LearnableConvGrandUnifiedEdge()
    x = torch.rand(2, 3, 640, 640)
    res = block(x)
    print(f"Residual shape: {res.shape} (Expected: [2, 3, 640, 640])")
    assert res.shape == x.shape
    # Check zero-init
    print(f"Initial max residual value: {res.abs().max().item():.6f} (Expected: 0.000000)")
    assert res.abs().max().item() < 1e-6
    print("✅ LearnableConvGrandUnifiedEdge passed self-test successfully!")
