#!/usr/bin/env python3
"""
Agaian-Sargis Physics-Guided Multi-Scale YOLO (AS-PGMS-YOLO) for Dense PCB AOI
==============================================================================
Unified Next-Generation Architecture synthesizing and upgrading all 4 Pillars:

1. Innovation 1 (Upgraded Pillars 1, 2, 3 — Fractional AMSR + Soft-AGC + True Sargis Weber Stem):
   - 3-Scale Bilateral-Gated Adaptive Multi-Scale Retinex (AMSR, Agaian IEEE Access 2023 Eq. 3 & 16)
     with learnable Fractional Exponents (lambda_c init 1.20, vartheta_k init 0.50, Eq. 10).
   - Fully Differentiable Vectorized Soft-Histogram Local-Global Agaian AGC (Eq. 23) + Sargis ABGC (Eq. 2).
   - Exact Differentiable Sargis Weber-Fechner Contrast Operator (Dissertation Eq. 6.3.8):
         W_k(R) = (u_k)^alpha * ln(1 + u_k)
     evaluated on BOTH Luminance Reflectance (3x3 & 5x5) AND Spectral Opponent Reflectance (Ceramic vs. FR-4).

2. Innovation 2 (Lossless Space-to-Depth Physics Highway into Stage P2):
   - Preserves 100% of pretrained Layer 0 and Layer 1 weights (loaded before wrapping).
   - Injects full-resolution 10-channel physical maps via PixelUnshuffle(2) into Stride 2 and Stride 4
     using zero-initialized residual adapters (exact pretrained identity at step 0).

3. Innovation 3 (Zero-Init P2-to-P3 High-Resolution Micro-Capacitor Neck Bridge):
   - Routes Stride-4 Stage P2 features (320x320) directly into the Stride-8 Neck P3 head input (160x160)
     via a Weber-gated PixelUnshuffle(2) residual bridge without altering Detect head structure.

4. Innovation 4 (Scale-Decoupled 3-Channel Micro-Cap / Boundary-Ring / Macro SDice + Full-1280px Physics SR):
   - Eliminates Macro-IC dominance in binary masks by dedicating 85% of P2/P3 auxiliary Squared-Dice
     supervision to Micro-Capacitors (Class 0) and 3x3 Morphological Boundary Rings.
   - Eliminates the sr_scale=2 train/test resolution mismatch by training and evaluating at native 1280px,
     using the auxiliary SRHead to reconstruct high-frequency Weber-Reflectance targets.

5. Innovation 5 (Scale-Adaptive 2D Gaussian NWD + CIoU Box Loss):
   - Blends Normalized Wasserstein Distance (NWD) with CIoU for sub-8px micro-capacitors.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from tools.scale_adaptive_nwd_loss import (
    ScaleDecoupledMultiStageLoss,
    inject_nwd_into_criterion,
)

# Register canonical module paths so checkpoints unpickle cleanly from any script
sys.modules["as_pgms_yolo"] = sys.modules[__name__]
sys.modules["models.as_pgms_yolo"] = sys.modules[__name__]

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def _gaussian_kernel_2d(ksize: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) * 0.5
    g1 = torch.exp(-(ax ** 2) / (2.0 * (sigma ** 2)))
    g2 = torch.outer(g1, g1)
    return g2 / g2.sum().clamp(min=1e-8)


def _inv_softplus(y: float) -> float:
    return math.log(max(math.exp(y) - 1.0, 1e-6))


# ---------------------------------------------------------------------------
# Innovation 1: Fractional AMSR + Soft-Histogram Agaian AGC + True Sargis Weber
# ---------------------------------------------------------------------------
class FractionalAMSRSoftAGCSargisWeberEdge(nn.Module):
    """
    100% Differentiable Physics Decomposition Module implementing:
      - Pillar 1: 3-Scale Bilateral-Gated AMSR + Fractional Exponents (Agaian Eq. 3, 10, 16)
      - Pillar 2: Soft-Histogram Local-Global Agaian AGC (Eq. 23) + Sargis ABGC (Eq. 2)
      - Pillar 3: True Sargis Weber-Fechner Contrast u^alpha * ln(1 + u) (Eq. 6.3.8)
                  on Luminance Reflectance (3x3, 5x5) and Spectral Opponent Reflectance (3x3)
    """
    __module__ = "models.as_pgms_yolo"

    def __init__(self, eps: float = 1e-3, num_bins: int = 32):
        super().__init__()
        self.eps = float(eps)
        self.num_bins = int(num_bins)

        # ── Pillar 1: 3-Scale Learnable Illumination Filters (k=5, 11, 21) ──
        self.illum_k5 = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False, padding_mode="replicate")
        self.illum_k11 = nn.Conv2d(1, 1, kernel_size=11, padding=5, bias=False, padding_mode="replicate")
        self.illum_k21 = nn.Conv2d(1, 1, kernel_size=21, padding=10, bias=False, padding_mode="replicate")
        with torch.no_grad():
            self.illum_k5.weight.copy_(_gaussian_kernel_2d(5, 1.5)[None, None])
            self.illum_k11.weight.copy_(_gaussian_kernel_2d(11, 3.5)[None, None])
            self.illum_k21.weight.copy_(_gaussian_kernel_2d(21, 7.0)[None, None])

        # Bilateral edge-stopping range parameter sigma_r (init 0.20)
        self.raw_sigma_r = nn.Parameter(torch.tensor(_inv_softplus(0.20), dtype=torch.float32))

        # Learnable Fractional Retinex Exponents (Agaian Case 2 Eq. 10: lambda=1.20, vartheta=0.50)
        self.raw_lambda = nn.Parameter(torch.full((1, 3, 1, 1), _inv_softplus(1.20), dtype=torch.float32))
        self.raw_vartheta = nn.Parameter(torch.full((3,), _inv_softplus(0.50), dtype=torch.float32))

        # ── Pillar 2: Soft-Histogram Centers & Sargis ABGC Parameters ──
        centers = torch.linspace(0.0, 1.0, steps=self.num_bins, dtype=torch.float32).view(1, self.num_bins, 1, 1)
        self.register_buffer("bin_centers", centers)
        self.abgc_alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.abgc_beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # ── Pillar 3: Learnable Sargis Weber Exponents alpha (init 0.20, Eq. 6.3.8) ──
        self.raw_alpha_w3 = nn.Parameter(torch.tensor(_inv_softplus(0.20), dtype=torch.float32))
        self.raw_alpha_w5 = nn.Parameter(torch.tensor(_inv_softplus(0.20), dtype=torch.float32))
        self.raw_alpha_spec = nn.Parameter(torch.tensor(_inv_softplus(0.20), dtype=torch.float32))

    def _sargis_weber_op(self, feat_1ch: torch.Tensor, ksize: int, alpha_param: torch.Tensor, c_stab: float = 0.02) -> torch.Tensor:
        """
        True Sargis Weber-Fechner Local Contrast Operator (Dissertation Chapter 6, Eq. 6.3.8 / Eq. 5):
            f(R) = (u)^alpha * ln(1 + u),   where u = (R_max - R_min) / (R_min + c)
        The ln(1 + u) factor guarantees zero gradient & zero noise amplification on flat FR-4 regions (u -> 0).
        """
        pad = ksize // 2
        f_max = F.max_pool2d(feat_1ch, kernel_size=ksize, stride=1, padding=pad)
        f_min = -F.max_pool2d(-feat_1ch, kernel_size=ksize, stride=1, padding=pad)
        diff = (f_max - f_min).clamp(min=0.0)
        u = (diff / (f_min.clamp(min=0.0) + c_stab)).clamp(min=0.0, max=20.0)

        alpha = F.softplus(alpha_param).clamp(min=0.05, max=0.60)
        weber_raw = ((u + 1e-6) ** alpha) * torch.log1p(u)
        # Per-board peak normalization into [0, 1]
        peak = weber_raw.amax(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        return (weber_raw / peak).clamp(0.0, 1.0)

    def _soft_agaian_agc(self, l_bar: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Differentiable Soft-Histogram Local-Global Agaian AGC (IEEE Access 2023 Eq. 23)
        coupled with Sargis Attention-Based Gamma Correction (ABGC, Eq. 2).
        """
        # Downsample 4x for fast kernel soft-histogram computation
        l_sub = F.avg_pool2d(l_bar, kernel_size=4, stride=4) if min(l_bar.shape[-2:]) >= 8 else l_bar
        bandwidth = 1.5 / float(self.num_bins)
        dist_sq = (l_sub - self.bin_centers.to(l_sub.device)) ** 2
        w_sub = F.softmax(-dist_sq / (2.0 * (bandwidth ** 2)), dim=1)  # [B, M, H/4, W/4]

        pdf = w_sub.mean(dim=(2, 3), keepdim=True)  # [B, M, 1, 1]
        cdf = torch.cumsum(pdf, dim=1)              # [B, M, 1, 1]

        # Evaluate soft PDF and CDF at subsampled grid then bilinearly upsample to full resolution
        p_sub = (w_sub * pdf).sum(dim=1, keepdim=True)
        c_glob_sub = (w_sub * cdf).sum(dim=1, keepdim=True)

        if p_sub.shape[-2:] != l_bar.shape[-2:]:
            p_L = F.interpolate(p_sub, size=l_bar.shape[-2:], mode="bilinear", align_corners=False)
            c_glob = F.interpolate(c_glob_sub, size=l_bar.shape[-2:], mode="bilinear", align_corners=False)
        else:
            p_L, c_glob = p_sub, c_glob_sub

        # Local shadow context CDF (detects local shadow pockets under tall Electrolytic Caps & Connectors)
        l_local_mean = F.avg_pool2d(l_bar, kernel_size=15, stride=1, padding=7)
        c_local = torch.sigmoid(4.0 * (l_bar - l_local_mean))
        c_L = 0.65 * c_glob + 0.35 * c_local

        # Agaian Eq. 23: gamma = ln( c(L*) / (p(L*)^-1 + 2) ) / ln(2)
        p_inv = 1.0 / (p_L.clamp(min=1e-5))
        ratio = (c_L / (p_inv + 2.0)).clamp(min=1e-5)
        raw_gamma = torch.log(ratio) / math.log(2.0)
        gamma = (1.0 + 0.35 * raw_gamma).clamp(min=0.30, max=2.00)

        l_max = l_bar.amax(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        norm_l = (l_bar / l_max).clamp(min=1e-5, max=1.0)
        l_agc = l_max * torch.pow(norm_l, gamma)

        # Sargis ABGC affine refinement
        l_out = self.abgc_alpha * l_agc + self.abgc_beta
        return l_out.clamp(0.0, 1.0)

    def forward(self, x32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x32: Float32 tensor [B, 3, H, W] in [0, 1].
        Returns:
            phys_10ch: [B, 10, H, W] physical representation [x, logR_frac, L_agc, W3_lum, W5_lum, W3_spec]
            sr_target_3ch: [B, 3, H, W] detached high-frequency target [W3_lum, W3_spec, R_lum_norm] for SRHead
        """
        l_init = x32.max(dim=1, keepdim=True)[0]
        sigma_r = F.softplus(self.raw_sigma_r).clamp(min=0.05, max=1.0)
        two_sigma_r_sq = 2.0 * (sigma_r ** 2) + 1e-6

        # 1. 3-Scale Bilateral-Gated Illumination Estimation
        l_scales = []
        for conv in (self.illum_k5, self.illum_k11, self.illum_k21):
            p = conv.kernel_size[0] // 2
            l_blur = F.conv2d(F.pad(l_init, (p, p, p, p), mode="replicate"), conv.weight.float())
            # Bilateral range gate prevents blurring across bright solder vs dark substrate edges
            gate = torch.exp(-((l_init - l_blur) ** 2) / two_sigma_r_sq)
            l_k = (gate * l_blur + (1.0 - gate) * l_init).clamp(min=self.eps, max=1.0)
            l_scales.append(l_k)

        # 2. Agaian AMSR Adaptive Content Likelihood Weights (Eq. 3)
        p_list = []
        for l_k in l_scales:
            mu_k = l_k.mean(dim=(2, 3), keepdim=True)
            std_k = l_k.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            p_k = torch.exp(-0.70710678 * (((l_k - mu_k) / std_k) ** 2))
            p_list.append(p_k)
        p_sum = (p_list[0] + p_list[1] + p_list[2]).clamp(min=1e-6)
        omegas = [p_k / p_sum for p_k in p_list]

        # 3. Agaian Fractional Logarithmic Retinex (Eq. 10 & 16)
        lam = F.softplus(self.raw_lambda).clamp(min=0.40, max=2.50)       # [1, 3, 1, 1]
        varthetas = F.softplus(self.raw_vartheta).clamp(min=0.10, max=1.50)  # [3]

        log_i = torch.log(x32.clamp(min=0.0) + self.eps)
        log_r_frac = torch.zeros_like(x32)
        l_bar = torch.zeros_like(l_init)

        for k in range(3):
            log_lk = torch.log(l_scales[k] + self.eps)
            log_r_frac = log_r_frac + omegas[k] * (lam * log_i - varthetas[k] * log_lk)
            l_bar = l_bar + omegas[k] * l_scales[k]

        # Smooth bounded log-reflectance and positive linear reflectance
        log_r_bounded = 2.0 * torch.tanh(log_r_frac * 0.5)
        r_pos = torch.exp(log_r_bounded)  # strictly positive in [e^-2, e^2], no sigmoid compression!

        # 4. Soft-Histogram Local-Global Agaian AGC + ABGC on Fused Illumination
        l_agc = self._soft_agaian_agc(l_bar.clamp(min=self.eps, max=1.0))

        # 5. True Sargis Weber-Fechner Local Contrast on Luminance & Spectral Opponent Reflectance
        r_lum = r_pos.mean(dim=1, keepdim=True)
        # Opponent material contrast: Ceramic/Copper (R+B) vs Green FR-4 Solder Mask (G)
        r_spec = (0.55 * r_pos[:, 0:1] + 0.45 * r_pos[:, 2:3] - 0.85 * r_pos[:, 1:2]).clamp(min=0.0) + 0.05

        w3_lum = self._sargis_weber_op(r_lum, ksize=3, alpha_param=self.raw_alpha_w3)
        w5_lum = self._sargis_weber_op(r_lum, ksize=5, alpha_param=self.raw_alpha_w5)
        w3_spec = self._sargis_weber_op(r_spec, ksize=3, alpha_param=self.raw_alpha_spec)

        # Normalize log_r_bounded to [-1, 1] for stable conv adapter input
        log_r_norm = log_r_bounded * 0.5
        phys_10ch = torch.cat([x32, log_r_norm, l_agc, w3_lum, w5_lum, w3_spec], dim=1)

        # Detached 3-channel high-frequency target in [0, 1] for auxiliary full-res SRHead
        r_lum_01 = (r_lum / (r_lum.amax(dim=(2, 3), keepdim=True).clamp(min=1e-4))).clamp(0.0, 1.0)
        sr_target_3ch = torch.cat([w3_lum, w3_spec, r_lum_01], dim=1).detach()

        return phys_10ch, sr_target_3ch


# ---------------------------------------------------------------------------
# Innovation 2A: Layer-0 Stem Wrapper with RGB Residual + Stride-2 SPD Injection
# ---------------------------------------------------------------------------
class ASPGMSStemBlock(nn.Module):
    """
    Wraps pretrained YOLO Layer 0 Conv (stride 2):
      1. Injects zero-init 3-channel RGB residual into orig_conv(x + res_rgb).
      2. Folds full-1280px 10-channel physics tensor via lossless Space-to-Depth (PixelUnshuffle(2))
         into 40 channels at Stride 2 and adds a zero-init residual directly to orig_conv's output.
    Starts at 100.000% exact pretrained YOLO behavior at initialization!
    """
    __module__ = "models.as_pgms_yolo"

    def __init__(self, orig_conv: nn.Module, c1_ch: int = 32, hidden: int = 24):
        super().__init__()
        self.orig_conv = orig_conv
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "Conv")
        self.np = getattr(orig_conv, "np", sum(p.numel() for p in orig_conv.parameters()))

        self.physics = FractionalAMSRSoftAGCSargisWeberEdge()

        # Adapter 1: 10ch physics -> 3ch RGB residual before orig_conv
        self.rgb_adapter = nn.Sequential(
            nn.Conv2d(10, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.rgb_adapter[-1].weight)

        # Adapter 2: Lossless PixelUnshuffle(2) of 10ch physics (40ch @ stride 2) -> c1_ch residual
        self.spd2_adapter = nn.Sequential(
            nn.Conv2d(40, c1_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1_ch, c1_ch, kernel_size=3, padding=1, groups=c1_ch, bias=False),
            nn.BatchNorm2d(c1_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1_ch, c1_ch, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.spd2_adapter[-1].weight)

        # Transient training-only reference for HighFreqPhysicsSRHead (only captured during loss())
        self.capture_sr_target: bool = False
        self.last_sr_target: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        dev_type = "cuda" if x.is_cuda else "cpu"

        with torch.autocast(device_type=dev_type, enabled=False):
            x32 = x.float()
            phys_10ch, sr_target = self.physics(x32)
            if self.training and self.capture_sr_target:
                self.last_sr_target = sr_target
            else:
                self.last_sr_target = None

            w_dtype = self.rgb_adapter[0].weight.dtype
            res_rgb = self.rgb_adapter(phys_10ch.to(w_dtype))

            # Lossless Space-to-Depth (PixelUnshuffle(2)) requires even H, W
            h, w = phys_10ch.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                phys_even = F.pad(phys_10ch, (0, w % 2, 0, h % 2), mode="replicate")
            else:
                phys_even = phys_10ch
            spd_40ch = F.pixel_unshuffle(phys_even, downscale_factor=2)
            res_s2 = self.spd2_adapter(spd_40ch.to(w_dtype))

        out_s2 = self.orig_conv(x + res_rgb.to(in_dtype))
        if res_s2.shape[-2:] != out_s2.shape[-2:]:
            res_s2 = F.interpolate(res_s2.float(), size=out_s2.shape[-2:], mode="bilinear", align_corners=False)
        return out_s2 + res_s2.to(out_s2.dtype)


# ---------------------------------------------------------------------------
# Innovation 2B: Layer-1 Lossless Space-to-Depth Bypass into Stage P2 (Stride 4)
# ---------------------------------------------------------------------------
class SPDConvBypassBlock(nn.Module):
    """
    Wraps pretrained YOLO Layer 1 Conv (Stride 2 -> Stride 4):
    Adds a zero-initialized Space-to-Depth (PixelUnshuffle(2)) residual bypass so
    high-frequency Stride-2 features reach Backbone Stage P2 (Stride 4) without
    strided-convolution aliasing.
    """
    __module__ = "models.as_pgms_yolo"

    def __init__(self, orig_conv: nn.Module, c1_ch: int, c2_ch: int):
        super().__init__()
        self.orig_conv = orig_conv
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 1)
        self.type = getattr(orig_conv, "type", "Conv")
        self.np = getattr(orig_conv, "np", sum(p.numel() for p in orig_conv.parameters()))

        mid_ch = max(c2_ch // 2, 16)
        self.spd4_adapter = nn.Sequential(
            nn.Conv2d(c1_ch * 4, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, c2_ch, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.spd4_adapter[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_s4 = self.orig_conv(x)
        h, w = x.shape[-2:]
        x_even = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate") if (h % 2 != 0 or w % 2 != 0) else x
        spd = F.pixel_unshuffle(x_even, downscale_factor=2)
        w_dtype = self.spd4_adapter[0].weight.dtype
        res_s4 = self.spd4_adapter(spd.to(w_dtype))
        if res_s4.shape[-2:] != out_s4.shape[-2:]:
            res_s4 = F.interpolate(res_s4.float(), size=out_s4.shape[-2:], mode="bilinear", align_corners=False)
        return out_s4 + res_s4.to(out_s4.dtype)


# ---------------------------------------------------------------------------
# Innovation 3: Zero-Init P2-to-P3 High-Resolution Micro-Capacitor Neck Bridge
# ---------------------------------------------------------------------------
class P2ToP3MicroBridgeBlock(nn.Module):
    """
    Wraps the Neck P3 output module (immediately feeding the P3 small-object Detect head).
    Uses Ultralytics' native `f = [-1, p2_idx]` graph routing to receive both:
      - x_main: standard Neck P3 input feature map (Stride 8, 160x160)
      - p2_feat: Backbone Stage P2 feature map (Stride 4, 320x320)
    Applies a spatial high-contrast gate + lossless Space-to-Depth (PixelUnshuffle(2))
    projection with zero-initialized final weights into the P3 output.
    """
    __module__ = "models.as_pgms_yolo"

    def __init__(self, orig_module: nn.Module, p2_ch: int, p3_ch: int, p2_idx: int):
        super().__init__()
        self.orig_module = orig_module
        orig_f = getattr(orig_module, "f", -1)
        if isinstance(orig_f, int):
            self.f = [orig_f, int(p2_idx)]
        else:
            self.f = list(orig_f) + [int(p2_idx)]
        self.i = getattr(orig_module, "i", 15)
        self.type = getattr(orig_module, "type", "C3k2")
        self.np = getattr(orig_module, "np", sum(p.numel() for p in orig_module.parameters()))

        # Spatial micro-edge attention gate on P2 (320x320)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(p2_ch, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        # PixelUnshuffle(2) maps [B, p2_ch, 320, 320] -> [B, 4*p2_ch, 160, 160]
        self.bridge_proj = nn.Sequential(
            nn.Conv2d(p2_ch * 4, p3_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(p3_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(p3_ch, p3_ch, kernel_size=3, padding=1, groups=p3_ch, bias=False),
            nn.BatchNorm2d(p3_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(p3_ch, p3_ch, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.bridge_proj[-1].weight)

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, (list, tuple)) and len(x) >= 2 and isinstance(x[-1], torch.Tensor):
            x_main = x[0] if len(x) == 2 else x[:-1]
            p2_feat = x[-1]
            p3_out = self.orig_module(x_main)

            w_dtype = self.bridge_proj[0].weight.dtype
            p2_cast = p2_feat.to(w_dtype)
            gated_p2 = p2_cast * (1.0 + self.spatial_gate(p2_cast))

            h, w = gated_p2.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                gated_p2 = F.pad(gated_p2, (0, w % 2, 0, h % 2), mode="replicate")
            p2_unshuffled = F.pixel_unshuffle(gated_p2, downscale_factor=2)
            bridge_res = self.bridge_proj(p2_unshuffled)

            if bridge_res.shape[-2:] != p3_out.shape[-2:]:
                bridge_res = F.interpolate(bridge_res.float(), size=p3_out.shape[-2:], mode="bilinear", align_corners=False)
            return p3_out + bridge_res.to(p3_out.dtype)

        return self.orig_module(x)


# ---------------------------------------------------------------------------
# Innovation 4: Training-Only Auxiliary Heads (3-Ch Decoupled Mask + Physics SR)
# ---------------------------------------------------------------------------
class MultiStageDecoupledMaskHead(nn.Module):
    """
    Projects intermediate backbone stages [P2, P3, P4, P5] into 3-channel masks:
      [Micro-Capacitor (ch 0), Boundary Ring (ch 1), Macro-Component (ch 2)]
    """
    __module__ = "models.as_pgms_yolo"

    def __init__(self, channels_list: List[int]):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=1),
                nn.Sigmoid(),
            )
            for ch in channels_list
        ])

    def forward(self, stage_features: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for head, feat in zip(self.heads, stage_features):
            w_dtype = head[0].weight.dtype
            out.append(head(feat.to(w_dtype)))
        return out


class HighFreqPhysicsSRHead(nn.Module):
    """
    Fuses P2 (Stride 4, 320x320) + P3 (Stride 8, 160x160) and upsamples 4x via two
    PixelShuffle(2) stages to reconstruct the full-1280px 3-channel Physics Target
    [W3_lum, W3_spec, R_lum] WITHOUT downsampling the input image during training.
    """
    __module__ = "models.as_pgms_yolo"

    def __init__(self, p2_ch: int = 64, p3_ch: int = 128, out_ch: int = 3):
        super().__init__()
        self.p3_lateral = nn.Sequential(
            nn.Conv2d(p3_ch, p2_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(p2_ch),
            nn.SiLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(p2_ch * 2, p2_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(p2_ch),
            nn.SiLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(p2_ch, 32 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(32, 16 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.recon = nn.Sequential(
            nn.Conv2d(16, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, p2_feat: torch.Tensor, p3_feat: torch.Tensor) -> torch.Tensor:
        w_dtype = self.p3_lateral[0].weight.dtype
        p2_c = p2_feat.to(w_dtype)
        p3_c = p3_feat.to(w_dtype)
        p3_up = F.interpolate(self.p3_lateral(p3_c), size=p2_c.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse_conv(torch.cat([p2_c, p3_up], dim=1))
        f = self.up2(self.up1(f))
        return self.recon(f)

    @staticmethod
    def compute_loss(recon_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
        recon_f = recon_phys.float()
        target_f = target_phys.float()
        if recon_f.shape[-2:] != target_f.shape[-2:]:
            recon_f = F.interpolate(recon_f, size=target_f.shape[-2:], mode="bilinear", align_corners=False)
        return F.l1_loss(recon_f, target_f)


# ---------------------------------------------------------------------------
# Unified AS-PGMS-YOLO DetectionModel
# ---------------------------------------------------------------------------
class ASPGMSDetectionModel(DetectionModel):
    """
    DetectionModel integrating:
      - Innovation 1 & 2: ASPGMSStemBlock (Layer 0) + SPDConvBypassBlock (Layer 1)
      - Innovation 3: P2ToP3MicroBridgeBlock (Neck P3)
      - Innovation 4: MultiStageDecoupledMaskHead + HighFreqPhysicsSRHead (Full 1280px)
      - Innovation 5: ScaleAdaptiveNWDBboxLoss in criterion
    """
    __module__ = "models.as_pgms_yolo"

    def __init__(
        self,
        cfg=None,
        ch: int = 3,
        nc: int = 4,
        verbose: bool = True,
        *,
        lambda_sr: float = 0.10,
        lambda_group: float = 0.20,
        use_squared_dice: bool = True,
        imgsz: int = 1280,
    ):
        # Initialize attributes BEFORE super().__init__ (Ultralytics runs dummy forward in super().__init__)
        self.p2_features = None
        self.p3_features = None
        self.stage_features = []
        self.p2_idx = 2
        self.p3_idx = 4
        self.stage_indices = [2, 4, 6, 8]
        self.sr_head = None
        self.mask_head = None
        self.group_criterion = None
        self.lambda_sr = float(lambda_sr)
        self.lambda_group = float(lambda_group)
        self.use_squared_dice = bool(use_squared_dice)
        self.imgsz = int(imgsz)
        self._bridges_injected = False
        self._capture_aux = False

        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _probe_architecture_taps(self) -> Tuple[int, Dict[int, Tuple[int, int]], int, int]:
        """
        Dynamically probes exact layer indices and channel widths across any YOLO backbone/neck
        (100% compatible with both YOLO26s and YOLO12s).
        Returns: (c1_ch, stride_map, p3_neck_idx, p3_neck_ch)
        """
        device = next(self.model.parameters()).device
        x = torch.zeros(1, 3, 128, 128, device=device)
        y = []
        stride_map: Dict[int, Tuple[int, int]] = {}
        c1_ch = 32

        detect_f = getattr(self.model[-1], "f", [15, 18, 21])
        p3_neck_idx = int(detect_f[0]) if isinstance(detect_f, (list, tuple)) else 15
        p3_neck_ch = 128

        with torch.no_grad():
            for i, m in enumerate(self.model[:-1]):
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x if m.i in self.save else None)
                if isinstance(x, torch.Tensor) and x.ndim == 4:
                    if i == 0:
                        c1_ch = int(x.shape[1])
                    if i == p3_neck_idx:
                        p3_neck_ch = int(x.shape[1])
                    s = 128 // max(int(x.shape[-1]), 1)
                    if s in (4, 8, 16, 32) and i <= 12:
                        stride_map[s] = (i, int(x.shape[1]))

        # Fallback if any stride was not recorded
        for s, default_idx in ((4, 2), (8, 4), (16, 6), (32, 8)):
            if s not in stride_map:
                stride_map[s] = (default_idx, 64 * (s // 4))

        return c1_ch, stride_map, p3_neck_idx, p3_neck_ch

    def inject_physics_and_micro_bridges(self):
        """
        Called AFTER loading pretrained weights into the standard YOLO layers so that
        100% of pretrained weights (including Layer 0, Layer 1, and Neck P3) are preserved!
        """
        if self._bridges_injected:
            return

        c1_ch, taps, p3_neck_idx, p3_neck_ch = self._probe_architecture_taps()
        self.p2_idx, p2_ch = taps[4]
        self.p3_idx, p3_ch = taps[8]
        p4_idx, p4_ch = taps[16]
        p5_idx, p5_ch = taps[32]
        self.stage_indices = [self.p2_idx, self.p3_idx, p4_idx, p5_idx]

        # Get Layer 1 output channels (input to Stage P2)
        c2_ch = p2_ch
        try:
            device = next(self.model.parameters()).device
            with torch.no_grad():
                x0 = self.model[0](torch.zeros(1, 3, 64, 64, device=device))
                x1 = self.model[1](x0)
                c1_ch = int(x0.shape[1])
                c2_ch = int(x1.shape[1])
        except Exception:
            pass

        # 1. Wrap Layer 0 with ASPGMSStemBlock (Fractional AMSR + Soft-AGC + True Sargis Weber + SPD2)
        if not isinstance(self.model[0], ASPGMSStemBlock):
            self.model[0] = ASPGMSStemBlock(self.model[0], c1_ch=c1_ch)

        # 2. Wrap Layer 1 with SPDConvBypassBlock (Lossless Stride-2 -> Stride-4 Bypass)
        if not isinstance(self.model[1], SPDConvBypassBlock):
            self.model[1] = SPDConvBypassBlock(self.model[1], c1_ch=c1_ch, c2_ch=c2_ch)

        # 3. Wrap Neck P3 with P2ToP3MicroBridgeBlock (Lossless Stride-4 P2 -> Stride-8 P3 Fusion)
        if not isinstance(self.model[p3_neck_idx], P2ToP3MicroBridgeBlock):
            self.model[p3_neck_idx] = P2ToP3MicroBridgeBlock(
                self.model[p3_neck_idx],
                p2_ch=p2_ch,
                p3_ch=p3_neck_ch,
                p2_idx=self.p2_idx,
            )
            if self.p2_idx not in self.save:
                self.save = sorted(list(set(self.save) | {int(self.p2_idx)}))

        # 4. Setup training-only auxiliary heads
        self.sr_head = HighFreqPhysicsSRHead(p2_ch=p2_ch, p3_ch=p3_ch, out_ch=3)
        self.mask_head = MultiStageDecoupledMaskHead([p2_ch, p3_ch, p4_ch, p5_ch])
        self.group_criterion = ScaleDecoupledMultiStageLoss(use_squared_dice=self.use_squared_dice)
        self._bridges_injected = True

        print(
            f"[AS-PGMS-YOLO] Injected Physics-Guided AMSR-AGC-Weber Stem (L0), SPD Bypass (L1), "
            f"P2->P3 Micro-Bridge (L{self.p2_idx}->L{p3_neck_idx}, {p2_ch}->{p3_neck_ch}ch), "
            f"and 3-Ch Scale-Decoupled SDice Heads @ stages {self.stage_indices}."
        )

    def init_criterion(self):
        criterion = super().init_criterion()
        return inject_nwd_into_criterion(criterion)

    def _predict_once(self, x, profile=False, embed=None):
        """
        Executes forward pass at FULL native 1280px resolution during both training and inference
        (eliminating the sr_scale=2 train-test resolution mismatch).
        """
        capture = bool(getattr(self, "_capture_aux", False)) and self.training
        sr_active = capture and getattr(self, "sr_head", None) is not None
        mask_active = capture and getattr(self, "mask_head", None) is not None

        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed else {-1}
        max_idx = max(embed_set)
        captured_stages = {}

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if sr_active:
                if m.i == self.p2_idx:
                    self.p2_features = x
                elif m.i == self.p3_idx:
                    self.p3_features = x

            if mask_active and m.i in self.stage_indices:
                captured_stages[m.i] = x

            y.append(x if m.i in self.save else None)
            if m.i in embed_set:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        if mask_active and len(captured_stages) == len(self.stage_indices):
            self.stage_features = [captured_stages[idx] for idx in self.stage_indices]
        return x

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        stem_mod = self.model[0] if len(self.model) > 0 else None
        try:
            if preds is None:
                self._capture_aux = True
                if stem_mod is not None and hasattr(stem_mod, "capture_sr_target"):
                    stem_mod.capture_sr_target = True
                preds = self.forward(batch["img"])

            det_loss, loss_items = self.criterion(preds, batch)
            bs = batch["img"].shape[0]

            # 1. Full-1280px High-Frequency Physics SR Reconstruction Loss
            sr_loss_val = torch.tensor(0.0, device=det_loss.device)
            sr_target = getattr(stem_mod, "last_sr_target", None)
            if (
                self.training
                and self.sr_head is not None
                and self.lambda_sr > 0
                and self.p2_features is not None
                and self.p3_features is not None
                and sr_target is not None
            ):
                recon_phys = self.sr_head(self.p2_features, self.p3_features)
                sr_loss_val = self.sr_head.compute_loss(recon_phys, sr_target)

            # 2. Scale-Decoupled 3-Channel Micro-Cap / Boundary-Ring / Macro Squared-Dice Loss
            group_loss_val = torch.tensor(0.0, device=det_loss.device)
            if (
                self.training
                and self.mask_head is not None
                and self.group_criterion is not None
                and self.lambda_group > 0
                and len(self.stage_features) == 4
            ):
                stage_preds = self.mask_head(self.stage_features)
                p2_hw = stage_preds[0].shape[-2:]
                target_masks_p2 = ScaleDecoupledMultiStageLoss.build_decoupled_gt_masks(
                    batch, target_hw=p2_hw, device=det_loss.device
                )
                group_loss_val, _ = self.group_criterion(stage_preds, target_masks_p2)
        finally:
            # Always clear transient references so deepcopy / EMA / checkpointing never sees graph tensors
            self._capture_aux = False
            self.p2_features = None
            self.p3_features = None
            self.stage_features = []
            if stem_mod is not None:
                if hasattr(stem_mod, "capture_sr_target"):
                    stem_mod.capture_sr_target = False
                if hasattr(stem_mod, "last_sr_target"):
                    stem_mod.last_sr_target = None

        total_aux = (self.lambda_sr * sr_loss_val + self.lambda_group * group_loss_val) * bs
        total_loss = det_loss + total_aux.to(det_loss.dtype) / max(len(det_loss), 1)
        return total_loss, loss_items


# ---------------------------------------------------------------------------
# Trainer with Pretrained-First Loading & Clean Checkpoint Stripping
# ---------------------------------------------------------------------------
class ASPGMSTrainer(DetectionTrainer):
    lambda_sr: float = 0.10
    lambda_group: float = 0.20
    use_squared_dice: bool = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        imgsz = getattr(self.args, "imgsz", 1280)
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]

        model = ASPGMSDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and int(os.environ.get("RANK", -1)) == -1,
            lambda_sr=self.lambda_sr,
            lambda_group=self.lambda_group,
            use_squared_dice=self.use_squared_dice,
            imgsz=imgsz,
        )
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)

        # CRITICAL: Load pretrained YOLO weights BEFORE wrapping Layer 0, Layer 1, and Neck P3
        # so that `model.0.conv.weight`, `model.1.conv.weight`, etc. match 100% by key name!
        if weights:
            model.load(weights)

        # Now wrap Layer 0, Layer 1, and Neck P3 with zero-initialized residual physics/micro bridges
        model.inject_physics_and_micro_bridges()

        # If `weights` was already an AS-PGMS checkpoint, load any matching wrapped parameters too
        if weights:
            try:
                model.load(weights, verbose=False)
            except Exception:
                pass

        return model

    def save_model(self):
        from ultralytics.utils.torch_utils import unwrap_model
        ema = unwrap_model(self.ema.ema)

        for model_ref in (ema, self.model):
            for attr in ("p2_features", "p3_features", "stage_features"):
                if hasattr(model_ref, attr):
                    setattr(model_ref, attr, None)
            if hasattr(model_ref, "model") and len(model_ref.model) > 0:
                if hasattr(model_ref.model[0], "last_sr_target"):
                    model_ref.model[0].last_sr_target = None

        orig_class = ema.__class__
        backup = {}
        strip_attrs = [
            "sr_head",
            "mask_head",
            "group_criterion",
            "p2_idx",
            "p3_idx",
            "stage_indices",
            "lambda_sr",
            "lambda_group",
            "use_squared_dice",
            "imgsz",
            "_bridges_injected",
            "_capture_aux",
        ]
        for attr in strip_attrs:
            if hasattr(ema, attr):
                backup[attr] = getattr(ema, attr)
                try:
                    delattr(ema, attr)
                except AttributeError:
                    pass

        orig_loss_names = getattr(ema, "loss_names", None)
        ema.__class__ = DetectionModel
        ema.loss_names = ["box_loss", "cls_loss", "dfl_loss"]

        try:
            super().save_model()
        finally:
            ema.__class__ = orig_class
            if orig_loss_names is not None:
                ema.loss_names = orig_loss_names
            for attr, val in backup.items():
                setattr(ema, attr, val)


# ---------------------------------------------------------------------------
# Main Runner (Train + Canonical Evaluation + Shared Results Export)
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train & Evaluate AS-PGMS-YOLO (All Approaches Upgraded)")
    p.add_argument("--run-key", default="yolo26s_exp7_all_approaches_together_1280")
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--data", default="datasets/pcb-native-res-cleaned-1280/data.yaml")
    p.add_argument("--eval-data", default=None, help="Optional canonical data.yaml for evaluation")
    p.add_argument("--lambda-sr", type=float, default=0.10)
    p.add_argument("--lambda-group", type=float, default=0.20)
    p.add_argument("--use-squared-dice", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--cls", type=float, default=1.5)
    p.add_argument("--box", type=float, default=5.0)
    p.add_argument("--dfl", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.10)
    p.add_argument("--copy-paste", type=float, default=0.30)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--device", default="0")
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--skip-train", action="store_true", help="Skip training and evaluate existing best.pt")
    args = p.parse_args()

    ASPGMSTrainer.lambda_sr = args.lambda_sr
    ASPGMSTrainer.lambda_group = args.lambda_group
    ASPGMSTrainer.use_squared_dice = args.use_squared_dice

    project_dir = Path("runs") / args.run_key
    if not args.skip_train:
        trainer = ASPGMSTrainer(overrides={
            "model": args.weights,
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "project": str(project_dir),
            "name": "pcb-filtered",
            "exist_ok": True,
            "save": True,
            "plots": True,
            "optimizer": args.optimizer,
            "cls": args.cls,
            "box": args.box,
            "dfl": args.dfl,
            "label_smoothing": args.label_smoothing,
            "copy_paste": args.copy_paste,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
        })
        trainer.train()

    candidates = [
        project_dir / "pcb-filtered" / "weights" / "best.pt",
        Path("runs/detect") / project_dir / "pcb-filtered" / "weights" / "best.pt",
        Path("runs/detect") / args.run_key / "pcb-filtered" / "weights" / "best.pt",
    ]
    best_pt = None
    for c in candidates:
        if c.exists():
            best_pt = c
            break

    assert best_pt is not None, f"Could not find best.pt in {candidates}"

    eval_data_path = args.eval_data if args.eval_data else args.data
    print(f"\n[EVAL] Evaluating {args.run_key} ({best_pt}) on '{args.eval_split}' split @ {args.imgsz}px (max_det={args.max_det})...")
    eval_model = YOLO(str(best_pt))
    metrics = eval_model.val(
        data=eval_data_path,
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.eval_conf,
        iou=args.eval_iou,
        max_det=args.max_det,
        device=args.device,
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    prec = float(metrics.box.mp)
    rec = float(metrics.box.mr)
    speed = metrics.speed
    total_time_ms = float(
        speed.get("preprocess", 0.0)
        + speed.get("inference", 0.0)
        + speed.get("postprocess", 0.0)
    )
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0

    # Extract true per-class AP@50 from metrics.box.ap50
    per_class_ap50 = {}
    names = metrics.names if hasattr(metrics, "names") else {i: CLASS_NAMES[i] for i in range(4)}
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None and len(metrics.box.ap50) > 0:
        for idx, ap in enumerate(metrics.box.ap50):
            c_name = names.get(idx, f"Class_{idx}")
            per_class_ap50[c_name] = round(float(ap), 4)

    summary = {
        "model": args.run_key,
        "weights": str(best_pt),
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": prec,
        "recall": rec,
        "total_time_ms": total_time_ms,
        "fps": fps,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "max_det": args.max_det,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_class_ap50": per_class_ap50,
    }
    for k, v in per_class_ap50.items():
        summary[f"{k}_ap50"] = v

    print("\n--- Overall Metrics ---")
    print(f"mAP@50:    {map50:.4f}")
    print(f"mAP@50-95: {map50_95:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"FPS:       {fps:.1f} ({total_time_ms:.1f} ms)")
    print("--- Per-Class AP@50 ---")
    for k, v in per_class_ap50.items():
        print(f"  {k}: {v:.4f}")

    for results_dir in (Path("/mnt/weka/etadevosyan/pcb-yolo/results"), Path("results")):
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            out_json = results_dir / f"{args.run_key}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Saved results JSON to: {out_json}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
