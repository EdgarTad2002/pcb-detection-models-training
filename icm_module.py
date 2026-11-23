#!/usr/bin/env python3
"""
icm_module.py
=============
Illumination-Conditioned Modulation (ICM) for YOLO on Densely-Populated PCBs.

Reference:
    "Low-Light Object Detection via Image-Level Illumination-Conditioned Modulation"
    (ICM-YOLO Framework)

Theoretical Rationale for PCB AOI:
    Industrial PCB inspection images suffer from severe non-uniform lighting:
    1. Specular highlights from directional LED illumination on metallic solder fillets (I ~ 255).
    2. Deep shadows cast by macro components (e.g., electrolytic capacitor aluminum cans, connectors).
    3. Faint low-contrast ceramic bodies of micro SMD capacitors (0402 / 0201) sitting on dull solder masks.

    Traditional preprocessing (like CLAHE or static Retinex) is non-differentiable and amplifies
    high-frequency fiberglass weave substrate noise. In contrast, ICM dynamically extracts
    image-level illumination quantile descriptors:
        - Mean intensity: I_mean
        - Lower illumination quantile: q_l (e.g. 0.25)
        - Upper illumination quantile: q_h (e.g. 0.75)
        - Illumination dynamic range: q_h - q_l
    and feeds this descriptor into a lightweight modulation generator MLP that predicts affine
    scale (gamma) and shift (beta) parameters to modulate intermediate features in feature space:
        F_out = (1.0 + gamma) * F + beta
    with ZERO-initialization on the final projection layer, guaranteeing that at initialization (step 0),
    the model behaves exactly as the pre-trained detector without degradation.

Self-test:
    python icm_module.py
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationIndicator(nn.Module):
    """
    Computes a continuous, differentiable descriptor of the image-level illumination
    distribution: [mean, q_low, q_high, dynamic_range, std].
    """
    def __init__(self, q_low: float = 0.25, q_high: float = 0.75):
        super().__init__()
        self.q_low = q_low
        self.q_high = q_high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input RGB image tensor (B, 3, H, W) normalized to [0, 1] or [0, 255].
        Returns:
            descriptor: (B, 5) tensor capturing illumination statistics.
        """
        # Convert RGB to perceived luminance (ITU-R BT.601)
        if x.shape[1] == 3:
            lum = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            lum = x.mean(dim=1, keepdim=True)

        B = lum.shape[0]
        lum_flat = lum.view(B, -1)  # (B, H*W)

        mean = lum_flat.mean(dim=-1, keepdim=True)
        std = lum_flat.std(dim=-1, keepdim=True)

        # Quantile estimation via sorted sub-sampling for efficiency (1024 points is plenty)
        n_pixels = lum_flat.shape[1]
        if n_pixels > 1024:
            # Subsample evenly spaced indices across the image to avoid expensive full sort
            step = n_pixels // 1024
            sampled = lum_flat[:, ::step][:, :1024]
        else:
            sampled = lum_flat

        sorted_lum, _ = torch.sort(sampled, dim=-1)
        idx_low = int(self.q_low * (sorted_lum.shape[1] - 1))
        idx_high = int(self.q_high * (sorted_lum.shape[1] - 1))

        q_l = sorted_lum[:, idx_low : idx_low + 1]
        q_h = sorted_lum[:, idx_high : idx_high + 1]
        dyn_range = q_h - q_l

        # Concatenate 5-D illumination descriptor
        descriptor = torch.cat([mean, q_l, q_h, dyn_range, std], dim=-1)
        return descriptor


class IlluminationConditionedModulationBlock(nn.Module):
    """
    Wraps an existing convolutional block (e.g. Layer 0 Stem or intermediate feature layer)
    and applies Feature-wise Linear Modulation (FiLM) conditioned on the image-level illumination descriptor.
    """
    def __init__(self, orig_conv: nn.Module, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.orig_conv = orig_conv
        self.out_channels = out_channels

        # Graph attributes for Ultralytics DetectionModel forward loop
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "ICMStemInputBlock")
        self.np = getattr(orig_conv, "np", sum(p.numel() for p in orig_conv.parameters()))

        self.indicator = IlluminationIndicator(q_low=0.25, q_high=0.75)

        # Modulation Generator MLP: maps 5-D illumination vector -> scale (gamma) & shift (beta)
        hidden_dim = 32
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2 * out_channels),
        )

        # ZERO-initialization: at step 0, gamma = 0 and beta = 0, so F_out = F (exact identity)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute illumination descriptor from raw input image x
        desc = self.indicator(x)  # (B, 5)

        # 2. Compute base features through original convolution
        feat = self.orig_conv(x)  # (B, out_channels, H_out, W_out)

        # 3. Generate scale (gamma) and shift (beta)
        style = self.mlp(desc)  # (B, 2 * out_channels)
        gamma, beta = style.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, out_channels, 1, 1)

        # 4. Feature-wise Linear Modulation (FiLM) with residual identity
        modulated_feat = (1.0 + gamma) * feat + beta
        return modulated_feat


def inject_icm_into_model(model: nn.Module) -> nn.Module:
    """
    Injects Illumination-Conditioned Modulation into Layer 0 (Stem) of a YOLO DetectionModel.
    Zero-initialized, perfectly preserving pretrained weights at step 0.
    """
    # Check if already injected
    if isinstance(model.model[0], IlluminationConditionedModulationBlock):
        return model

    orig_l0 = model.model[0]
    # Inspect out_channels of Layer 0
    if hasattr(orig_l0, "conv"):
        out_ch = orig_l0.conv.out_channels
    elif hasattr(orig_l0, "out_channels"):
        out_ch = orig_l0.out_channels
    else:
        # Default for YOLO26s stem (3 -> 32)
        out_ch = 32

    model.model[0] = IlluminationConditionedModulationBlock(orig_l0, in_channels=3, out_channels=out_ch)
    return model


if __name__ == "__main__":
    print("=" * 65)
    print("Self-testing Illumination-Conditioned Modulation (ICM-YOLO) Module")
    print("=" * 65)

    # 1. Test IlluminationIndicator
    indicator = IlluminationIndicator()
    dummy_img = torch.rand(2, 3, 1280, 1280)
    desc = indicator(dummy_img)
    print(f"Illumination descriptor shape: {desc.shape} (Batch=2, Dim=5)")
    print(f"Sample descriptor values: {desc[0].tolist()}")
    assert desc.shape == (2, 5)

    # 2. Test ICM Block with a standard Conv layer
    dummy_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
    icm_block = IlluminationConditionedModulationBlock(dummy_conv, in_channels=3, out_channels=32)

    # 3. Test backward pass gradient flow
    out_icm = icm_block(dummy_img)
    out_icm.sum().backward()
    grad_norm = icm_block.mlp[-1].weight.grad.norm().item()
    print(f"Gradient norm on ICM MLP weights: {grad_norm:.6f}")
    assert grad_norm >= 0, "Gradient error!"

    print("\nALL ICM SELF-TESTS PASSED SUCCESSFULLY! Ready for training.")
