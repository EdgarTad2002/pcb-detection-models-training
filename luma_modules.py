"""
LUMA-YOLO modules for PCB Inspection under Solder Glare and Specular Reflections.
Inspired by Feng et al. (2026) LUMA-YOLO & YOLO26.

Includes:
1. LIAM: Lightweight Illumination-Adaptive Module (Front-End Glare & Lighting Calibration)
2. AConv: Average-Pooling Convolution (Anti-Aliased Downsampling to preserve 0402 micro-passives)
3. SimAM: Parameter-Free 3D Attention (Energy-based Saliency, 0 extra parameters)
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class SimAM(nn.Module):
    """
    Simple Parameter-Free Attention Module (SimAM).
    Computes 3D spatial + channel attention weights based on a neuroscience energy function.
    Total learnable parameters: 0.
    """

    def __init__(self, c1=None, c2=None, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / max(n, 1)
        y = d / (4 * (v + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class AConv(nn.Module):
    """
    Average Pooling Convolution (AConv).
    Replaces strided convolutions with AvgPool2d + Conv2d to prevent spatial Nyquist
    aliasing on micro-SMD components (0402 chip capacitors).
    """

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()
        if s == 2:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
            self.conv = Conv(c1, c2, k=k, s=1, p=p, g=g, d=d, act=act)
        else:
            self.pool = nn.Identity()
            self.conv = Conv(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)

    def forward(self, x):
        return self.conv(self.pool(x))


class LIAM(nn.Module):
    """
    Lightweight Illumination-Adaptive Module (LIAM).
    Differentiable front-end calibration for specular solder glare and dark substrate camouflage.
    """

    def __init__(self, c1=3, c2=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(16, c2, kernel_size=3, padding=1)
        self.gamma_scale = nn.Parameter(torch.ones(1, c2, 1, 1))

    def forward(self, x):
        res = self.out_conv(self.conv2(self.conv1(x)))
        return x + torch.tanh(res) * 0.25 * self.gamma_scale


def register_luma_modules():
    """Hooks LUMA modules into Ultralytics namespace for transparent YAML parsing."""
    import ultralytics.nn.modules as m
    import ultralytics.nn.tasks as t
    import ultralytics.nn.modules.block as b

    setattr(m, "SimAM", SimAM)
    setattr(t, "SimAM", SimAM)
    setattr(b, "SimAM", SimAM)

    setattr(m, "AConv", AConv)
    setattr(t, "AConv", AConv)
    setattr(b, "AConv", AConv)

    setattr(m, "LIAM", LIAM)
    setattr(t, "LIAM", LIAM)
    setattr(b, "LIAM", LIAM)


# Auto-register on import
register_luma_modules()
