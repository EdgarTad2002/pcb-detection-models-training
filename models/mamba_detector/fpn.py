"""
Feature Pyramid Network (FPN) for Multi-Scale PCB Inspection
=============================================================
Fuses hierarchical feature representations from VMamba backbone.
Includes P2 (stride 4) to explicitly preserve high-frequency spatial detail
for micro-capacitors (0402 / 0201 packages).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Standard Conv2D + GroupNorm + SiLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class MambaFPN(nn.Module):
    """
    Feature Pyramid Network.
    Inputs: [C2, C3, C4, C5] with strides [4, 8, 16, 32]
    Outputs: [P2, P3, P4, P5] all with uniform channel dimension (out_channels).
    """

    def __init__(
        self,
        in_channels_list: List[int] = [96, 192, 384, 768],
        out_channels: int = 128,
        include_p2: bool = True,
    ):
        super().__init__()
        self.include_p2 = include_p2
        self.out_channels = out_channels

        # Lateral 1x1 convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # Smooth 3x3 convolutions
        self.fpn_convs = nn.ModuleList([
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            inputs: [C2, C3, C4, C5]
        Returns:
            outputs: [P2, P3, P4, P5]
        """
        assert len(inputs) == len(self.lateral_convs), (
            f"Expected {len(self.lateral_convs)} feature maps, got {len(inputs)}"
        )

        # 1. Compute lateral projections
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. Top-down pathway (from deepest C5 down to finest C2)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="nearest"
            )

        # 3. Smooth with 3x3 convolutions
        outputs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        return outputs
