"""
VMamba (Visual State Space Model) Hierarchical Backbone
=========================================================
Extracts multi-scale hierarchical feature maps from 2D images using stacked
Visual State Space Blocks (VSSBlocks).

Strides and Output Shapes (for input 640x640):
- Stage 1 (C2): Stride 4,  160x160, channels 96
- Stage 2 (C3): Stride 8,   80x80,  channels 192
- Stage 3 (C4): Stride 16,  40x40,  channels 384
- Stage 4 (C5): Stride 32,  20x20,  channels 768
"""

import os
from typing import List, Optional
import urllib.request

import torch
import torch.nn as nn
from .ss2d import VSSBlock


class PatchEmbed2D(nn.Module):
    """Patch Embedding: projects 3-channel image into 2D token grid (stride 4)."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)


class Downsample2D(nn.Module):
    """Downsamples spatial dimensions by 2x while doubling channel dimension."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(1, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x)


class VMambaStage(nn.Module):
    """A stage consisting of stacked VSSBlocks."""

    def __init__(self, dim: int, depth: int, d_state: int = 16, ssm_ratio: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(dim=dim, d_state=d_state, ssm_ratio=ssm_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VMambaBackbone(nn.Module):
    """
    Hierarchical Visual State Space Model Backbone.
    Outputs multi-scale feature maps [C2, C3, C4, C5] for dense object detection.
    """

    def __init__(
        self,
        in_chans: int = 3,
        dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 9, 2],
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        out_indices: List[int] = [0, 1, 2, 3],
    ):
        super().__init__()
        self.dims = dims
        self.out_indices = out_indices
        self.num_stages = len(depths)

        # 1. Stem: Patch embedding (stride 4)
        self.patch_embed = PatchEmbed2D(in_chans=in_chans, embed_dim=dims[0], patch_size=4)

        # 2. Hierarchical stages + downsampling transitions
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_stages):
            stage = VMambaStage(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                ssm_ratio=ssm_ratio,
            )
            self.stages.append(stage)

            if i < self.num_stages - 1:
                down = Downsample2D(in_dim=dims[i], out_dim=dims[i + 1])
                self.downsamples.append(down)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: List of feature maps corresponding to out_indices
        """
        x = self.patch_embed(x) # (B, dims[0], H/4, W/4)

        outputs = []
        for i in range(self.num_stages):
            x = self.stages[i](x)
            if i in self.out_indices:
                outputs.append(x)
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        return outputs

    def load_pretrained(self, weights_path: str, strict: bool = False):
        """
        Loads official VMamba ImageNet pretrained weights.
        Safely filters out classifier head and mismatching tensors.
        """
        if not os.path.exists(weights_path):
            print(f"Warning: Pretrained weights not found at {weights_path}")
            return False

        print(f"Loading pretrained VMamba weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]

        model_dict = self.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            # Strip potential backbone prefix
            clean_k = k.replace("backbone.", "").replace("patch_embed.", "patch_embed.")
            if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                filtered_dict[clean_k] = v

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"Successfully transferred {len(filtered_dict)} layers from pretrained weights.")
        return True
