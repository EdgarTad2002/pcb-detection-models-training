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


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block for High-Resolution Early Stages (MambaVision architecture).
    Captures fine spatial patterns (e.g. 0402 ceramic chip capacitors) with zero
    sequential recurrence overhead, O(1) memory, and high throughput.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(int(mlp_ratio * dim), dim, kernel_size=1)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop(x)
        return residual + x


class VMambaStage(nn.Module):
    """A stage consisting of stacked ConvNeXtBlocks or VSSBlocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        stage_type: str = "mamba",
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            if stage_type == "conv":
                block = ConvNeXtBlock(dim=dim)
            else:
                block = VSSBlock(dim=dim, d_state=d_state, ssm_ratio=ssm_ratio)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class VMambaBackbone(nn.Module):
    """
    Hierarchical Visual State Space Model Backbone (MambaVision Hybrid Architecture).
    - Early stages (stride 4 & 8): ConvNeXt blocks for micro-geometry & 0402 chip capacitors.
    - Deep stages (stride 16 & 32): 2D Selective Scan (SS2D) for global context across the PCB.
    Outputs multi-scale feature maps [C2, C3, C4, C5] for dense object detection.
    """

    def __init__(
        self,
        in_chans: int = 3,
        dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 9, 2],
        stage_types: Optional[List[str]] = None,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        out_indices: List[int] = [0, 1, 2, 3],
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self.out_indices = out_indices
        self.num_stages = len(depths)
        self.stage_types = stage_types or ["conv", "conv", "mamba", "mamba"]

        # 1. Stem: Patch embedding (stride 4)
        self.patch_embed = PatchEmbed2D(in_chans=in_chans, embed_dim=dims[0], patch_size=4)

        # 2. Hierarchical stages + downsampling transitions
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(self.num_stages):
            st = self.stage_types[i] if i < len(self.stage_types) else "mamba"
            stage = VMambaStage(
                dim=dims[i],
                depth=depths[i],
                stage_type=st,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                use_checkpoint=use_checkpoint,
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
