#!/usr/bin/env python3
"""
matched_filter_yolo26.py
========================
Matched-Filter YOLO26s: Integrating Normalized Cross-Correlation (NCC)
Spatial Saliency Priors as an Auxiliary 4th Input Channel for Micro-Capacitor AOI.

Key Innovation:
- Micro-capacitors on PCBs suffer from severe spatial downsampling in deep backbones.
- MatchedFilterInputBlock replaces Layer 0 stem convolution:
  1. Computes multi-scale horizontal and vertical Normalized Cross-Correlation (NCC)
     heatmaps R_max(x, y) at pixel-resolution on-the-fly directly on GPU.
  2. Concatenates the correlation heatmap as an auxiliary 4th channel: [R, G, B, R_max].
  3. Projects through a 4-channel stem convolution initialized with 100% of COCO
     pretrained weights for the RGB channels and warm-started weights for Channel 4.
  4. Preserves 100% of the rest of the YOLO26s backbone and head weights.

Usage:
  python matched_filter_yolo26.py \
      --run-key rectified_yolov26s_matched_filter_640 \
      --weights yolo26s.pt \
      --data datasets/pcb-unified-4class/data.yaml \
      --epochs 100 --imgsz 640 --batch 16 --workers 8 --eval-conf 0.001
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

# Canonical module registration for safe unpickling
if __name__ == "__main__":
    sys.modules["matched_filter_yolo26"] = sys.modules[__name__]

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


# ---------------------------------------------------------------------------
# 1. GPU-Accelerated Matched Filter Cross-Correlation Module
# ---------------------------------------------------------------------------
class MatchedFilterModule(nn.Module):
    """
    Computes spatial correlation saliency maps against canonical capacitor templates.
    Operates on grayscale luminance: L = 0.299*R + 0.587*G + 0.114*B.
    Outputs a single-channel spatial heatmap [B, 1, H, W] in [0, 1].
    """
    __module__ = "matched_filter_yolo26"

    def __init__(self, kernel_size=(15, 31)):
        super().__init__()
        # Standard EIA 0402 capacitor aspect ratio is ~2:1
        # Horizontal template: dark ceramic center, bright silver solder terminations on left/right
        kh, kw = kernel_size
        th = torch.zeros(1, 1, kh, kw, dtype=torch.float32)
        end_w = max(2, kw // 4)
        th[:, :, :, :end_w] = 1.0       # Left solder terminal (bright silver)
        th[:, :, :, -end_w:] = 1.0      # Right solder terminal (bright silver)
        th[:, :, :, end_w:-end_w] = -1.0 # Center dielectric body (dark ceramic)

        # Normalize kernel (zero-mean, unit-variance)
        th = (th - th.mean()) / (th.std() + 1e-6)

        # Vertical template: transpose of horizontal
        tv = th.transpose(2, 3).contiguous()

        # Register as buffers (not updated by SGD, fixed physical template geometry)
        self.register_buffer("kernel_h", th)
        self.register_buffer("kernel_v", tv)

        # Grayscale weights (standard Rec. 601 luma)
        luma_w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("luma_weight", luma_w)

    def forward(self, x):
        """
        Args:
            x: Input image tensor [B, 3, H, W] (normalized to [0, 1])
        Returns:
            corr_map: Spatial correlation response [B, 1, H, W] in [0, 1]
        """
        # 1. Convert RGB to single-channel luminance
        luma = F.conv2d(x, self.luma_weight)  # [B, 1, H, W]

        # 2. Local mean and variance for normalized cross-correlation
        kh_h, kw_h = self.kernel_h.shape[2:]
        kh_v, kw_v = self.kernel_v.shape[2:]

        # Convolve with horizontal and vertical prototype kernels
        resp_h = F.conv2d(luma, self.kernel_h, padding="same")
        resp_v = F.conv2d(luma, self.kernel_v, padding="same")

        # Local std dev approximation via Average pooling of luma and luma^2
        mean_luma_h = F.avg_pool2d(luma, (kh_h, kw_h), stride=1, padding=(kh_h // 2, kw_h // 2))
        sq_mean_luma_h = F.avg_pool2d(luma ** 2, (kh_h, kw_h), stride=1, padding=(kh_h // 2, kw_h // 2))
        std_luma_h = torch.sqrt(torch.clamp(sq_mean_luma_h - mean_luma_h ** 2, min=1e-5))

        mean_luma_v = F.avg_pool2d(luma, (kh_v, kw_v), stride=1, padding=(kh_v // 2, kw_v // 2))
        sq_mean_luma_v = F.avg_pool2d(luma ** 2, (kh_v, kw_v), stride=1, padding=(kh_v // 2, kw_v // 2))
        std_luma_v = torch.sqrt(torch.clamp(sq_mean_luma_v - mean_luma_v ** 2, min=1e-5))

        # Normalized correlation responses
        norm_resp_h = resp_h / (std_luma_h * (kh_h * kw_h) + 1e-4)
        norm_resp_v = resp_v / (std_luma_v * (kh_v * kw_v) + 1e-4)

        # Max response across horizontal and vertical orientations
        corr_max = torch.maximum(norm_resp_h, norm_resp_v)

        # Bounded between [0.0, 1.0]
        corr_saliency = torch.clamp(corr_max, min=0.0, max=1.0)
        return corr_saliency


# ---------------------------------------------------------------------------
# 2. Matched Filter 4-Channel Input Block (Wraps Layer 0)
# ---------------------------------------------------------------------------
class MatchedFilterInputBlock(nn.Module):
    """
    Wraps YOLO26s Layer 0.
    Takes 3-channel RGB image x [B, 3, H, W],
    computes matched filter saliency map R_max [B, 1, H, W],
    concatenates into 4-channel input [B, 4, H, W],
    and projects through modified stem convolution.
    """
    __module__ = "matched_filter_yolo26"

    def __init__(self, orig_conv):
        super().__init__()
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "Conv")

        self.matched_filter = MatchedFilterModule(kernel_size=(15, 31))

        # Build 4-channel stem convolution with identical hyper-parameters
        out_channels = orig_conv.conv.out_channels
        kernel_size = orig_conv.conv.kernel_size
        stride = orig_conv.conv.stride
        padding = orig_conv.conv.padding
        bias = orig_conv.conv.bias is not None

        self.conv = nn.Conv2d(4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = orig_conv.bn
        self.act = orig_conv.act

        # Initialize weights:
        # Channels 0, 1, 2 copy 100% of the COCO pretrained RGB filters
        with torch.no_grad():
            self.conv.weight[:, :3, :, :].copy_(orig_conv.conv.weight)
            # Channel 3 (Matched Filter correlation channel):
            # Soft positive initialization to prime network without shocking early gradients
            nn.init.normal_(self.conv.weight[:, 3:4, :, :], mean=0.01, std=0.005)
            if bias and orig_conv.conv.bias is not None:
                self.conv.bias.copy_(orig_conv.conv.bias)

    def forward(self, x):
        # 1. Compute correlation saliency map
        corr_saliency = self.matched_filter(x)
        # 2. Concatenate as 4th channel
        x_4ch = torch.cat([x, corr_saliency], dim=1)
        # 3. Pass through stem
        return self.act(self.bn(self.conv(x_4ch)))


# Register safe globals for PyTorch weights_only unpickling
try:
    torch.serialization.add_safe_globals([
        MatchedFilterInputBlock,
        MatchedFilterModule,
    ])
except Exception:
    pass


# ---------------------------------------------------------------------------
# 3. Custom Trainer
# ---------------------------------------------------------------------------
class MatchedFilterDetectionTrainer(DetectionTrainer):
    """
    Subclasses Ultralytics DetectionTrainer to inject MatchedFilterInputBlock
    into Layer 0 after model architecture is initialized.
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        layer0 = model.model[0]
        if not isinstance(layer0, MatchedFilterInputBlock):
            model.model[0] = MatchedFilterInputBlock(layer0)
            if verbose:
                print("✨ Injected MatchedFilterInputBlock into Layer 0 (4-channel stem)")
        return model


# ---------------------------------------------------------------------------
# 4. Evaluation & Results Logging
# ---------------------------------------------------------------------------
def evaluate_and_save(weights_path, args):
    print(f"\nEvaluating clean checkpoint: {weights_path} at conf={args.eval_conf} on '{args.eval_split}' split")
    clean = YOLO(str(weights_path))

    metrics = clean.val(
        data=args.data,
        split=args.eval_split,
        conf=args.eval_conf,
        iou=args.eval_iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
    )

    speed = metrics.speed
    total_time_ms = (
        speed.get("preprocess", 0.0)
        + speed.get("inference", 0.0)
        + speed.get("postprocess", 0.0)
    )
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0

    ap50_list = metrics.box.ap50.tolist() if hasattr(metrics.box, "ap50") else []
    maps_list = metrics.box.maps.tolist() if hasattr(metrics.box, "maps") else []
    per_class_ap50 = {
        name: float(ap50_list[i]) for i, name in enumerate(CLASS_NAMES) if i < len(ap50_list)
    }
    per_class_ap50_95 = {
        name: float(maps_list[i]) for i, name in enumerate(CLASS_NAMES) if i < len(maps_list)
    }

    summary = {
        "model": args.run_key,
        "weights": str(weights_path),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "total_time_ms": float(total_time_ms),
        "fps": float(fps),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap50_95,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "architecture": "Matched-Filter YOLO26s (4-Channel NCC Saliency Stem)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n--- Overall Metrics ---")
    for k, v in summary.items():
        if k != "per_class_ap50":
            print(f"{k}: {v}")
    print("\n--- Per-Class AP@0.5 ---")
    for name, ap in per_class_ap50.items():
        print(f"{name}: {ap:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{args.run_key}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results JSON to: {result_path}")


# ---------------------------------------------------------------------------
# 5. Main CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-key", type=str, default="rectified_yolov26s_matched_filter_640")
    p.add_argument("--weights", type=str, default="yolo26s.pt")
    p.add_argument("--data", type=str, default="datasets/pcb-unified-4class/data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.5)
    p.add_argument("--eval-split", type=str, default="test")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--results-dir", type=str, default=None)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent
    if args.results_dir:
        args.results_dir = Path(args.results_dir)
    elif Path("/mnt/weka/etadevosyan/pcb-yolo/results").exists():
        args.results_dir = Path("/mnt/weka/etadevosyan/pcb-yolo/results")
    else:
        args.results_dir = project_root / "results"

    run_dir = project_root / "runs" / args.run_key / "pcb-filtered"
    weights_path = run_dir / "weights" / "best.pt"

    if not args.skip_train:
        overrides = dict(
            model=args.weights,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer="SGD",
            project=str(project_root / "runs" / args.run_key),
            name="pcb-filtered",
            exist_ok=True,
            val=True,
        )
        trainer = MatchedFilterDetectionTrainer(overrides=overrides)
        trainer.train()
        print("Training finished. Run saved to:", trainer.save_dir)
        weights_path = trainer.save_dir / "weights" / "best.pt"

    evaluate_and_save(weights_path, args)


if __name__ == "__main__":
    main()
