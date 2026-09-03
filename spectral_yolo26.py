#!/usr/bin/env python3
"""
Spectral-Enhanced YOLO26 (Run T).

Incorporates physics-based 31-band pseudo-hyperspectral reconstruction
and a trainable 1x1 spectral adapter into YOLO26s.

Key Architecture:
1. SpectralReconstructor: Expands 3-channel RGB into a 31-band continuous
   spectral cube (400nm - 700nm in 10nm increments) capturing wavelength-specific
   material reflectance (copper traces, ceramic capacitor bodies, tin solder).
2. SpectralAdapter: A trainable sub-network that projects the 31 spectral bands
   into a 3-channel residual enhancement:
       adapted_RGB = raw_RGB + adapter(spectral_cube)
   Initialized with zero residual so it smoothly transitions from standard
   pre-trained COCO initialization.
3. SpectralInputBlock: Elegantly replaces Layer 0 of YOLO26s, preserving
   all downstream backbone and neck weights.

Usage:
    python spectral_yolo26.py \
        --run-key yolov26s_spectral_native \
        --data datasets/pcb-native-res/data.yaml \
        --epochs 100 --imgsz 1280 --batch 8 --workers 8 --eval-conf 0.001
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
DEFAULT_CLASSES = [2, 4, 7, 9]


# ---------------------------------------------------------------------------
# 1. Physics-Inspired 31-Band Spectral Reconstructor
# ---------------------------------------------------------------------------
class SpectralReconstructor(nn.Module):
    """
    Expands 3-channel RGB into 31 discrete spectral bands (400nm - 700nm)
    using physics-based spectral sensitivity basis initialization
    and spatial contextual refinement.
    """

    def __init__(self, num_bands=31):
        super().__init__()
        self.num_bands = num_bands

        # Spatial spectral mapping: captures local neighborhood material variations
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(3, 31, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(31),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(31, 31, kernel_size=1, bias=False),
            nn.BatchNorm2d(31),
            nn.Sigmoid(),
        )
        self._init_spectral_weights()

    def _init_spectral_weights(self):
        """Initializes the first conv layer with optical wavelength response curves:
        Bands 0-9 (400-490nm) respond primarily to Blue.
        Bands 10-18 (500-580nm) respond primarily to Green.
        Bands 19-30 (590-700nm) respond primarily to Red.
        """
        w = torch.zeros(31, 3, 3, 3)
        for band in range(31):
            if band < 10:  # Blue dominant
                w[band, 2, 1, 1] = 0.8
                w[band, 1, 1, 1] = 0.2 * (band / 10.0)
            elif band < 19:  # Green dominant
                w[band, 1, 1, 1] = 0.8
                w[band, 0, 1, 1] = 0.2 * ((band - 10) / 9.0)
            else:  # Red / Near-IR dominant
                w[band, 0, 1, 1] = 0.8
                w[band, 1, 1, 1] = 0.2 * (1.0 - (band - 19) / 12.0)
        self.spectral_conv[0].weight.data.copy_(w)

    def forward(self, x):
        return self.spectral_conv(x)


# ---------------------------------------------------------------------------
# 2. Spectral Input Block (Replaces Layer 0 seamlessly)
# ---------------------------------------------------------------------------
class SpectralInputBlock(nn.Module):
    """
    Wraps YOLO26s Layer 0. Takes incoming RGB, generates 31 spectral bands,
    adapts them via trainable 1x1 convolutions, and adds a residual enhancement
    before feeding into the original pretrained convolution.
    """

    def __init__(self, orig_conv):
        super().__init__()
        self.orig_conv = orig_conv
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "Conv")

        self.spectral_net = SpectralReconstructor(num_bands=31)
        self.adapter = nn.Sequential(
            nn.Conv2d(31, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
        )
        # Initialize final adapter layer to 0 so the model starts as an exact identity residual
        nn.init.zeros_(self.adapter[-1].weight)

    def forward(self, x):
        # x is (B, 3, H, W) normalized to [0, 1] in Ultralytics pipeline
        spectral_cube = self.spectral_net(x)
        spectral_residual = self.adapter(spectral_cube)
        enhanced_x = x + spectral_residual
        return self.orig_conv(enhanced_x)


# ---------------------------------------------------------------------------
# 3. Custom Trainer
# ---------------------------------------------------------------------------
class SpectralDetectionTrainer(DetectionTrainer):
    """
    Ultralytics DetectionTrainer that injects SpectralInputBlock into
    the DetectionModel at initialization.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        print("Injecting SpectralInputBlock (31-band HSI Reconstructor + 1x1 Adapter) into Layer 0...")
        model.model[0] = SpectralInputBlock(model.model[0])
        return model


# ---------------------------------------------------------------------------
# 4. CLI Arguments & Execution
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Spectral-Enhanced YOLO26s.")
    p.add_argument("--run-key", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--results-dir", type=Path, default=Path("/mnt/weka/etadevosyan/pcb-yolo/results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.5)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def evaluate_and_save(weights_path, args):
    print(f"\nEvaluating clean checkpoint: {weights_path} at conf={args.eval_conf} on '{args.eval_split}' split")
    clean = YOLO(str(weights_path))

    metrics = clean.val(
        data=args.data,
        split=args.eval_split,
        classes=DEFAULT_CLASSES,
        conf=args.eval_conf,
        iou=args.eval_iou,
        device=args.device,
    )

    speed = metrics.speed
    total_time_ms = (
        speed.get("preprocess", 0.0)
        + speed.get("inference", 0.0)
        + speed.get("postprocess", 0.0)
    )
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
    per_class_ap = {
        name: float(ap) for name, ap in zip(CLASS_NAMES, metrics.box.ap50)
    }

    summary = {
        "model": args.run_key,
        "weights": str(weights_path),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.p.mean()),
        "recall": float(metrics.box.r.mean()),
        "total_time_ms": float(total_time_ms),
        "fps": float(fps),
        "per_class_ap50": per_class_ap,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n--- Overall Metrics ---")
    for k, v in summary.items():
        if k != "per_class_ap50":
            print(f"{k}: {v}")
    print("\n--- Per-Class AP@0.5 ---")
    for name, ap in per_class_ap.items():
        print(f"{name}: {ap:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{args.run_key}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results JSON to: {result_path}")


def main():
    args = parse_args()
    run_dir = args.project_root / "runs" / args.run_key / "pcb-filtered"
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
            classes=DEFAULT_CLASSES,
            optimizer="SGD",
            project=str(args.project_root / "runs" / args.run_key),
            name="pcb-filtered",
            exist_ok=True,
            val=True,
        )
        trainer = SpectralDetectionTrainer(overrides=overrides)
        trainer.train()
        print("Training finished. Run saved to:", trainer.save_dir)
        weights_path = trainer.save_dir / "weights" / "best.pt"

    evaluate_and_save(weights_path, args)


if __name__ == "__main__":
    main()
