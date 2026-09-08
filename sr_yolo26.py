#!/usr/bin/env python3
"""
SuperYOLO-style auxiliary super-resolution training for YOLO26 (Corrected v3).

Reimplements the core idea from icey-zhang/SuperYOLO (Zhang et al., TGRS 2023)
adapted for modern Ultralytics and YOLO26:

Key Improvements in v3:
  1. Multi-Scale Feature Fusion: The SR head now captures BOTH Layer 2 (stride 4,
     128ch spatial detail) and Layer 4 (stride 8, 256ch semantic detail). Both
     low-level edge layers and mid-level detection layers receive SR gradient supervision.
  2. Smooth L1 Loss (Huber Loss): Replaces raw unscaled L1 with Smooth L1 (beta=0.01)
     to avoid huge gradient spikes from outlier pixels.
  3. Stable Loss Scaling: Default sr-lambda set to 0.5 (calibrated auxiliary regularizer),
     fixing the severe gradient instability caused by previous sr-lambda=100.0.
  4. Spatial Alignment: Enforces exact 1:1 pixel coordinate matching with letterbox
     transformation and disables desynchronizing spatial augmentations.
  5. Loss Reweight Compatibility: Supports --cls, --box, and --dfl overrides so
     SuperYOLO can synergize with class loss reweighting.
  6. Zero-Cost Inference: The auxiliary SR head is completely detached at test time.

Usage: see sbatch/train_run_r_superyolo.sh or sbatch/train_run_s_superyolo_native.sh
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

LAYER_P2_IDX = 2       # Backbone Layer 2: stride 4, 128 channels (fine spatial detail)
LAYER_P3_IDX = 4       # Backbone Layer 4: stride 8, 256 channels (semantic context)
P2_CHANNELS = 128
P3_CHANNELS = 256


def letterbox_hr(img, target_size=640):
    """Resizes and center-pads an image to (target_size, target_size) matching
    Ultralytics' standard LetterBox transformation without aspect ratio distortion."""
    h, w = img.shape[:2]
    r = min(target_size / h, target_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = target_size - nw
    pad_h = target_size - nh
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


# ---------------------------------------------------------------------------
# 1. Multi-Scale SR Decoder Head (Fuses P2 spatial detail + P3 semantic context)
# ---------------------------------------------------------------------------
class MultiScaleSRHead(nn.Module):
    """Reconstructs a high-resolution 3-channel image by fusing low-level spatial
    features (stride 4) and mid-level semantic features (stride 8)."""

    def __init__(self, in_ch_p2=128, in_ch_p3=256, upsample_factor=4):
        super().__init__()
        # Align P3 (stride 8) to P2 (stride 4)
        self.p3_to_p2 = nn.Sequential(
            nn.Conv2d(in_ch_p3, in_ch_p2, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Multi-scale fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch_p2 * 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Progressive PixelShuffle upsamplers
        # upsample_factor=4: 160 -> 320 -> 640 (2 stages)
        # upsample_factor=8: 160 -> 320 -> 640 -> 1280 (3 stages)
        n_upsamples = max(1, int(round(math.log2(upsample_factor))))
        layers = []
        ch = 128
        for _ in range(n_upsamples):
            out_ch = max(ch // 2, 32)
            layers += [
                nn.Conv2d(ch, out_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, 3, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat_p2, feat_p3):
        p3_aligned = self.p3_to_p2(feat_p3)
        if p3_aligned.shape[-2:] != feat_p2.shape[-2:]:
            p3_aligned = F.interpolate(
                p3_aligned, size=feat_p2.shape[-2:], mode="bilinear", align_corners=False
            )
        fused = self.fuse(torch.cat([feat_p2, p3_aligned], dim=1))
        return torch.sigmoid(self.net(fused))


# ---------------------------------------------------------------------------
# 2. Multi-Layer Feature Capture Hook (Picklable for PyTorch Model Checkpointing)
# ---------------------------------------------------------------------------
class LayerHook:
    def __init__(self, storage, key):
        self.storage = storage
        self.key = key

    def __call__(self, module, inp, out):
        self.storage[self.key] = out


class MultiLayerCapture:
    def __init__(self, model, layer_indices):
        self.features = {}
        for idx in layer_indices:
            target_layer = model.model[idx]
            target_layer.register_forward_hook(LayerHook(self.features, idx))


# ---------------------------------------------------------------------------
# 3. Wrapped Model with Auxiliary SR Loss Regularization
# ---------------------------------------------------------------------------
class SRWrappedModel(nn.Module):
    def __init__(self, detection_model, sr_lambda=0.5, imgsz=640, target_imgsz=640):
        super().__init__()
        self.detection_model = detection_model

        # Auto-detect P2 and P3 channel dimensions for any model (YOLOv5s: 64/128, YOLO26s: 128/256)
        dev = next(detection_model.parameters()).device
        with torch.no_grad():
            cur = torch.zeros(1, 3, 64, 64, device=dev)
            ch_p2, ch_p3 = 128, 256
            for i in range(LAYER_P3_IDX + 1):
                cur = detection_model.model[i](cur)
                if i == LAYER_P2_IDX:
                    ch_p2 = cur.shape[1]
                elif i == LAYER_P3_IDX:
                    ch_p3 = cur.shape[1]

        upsample_factor = max(1, round(target_imgsz / (imgsz / 4)))  # Stride 4 footprint
        self.sr_head = MultiScaleSRHead(
            in_ch_p2=ch_p2, in_ch_p3=ch_p3, upsample_factor=upsample_factor
        )
        self.capture = MultiLayerCapture(detection_model, [LAYER_P2_IDX, LAYER_P3_IDX])
        self.sr_lambda = sr_lambda

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("args", "nc", "names", "hyp"):
            try:
                object.__setattr__(self.detection_model, name, value)
            except AttributeError:
                pass

    def forward(self, batch, *args, **kwargs):
        if self.training and isinstance(batch, dict) and "hr_img" in batch:
            det_loss, loss_items = self.detection_model(batch, *args, **kwargs)

            feat_p2 = self.capture.features.get(LAYER_P2_IDX)
            feat_p3 = self.capture.features.get(LAYER_P3_IDX)

            if feat_p2 is not None and feat_p3 is not None:
                sr_out = self.sr_head(feat_p2, feat_p3)

                hr_target = batch["hr_img"].to(sr_out.device).float() / 255.0
                if hr_target.shape[-2:] != sr_out.shape[-2:]:
                    hr_target = F.interpolate(
                        hr_target,
                        size=sr_out.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Smooth L1 loss provides well-behaved, bounded gradients
                sr_loss = F.smooth_l1_loss(sr_out, hr_target, beta=0.01)
                total_loss = det_loss + self.sr_lambda * sr_loss
                return total_loss, loss_items
            return det_loss, loss_items
        else:
            return self.detection_model(batch, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.detection_model, name)


# ---------------------------------------------------------------------------
# 4. Dataset with Pixel-Aligned HR Reconstruction Target
# ---------------------------------------------------------------------------
class SRYOLODataset(YOLODataset):
    target_imgsz = 640

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img_path = Path(self.im_files[index])
        hr_path = img_path.parent.parent / "images_hr" / img_path.name
        hr_img = cv2.imread(str(hr_path))
        if hr_img is None:
            hr_img = cv2.cvtColor(item["img"].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = letterbox_hr(hr_img, target_size=self.target_imgsz)
        item["hr_img"] = torch.from_numpy(hr_img).permute(2, 0, 1).contiguous()
        return item

    @staticmethod
    def collate_fn(batch):
        hr_imgs = [b.pop("hr_img") for b in batch]
        collated = YOLODataset.collate_fn(batch)
        collated["hr_img"] = torch.stack(hr_imgs)
        return collated


# ---------------------------------------------------------------------------
# 5. Detection Trainer Wiring
# ---------------------------------------------------------------------------
class SRDetectionTrainer(DetectionTrainer):
    sr_lambda = 0.5
    sr_target_imgsz = 640

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            ds.__class__ = SRYOLODataset
            ds.target_imgsz = self.sr_target_imgsz
        return ds

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode == "train":
            loader.collate_fn = SRYOLODataset.collate_fn
        return loader

    def get_model(self, cfg=None, weights=None, verbose=True):
        detection_model = super().get_model(cfg, weights, verbose)
        imgsz = getattr(self.args, "imgsz", 640)
        return SRWrappedModel(
            detection_model,
            sr_lambda=self.sr_lambda,
            imgsz=imgsz,
            target_imgsz=self.sr_target_imgsz,
        )


def parse_args():
    p = argparse.ArgumentParser(description="SuperYOLO v3: Multi-Scale Auxiliary SR Training for YOLO26")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--run-key", required=True, help="Run key name")
    p.add_argument("--weights", default="yolo26s.pt", help="Base model weights")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sr-lambda", type=float, default=0.5, help="Auxiliary SR loss scale (default 0.5)")
    p.add_argument("--sr-target-imgsz", type=int, default=640, help="Target resolution for HR reconstruction")
    p.add_argument("--cls", type=float, default=None, help="Optional class loss gain override (e.g. 1.5)")
    p.add_argument("--box", type=float, default=None, help="Optional box loss gain override (e.g. 5.0)")
    p.add_argument("--dfl", type=float, default=None, help="Optional DFL loss gain override")
    p.add_argument("--eval-conf", type=float, default=0.001, help="Confidence threshold for evaluation")
    p.add_argument("--eval-iou", type=float, default=0.5, help="IoU threshold for evaluation")
    p.add_argument("--eval-split", default="test", help="Evaluation dataset split")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def evaluate_and_save(trainer, args):
    CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
    results_dir = Path("/mnt/weka/etadevosyan/pcb-yolo/results")

    clean_weights_path = trainer.save_dir / "weights" / "best.pt"
    assert clean_weights_path.exists(), f"best.pt not found at {clean_weights_path}"
    clean = YOLO(str(clean_weights_path))
    print(f"Loaded clean checkpoint from: {clean_weights_path}")

    metrics = clean.val(
        data=args.data,
        split=args.eval_split,
        classes=[2, 4, 7, 9],
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
        "weights": str(clean_weights_path),
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
        "sr_lambda": args.sr_lambda,
        "sr_target_imgsz": args.sr_target_imgsz,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n--- Overall Metrics ---")
    for k, v in summary.items():
        if k != "per_class_ap50":
            print(f"{k}: {v}")
    print("\n--- Per-Class AP@0.5 ---")
    for name, ap in per_class_ap.items():
        print(f"{name}: {ap:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{args.run_key}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results JSON to: {result_path}")


def main():
    args = parse_args()
    SRDetectionTrainer.sr_lambda = args.sr_lambda
    SRDetectionTrainer.sr_target_imgsz = args.sr_target_imgsz

    overrides = dict(
        model=args.weights,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        classes=[2, 4, 7, 9],
        optimizer="SGD",
        project=str(args.project_root / "runs" / args.run_key),
        name="pcb-filtered",
        exist_ok=True,
        val=True,
        # Preserve exact pixel alignment for SR pairs:
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
    )

    if args.cls is not None:
        overrides["cls"] = args.cls
    if args.box is not None:
        overrides["box"] = args.box
    if args.dfl is not None:
        overrides["dfl"] = args.dfl

    trainer = SRDetectionTrainer(overrides=overrides)
    trainer.train()
    print("Training finished. Run saved to:", trainer.save_dir)

    evaluate_and_save(trainer, args)


if __name__ == "__main__":
    main()
