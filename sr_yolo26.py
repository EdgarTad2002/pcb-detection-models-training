#!/usr/bin/env python3
"""
SuperYOLO-style auxiliary super-resolution training for YOLO26.

Reimplements the core idea from icey-zhang/SuperYOLO (Zhang et al., TGRS 2023)
using Ultralytics' own extension points, since SuperYOLO's actual code is
built on the old standalone YOLOv5 repo and isn't compatible with the
modern `ultralytics` package YOLO26 lives in.

Core idea: an auxiliary decoder branch attached to an intermediate backbone
feature map learns to reconstruct a high-resolution version of the input
during training (SR loss, L1), on top of the normal detection loss. This
forces the backbone to preserve fine detail relevant to small objects.
At inference, the SR branch is never invoked -- ZERO added inference cost,
matching the paper's own real-time deployment framing.

Uses your native-res dataset as the natural HR target (downsampled for the
actual detector input, original used as the SR reconstruction target) --
see build_sr_dataset.py to prepare the paired data.

IMPORTANT: This wires into Ultralytics 8.4.x via three well-established,
stable extension points:
  1. A forward hook on an intermediate backbone layer (does not modify
     DetectionModel.forward() at all -- very version-stable)
  2. A custom YOLODataset subclass that also returns the HR target image
  3. A wrapped model whose forward() adds the SR loss to whatever the
     underlying DetectionModel already returns -- the trainer only ever
     sees (loss, loss_items), same contract regardless of what's inside

The one thing to verify once actually running: SR_SOURCE_LAYER_IDX below is
set for YOLO26s's architecture as printed in earlier training logs (layer 4,
C3k2 block, 256 channels, stride 8). If Ultralytics prints a different
layer numbering for your installed version, adjust it -- run:
    python -c "from ultralytics import YOLO; m = YOLO('yolo26s.pt'); print(m.model.model)"
and confirm layer 4 is still the 256-channel stride-8 C3k2 block.

Usage: see sbatch/train_run_r_superyolo.sh
"""

import argparse
import math
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

SR_SOURCE_LAYER_IDX = 2       # Early layer with spatial resolution (stride 4)
SR_SOURCE_CHANNELS = 128      # Standard channel count at stride 4 for YOLOv26s
SR_SOURCE_STRIDE = 4          # Stride 4 footprint


def letterbox_hr(img, target_size=1280):
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
# 1. SR decoder head -- upsamples from feature map to target HR resolution
# ---------------------------------------------------------------------------
class SRHead(nn.Module):
    """Reconstructs a 3-channel image from an intermediate feature map via
    conv + pixel-shuffle upsampling blocks up to the target HR resolution."""

    def __init__(self, in_channels, upsample_factor=8):
        super().__init__()
        n_upsamples = max(1, int(round(math.log2(upsample_factor))))
        layers = []
        ch = in_channels
        for _ in range(n_upsamples):
            out_ch = max(ch // 2, 16)
            layers += [
                nn.Conv2d(ch, out_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, 3, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        return torch.sigmoid(self.net(feat))  # output in [0,1], compare against normalized HR target


# ---------------------------------------------------------------------------
# 2. Feature capture via forward hook -- doesn't touch DetectionModel.forward()
# ---------------------------------------------------------------------------
class FeatureCapture:
    def __init__(self, model, layer_idx):
        self.feature = None
        target_layer = model.model[layer_idx]
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.feature = out


# ---------------------------------------------------------------------------
# 3. Wrapped model -- adds SR loss on top of whatever DetectionModel returns
# ---------------------------------------------------------------------------
class SRWrappedModel(nn.Module):
    """
    Wraps a standard Ultralytics DetectionModel. The trainer only ever calls
    self.model(batch) expecting (loss, loss_items) back during training --
    this wrapper preserves that contract exactly, just adding the SR loss
    term before returning.
    """

    def __init__(self, detection_model, sr_lambda=1.0, imgsz=640, target_imgsz=1280):
        super().__init__()
        self.detection_model = detection_model
        upsample_factor = max(1, round(target_imgsz / (imgsz / SR_SOURCE_STRIDE)))
        self.sr_head = SRHead(SR_SOURCE_CHANNELS, upsample_factor=upsample_factor)
        self.capture = FeatureCapture(detection_model, SR_SOURCE_LAYER_IDX)
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

            feat = self.capture.feature
            sr_out = self.sr_head(feat)  # (B, 3, H', W')

            hr_target = batch["hr_img"].to(sr_out.device).float() / 255.0
            if hr_target.shape[-2:] != sr_out.shape[-2:]:
                hr_target = F.interpolate(
                    hr_target,
                    size=sr_out.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            sr_loss = F.l1_loss(sr_out, hr_target)
            total_loss = det_loss + self.sr_lambda * sr_loss
            return total_loss, loss_items
        else:
            return self.detection_model(batch, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.detection_model, name)


# ---------------------------------------------------------------------------
# 4. Dataset that also returns the paired HR target image
# ---------------------------------------------------------------------------
class SRYOLODataset(YOLODataset):
    """
    Expects a sibling directory next to `images/` called `images_hr/` with
    identically-named files -- the high-resolution reconstruction target
    for each training image.
    """
    target_imgsz = 1280

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
# 5. Custom trainer wiring it together
# ---------------------------------------------------------------------------
class SRDetectionTrainer(DetectionTrainer):
    sr_lambda = 1.0
    sr_target_imgsz = 1280

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
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run-key", required=True)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sr-lambda", type=float, default=1.0)
    p.add_argument("--sr-target-imgsz", type=int, default=1280, help="Target resolution for HR reconstruction")
    p.add_argument("--eval-conf", type=float, default=0.001, help="Confidence threshold for post-training evaluation")
    p.add_argument("--eval-iou", type=float, default=0.5, help="IoU threshold for post-training evaluation")
    p.add_argument("--eval-split", default="test", help="Dataset split for post-training evaluation")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def evaluate_and_save(trainer, args):
    """
    Re-saves a clean checkpoint, evaluates it under the project's standard
    protocol, and writes the results JSON.
    """
    import json
    import time

    CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
    results_dir = Path("/mnt/weka/etadevosyan/pcb-yolo/results")

    clean_weights_path = trainer.save_dir / "weights" / "best.pt"
    assert clean_weights_path.exists(), f"best.pt not found at {clean_weights_path}"
    clean = YOLO(str(clean_weights_path))
    print(f"Loaded clean checkpoint from: {clean_weights_path}")

    # Evaluate under the specified protocol
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
        model="yolo26s.pt",
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
        # Enforce spatial 1-to-1 pixel alignment with the HR target:
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
    trainer = SRDetectionTrainer(overrides=overrides)
    trainer.train()
    print("Training finished. Run saved to:", trainer.save_dir)

    evaluate_and_save(trainer, args)


if __name__ == "__main__":
    main()

