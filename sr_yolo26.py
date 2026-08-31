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
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

SR_SOURCE_LAYER_IDX = 4       # see docstring -- verify against your installed version
SR_SOURCE_CHANNELS = 256      # channel count at that layer for YOLO26s
SR_SOURCE_STRIDE = 8          # spatial downsampling factor at that layer


# ---------------------------------------------------------------------------
# 1. SR decoder head -- a small upsampling stack, discarded at inference
# ---------------------------------------------------------------------------
class SRHead(nn.Module):
    """Reconstructs a 3-channel image from an intermediate feature map via
    a few conv + pixel-shuffle upsampling blocks, back up to roughly the
    original input resolution (undoing the feature map's stride)."""

    def __init__(self, in_channels, stride):
        super().__init__()
        n_upsamples = int(torch.log2(torch.tensor(float(stride))).item())  # e.g. stride 8 -> 3 upsample steps
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

    def __init__(self, detection_model, sr_lambda=1.0):
        super().__init__()
        self.detection_model = detection_model
        self.sr_head = SRHead(SR_SOURCE_CHANNELS, SR_SOURCE_STRIDE)
        self.capture = FeatureCapture(detection_model, SR_SOURCE_LAYER_IDX)
        self.sr_lambda = sr_lambda
        # NOTE: .args/.stride/.nc/.names are intentionally NOT copied here --
        # DetectionModel doesn't have .args set yet at this point in the
        # trainer's setup (the trainer attaches it later). __getattr__ below
        # forwards any of these to self.detection_model lazily, whenever
        # they're actually accessed, by which point they exist.

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Ultralytics' trainer sets .args/.nc/.names/.hyp on "self.model"
        # (which is this wrapper) -- but the actual loss computation runs
        # on self.detection_model, one level deeper, which never receives
        # these assignments unless we mirror them down explicitly.
        if name in ("args", "nc", "names", "hyp"):
            try:
                object.__setattr__(self.detection_model, name, value)
            except AttributeError:
                pass  # detection_model not registered yet (during __init__)

    def forward(self, batch, *args, **kwargs):
        if self.training and isinstance(batch, dict) and "hr_img" in batch:
            det_loss, loss_items = self.detection_model(batch, *args, **kwargs)

            feat = self.capture.feature
            sr_out = self.sr_head(feat)  # (B, 3, H', W')

            hr_target = batch["hr_img"].to(sr_out.device).float() / 255.0
            if hr_target.shape[-2:] != sr_out.shape[-2:]:
                hr_target = F.interpolate(hr_target, size=sr_out.shape[-2:], mode="bilinear", align_corners=False)

            sr_loss = F.l1_loss(sr_out, hr_target)
            total_loss = det_loss + self.sr_lambda * sr_loss
            return total_loss, loss_items
        else:
            # inference / validation: plain detection forward, SR branch never runs
            return self.detection_model(batch, *args, **kwargs)

    def __getattr__(self, name):
        # forward any attribute Ultralytics looks for (e.g. .criterion, .args)
        # to the underlying DetectionModel if not found on the wrapper itself
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
    for each training image. Built by build_sr_dataset.py.
    """

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img_path = Path(self.im_files[index])
        hr_path = img_path.parent.parent / "images_hr" / img_path.name
        hr_img = cv2.imread(str(hr_path))
        if hr_img is None:
            hr_img = cv2.cvtColor(item["img"].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)  # fallback: reuse LR img
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        item["hr_img"] = torch.from_numpy(hr_img).permute(2, 0, 1).contiguous()
        return item

    @staticmethod
    def collate_fn(batch):
        hr_imgs = [b.pop("hr_img") for b in batch]
        collated = YOLODataset.collate_fn(batch)
        # pad/resize HR images to a common size for stacking
        max_h = max(im.shape[1] for im in hr_imgs)
        max_w = max(im.shape[2] for im in hr_imgs)
        padded = []
        for im in hr_imgs:
            pad_h, pad_w = max_h - im.shape[1], max_w - im.shape[2]
            padded.append(F.pad(im, (0, pad_w, 0, pad_h)))
        collated["hr_img"] = torch.stack(padded)
        return collated


# ---------------------------------------------------------------------------
# 5. Custom trainer wiring it together
# ---------------------------------------------------------------------------
class SRDetectionTrainer(DetectionTrainer):
    sr_lambda = 1.0  # set via CLI, see main() below

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            ds.__class__ = SRYOLODataset
        return ds

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode == "train":
            loader.collate_fn = SRYOLODataset.collate_fn
        return loader

    def get_model(self, cfg=None, weights=None, verbose=True):
        detection_model = super().get_model(cfg, weights, verbose)
        return SRWrappedModel(detection_model, sr_lambda=self.sr_lambda)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run-key", required=True)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sr-lambda", type=float, default=1.0)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def evaluate_and_save(trainer, args):
    """
    Re-saves a CLEAN checkpoint (just the trained detection weights, no
    SRWrappedModel wrapper -- the raw saved best.pt may not load correctly
    through a plain YOLO(path) call, since it pickles the custom wrapper),
    then evaluates it under the project's standard protocol and writes the
    same JSON schema every other run uses, so it shows up in
    aggregate_results.py automatically.
    """
    import json
    import time

    import numpy as np

    CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
    results_dir = Path("/mnt/weka/etadevosyan/pcb-yolo/results")

    # 1. Load the clean unwrapped checkpoint that Ultralytics already saved
    #    during training (best.pt). It has the correct adapted architecture
    #    (nc=23, fine-tuned head) baked in -- no need to re-create a fresh
    #    COCO model and transfer weights (which fails due to class count mismatch).
    clean_weights_path = trainer.save_dir / "weights" / "best.pt"
    assert clean_weights_path.exists(), f"best.pt not found at {clean_weights_path}"
    clean = YOLO(str(clean_weights_path))
    print(f"Loaded clean checkpoint from: {clean_weights_path}")

    # 2. Evaluate it under the exact same protocol as every other run
    metrics = clean.val(
        data=args.data,
        split="test",
        classes=[2, 4, 7, 9],
        conf=0.25,
        iou=0.5,
        device=args.device,
    )

    speed = metrics.speed
    total_time_ms = speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0)
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
    per_class_ap = {name: float(ap) for name, ap in zip(CLASS_NAMES, metrics.box.ap50)}

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
        "eval_conf": 0.25,
        "eval_iou": 0.5,
        "eval_split": "test",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "sr_lambda": args.sr_lambda,
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
    )
    trainer = SRDetectionTrainer(overrides=overrides)
    trainer.train()
    print("Training finished. Run saved to:", trainer.save_dir)

    evaluate_and_save(trainer, args)


if __name__ == "__main__":
    main()
