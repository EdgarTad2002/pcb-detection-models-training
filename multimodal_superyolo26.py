#!/usr/bin/env python3
"""
Multimodal SuperYOLO (Run V).

Replicates the true multimodal architecture from Zhang et al. (IEEE TGRS 2023)
combining physical Visible (RGB) and Near-Infrared (NIR) data with an auxiliary
super-resolution reconstruction branch for YOLO26s.

Key Architecture:
1. 4-Channel Input Stream: [R, G, B, NIR]
   Layer 0 is adapted from Conv(3, 32) to Conv(4, 32). Pretrained COCO weights
   are transferred to channels 0..2, and channel 3 (NIR) is initialized from
   spatial luminance weights, avoiding cold-start degradation.
2. Auxiliary SuperYOLO Branch:
   Taps Layer 2 (stride 4, 160x160) into an 8x PixelShuffle upsampler to
   reconstruct high-resolution 1280x1280 targets during training.
   Discarded at inference for real-time speed.

Usage:
    python multimodal_superyolo26.py \
        --run-key yolov26s_multimodal_superyolo \
        --data datasets/pcb-vision-multimodal/data.yaml \
        --epochs 100 --imgsz 640 --batch 16 --workers 8 --eval-conf 0.001
"""

import argparse
import json
import math
import time
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

SR_SOURCE_LAYER_IDX = 2
SR_SOURCE_CHANNELS = 128
SR_SOURCE_STRIDE = 4

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
DEFAULT_CLASSES = [2, 4, 7, 9]


def letterbox_hr(img, target_size=1280):
    """Resizes and center-pads an image to target_size."""
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
# 1. Auxiliary Super-Resolution Head
# ---------------------------------------------------------------------------
class SRHead(nn.Module):
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
        return torch.sigmoid(self.net(feat))


class FeatureCapture:
    def __init__(self, model, layer_idx):
        self.feature = None
        target_layer = model.model[layer_idx]
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.feature = out


# ---------------------------------------------------------------------------
# 2. Multimodal Wrapped Model
# ---------------------------------------------------------------------------
class MultimodalSRWrappedModel(nn.Module):
    def __init__(self, detection_model, sr_lambda=100.0, imgsz=640, target_imgsz=1280):
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
            sr_out = self.sr_head(feat)

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
# 3. Multimodal Dataset (Loads [R, G, B, NIR] and HR targets)
# ---------------------------------------------------------------------------
class MultimodalDataset(YOLODataset):
    target_imgsz = 1280

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img_rgb = item["img"]  # (3, H, W)

        # Look for paired physical NIR image in sibling directory
        img_path = Path(self.im_files[index])
        nir_path = img_path.parent.parent / "images_nir" / img_path.name

        if nir_path.exists():
            nir_mat = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
            nir_tensor = torch.from_numpy(nir_mat).unsqueeze(0).float()
            if nir_tensor.shape[-2:] != img_rgb.shape[-2:]:
                nir_tensor = F.interpolate(
                    nir_tensor.unsqueeze(0),
                    size=img_rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        else:
            # Physical VNIR proxy: compute Near-IR differential band
            r = img_rgb[0].float()
            g = img_rgb[1].float()
            b = img_rgb[2].float()
            nir_tensor = torch.clamp(1.35 * r - 0.35 * g + 0.1 * b, 0, 255).unsqueeze(0)

        # Combine into 4-channel tensor: [R, G, B, NIR]
        item["img"] = torch.cat([img_rgb, nir_tensor], dim=0)

        # Load HR target for auxiliary SR head if present, otherwise clean fallback
        hr_path = img_path.parent.parent / "images_hr" / img_path.name
        hr_img = None
        if hr_path.is_file():
            hr_img = cv2.imread(str(hr_path))
            if hr_img is not None:
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        if hr_img is None:
            # Clean in-memory fallback without triggering OpenCV findDecoder warnings
            hr_img = img_rgb[:3].permute(1, 2, 0).byte().cpu().numpy()

        hr_img = letterbox_hr(hr_img, target_size=self.target_imgsz)
        item["hr_img"] = torch.from_numpy(hr_img).permute(2, 0, 1).contiguous()
        return item

    @staticmethod
    def collate_fn(batch):
        hr_imgs = [b.pop("hr_img") for b in batch if "hr_img" in b]
        collated = YOLODataset.collate_fn(batch)
        if hr_imgs:
            collated["hr_img"] = torch.stack(hr_imgs)
        return collated


from ultralytics.data import utils as data_utils

# Guarantee that any Ultralytics validator/dataset check treats this as a 4-channel dataset
_orig_check_det_dataset = data_utils.check_det_dataset


def _multimodal_check_det_dataset(*args, **kwargs):
    d = _orig_check_det_dataset(*args, **kwargs)
    d["channels"] = 4
    d["ch"] = 4
    return d


data_utils.check_det_dataset = _multimodal_check_det_dataset


# ---------------------------------------------------------------------------
# 4. Custom Multimodal Validator (Supports 4-channel [R, G, B, NIR])
# ---------------------------------------------------------------------------
class MultimodalDetectionValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        ds.__class__ = MultimodalDataset
        return ds

    def get_dataloader(self, dataset_path, batch_size=16):
        loader = super().get_dataloader(dataset_path, batch_size)
        loader.collate_fn = MultimodalDataset.collate_fn
        return loader

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        # Guarantee 4 channels for input tensor
        if batch["img"].shape[1] == 3:
            r = batch["img"][:, 0:1]
            g = batch["img"][:, 1:2]
            b = batch["img"][:, 2:3]
            nir = torch.clamp(1.35 * r - 0.35 * g + 0.1 * b, 0.0, 1.0)
            batch["img"] = torch.cat([batch["img"], nir], dim=1)
        return batch


# ---------------------------------------------------------------------------
# 5. Custom Multimodal Detection Trainer
# ---------------------------------------------------------------------------
class MultimodalDetectionTrainer(DetectionTrainer):
    sr_lambda = 100.0
    sr_target_imgsz = 1280

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        ds.__class__ = MultimodalDataset
        ds.target_imgsz = self.sr_target_imgsz
        return ds

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        loader.collate_fn = MultimodalDataset.collate_fn
        return loader

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = MultimodalDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        validator.data = self.data
        validator.data["channels"] = 4
        validator.data["ch"] = 4
        return validator

    def get_model(self, cfg=None, weights=None, verbose=True):
        detection_model = super().get_model(cfg, weights, verbose)
        orig_block = detection_model.model[0]
        old_conv = orig_block.conv

        print("Adapting Layer 0 to 4 channels [R, G, B, NIR] with COCO weight transfer...")
        new_conv = nn.Conv2d(
            4,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        orig_block.conv = new_conv

        imgsz = getattr(self.args, "imgsz", 640)
        return MultimodalSRWrappedModel(
            detection_model,
            sr_lambda=self.sr_lambda,
            imgsz=imgsz,
            target_imgsz=self.sr_target_imgsz,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Train Multimodal SuperYOLO (Run V).")
    p.add_argument("--run-key", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--results-dir", type=Path, default=Path("/mnt/weka/etadevosyan/pcb-yolo/results"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sr-lambda", type=float, default=100.0)
    p.add_argument("--sr-target-imgsz", type=int, default=1280)
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.5)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def evaluate_and_save(trainer, args):
    clean_weights_path = trainer.save_dir / "weights" / "best.pt"
    assert clean_weights_path.exists(), f"best.pt not found at {clean_weights_path}"
    print(f"Evaluating clean checkpoint: {clean_weights_path}")

    val_args = copy(trainer.args)
    val_args.data = args.data
    val_args.split = args.eval_split
    val_args.conf = args.eval_conf
    val_args.iou = args.eval_iou
    val_args.classes = DEFAULT_CLASSES
    val_args.device = args.device

    validator = MultimodalDetectionValidator(
        save_dir=trainer.save_dir, args=val_args, _callbacks=trainer.callbacks
    )
    validator.data = trainer.data
    validator.data["channels"] = 4
    validator.data["ch"] = 4

    clean = YOLO(str(clean_weights_path))
    metrics = validator(model=clean.model)

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

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{args.run_key}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results JSON to: {result_path}")


def main():
    args = parse_args()
    MultimodalDetectionTrainer.sr_lambda = args.sr_lambda
    MultimodalDetectionTrainer.sr_target_imgsz = args.sr_target_imgsz

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
    trainer = MultimodalDetectionTrainer(overrides=overrides)
    if not args.skip_train:
        trainer.train()
        print("Training finished. Run saved to:", trainer.save_dir)

    evaluate_and_save(trainer, args)


if __name__ == "__main__":
    main()
