#!/usr/bin/env python3
"""
Case 1: YOLO26s with In-Network CTDN Feature Stem & Group Multi-Stage SDice Loss.
================================================================================
Embeds the CTDN attention decomposition stem directly into YOLO26s.
Replaces layers 0 & 1 with a differentiable stem decomposing raw RGB into
Reflectance and AGC-Illumination in latent feature space at Stride 4.
Intermediate stages P1..P4 are supervised by the Group Multi-Stage BCE + SDice loss.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from models.ctdn_yolo_stem import CTDNInNetworkStem
from models.superyolo_multistage import MultiStageMaskHead
from tools.group_multistage_loss import GroupMultiStageLoss

if __name__ == "__main__":
    sys.modules["train_case1_ctdn_stem"] = sys.modules[__name__]


class YOLO26sCTDNStemModel(DetectionModel):
    """
    Detection model with In-Network CTDN Feature Stem.
    Replaces static stride-2 and stride-4 conv downsamplers with CTDN attention decomposition.
    """
    __module__ = "train_case1_ctdn_stem"

    def __init__(self, cfg=None, ch=3, nc=4, verbose=True, *,
                 lambda_group: float = 0.15, use_squared_dice: bool = True, imgsz: int = 1280):
        # Initialize attributes BEFORE super().__init__ because Ultralytics probes strides
        self.ctdn_stem = None
        self.mask_head = None
        self.stage_features = []
        self.stage_indices = [2, 4, 6, 8]
        self.lambda_group = float(lambda_group)
        self.use_squared_dice = bool(use_squared_dice)
        self.imgsz = int(imgsz)
        self.group_criterion = None

        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Build in-network CTDN stem (outputs 64 channels at stride 4)
        self.ctdn_stem = CTDNInNetworkStem(in_channels=ch, out_channels=64, internal_dim=32)

        # Setup 4-stage auxiliary mask heads for Group Multi-Stage Loss
        p2_ch = 64
        p3_ch = 128
        p4_ch = 256
        p5_ch = 512
        self.mask_head = MultiStageMaskHead([p2_ch, p3_ch, p4_ch, p5_ch])
        self.group_criterion = GroupMultiStageLoss(use_squared_dice=self.use_squared_dice)

    def _rasterize_gt_mask(self, batch: dict, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        bs, _, h, w = batch["img"].shape
        mask = torch.zeros((bs, 1, h, w), device=device, dtype=torch.float32)
        if "bboxes" in batch and len(batch["bboxes"]) > 0:
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"].long()
            for b_idx in range(bs):
                box_subset = bboxes[batch_idx == b_idx]
                for box in box_subset:
                    xc, yc, bw, bh = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                    x1 = max(0, int((xc - bw / 2.0) * w))
                    y1 = max(0, int((yc - bh / 2.0) * h))
                    x2 = min(w, int((xc + bw / 2.0) * w))
                    y2 = min(h, int((yc + bh / 2.0) * h))
                    if x2 > x1 and y2 > y1:
                        mask[b_idx, 0, y1:y2, x1:x2] = 1.0
        return mask

    def _predict_once(self, x, profile=False, embed=None):
        stem_active = getattr(self, "ctdn_stem", None) is not None
        mask_active = self.training and getattr(self, "mask_head", None) is not None

        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed else {-1}
        max_idx = max(embed_set)
        captured_stages = {}

        if stem_active:
            # Pass raw image through CTDN stem directly to stride 4 (64 channels)
            x_stem, _, _ = self.ctdn_stem(x)
            y.append(None)       # layer 0 skipped
            y.append(x_stem)     # layer 1 replaced with CTDN stem output
            start_layer = 2
            x = x_stem
        else:
            start_layer = 0

        for idx in range(start_layer, len(self.model)):
            m = self.model[idx]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if mask_active and m.i in self.stage_indices:
                captured_stages[m.i] = x

            y.append(x if m.i in self.save else None)
            if m.i in embed_set:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        if mask_active and len(captured_stages) == len(self.stage_indices):
            self.stage_features = [captured_stages[k] for k in self.stage_indices]
        return x

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])

        det_loss, loss_items = self.criterion(preds, batch)
        bs = batch["img"].shape[0]

        group_loss_val = torch.tensor(0.0, device=det_loss.device)
        if self.training and self.mask_head is not None and self.lambda_group > 0 and len(self.stage_features) == 4:
            stage_preds = self.mask_head(self.stage_features)
            target_mask = self._rasterize_gt_mask(batch, shape=batch["img"].shape[-2:], device=det_loss.device)
            group_loss_val, _ = self.group_criterion(stage_preds, target_mask)

        self.stage_features = []
        total_loss = det_loss + (self.lambda_group * group_loss_val * bs) / len(det_loss)
        loss_items["group_sdice_loss"] = group_loss_val.detach()
        return total_loss, loss_items


class Case1CTDNTrainer(DetectionTrainer):
    lambda_group: float = 0.15
    use_squared_dice: bool = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        imgsz = getattr(self.args, "imgsz", 1280)
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        model = YOLO26sCTDNStemModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and int(os.environ.get("RANK", -1)) == -1,
            lambda_group=self.lambda_group,
            use_squared_dice=self.use_squared_dice,
            imgsz=imgsz,
        )
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def save_model(self):
        from ultralytics.utils.torch_utils import unwrap_model
        ema = unwrap_model(self.ema.ema)
        for attr in ("stage_features",):
            if hasattr(ema, attr):
                setattr(ema, attr, [])

        orig_class = ema.__class__
        backup = {}
        strip_attrs = ["mask_head", "group_criterion", "stage_indices", "lambda_group", "use_squared_dice"]
        for attr in strip_attrs:
            if hasattr(ema, attr):
                backup[attr] = getattr(ema, attr)
                try:
                    delattr(ema, attr)
                except AttributeError:
                    pass

        orig_loss_names = getattr(ema, "loss_names", None)
        ema.__class__ = DetectionModel
        ema.loss_names = ["box_loss", "cls_loss", "dfl_loss"]

        try:
            super().save_model()
        finally:
            ema.__class__ = orig_class
            if orig_loss_names is not None:
                ema.loss_names = orig_loss_names
            for attr, val in backup.items():
                setattr(ema, attr, val)


def main():
    p = argparse.ArgumentParser(description="Case 1: YOLO26s with In-Network CTDN Feature Stem")
    p.add_argument("--run-key", default="yolov26s_case1_ctdn_stem_1280")
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--data", default="datasets/pcb-unified-4class-cleaned/data.yaml")
    p.add_argument("--lambda-group", type=float, default=0.15)
    p.add_argument("--use-squared-dice", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--max-det", type=int, default=1000)
    args = p.parse_args()

    Case1CTDNTrainer.lambda_group = args.lambda_group
    Case1CTDNTrainer.use_squared_dice = args.use_squared_dice

    project_dir = Path("runs") / args.run_key
    trainer = Case1CTDNTrainer(overrides={
        "model": args.weights,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(project_dir),
        "name": "pcb-filtered",
        "exist_ok": True,
        "save": True,
        "plots": True,
        "optimizer": "auto",
        "copy_paste": 0.30,
        "mosaic": 1.0,
        "mixup": 0.15,
    })
    trainer.train()

    best_pt = project_dir / "pcb-filtered" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n[EVAL] Running evaluation on '{args.eval_split}' split with max_det={args.max_det} at {args.imgsz}px...")
        eval_model = YOLO(str(best_pt))
        metrics = eval_model.val(
            data=args.data,
            split=args.eval_split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.eval_conf,
            iou=args.eval_iou,
            max_det=args.max_det,
            device=args.device,
        )
        print(f"Case 1 Evaluation mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
