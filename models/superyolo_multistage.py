#!/usr/bin/env python3
"""
True SuperYOLO with Group Multi-Stage Squared-Dice Loss for PCB Inspection.
===========================================================================
Integrates:
1. SuperYOLO auxiliary high-resolution super-resolution head (IEEE TGRS 2023).
2. Group Multi-Stage Loss (TransFuse / EGE-UNet) with:
   - Binary Cross-Entropy (BCE)
   - Squared-Dice Loss (SDice)
   over multi-scale backbone feature maps [P1, P2, P3, P4] (strides 4, 8, 16, 32).
3. The quadratic penalty in Squared-Dice strongly suppresses edge bleeding for
   tiny 0402/0201 ceramic chip capacitors without degrading inference FPS.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from tools.group_multistage_loss import GroupMultiStageLoss

# ── Canonical module registration ──────────────────────────────────────────
if __name__ == "__main__":
    sys.modules["superyolo_multistage"] = sys.modules[__name__]

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


# ---------------------------------------------------------------------------
# 1. Auxiliary Super-Resolution Head
# ---------------------------------------------------------------------------
class SRHead(nn.Module):
    """Fuses P2 + P3 and upsamples to reconstruct high-resolution target."""
    _STAGE_CHANNELS = [64, 32, 16, 8, 4]

    def __init__(self, p2_ch: int = 64, p3_ch: int = 128, out_ch: int = 3,
                 edge_weight: float = 0.5, n_upsample: int = 3):
        super().__init__()
        self.edge_weight = edge_weight
        self.n_upsample = n_upsample

        self.p3_lateral = nn.Sequential(
            nn.Conv2d(p3_ch, p2_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(p2_ch),
            nn.SiLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(p2_ch * 2, p2_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(p2_ch),
            nn.PReLU(),
        )

        ups = []
        in_c = p2_ch
        for i in range(n_upsample):
            out_c = self._STAGE_CHANNELS[i] if i < len(self._STAGE_CHANNELS) else 4
            ups.append(nn.Sequential(
                nn.Conv2d(in_c, out_c * 4, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(out_c),
                nn.PReLU(),
            ))
            in_c = out_c
        self.ups = nn.ModuleList(ups)
        self.recon = nn.Sequential(
            nn.Conv2d(in_c, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_x", sobel_x.expand(3, 1, 3, 3).clone())
        self.register_buffer(
            "sobel_y", sobel_x.transpose(-2, -1).expand(3, 1, 3, 3).clone()
        )

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(pad, self.sobel_x, groups=3)
        gy = F.conv2d(pad, self.sobel_y, groups=3)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, p2_feat: torch.Tensor, p3_feat: torch.Tensor) -> torch.Tensor:
        p3_up = F.interpolate(
            self.p3_lateral(p3_feat), size=p2_feat.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        f = self.fuse_conv(torch.cat([p2_feat, p3_up], dim=1))
        for up in self.ups:
            f = up(f)
        return self.recon(f)

    def compute_sr_loss(self, recon_hr: torch.Tensor, target_hr: torch.Tensor) -> torch.Tensor:
        if recon_hr.shape[-2:] != target_hr.shape[-2:]:
            recon_hr = F.interpolate(recon_hr, size=target_hr.shape[-2:], mode="bilinear", align_corners=False)
        l1 = F.l1_loss(recon_hr, target_hr)
        if self.edge_weight > 0:
            edge = F.l1_loss(self._sobel_edges(recon_hr), self._sobel_edges(target_hr))
            return l1 + self.edge_weight * edge
        return l1


# ---------------------------------------------------------------------------
# 2. Multi-Stage Auxiliary Mask Head
# ---------------------------------------------------------------------------
class MultiStageMaskHead(nn.Module):
    """Lightweight 1x1 convs projecting backbone stages into component binary masks."""
    def __init__(self, channels_list: List[int]):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            for ch in channels_list
        ])

    def forward(self, stage_features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(feat) for head, feat in zip(self.heads, stage_features)]


# ---------------------------------------------------------------------------
# 3. SuperYOLO with Multi-Stage Squared-Dice Loss Model
# ---------------------------------------------------------------------------
class SuperYOLOMultiStageModel(DetectionModel):
    """DetectionModel with Super-Resolution Head AND Group Multi-Stage SDice Heads."""
    __module__ = "superyolo_multistage"

    def __init__(self, cfg=None, ch=3, nc=4, verbose=True, *,
                 lambda_sr: float = 0.10, sr_scale: int = 2,
                 edge_weight: float = 0.5, lambda_group: float = 0.15,
                 use_squared_dice: bool = True, imgsz: int = 1280):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.lambda_sr = float(lambda_sr)
        self.sr_scale = int(sr_scale)
        self.edge_weight = float(edge_weight)
        self.lambda_group = float(lambda_group)
        self.use_squared_dice = bool(use_squared_dice)
        self.imgsz = int(imgsz)

        self.p2_features = None
        self.p3_features = None
        self.stage_features = []
        self.img_hr = None

        self.p2_idx = 2
        self.p3_idx = 4
        self.stage_indices = [2, 4, 6, 8]  # default candidate indices for P2, P3, P4, P5

        self.sr_head = None
        self.mask_head = None
        self.group_criterion = GroupMultiStageLoss(use_squared_dice=self.use_squared_dice)
        self._setup_branches()

    def _probe_layer_channels(self, candidates):
        for idx in candidates:
            if idx < len(self.model):
                mod = self.model[idx]
                if hasattr(mod, "cv2") and hasattr(mod.cv2, "conv"):
                    return idx, mod.cv2.conv.out_channels
                if hasattr(mod, "conv") and hasattr(mod.conv, "out_channels"):
                    return idx, mod.conv.out_channels
        return candidates[0], 64

    def _setup_branches(self):
        # 1. Setup SR Branch
        self.p2_idx, p2_ch = self._probe_layer_channels([2, 1, 3])
        self.p3_idx, p3_ch = self._probe_layer_channels([4, 3, 5])
        lr_size = self.imgsz // max(self.sr_scale, 1)
        p2_size = lr_size // 4
        upscale = max(1, self.imgsz // max(p2_size, 1))
        n_up = max(1, int(round(math.log2(upscale))))

        self.sr_head = SRHead(p2_ch=p2_ch, p3_ch=p3_ch, out_ch=3,
                              edge_weight=self.edge_weight, n_upsample=n_up)

        # 2. Setup 4-Stage Mask Heads for Group Multi-Stage Loss
        p4_idx, p4_ch = self._probe_layer_channels([6, 5, 7])
        p5_idx, p5_ch = self._probe_layer_channels([8, 7, 9])
        self.stage_indices = [self.p2_idx, self.p3_idx, p4_idx, p5_idx]
        stage_channels = [p2_ch, p3_ch, p4_ch, p5_ch]
        self.mask_head = MultiStageMaskHead(stage_channels)

    def _rasterize_gt_mask(self, batch: dict, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Rasterizes bounding boxes into a full-resolution binary component mask [B, 1, H, W]."""
        bs, _, h, w = batch["img"].shape
        mask = torch.zeros((bs, 1, h, w), device=device, dtype=torch.float32)

        if "bboxes" in batch and len(batch["bboxes"]) > 0:
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"].long()
            for b_idx in range(bs):
                box_subset = bboxes[batch_idx == b_idx]
                for box in box_subset:
                    # box is [x_center, y_center, width, height] normalized in [0, 1]
                    xc, yc, bw, bh = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                    x1 = max(0, int((xc - bw / 2.0) * w))
                    y1 = max(0, int((yc - bh / 2.0) * h))
                    x2 = min(w, int((xc + bw / 2.0) * w))
                    y2 = min(h, int((yc + bh / 2.0) * h))
                    if x2 > x1 and y2 > y1:
                        mask[b_idx, 0, y1:y2, x1:x2] = 1.0
        return mask

    def _predict_once(self, x, profile=False, embed=None):
        if self.training and hasattr(self, "sr_scale") and self.sr_scale > 1:
            self.img_hr = x
            h, w = x.shape[-2:]
            x = F.interpolate(x, size=(h // self.sr_scale, w // self.sr_scale),
                              mode="bilinear", align_corners=False)

        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed else {-1}
        max_idx = max(embed_set)
        captured_stages = {}

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if self.training:
                if m.i == self.p2_idx:
                    self.p2_features = x
                if m.i == self.p3_idx:
                    self.p3_features = x
                if m.i in self.stage_indices:
                    captured_stages[m.i] = x

            y.append(x if m.i in self.save else None)
            if m.i in embed_set:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)

        if self.training and len(captured_stages) == 4:
            self.stage_features = [captured_stages[idx] for idx in self.stage_indices]
        return x

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])

        det_loss, loss_items = self.criterion(preds, batch)
        bs = batch["img"].shape[0]

        # ── 1. Auxiliary Super-Resolution Loss ──
        sr_loss_val = torch.tensor(0.0, device=det_loss.device)
        if self.training and self.sr_head is not None and self.lambda_sr > 0 and self.p2_features is not None:
            recon_hr = self.sr_head(self.p2_features, self.p3_features)
            target_hr = self.img_hr.float() / 255.0 if self.img_hr.max() > 1.5 else self.img_hr.float()
            sr_loss_val = self.sr_head.compute_sr_loss(recon_hr, target_hr)

        # ── 2. Group Multi-Stage BCE + Squared-Dice Loss ──
        group_loss_val = torch.tensor(0.0, device=det_loss.device)
        if self.training and self.mask_head is not None and self.lambda_group > 0 and len(self.stage_features) == 4:
            stage_preds = self.mask_head(self.stage_features)
            target_mask = self._rasterize_gt_mask(batch, shape=batch["img"].shape[-2:], device=det_loss.device)
            group_loss_val, _ = self.group_criterion(stage_preds, target_mask)

        # Clear references
        self.p2_features = None
        self.p3_features = None
        self.stage_features = []
        self.img_hr = None

        # Combine losses scaled by batch size
        total_aux = (self.lambda_sr * sr_loss_val + self.lambda_group * group_loss_val) * bs
        total_loss = det_loss + total_aux / len(det_loss)

        loss_items["sr_loss"] = sr_loss_val.detach()
        loss_items["group_sdice_loss"] = group_loss_val.detach()
        return total_loss, loss_items


# ---------------------------------------------------------------------------
# 4. Trainer with Strip-Checkpoint Logic
# ---------------------------------------------------------------------------
class SuperYOLOMultiStageTrainer(DetectionTrainer):
    lambda_sr: float = 0.10
    sr_scale: int = 2
    edge_weight: float = 0.5
    lambda_group: float = 0.15
    use_squared_dice: bool = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        imgsz = getattr(self.args, "imgsz", 1280)
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        model = SuperYOLOMultiStageModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and int(os.environ.get("RANK", -1)) == -1,
            lambda_sr=self.lambda_sr,
            sr_scale=self.sr_scale,
            edge_weight=self.edge_weight,
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

        for model_ref in [ema, self.model]:
            for attr in ("p2_features", "p3_features", "stage_features", "img_hr"):
                if hasattr(model_ref, attr):
                    setattr(model_ref, attr, None)

        orig_class = ema.__class__
        backup = {}
        strip_attrs = [
            "sr_head", "mask_head", "group_criterion", "p2_idx", "p3_idx",
            "stage_indices", "lambda_sr", "sr_scale", "edge_weight",
            "lambda_group", "use_squared_dice", "imgsz",
        ]
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
    p = argparse.ArgumentParser(description="True SuperYOLO with Group Multi-Stage SDice Loss")
    p.add_argument("--run-key", default="superyolo26s_grand_unified_1280")
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--data", default="datasets/pcb-grand-unified-1280/data.yaml")
    p.add_argument("--lambda-sr", type=float, default=0.10)
    p.add_argument("--lambda-group", type=float, default=0.15)
    p.add_argument("--use-squared-dice", action="store_true", default=True)
    p.add_argument("--sr-scale", type=int, default=2)
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

    # Pass class attributes to Trainer
    SuperYOLOMultiStageTrainer.lambda_sr = args.lambda_sr
    SuperYOLOMultiStageTrainer.lambda_group = args.lambda_group
    SuperYOLOMultiStageTrainer.use_squared_dice = args.use_squared_dice
    SuperYOLOMultiStageTrainer.sr_scale = args.sr_scale

    project_dir = Path("runs") / args.run_key
    trainer = SuperYOLOMultiStageTrainer(overrides={
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

    # Post-train evaluation
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
        print(f"Grand Unified Evaluation mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
