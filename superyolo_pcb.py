#!/usr/bin/env python3
"""
True SuperYOLO (1280px Resolution-Preserving): Super-Resolution Assisted Object Detection
for Dense PCB Inspection (Experiment E2 Fix).
=========================================================================================
Adapted from Zhang et al. (IEEE TGRS 2023) for Surface-Mount Technology (SMT)
micro-component inspection on dense circuit boards.

Diagnostic Fix (Experiment E2)
------------------------------
Earlier iterations of `superyolo_pcb.py` silently downsampled 1280px inputs by
`sr_scale=2` down to 640px before the backbone and evaluated at `eval_imgsz=640`,
which violated the spatial Nyquist limit on 0402/0201 capacitors (<2x2 px at 640px)
and collapsed Capacitor AP from 35.7% down to 14.7%-22.9%.

This rectified implementation guarantees:
  1. Zero Silent Downsampling: At `--imgsz 1280`, the detection backbone processes
     the full 1280px tensor (producing 320x320 P2 and 160x160 P3 feature maps), and
     `--eval-imgsz` defaults to `1280` (not `640`).
  2. Multi-Scale Stride & Anchor Alignment: If explicit HR->LR pre-downsampling is
     requested (`--downsample-input`), the target LR dimensions are rounded to the
     nearest multiple of the maximum detector stride (32), preventing PANet concat
     and anchor grid shape mismatches under multi-scale training.
  3. High-Frequency SR Reconstruction: `SRHead` fuses P2 (stride 4) + P3 (stride 8)
     features and upsamples via PixelShuffle(2) stages to reconstruct the full-resolution
     (or `sr_scale * imgsz` super-resolved) target with L1 + Sobel edge loss.
  4. Standardized Dense Evaluation: Enforces `--max-det 1000` (Experiment E1) and
     cleanly strips `SRHead` at checkpoint serialization for zero-overhead inference.

Usage (1280px Native Backbone + Auxiliary SR Head):
    python superyolo_pcb.py \
        --run-key superyolo26s_rectified_1280 \
        --weights yolo26s.pt \
        --data datasets/pcb-native-res-unified-4class/data.yaml \
        --imgsz 1280 --eval-imgsz 1280 --sr-scale 2 \
        --lambda-sr 0.10 --edge-weight 0.5 \
        --max-det 1000 --epochs 100 --batch 8 --workers 8
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

# ── Canonical module registration for clean pickling ────────────────────────
if __name__ == "__main__":
    sys.modules["superyolo_pcb"] = sys.modules[__name__]

# ── Constants (matches train.py) ───────────────────────────────────────────
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
LEGACY_23_CLASSES = [2, 4, 7, 9]


# ---------------------------------------------------------------------------
# 1. Auxiliary Super-Resolution Reconstruction Head with P2+P3 Fusion
# ---------------------------------------------------------------------------
class SRHead(nn.Module):
    """
    Fuses P2 (stride 4) + P3 (stride 8) backbone features and reconstructs
    an HR image via n_upsample PixelShuffle(2) stages.

    Each PixelShuffle(2) stage doubles spatial resolution from P2 (stride 4):
      - n_upsample = 2 -> 4x upscaling from P2 (matches 1280px input from 320x320 P2)
      - n_upsample = 3 -> 8x upscaling from P2 (2x super-resolution from P2, or 1280px from 160x160 P2)

    SR loss = L1(recon, target) + edge_weight * Sobel_edge_loss(recon, target).
    """

    __module__ = "superyolo_pcb"

    _STAGE_CHANNELS = [64, 32, 16, 8, 4]

    def __init__(
        self,
        p2_ch: int = 64,
        p3_ch: int = 128,
        out_ch: int = 3,
        edge_weight: float = 0.5,
        n_upsample: int = 2,
    ):
        super().__init__()
        self.edge_weight = float(edge_weight)
        self.n_upsample = int(n_upsample)

        # Lateral projection: P3 -> P2 channel space
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

        # Dynamic PixelShuffle stages
        ups = []
        in_c = p2_ch
        for i in range(self.n_upsample):
            out_c = self._STAGE_CHANNELS[i] if i < len(self._STAGE_CHANNELS) else 4
            ups.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c * 4, kernel_size=3, padding=1, bias=False),
                    nn.PixelShuffle(2),
                    nn.BatchNorm2d(out_c),
                    nn.PReLU(),
                )
            )
            in_c = out_c
        self.ups = nn.ModuleList(ups)

        self.recon = nn.Sequential(
            nn.Conv2d(in_c, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(
                1, 1, 3, 3
            )
            / 4.0
        )
        self.register_buffer("sobel_x", sobel_x.expand(3, 1, 3, 3).clone())
        self.register_buffer(
            "sobel_y", sobel_x.transpose(-2, -1).expand(3, 1, 3, 3).clone()
        )

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude per channel in FP32 for numerical stability."""
        x32 = x.float()
        pad = F.pad(x32, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(pad, self.sobel_x.float(), groups=3)
        gy = F.conv2d(pad, self.sobel_y.float(), groups=3)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, p2_feat: torch.Tensor, p3_feat: torch.Tensor) -> torch.Tensor:
        """Fuse P2 + P3 features and reconstruct HR image."""
        p3_up = F.interpolate(
            self.p3_lateral(p3_feat),
            size=p2_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        f = self.fuse_conv(torch.cat([p2_feat, p3_up], dim=1))
        for up in self.ups:
            f = up(f)
        return self.recon(f)

    def compute_sr_loss(
        self, recon_hr: torch.Tensor, target_hr: torch.Tensor
    ) -> torch.Tensor:
        """L1 + edge-aware Sobel gradient loss with automatic spatial alignment."""
        if recon_hr.shape[-2:] != target_hr.shape[-2:]:
            target_hr = F.interpolate(
                target_hr.float(),
                size=recon_hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
        l1 = F.l1_loss(recon_hr.float(), target_hr.float())
        if self.edge_weight > 0:
            edge = F.l1_loss(
                self._sobel_edges(recon_hr), self._sobel_edges(target_hr)
            )
            return l1 + self.edge_weight * edge
        return l1


# ---------------------------------------------------------------------------
# 2. True SuperYOLO Model: Full-Resolution 1280px Detection + Auxiliary SR
# ---------------------------------------------------------------------------
class TrueSuperYOLOModel(DetectionModel):
    """
    Subclasses DetectionModel to inject:
    1. Resolution-preserving 1280px forward pass (or stride-aligned downsampling only
       if `downsample_input=True`).
    2. P2 + P3 feature capture for the SRHead.
    3. Multi-task loss with batch-scaled SR term.
    """

    __module__ = "superyolo_pcb"

    def __init__(
        self,
        cfg=None,
        ch=3,
        nc=4,
        verbose=True,
        *,
        lambda_sr: float = 0.10,
        sr_scale: int = 2,
        edge_weight: float = 0.5,
        imgsz: int = 1280,
        downsample_input: bool = False,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.lambda_sr = float(lambda_sr)
        self.sr_scale = int(sr_scale)
        self.edge_weight = float(edge_weight)
        self.imgsz = int(imgsz)
        self.downsample_input = bool(downsample_input)

        # Transient state (never serialized into best.pt)
        self.p2_features = None
        self.p3_features = None
        self.img_hr = None

        self.p2_idx = 2
        self.p3_idx = 4
        self.sr_head = None
        self._setup_sr_branch()

    def _probe_layer_channels(self, candidates):
        """Return (layer_index, out_channels) for the first matching layer."""
        for idx in candidates:
            if idx >= len(self.model):
                continue
            mod = self.model[idx]
            if hasattr(mod, "cv2") and hasattr(mod.cv2, "conv"):
                return idx, mod.cv2.conv.out_channels
            if hasattr(mod, "conv") and hasattr(mod.conv, "out_channels"):
                return idx, mod.conv.out_channels
        return candidates[0], 64

    def _setup_sr_branch(self):
        """Auto-detect P2/P3 layers, compute n_upsample, attach SRHead."""
        self.p2_idx, p2_ch = self._probe_layer_channels([2, 1])
        self.p3_idx, p3_ch = self._probe_layer_channels([4, 6, 2, 8])

        if self.downsample_input and self.sr_scale > 1:
            backbone_size = max((self.imgsz // self.sr_scale // 32) * 32, 32)
            p2_size = max(backbone_size // 4, 1)
            upscale = self.imgsz // p2_size
            n_upsample = max(int(round(math.log2(upscale))), 1)
            mode_desc = f"Downsample HR={self.imgsz}px -> Backbone={backbone_size}px"
        else:
            # Native 1280px mode: backbone runs at full imgsz (1280px), P2 is stride 4 (320px).
            # 2 PixelShuffle stages reconstruct 1280px; 3 stages super-resolve 2x (2560px) if sr_scale=2.
            n_upsample = 3 if self.sr_scale >= 2 else 2
            backbone_size = self.imgsz
            mode_desc = f"Native Backbone={backbone_size}px (Zero pre-downsampling)"

        print(
            f"  [SuperYOLO E2] P2 @ layer {self.p2_idx} ({p2_ch}ch), "
            f"P3 @ layer {self.p3_idx} ({p3_ch}ch) | {mode_desc}"
        )
        print(
            f"  [SuperYOLO E2] SR config: lambda_sr={self.lambda_sr}, "
            f"n_upsample={n_upsample} (x{2**n_upsample} from P2), edge_w={self.edge_weight}"
        )

        if self.lambda_sr > 0:
            self.sr_head = SRHead(
                p2_ch=p2_ch,
                p3_ch=p3_ch,
                out_ch=3,
                edge_weight=self.edge_weight,
                n_upsample=n_upsample,
            )

    @staticmethod
    def _align_stride(val: int, stride: int = 32) -> int:
        return max(stride, int(round(val / stride)) * stride)

    def _predict_once(self, x, profile=False, embed=None):
        """Forward pass preserving 1280px resolution (or stride-aligned downsampling)."""
        sr_active = self.training and getattr(self, "sr_head", None) is not None
        if sr_active:
            self.img_hr = x

        if (
            self.training
            and getattr(self, "downsample_input", False)
            and getattr(self, "sr_scale", 1) > 1
        ):
            self.img_hr = x
            h, w = x.shape[-2:]
            max_stride = int(max(self.stride)) if hasattr(self, "stride") else 32
            target_h = self._align_stride(h // self.sr_scale, max_stride)
            target_w = self._align_stride(w // self.sr_scale, max_stride)
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed else {-1}
        max_idx = max(embed_set)

        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            if sr_active:
                if m.i == self.p2_idx:
                    self.p2_features = x
                elif m.i == self.p3_idx:
                    self.p3_features = x

            y.append(x if m.i in self.save else None)
            if m.i in embed_set:
                embeddings.append(
                    F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                )
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Multi-task loss: Detection + Auxiliary Super-Resolution."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])

        det_loss, loss_items = self.criterion(preds, batch)

        p2_feat = self.p2_features
        p3_feat = self.p3_features
        img_hr = self.img_hr

        self.p2_features = None
        self.p3_features = None
        self.img_hr = None

        sr_can_compute = (
            self.training
            and getattr(self, "sr_head", None) is not None
            and self.lambda_sr > 0
            and p2_feat is not None
            and p3_feat is not None
            and img_hr is not None
        )

        if sr_can_compute:
            recon_hr = self.sr_head(p2_feat, p3_feat)
            target_hr = img_hr.float()
            if target_hr.max() > 1.5:
                target_hr = target_hr / 255.0

            sr_loss = self.sr_head.compute_sr_loss(recon_hr, target_hr)
            bs = batch["img"].shape[0]
            sr_loss_scaled = self.lambda_sr * sr_loss * bs
            total_loss = det_loss + sr_loss_scaled / len(det_loss)
            loss_items["sr_loss"] = sr_loss.detach()
            return total_loss, loss_items

        loss_items["sr_loss"] = torch.tensor(0.0, device=det_loss.device)
        return det_loss, loss_items


# ---------------------------------------------------------------------------
# 3. Custom Trainer with Clean Checkpoint Saving
# ---------------------------------------------------------------------------
class TrueSuperYOLOTrainer(DetectionTrainer):
    """
    Extends DetectionTrainer to construct TrueSuperYOLOModel and serialize
    clean DetectionModel checkpoints with SRHead stripped at inference time.
    """

    lambda_sr: float = 0.10
    sr_scale: int = 2
    edge_weight: float = 0.5
    downsample_input: bool = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        imgsz = getattr(self.args, "imgsz", 1280)
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        model = TrueSuperYOLOModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and int(os.environ.get("RANK", -1)) == -1,
            lambda_sr=self.lambda_sr,
            sr_scale=self.sr_scale,
            edge_weight=self.edge_weight,
            imgsz=imgsz,
            downsample_input=self.downsample_input,
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
            for attr in ("p2_features", "p3_features", "img_hr"):
                if hasattr(model_ref, attr):
                    setattr(model_ref, attr, None)

        orig_class = ema.__class__
        sr_backup = {}
        sr_attrs = [
            "sr_head",
            "p2_idx",
            "p3_idx",
            "lambda_sr",
            "sr_scale",
            "edge_weight",
            "imgsz",
            "downsample_input",
        ]
        for attr in sr_attrs:
            if hasattr(ema, attr):
                sr_backup[attr] = getattr(ema, attr)
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
            for attr, val in sr_backup.items():
                setattr(ema, attr, val)


# ---------------------------------------------------------------------------
# 4. CLI + Standardized 1280px Evaluation
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="True SuperYOLO (E2 1280px Fix): Super-Resolution Assisted Detection for Dense PCBs"
    )
    p.add_argument("--run-key", default="superyolo26s_rectified_1280")
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--data", default="datasets/pcb-native-res-unified-4class/data.yaml")
    p.add_argument("--classes", type=int, nargs="+", default=None)

    p.add_argument("--lambda-sr", type=float, default=0.10)
    p.add_argument("--sr-scale", type=int, default=2)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument(
        "--downsample-input",
        action="store_true",
        help="Explicitly downsample input by sr_scale before backbone (default False: keep full 1280px backbone).",
    )

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cls-weight", type=float, default=None)
    p.add_argument("--box-weight", type=float, default=None)
    p.add_argument("--dfl-weight", type=float, default=None)

    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--eval-split", default="test")
    p.add_argument(
        "--eval-imgsz",
        type=int,
        default=None,
        help="Evaluation resolution (E2 fix: defaults to full imgsz=1280 unless --downsample-input is set).",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="Max detections per image at validation/test time (E1 standard: 1000).",
    )

    default_weka = Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training")
    p.add_argument(
        "--project-root",
        type=Path,
        default=default_weka if default_weka.exists() else Path("."),
    )
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def resolve_eval_imgsz(args) -> int:
    """E2 Resolution Fix: Never silently downsample 1280px evaluation to 640px."""
    if args.eval_imgsz is not None:
        return int(args.eval_imgsz)
    if args.downsample_input and args.sr_scale > 1:
        return max(32, (args.imgsz // args.sr_scale // 32) * 32)
    return int(args.imgsz)


def evaluate_and_save(weights_path, args, effective_classes=None):
    """Evaluate the clean (SR-stripped) checkpoint at full 1280px detection resolution."""
    eval_imgsz = resolve_eval_imgsz(args)

    print(f"\n{'=' * 75}")
    print(f"📊 Evaluating SuperYOLO Checkpoint (E2 1280px Standardized): {weights_path}")
    print(
        f"   Eval Resolution: {eval_imgsz}px | max_det: {args.max_det} | "
        f"Split: '{args.eval_split}' | Conf: {args.eval_conf}"
    )
    print(f"{'=' * 75}")

    model = YOLO(str(weights_path))

    val_kwargs = dict(
        data=args.data,
        split=args.eval_split,
        conf=args.eval_conf,
        iou=args.eval_iou,
        imgsz=eval_imgsz,
        max_det=args.max_det,
        device=args.device,
    )
    if effective_classes is not None:
        val_kwargs["classes"] = effective_classes

    metrics = model.val(**val_kwargs)

    speed = metrics.speed
    total_time_ms = (
        speed.get("preprocess", 0.0)
        + speed.get("inference", 0.0)
        + speed.get("postprocess", 0.0)
    )
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0

    per_class_ap = {}
    for idx, name in enumerate(CLASS_NAMES):
        if idx < len(metrics.box.ap50):
            per_class_ap[name] = float(metrics.box.ap50[idx])

    summary = {
        "model": args.run_key,
        "weights": str(weights_path),
        "imgsz": eval_imgsz,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.p.mean()),
        "recall": float(metrics.box.r.mean()),
        "total_time_ms": float(total_time_ms),
        "fps": float(fps),
        "per_class_ap50": per_class_ap,
        "lambda_sr": float(args.lambda_sr),
        "sr_scale": args.sr_scale,
        "edge_weight": args.edge_weight,
        "downsample_input": bool(args.downsample_input),
        "train_imgsz": args.imgsz,
        "eval_imgsz": eval_imgsz,
        "max_det": args.max_det,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n--- Summary ---")
    for k, v in summary.items():
        if k != "per_class_ap50":
            print(f"  {k:20s}: {v}")
    print("\n--- Per-Class AP50 ---")
    for name, ap in per_class_ap.items():
        print(f"  {name:24s}: {ap:.4f}")

    results_dir = args.results_dir or (args.project_root / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.run_key}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to: {out_json}")


def main():
    args = parse_args()
    eval_imgsz = resolve_eval_imgsz(args)

    print("=" * 75)
    print("🚀 True SuperYOLO Training for PCB Inspection (E2 1280px Standardized)")
    print(f"  Run key:          {args.run_key}")
    print(f"  Backbone:         {args.weights}")
    print(f"  Dataset:          {args.data}")
    print(f"  Train imgsz:      {args.imgsz}px | Eval imgsz: {eval_imgsz}px")
    print(f"  Downsample input: {args.downsample_input} (sr_scale={args.sr_scale})")
    print(f"  lambda_sr:        {args.lambda_sr} | edge_weight: {args.edge_weight}")
    print(f"  max_det:          {args.max_det}")
    print("=" * 75)

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = args.project_root / data_yaml
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"

    import yaml

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg.get("nc", len(data_cfg.get("names", [])))

    if args.classes is not None:
        effective_classes = args.classes
    elif nc == 4:
        effective_classes = None
    else:
        effective_classes = LEGACY_23_CLASSES

    TrueSuperYOLOTrainer.lambda_sr = args.lambda_sr
    TrueSuperYOLOTrainer.sr_scale = args.sr_scale
    TrueSuperYOLOTrainer.edge_weight = args.edge_weight
    TrueSuperYOLOTrainer.downsample_input = args.downsample_input

    project_dir = args.project_root / "runs" / args.run_key
    overrides = {
        "model": args.weights,
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "max_det": args.max_det,
        "project": str(project_dir),
        "name": "pcb-filtered",
        "exist_ok": True,
        "cache": True,
        "plots": True,
        "save": True,
    }
    if args.cls_weight is not None:
        overrides["cls"] = args.cls_weight
    if args.box_weight is not None:
        overrides["box"] = args.box_weight
    if args.dfl_weight is not None:
        overrides["dfl"] = args.dfl_weight
    if effective_classes is not None:
        overrides["classes"] = effective_classes

    if not args.skip_train:
        trainer = TrueSuperYOLOTrainer(overrides=overrides)
        trainer.train()

    weights_path = project_dir / "pcb-filtered" / "weights" / "best.pt"
    if not weights_path.exists():
        alt = project_dir / "weights" / "best.pt"
        if alt.exists():
            weights_path = alt
        elif Path(args.weights).exists() and args.skip_train:
            weights_path = Path(args.weights)

    if weights_path.exists():
        evaluate_and_save(weights_path, args, effective_classes)
    else:
        print(f"⚠️  Could not locate best.pt at {weights_path}")


if __name__ == "__main__":
    main()
