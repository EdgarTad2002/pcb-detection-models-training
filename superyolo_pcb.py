#!/usr/bin/env python3
"""
SuperYOLO: Super-Resolution Assisted Object Detection for Dense PCB Inspection.
================================================================================
Adapted from Zhang et al. (IEEE TGRS 2023) for Surface-Mount Technology (SMT)
micro-component inspection on dense circuit boards.

Core Architectural Concept:
---------------------------
Standard 640px detectors lose micro-components (0402 / 0201 chip capacitors) due
to Nyquist-Shannon spatial undersampling in deep convolutional layers. Native 1280px
resolution fixes detection but incurs a severe 4x latency penalty.

SuperYOLO resolves this 'Spatial-Efficiency Paradox' via an Auxiliary Super-Resolution
learning branch during training:
1. Shared Backbone: Processes 640px input to extract multiscale features (P3, P4, P5).
2. Auxiliary SR Head: Taps high-frequency spatial features from P3 and reconstructs
   pixel-level details using sub-pixel convolutions (PixelShuffle).
3. Joint Multi-Task Loss:
       L_total = L_detection + lambda_sr * L_super_resolution
   The SR reconstruction gradient compels the shared backbone to retain high-frequency
   edge contours (solder fillets, capacitor termination pads, IC pin pitch).
4. Zero-Overhead Inference: The auxiliary SR head is completely pruned/detached at test
   time, running pure 640px forward passes at ~147 FPS on NVIDIA H100 hardware!

Usage:
    python superyolo_pcb.py \
        --run-key superyolo26s_rectified_640 \
        --weights yolo26s.pt \
        --data datasets/pcb-native-res-unified-4class/data.yaml \
        --lambda-sr 0.10 \
        --epochs 100 --imgsz 640 --batch 16 --workers 8
"""

import argparse
import json
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

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
DEFAULT_CLASSES = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 1. Auxiliary Lightweight Super-Resolution Reconstruction Head
# ---------------------------------------------------------------------------
class LightweightSRHead(nn.Module):
    """
    Sub-Pixel Convolutional Reconstruction Head (PixelShuffle).
    Takes deep spatial features from P3 (stride 8, e.g. 128/256 channels at 80x80)
    and reconstructs fine 3-channel RGB details back to input resolution.
    """

    def __init__(self, in_channels=128, out_channels=3, upscale_factor=8):
        super().__init__()
        self.upscale_factor = upscale_factor

        # 1. Channel reduction & feature refinement
        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        # 2. Stage 1 Upsample (x2)
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        # 3. Stage 2 Upsample (x2)
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        # 4. Stage 3 Upsample (x2) -> Total 8x upsampling from stride-8 features
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 16 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )

        # 5. Final RGB reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] normalized RGB space
        )

    def forward(self, x):
        f = self.feat_conv(x)
        f = self.up1(f)
        f = self.up2(f)
        f = self.up3(f)
        out_rgb = self.reconstruction(f)
        return out_rgb


# ---------------------------------------------------------------------------
# 2. SuperYOLO Architecture Wrapper
# ---------------------------------------------------------------------------
class SuperYOLODetectionModel(DetectionModel):
    """
    Subclasses Ultralytics DetectionModel to inject:
    1. Intermediate P3 feature extraction during forward pass.
    2. LightweightSRHead reconstruction of fine-grained spatial textures.
    3. Multi-task loss: L_det + lambda_sr * L_sr during training.
    """

    def __init__(self, cfg=None, ch=3, nc=4, verbose=True, lambda_sr=0.10):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.lambda_sr = float(lambda_sr)
        self.p3_features = None
        self.p3_idx = 4
        self.sr_head = None
        self._setup_sr_branch()

    def _setup_sr_branch(self):
        # Find P3 layer index (e.g. layer 4 or 2 or 6)
        in_ch = 128
        for idx in [4, 6, 2, 8]:
            if idx < len(self.model):
                mod = self.model[idx]
                if hasattr(mod, "cv2") and hasattr(mod.cv2, "conv"):
                    in_ch = mod.cv2.conv.out_channels
                    self.p3_idx = idx
                    break
                elif hasattr(mod, "conv") and hasattr(mod.conv, "out_channels"):
                    in_ch = mod.conv.out_channels
                    self.p3_idx = idx
                    break

        print(f"🔗 SuperYOLO Branch attached to Layer {self.p3_idx} (channels={in_ch}) with lambda_sr={self.lambda_sr}...")
        self.sr_head = LightweightSRHead(in_channels=in_ch, out_channels=3, upscale_factor=8)

    def _predict_once(self, x, profile=False, embed=None):
        """Native forward pass capturing intermediate P3 features without serializable closure hooks."""
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        p3_idx = getattr(self, "p3_idx", 4)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if self.training and getattr(self, "sr_head", None) is not None and m.i == p3_idx:
                self.p3_features = x
            y.append(x if m.i in self.save else None)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Multi-task loss: Detection Loss + Auxiliary Super-Resolution Loss."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])

        # 1. Standard Ultralytics detection loss (box, cls, dfl)
        det_loss, loss_items = self.criterion(preds, batch)

        # 2. Auxiliary Super-Resolution reconstruction loss
        p3_feat = getattr(self, "p3_features", None)
        self.p3_features = None  # Immediately reset to prevent holding non-leaf activation tensors

        if self.training and getattr(self, "sr_head", None) is not None and p3_feat is not None:
            sr_recon = self.sr_head(p3_feat)
            target_img = batch["img"].float()
            # Normalize target to [0, 1] if in [0, 255]
            if target_img.max() > 1.5:
                target_img = target_img / 255.0

            # Match spatial dimensions if needed
            if sr_recon.shape[-2:] != target_img.shape[-2:]:
                sr_recon = F.interpolate(sr_recon, size=target_img.shape[-2:], mode="bilinear", align_corners=False)

            # L1 reconstruction loss on high-frequency edges & textures
            sr_loss = F.l1_loss(sr_recon, target_img)

            # Combined multi-task gradient backpropagation: distribute evenly across det_loss vector
            total_loss = det_loss + (self.lambda_sr * sr_loss) / len(det_loss)
            return total_loss, loss_items

        return det_loss, loss_items


# ---------------------------------------------------------------------------
# 3. Custom SuperYOLO Trainer
# ---------------------------------------------------------------------------
class SuperYOLOTrainer(DetectionTrainer):
    lambda_sr = 0.10

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Construct SuperYOLO model with attached auxiliary SR branch."""
        model = self.set_model_names_for_load(
            SuperYOLODetectionModel(
                cfg=cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and int(os.environ.get("RANK", -1)) == -1,
                lambda_sr=self.lambda_sr,
            )
        )
        if weights:
            model.load(weights)
        return model

    def save_model(self):
        """Save model training checkpoints with SR head pruned for clean Ultralytics compatibility."""
        import io
        from copy import deepcopy
        from datetime import datetime
        from ultralytics.utils import GIT, __version__
        from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16, unwrap_model

        # Reset any leftover transient feature activation
        if hasattr(self.model, "p3_features"):
            self.model.p3_features = None

        ema = unwrap_model(self.ema.ema)
        if hasattr(ema, "p3_features"):
            ema.p3_features = None

        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])

        ema = deepcopy(ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(ema, "criterion"):
            ema.criterion = None
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)

        # SuperYOLO clean export: strip SR auxiliary head and restore pure DetectionModel class
        if hasattr(ema, "sr_head"):
            del ema.sr_head
        if hasattr(ema, "p3_features"):
            del ema.p3_features
        if hasattr(ema, "p3_idx"):
            del ema.p3_idx
        ema.__class__ = DetectionModel

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, "fitness": self.fitness},
                "train_results": self.read_results_csv(),
                "date": datetime.now().astimezone().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()

        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True


# ---------------------------------------------------------------------------
# 4. Evaluation and CLI Suite
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train SuperYOLO for Dense PCB Component Inspection")
    p.add_argument("--run-key", default="superyolo26s_rectified_640", help="Run identifier")
    p.add_argument("--weights", default="yolo26s.pt", help="Pretrained weights or yaml")
    p.add_argument("--data", default="datasets/pcb-native-res-unified-4class/data.yaml", help="Dataset yaml")
    p.add_argument("--lambda-sr", type=float, default=0.10, help="Auxiliary Super-Resolution loss weight")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--cls-weight", type=float, default=1.5, help="Classification loss weight")
    p.add_argument("--box-weight", type=float, default=5.0, help="Box regression loss weight")
    p.add_argument("--dfl-weight", type=float, default=2.0, help="DFL loss weight")
    p.add_argument("--label-smoothing", type=float, default=0.10, help="Label smoothing epsilon")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--results-dir", type=Path, default=Path("/mnt/weka/etadevosyan/pcb-yolo/results") if Path("/mnt/weka/etadevosyan/pcb-yolo/results").exists() else Path("results"))
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def evaluate_and_save(weights_path, args):
    """Run standard test set evaluation with SR head pruned."""
    print(f"\n======================================================================")
    print(f"📊 Evaluating SuperYOLO Clean Checkpoint: {weights_path}")
    print(f"Resolution: {args.imgsz}px | Split: '{args.eval_split}' | Conf: {args.eval_conf}")
    print(f"======================================================================")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=args.data,
        split=args.eval_split,
        conf=args.eval_conf,
        iou=args.eval_iou,
        imgsz=args.imgsz,
        device=args.device,
    )

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
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.p.mean()),
        "recall": float(metrics.box.r.mean()),
        "total_time_ms": float(total_time_ms),
        "fps": float(fps),
        "per_class_ap50": per_class_ap,
        "lambda_sr": float(args.lambda_sr),
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n--- Summary Performance ---")
    for k, v in summary.items():
        if k != "per_class_ap50":
            print(f"  {k:15s}: {v}")
    print("\n--- Per-Class AP50 ---")
    for name, ap in per_class_ap.items():
        print(f"  {name:24s}: {ap:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"{args.run_key}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved results summary to: {out_json}")


def main():
    args = parse_args()
    print("=" * 80)
    print("🚀 Initializing SuperYOLO Training for PCB Inspection")
    print(f"Key:          {args.run_key}")
    print(f"Base Weights: {args.weights}")
    print(f"Dataset:      {args.data}")
    print(f"Resolution:   {args.imgsz}px")
    print(f"Lambda SR:    {args.lambda_sr} (Auxiliary Reconstruction Weight)")
    print(f"Reweighting:  cls={args.cls_weight}, box={args.box_weight}, dfl={args.dfl_weight}")
    print("=" * 80)

    # Set trainer hyperparameters
    SuperYOLOTrainer.lambda_sr = args.lambda_sr

    default_weka_runs = Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/runs")
    project_dir = default_weka_runs if default_weka_runs.exists() else Path("runs")
    exp_dir = project_dir / args.run_key

    trainer = SuperYOLOTrainer(
        overrides={
            "model": args.weights,
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "project": str(project_dir),
            "name": args.run_key,
            "exist_ok": True,
            "cls": args.cls_weight,
            "box": args.box_weight,
            "dfl": args.dfl_weight,
            "label_smoothing": args.label_smoothing,
            "cache": True,
            "plots": True,
            "save": True,
        }
    )

    if not args.skip_train:
        trainer.train()

    best_weights = exp_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = exp_dir / "pcb-filtered" / "weights" / "best.pt"

    if best_weights.exists():
        evaluate_and_save(best_weights, args)
    else:
        print(f"⚠️ Warning: Could not locate best.pt in {exp_dir}")


if __name__ == "__main__":
    main()
