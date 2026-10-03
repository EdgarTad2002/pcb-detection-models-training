#!/usr/bin/env python3
"""
True SuperYOLO: Super-Resolution Assisted Object Detection for Dense PCB Inspection.
=====================================================================================
Adapted from Zhang et al. (IEEE TGRS 2023) for Surface-Mount Technology (SMT)
micro-component inspection on dense circuit boards.

How it works
------------
The fundamental insight is that tiny components (0402/0201 chip caps, fine-pitch
QFP pins) vanish at 640px because their features fall below the spatial Nyquist
limit in deep layers. Native 1280px training restores detail but quadruples
latency. True SuperYOLO resolves this by training on 1280px HR images while
running detection at 640px:

  1.  Dataloader delivers images at ``--imgsz 1280`` (the HR target resolution).
  2.  In ``_predict_once``, the 1280px input is stored as the HR ground truth,
      then *bilinear-downsampled* to 640px (LR) before entering the backbone.
  3.  The backbone processes the 640px LR input at standard cost, producing
      P2 (stride 4 → 160×160) and P3 (stride 8 → 80×80) feature maps.
  4.  An auxiliary **SRHead** fuses P2 + P3 features and upsamples 8× via three
      PixelShuffle(2) stages to reconstruct the 1280px HR image.
  5.  Joint multi-task loss:
          L_total = L_det(640px predictions, labels)
                  + λ_sr · [L1(recon_HR, target_HR)
                           + edge_w · Sobel_edge_loss(recon_HR, target_HR)]
      × batch_size  (to match Ultralytics' det_loss scaling convention).
  6.  At checkpoint time, the SR head is stripped → pure DetectionModel that
      runs 640px inference at ~147 FPS with zero overhead.

The SR reconstruction gradient forces the shared backbone to *retain*
high-frequency edge contours (solder fillets, capacitor termination pads, IC
pin pitch) even at 640px, giving the detection heads richer feature maps.

Usage
-----
    python superyolo_pcb.py \\
        --run-key true_superyolo26s_rectified_640 \\
        --weights yolo26s.pt \\
        --data datasets/pcb-native-res-unified-4class/data.yaml \\
        --imgsz 1280 --sr-scale 2 \\
        --lambda-sr 0.10 --edge-weight 0.5 \\
        --epochs 100 --batch 8 --workers 8 \\
        --eval-conf 0.001

Ablation (λ_sr=0, same HR→LR pipeline but no SR gradient):
    python superyolo_pcb.py \\
        --run-key true_superyolo26s_ablation_no_sr \\
        --weights yolo26s.pt \\
        --data datasets/pcb-native-res-unified-4class/data.yaml \\
        --imgsz 1280 --sr-scale 2 \\
        --lambda-sr 0.0 \\
        --epochs 100 --batch 8 --workers 8
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

    Total upscaling from P2: 2^n_upsample.

    n_upsample is computed automatically from sr_scale and imgsz:
        lr_size   = imgsz // sr_scale        (backbone input size)
        p2_size   = lr_size // 4             (P2 stride-4 feature map)
        upscale   = imgsz // p2_size         (target_size / p2_size)
        n_upsample = log2(upscale)

    Examples:
        sr_scale=2, imgsz=1280 → lr=640, p2=160, upscale=8,  n=3 (×8  from P2)
        sr_scale=4, imgsz=1280 → lr=320, p2=80,  upscale=16, n=4 (×16 from P2)

    SR loss = L1(recon, target) + edge_weight · Sobel_edge_loss(recon, target).
    """

    __module__ = "superyolo_pcb"

    # Channel widths after each PixelShuffle stage (tapering down)
    _STAGE_CHANNELS = [64, 32, 16, 8, 4]

    def __init__(self, p2_ch: int = 64, p3_ch: int = 128, out_ch: int = 3,
                 edge_weight: float = 0.5, n_upsample: int = 3):
        super().__init__()
        self.edge_weight = edge_weight
        self.n_upsample = n_upsample

        # ── Lateral fusion: project P3 → P2 channel space, upsample, concat
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

        # ── Dynamic PixelShuffle stages
        # Each stage doubles spatial resolution: (H, W) → (2H, 2W)
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

        # ── Final RGB reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(in_c, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output in [0, 1]
        )

        # ── Sobel edge kernels (non-learnable, move with the model)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_x", sobel_x.expand(3, 1, 3, 3).clone())
        self.register_buffer(
            "sobel_y", sobel_x.transpose(-2, -1).expand(3, 1, 3, 3).clone()
        )

    # --------------------------------------------------------------------- #
    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude per channel."""
        pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(pad, self.sobel_x, groups=3)
        gy = F.conv2d(pad, self.sobel_y, groups=3)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, p2_feat: torch.Tensor, p3_feat: torch.Tensor) -> torch.Tensor:
        """Fuse P2 + P3 features and reconstruct HR image."""
        p3_up = F.interpolate(
            self.p3_lateral(p3_feat), size=p2_feat.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        f = self.fuse_conv(torch.cat([p2_feat, p3_up], dim=1))
        for up in self.ups:
            f = up(f)
        return self.recon(f)

    def compute_sr_loss(self, recon_hr: torch.Tensor,
                        target_hr: torch.Tensor) -> torch.Tensor:
        """L1 + edge-aware Sobel gradient loss."""
        # Ensure matching spatial dimensions
        if recon_hr.shape[-2:] != target_hr.shape[-2:]:
            recon_hr = F.interpolate(
                recon_hr, size=target_hr.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        l1 = F.l1_loss(recon_hr, target_hr)
        if self.edge_weight > 0:
            edge = F.l1_loss(self._sobel_edges(recon_hr),
                             self._sobel_edges(target_hr))
            return l1 + self.edge_weight * edge
        return l1


# ---------------------------------------------------------------------------
# 2. True SuperYOLO Model: Detection @ LR + Auxiliary SR @ HR
# ---------------------------------------------------------------------------
class TrueSuperYOLOModel(DetectionModel):
    """
    Subclasses DetectionModel to inject:
    1. Internal HR → LR downsampling (training only).
    2. P2 + P3 feature capture for the SRHead.
    3. Multi-task loss with properly scaled SR term.
    """

    __module__ = "superyolo_pcb"

    def __init__(self, cfg=None, ch=3, nc=4, verbose=True, *,
                 lambda_sr: float = 0.10, sr_scale: int = 2,
                 edge_weight: float = 0.5, imgsz: int = 1280):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.lambda_sr = float(lambda_sr)
        self.sr_scale = int(sr_scale)
        self.edge_weight = float(edge_weight)
        self.imgsz = int(imgsz)

        # Transient state (not saved in checkpoint)
        self.p2_features = None
        self.p3_features = None
        self.img_hr = None

        # Layer indices (auto-detected in _setup_sr_branch)
        self.p2_idx = 2
        self.p3_idx = 4

        self.sr_head = None
        self._setup_sr_branch()

    # --------------------------------------------------------------------- #
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
        return candidates[0], 64  # fallback

    def _setup_sr_branch(self):
        """Auto-detect P2/P3 layers, compute n_upsample, attach SRHead."""
        import math
        self.p2_idx, p2_ch = self._probe_layer_channels([2, 1])
        self.p3_idx, p3_ch = self._probe_layer_channels([4, 6, 2, 8])

        # Number of PixelShuffle(×2) stages needed to go from P2 → HR target
        #   lr_size   = imgsz // sr_scale   (what backbone actually sees)
        #   p2_size   = lr_size // 4        (stride-4 feature map)
        #   upscale   = imgsz // p2_size    (factor needed to reach HR)
        #   n_upsample = log2(upscale)
        lr_size = max(self.imgsz // self.sr_scale, 1)
        p2_size = max(lr_size // 4, 1)
        upscale = self.imgsz // p2_size
        n_upsample = max(int(round(math.log2(upscale))), 1)

        print(f"  True SuperYOLO: P2 @ layer {self.p2_idx} ({p2_ch}ch), "
              f"P3 @ layer {self.p3_idx} ({p3_ch}ch)")
        print(f"  SR config: λ_sr={self.lambda_sr}, ×{self.sr_scale} scale "
              f"(LR={lr_size}px→HR={self.imgsz}px), "
              f"n_upsample={n_upsample} (×{2**n_upsample} from P2), "
              f"edge_w={self.edge_weight}")

        if self.lambda_sr > 0:
            self.sr_head = SRHead(
                p2_ch=p2_ch, p3_ch=p3_ch, out_ch=3,
                edge_weight=self.edge_weight,
                n_upsample=n_upsample,
            )
        else:
            print("  SR head disabled (λ_sr=0): ablation mode, "
                  "still downsamples HR→LR internally")

    # --------------------------------------------------------------------- #
    def _predict_once(self, x, profile=False, embed=None):
        """Forward pass with HR storage + LR downsampling during training."""
        # ── During TRAINING: store HR (1280px), downsample to LR (640px) ──
        sr_active = self.training and getattr(self, "sr_head", None) is not None
        if self.training and hasattr(self, "sr_scale") and self.sr_scale > 1:
            # Always downsample during training (both when sr_head exists
            # and in ablation mode with lambda_sr=0) so the backbone only
            # ever sees LR images, keeping BN stats consistent.
            self.img_hr = x  # reference, not copy (no grad through target)
            h, w = x.shape[-2:]
            x = F.interpolate(
                x, size=(h // self.sr_scale, w // self.sr_scale),
                mode="bilinear", align_corners=False,
            )

        # ── Standard YOLO sequential forward ──
        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed else {-1}
        max_idx = max(embed_set)

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else \
                    [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)

            # Capture P2/P3 for SR branch
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

    # --------------------------------------------------------------------- #
    def loss(self, batch, preds=None):
        """Multi-task loss: Detection + Auxiliary Super-Resolution."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])

        # ── 1. Standard detection loss (box, cls, dfl) × batch_size ──
        det_loss, loss_items = self.criterion(preds, batch)

        # ── 2. Auxiliary SR loss ──
        p2_feat = self.p2_features
        p3_feat = self.p3_features
        img_hr = self.img_hr

        # Immediately release activation references
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

            # Normalise HR target to [0, 1]
            target_hr = img_hr.float()
            if target_hr.max() > 1.5:
                target_hr = target_hr / 255.0

            sr_loss = self.sr_head.compute_sr_loss(recon_hr, target_hr)

            # Scale SR loss by batch_size to match Ultralytics det_loss scaling
            bs = batch["img"].shape[0]
            sr_loss_scaled = self.lambda_sr * sr_loss * bs

            # Distribute evenly across the 3 det_loss components
            total_loss = det_loss + sr_loss_scaled / len(det_loss)
            loss_items["sr_loss"] = sr_loss.detach()
            return total_loss, loss_items

        # No SR loss (lambda_sr=0, not training, or no features captured)
        loss_items["sr_loss"] = torch.tensor(0.0, device=det_loss.device)
        return det_loss, loss_items


# ---------------------------------------------------------------------------
# 3. Custom Trainer with Clean Checkpoint Saving
# ---------------------------------------------------------------------------
class TrueSuperYOLOTrainer(DetectionTrainer):
    """
    Extends DetectionTrainer to:
    1. Construct a TrueSuperYOLOModel with SR branch.
    2. Save clean checkpoints by temporarily stripping all SR-specific state
       from the EMA model, calling the parent save_model(), then restoring.
    """

    # Set by CLI before instantiation
    lambda_sr: float = 0.10
    sr_scale: int = 2
    edge_weight: float = 0.5

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Construct TrueSuperYOLOModel with attached SR branch."""
        # self.args.imgsz is the HR resolution set in overrides
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
        )
        # set_model_names_for_load was added in later Ultralytics versions
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model

    def save_model(self):
        """Save a pure DetectionModel checkpoint with the SR head stripped.

        The parent ``save_model()`` deepcopies ``self.ema.ema`` internally,
        so we temporarily strip SR-specific attributes, delegate to the parent,
        and restore them afterward.  This is version-resilient — we never
        duplicate the serialisation logic.
        """
        from ultralytics.utils.torch_utils import unwrap_model

        ema = unwrap_model(self.ema.ema)

        # Clear transient activation tensors (prevent OOM and stale refs)
        for model_ref in [ema, self.model]:
            for attr in ("p2_features", "p3_features", "img_hr"):
                if hasattr(model_ref, attr):
                    setattr(model_ref, attr, None)

        # ── Backup and strip SR attributes ──
        orig_class = ema.__class__
        sr_backup = {}
        sr_attrs = [
            "sr_head", "p2_idx", "p3_idx",
            "lambda_sr", "sr_scale", "edge_weight", "imgsz",
        ]
        for attr in sr_attrs:
            if hasattr(ema, attr):
                sr_backup[attr] = getattr(ema, attr)
                try:
                    delattr(ema, attr)
                except AttributeError:
                    pass

        # Swap class + loss_names to standard DetectionModel
        orig_loss_names = getattr(ema, "loss_names", None)
        ema.__class__ = DetectionModel
        ema.loss_names = ["box_loss", "cls_loss", "dfl_loss"]

        try:
            super().save_model()
        finally:
            # ── Always restore, even if save fails ──
            ema.__class__ = orig_class
            if orig_loss_names is not None:
                ema.loss_names = orig_loss_names
            for attr, val in sr_backup.items():
                setattr(ema, attr, val)


# ---------------------------------------------------------------------------
# 4. CLI + Evaluation
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="True SuperYOLO: HR→LR Super-Resolution Assisted Detection "
                    "for Dense PCB Inspection",
    )
    # ── Core ──
    p.add_argument("--run-key", default="true_superyolo26s_rectified_640",
                    help="Run identifier (used for output dirs and results JSON)")
    p.add_argument("--weights", default="yolo26s.pt",
                    help="Pretrained backbone weights (.pt) or architecture (.yaml)")
    p.add_argument("--data",
                    default="datasets/pcb-native-res-unified-4class/data.yaml",
                    help="Dataset YAML")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                    help="Explicit class indices (overrides auto-detection)")

    # ── SuperYOLO-specific ──
    p.add_argument("--lambda-sr", type=float, default=0.10,
                    help="SR reconstruction loss weight (0 = ablation)")
    p.add_argument("--sr-scale", type=int, default=2,
                    help="HR/LR downsampling factor (imgsz → imgsz//sr_scale)")
    p.add_argument("--edge-weight", type=float, default=0.5,
                    help="Sobel edge loss weight inside SRHead")

    # ── Training ──
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280,
                    help="HR resolution loaded by the dataloader")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    # Optional loss gains — use Ultralytics defaults if not specified
    p.add_argument("--cls-weight", type=float, default=None,
                    help="Classification loss gain (default: Ultralytics default)")
    p.add_argument("--box-weight", type=float, default=None,
                    help="Box regression loss gain (default: Ultralytics default)")
    p.add_argument("--dfl-weight", type=float, default=None,
                    help="DFL loss gain (default: Ultralytics default)")

    # ── Evaluation ──
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--eval-imgsz", type=int, default=None,
                    help="Evaluation resolution (default: imgsz // sr_scale)")

    # ── Paths ──
    default_weka = Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training")
    p.add_argument(
        "--project-root", type=Path,
        default=default_weka if default_weka.exists() else Path("."),
        help="Root directory for runs/ and results/",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Results JSON directory (default: <project-root>/results)",
    )
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def evaluate_and_save(weights_path, args, effective_classes=None):
    """Evaluate the clean (SR-stripped) checkpoint at the detection resolution."""
    eval_imgsz = args.eval_imgsz or (args.imgsz // args.sr_scale)

    print(f"\n{'=' * 70}")
    print(f"📊 Evaluating True SuperYOLO Clean Checkpoint: {weights_path}")
    print(f"   Detection resolution: {eval_imgsz}px | Split: '{args.eval_split}' "
          f"| Conf: {args.eval_conf}")
    print(f"{'=' * 70}")

    model = YOLO(str(weights_path))

    val_kwargs = dict(
        data=args.data,
        split=args.eval_split,
        conf=args.eval_conf,
        iou=args.eval_iou,
        imgsz=eval_imgsz,
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
        "train_imgsz": args.imgsz,
        "eval_imgsz": eval_imgsz,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "batch": args.batch,
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


# ---------------------------------------------------------------------------
# 5. Main Entry Point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    det_imgsz = args.imgsz // args.sr_scale
    print("=" * 70)
    print("🚀 True SuperYOLO Training for PCB Inspection")
    print(f"  Run key:       {args.run_key}")
    print(f"  Backbone:      {args.weights}")
    print(f"  Dataset:       {args.data}")
    print(f"  HR resolution: {args.imgsz}px → LR detection: {det_imgsz}px "
          f"(×{args.sr_scale})")
    print(f"  λ_sr:          {args.lambda_sr} "
          f"{'(ablation: SR gradient disabled)' if args.lambda_sr == 0 else ''}")
    print(f"  Edge weight:   {args.edge_weight}")
    print("=" * 70)

    # ── Auto-detect native 4-class vs legacy 23-class ──
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
        effective_classes = None  # native 4-class
    else:
        effective_classes = LEGACY_23_CLASSES

    print(f"  Classes:       nc={nc}, effective={effective_classes or 'all'}")

    # ── Propagate config to trainer class ──
    TrueSuperYOLOTrainer.lambda_sr = args.lambda_sr
    TrueSuperYOLOTrainer.sr_scale = args.sr_scale
    TrueSuperYOLOTrainer.edge_weight = args.edge_weight

    # ── Build overrides ──
    project_dir = args.project_root / "runs" / args.run_key
    overrides = {
        "model": args.weights,
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,     # dataloader loads at HR resolution
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(project_dir),
        "name": "pcb-filtered",   # matches train.py convention
        "exist_ok": True,
        "cache": True,
        "plots": True,
        "save": True,
    }
    # Only override loss gains if explicitly requested
    if args.cls_weight is not None:
        overrides["cls"] = args.cls_weight
    if args.box_weight is not None:
        overrides["box"] = args.box_weight
    if args.dfl_weight is not None:
        overrides["dfl"] = args.dfl_weight
    # Class filtering for legacy 23-class datasets
    if effective_classes is not None:
        overrides["classes"] = effective_classes

    trainer = TrueSuperYOLOTrainer(overrides=overrides)

    if not args.skip_train:
        trainer.train()

    # ── Evaluate ──
    weights_path = project_dir / "pcb-filtered" / "weights" / "best.pt"
    if weights_path.exists():
        evaluate_and_save(weights_path, args, effective_classes)
    else:
        print(f"⚠️  Could not locate best.pt at {weights_path}")
        # Fallback: check for old-style path
        alt = project_dir / "weights" / "best.pt"
        if alt.exists():
            print(f"  Found weights at legacy path: {alt}")
            evaluate_and_save(alt, args, effective_classes)


if __name__ == "__main__":
    main()
