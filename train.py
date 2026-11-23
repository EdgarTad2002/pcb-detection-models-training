#!/usr/bin/env python3
"""
Shared training + evaluation script for the PCB YOLO ensemble project,
built to run as a Slurm batch job on the YSU/YerevaNN cluster instead of
Colab notebook cells.

One script, many models: every baseline model (YOLOv8s, v10s, v11s, YOLO26s)
and every ablation run (Run D/E/G/H/...) is just a different set of CLI flags
against this same script. This keeps evaluation logic (conf/iou thresholds,
class filter, per-class AP, timing) identical across every run by construction
-- no risk of one run's .py file quietly drifting from another's.

Usage (see sbatch/*.sh for full examples):
    python train.py \
        --run-key yolov26s_p2_only \
        --weights yolo26s-p2.yaml \
        --epochs 100 --imgsz 640 --batch 8

Results are written as JSON to $RESULTS_DIR/<run-key>.json so that
aggregate_results.py can build the comparison table afterward without any
read-modify-write race condition between concurrently running Slurm jobs.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

# Register LUMA modules (SimAM, AConv, LIAM) and PhysOT-YOLO modules
for _mod_name in (
    "luma_modules",
    "retinex_stem",
    "weber_stem",
    "squared_dice_loss",
    "agwd_ota_loss",
    "igr_stem",
    "fg_csfa_neck",
    "aca_supcon",
    "icm_module",
    "superyolo_pcb",
    "physics_spectral_yolo26",
):
    try:
        __import__(_mod_name)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Model Dispatch Registry (Phase 3)
# ---------------------------------------------------------------------------
MODEL_DISPATCH_REGISTRY = {
    "physot-yolo26s": {
        "weights": "yolo26s.pt",
        "description": "Flagship PhysOT-YOLO26s (Proposals 1-4: AGWD-OTA + IGR-Net + FG-CSFA + ACA-SupCon)",
        "defaults": {
            "agwd_ota": True,
            "use_igr_stem": True,
            "use_fg_csfa": True,
            "use_aca_supcon": True,
        },
    },
    "physot-yolo26m": {
        "weights": "yolo26m.pt",
        "description": "Flagship PhysOT-YOLO26m (Medium backbone with Proposals 1-4)",
        "defaults": {
            "agwd_ota": True,
            "use_igr_stem": True,
            "use_fg_csfa": True,
            "use_aca_supcon": True,
        },
    },
    "physot-yolov11s": {
        "weights": "yolo11s.pt",
        "description": "PhysOT-YOLOv11s with Proposals 1-4",
        "defaults": {
            "agwd_ota": True,
            "use_igr_stem": True,
            "use_fg_csfa": True,
            "use_aca_supcon": True,
        },
    },
    "physot-yolov8s": {
        "weights": "yolov8s.pt",
        "description": "PhysOT-YOLOv8s with Proposals 1-4",
        "defaults": {
            "agwd_ota": True,
            "use_igr_stem": True,
            "use_fg_csfa": True,
            "use_aca_supcon": True,
        },
    },
    "yolo26s": {"weights": "yolo26s.pt", "description": "Standard YOLO26s baseline", "defaults": {}},
    "yolo26s-p2": {"weights": "yolo26s-p2.yaml", "description": "YOLO26s with native P2 head", "defaults": {}},
    "yolo11s": {"weights": "yolo11s.pt", "description": "Standard YOLOv11s baseline", "defaults": {}},
    "yolov10s": {"weights": "yolov10s.pt", "description": "Standard YOLOv10s baseline", "defaults": {}},
    "yolov8s": {"weights": "yolov8s.pt", "description": "Standard YOLOv8s baseline", "defaults": {}},
}

# ---------------------------------------------------------------------------
# Paths -- override via environment variables if your layout differs.
# Large files (datasets, checkpoints, results) belong under /mnt/weka/, per
# the cluster onboarding email. Do not point these at /home/.
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get(
        "PCB_PROJECT_ROOT",
        "/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training",
    )
)
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("PCB_RESULTS_DIR", "/mnt/weka/etadevosyan/pcb-yolo/results")
)
LEGACY_23_CLASSES = [2, 4, 7, 9]  # Capacitor, Connector, Electrolytic Capacitor, IC in 23-class dataset
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def _resolve_default_data_yaml() -> Path:
    """Finds the best default data.yaml if --data is not explicitly passed."""
    candidates = [
        Path("datasets/pcb-unified-4class-cleaned/data.yaml"),
        Path("datasets/pcb-unified-4class/data.yaml"),
        Path("datasets/pcb-filtered-yolov8/data.yaml"),
        DEFAULT_PROJECT_ROOT / "datasets/pcb-unified-4class-cleaned/data.yaml",
        DEFAULT_PROJECT_ROOT / "datasets/pcb-unified-4class/data.yaml",
        DEFAULT_PROJECT_ROOT / "datasets/pcb-filtered-yolov8/data.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("datasets/pcb-unified-4class-cleaned/data.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate one YOLO / PhysOT-YOLO PCB model.")

    # --- identity & model dispatch ---
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model dispatch key or weights path (e.g. 'physot-yolo26s', 'yolo26s', 'yolo26s.pt'). "
            f"Registered keys: {', '.join(MODEL_DISPATCH_REGISTRY.keys())}"
        ),
    )
    p.add_argument(
        "--run-key",
        type=str,
        default=None,
        help="Experiment identifier (e.g. physot_yolo26s_flagship). Auto-generated if --model is used.",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="e.g. yolo26s.pt or yolo26s-p2.yaml (optional if --model is specified).",
    )
    p.add_argument(
        "--pretrained-weights",
        type=str,
        default=None,
        help="Optional pretrained weights to transfer (e.g. yolo26s.pt) when --weights is a YAML config",
    )

    # --- paths ---
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data.yaml (defaults to datasets/pcb-unified-4class/data.yaml if present).",
    )

    # --- core training hyperparameters ---
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--classes", type=int, nargs="+", default=None,
        help="Explicit class indices to train/val on. Defaults to None (all classes if nc==4, or [2,4,7,9] if legacy 23-class).",
    )
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--lrf", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-epochs", type=float, default=None)
    p.add_argument("--nbs", type=int, default=None, help="Nominal batch size controlling gradient accumulation.")
    p.add_argument("--architecture", choices=["detection", "superyolo", "spectral"], default="detection")
    p.add_argument("--spectral-alpha", type=float, default=0.0)
    p.add_argument("--lambda-sr", type=float, default=0.10)
    p.add_argument("--sr-scale", type=int, default=2)
    p.add_argument("--edge-weight", type=float, default=0.5)
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Cluster has far more CPU headroom than Colab's workers=2 -- bump this freely.",
    )
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--save-period", type=int, default=-1)
    p.add_argument("--device", default="0")
    p.add_argument("--no-cache", action="store_true", help="Disable image caching.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP/FP16). Enabled by default.",
    )
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable activation gradient checkpointing in custom PhysOT modules to save VRAM at 1280px.",
    )

    # --- optional augmentation overrides (only applied if explicitly passed) ---
    p.add_argument("--copy-paste", type=float, default=None)
    p.add_argument("--mixup", type=float, default=None)
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--shear", type=float, default=None)
    p.add_argument("--perspective", type=float, default=None)
    p.add_argument("--hsv-h", type=float, default=None)
    p.add_argument("--hsv-s", type=float, default=None)
    p.add_argument("--hsv-v", type=float, default=None)
    p.add_argument("--mosaic", type=float, default=None)
    p.add_argument("--close-mosaic", type=int, default=None)
    p.add_argument("--fliplr", type=float, default=None)
    p.add_argument("--flipud", type=float, default=None)

    # --- optional loss reweighting overrides ---
    p.add_argument("--box", type=float, default=None, help="Box loss gain override (default 7.5)")
    p.add_argument("--cls", type=float, default=None, help="Class loss gain override (default 0.5)")
    p.add_argument("--dfl", type=float, default=None, help="Distribution Focal Loss gain override (default 1.5)")
    p.add_argument("--label-smoothing", type=float, default=None, help="Label smoothing override (default 0.0)")
    p.add_argument("--cos-lr", action="store_true", help="Enable cosine learning rate annealing schedule.")

    # --- Experiment E1: Dense-scene evaluation standardization ---
    p.add_argument("--seed", type=int, default=0, help="Training seed (Ultralytics default 0). Use several seeds to measure run-to-run noise.")
    p.add_argument(
        "--max-det", type=int, default=1000,
        help="Experiment E1: Max detections per image at val time (standardized to 1000; dense PCBs exceed 300).",
    )

    # --- Experiment E3 & Proposal 1: NWD / AGWD-OTA Label Assignment & Box Loss ---
    p.add_argument(
        "--nwd-loss",
        action="store_true",
        help="Experiment E3: Enable foundational Isotropic Normalized Wasserstein Distance (NWD) loss & TAL assigner.",
    )
    p.add_argument(
        "--agwd-ota",
        action="store_true",
        help="Proposal 1: Enable Anisotropic Gaussian Wasserstein + Symmetrized Jeffreys KLD + Dynamic-k SimOTA assigner.",
    )
    p.add_argument(
        "--agwd-kappa",
        type=float,
        default=12.8,
        help="Scale constant kappa (in pixels) for NWD/AGWD exponential kernel (default: 12.8).",
    )
    p.add_argument(
        "--agwd-tau",
        type=float,
        default=576.0,
        help="Scale-gate transition area tau_A in px^2 (default: 576.0 = 24x24 px).",
    )
    p.add_argument(
        "--agwd-alpha-kld",
        type=float,
        default=0.25,
        help="Weight alpha_kld for Symmetrized Jeffreys KLD aspect-ratio penalty in AGWD (default: 0.25).",
    )
    p.add_argument(
        "--tal-beta",
        type=float,
        default=3.0,
        help="Task-Aligned Assigner localization exponent beta (default: 3.0).",
    )
    p.add_argument(
        "--dynamic-k-max",
        type=int,
        default=15,
        help="Maximum dynamic-k candidate anchors per GT in SimOTA assignment (default: 15).",
    )

    # --- Proposal 2: Dual-Stream Illumination-Gated Retinex & Differentiable Gamma (IGR-Net) ---
    p.add_argument(
        "--use-igr-stem",
        action="store_true",
        help="Proposal 2: Inject Dual-Stream Illumination-Gated Retinex & Differentiable Pixel-Wise Gamma (igr_stem.py) into Layer 0.",
    )

    # --- Proposal 3: Frequency-Gated Cross-Scale Alignment & Quaternion Spectral Stem (FG-CSFA) ---
    p.add_argument(
        "--use-fg-csfa",
        action="store_true",
        help="Proposal 3: Inject Quaternion Radiometric Stem + FR-4 Weave Suppressor + SPD/Deformable P2->P3 Neck (fg_csfa_neck.py).",
    )
    p.add_argument(
        "--spectral-priors",
        type=Path,
        default=Path("data/pcb_spectral_priors.json"),
        help="Path to empirical PCB spectral priors JSON for FG-CSFA initialization.",
    )

    # --- Proposal 4: Axial Manhattan Context & Auxiliary Supervised Contrastive Head (ACA-SupCon) ---
    p.add_argument(
        "--use-aca-supcon",
        action="store_true",
        help="Proposal 4: Inject Orthogonal Manhattan Strip Pooling + Training-Only SupCon Hard-Negative Head (aca_supcon.py).",
    )
    p.add_argument(
        "--aca-kernel",
        type=int,
        default=11,
        help="Orthogonal strip convolution kernel length K (1xK and Kx1) for ACA (default: 11).",
    )
    p.add_argument(
        "--supcon-lambda",
        type=float,
        default=0.15,
        help="Auxiliary loss weight lambda_supcon for L_SupCon-HN (default: 0.15).",
    )
    p.add_argument(
        "--supcon-temp",
        type=float,
        default=0.10,
        help="Temperature tau_c for L_SupCon-HN contrastive similarity (default: 0.10).",
    )

    # --- Legacy / baseline custom modules ---
    p.add_argument(
        "--retinex-stem", action="store_true",
        help="Inject the legacy learnable in-network Retinex-Edge stem (retinex_stem.py) into Layer 0.",
    )
    p.add_argument(
        "--weber-stem", action="store_true",
        help="Inject the Professor's Extended Weber Contrast Law 3x3 edge detection stem (Eq. 5, weber_stem.py) into Layer 0.",
    )
    p.add_argument(
        "--use-icm", action="store_true",
        help="Inject Image-Level Illumination-Conditioned Modulation (ICM-YOLO, icm_module.py) into Layer 0.",
    )
    p.add_argument(
        "--squared-dice", action="store_true",
        help="Use BCE + Squared-Dice composite classification loss (Eq. 3/4).",
    )
    p.add_argument(
        "--lambda-dice", type=float, default=0.5,
        help="Weight for Squared-Dice classification loss component (default: 0.5).",
    )
    p.add_argument(
        "--smooth-dice", type=float, default=1.0,
        help="Smoothing epsilon for Squared-Dice loss (default: 1.0).",
    )
    p.add_argument(
        "--stage-weights", type=float, nargs="+", default=None,
        help="Stage / deep-supervision weights lambda_i for multi-scale Squared-Dice (Eq. 8, e.g. 0.5 0.3 0.1 0.1).",
    )
    p.add_argument(
        "--squared-box", action="store_true",
        help="Enable Squared-IoU ablation in bounding box loss (default: False, uses plain CIoU).",
    )

    # --- evaluation protocol -- keep these fixed across every run for a fair comparison ---
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.5)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--eval-data", type=Path, default=None, help="Independent fixed evaluation dataset YAML.")

    # --- misc ---
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and evaluate an existing best.pt for this run-key instead.",
    )
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional: copy the run directory here after training (e.g. a second Weka path).",
    )

    args = p.parse_args()

    # Resolve --model dispatch registry
    if args.model is not None:
        model_key = args.model.strip().lower()
        if model_key in MODEL_DISPATCH_REGISTRY:
            entry = MODEL_DISPATCH_REGISTRY[model_key]
            if args.weights is None:
                args.weights = entry["weights"]
            # If user invoked --model physot-yolo26s without specifying individual paradigm flags,
            # enable the registered defaults; if they explicitly passed flags, those are already True.
            any_paradigm_flag = (
                args.nwd_loss
                or args.agwd_ota
                or args.use_igr_stem
                or args.use_fg_csfa
                or args.use_aca_supcon
            )
            if not any_paradigm_flag:
                for k, v in entry.get("defaults", {}).items():
                    setattr(args, k, v)
        else:
            if args.weights is None:
                args.weights = args.model

    if args.weights is None:
        p.error("Either --weights or --model must be specified (e.g., --model physot-yolo26s or --weights yolo26s.pt).")

    if args.run_key is None:
        base_name = (args.model or Path(args.weights).stem).replace("-", "_")
        tags = []
        if args.nwd_loss:
            tags.append("nwd")
        if args.agwd_ota:
            tags.append("agwd")
        if args.use_igr_stem:
            tags.append("igr")
        if args.use_fg_csfa:
            tags.append("fgcsfa")
        if args.use_aca_supcon:
            tags.append("acasupcon")
        suffix = ("_" + "_".join(tags)) if tags else ""
        args.run_key = f"{base_name}{suffix}_{args.imgsz}px"

    if args.data is None:
        args.data = _resolve_default_data_yaml()

    return args


def build_train_kwargs(args, data_yaml, effective_classes=None):
    kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project_root / "runs" / args.run_key),
        name="pcb-filtered",
        exist_ok=True,
        pretrained=args.pretrained_weights if args.pretrained_weights else (not args.no_pretrained),
        optimizer=args.optimizer,
        verbose=True,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=not args.no_cache,
        plots=True,
        save=True,
        save_period=args.save_period,
        val=True,
        split="val",
        conf=args.eval_conf,
        iou=args.eval_iou,
        seed=args.seed,
        amp=not args.no_amp,
    )
    if args.max_det is not None:
        kwargs["max_det"] = args.max_det
    if effective_classes is not None:
        kwargs["classes"] = effective_classes

    # Only pass augmentation and loss params the user explicitly set, so everything
    # else falls back to Ultralytics' own defaults rather than us silently
    # re-specifying (and potentially drifting from) them here.
    optional_overrides = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "nbs": args.nbs,
        "copy_paste": args.copy_paste,
        "mixup": args.mixup,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "label_smoothing": args.label_smoothing,
        "cos_lr": True if args.cos_lr else None,
    }
    for key, val in optional_overrides.items():
        if val is not None:
            kwargs[key] = val

    return kwargs


def get_physot_composite_trainer(args):
    """
    Builds a composable Ultralytics DetectionTrainer subclass that wires any
    subset (or the full flagship suite) of:
      - Proposal 2: IGR-Net (`--use-igr-stem`) or legacy Retinex-Edge (`--retinex-stem`)
      - Proposal 3: FG-CSFA (`--use-fg-csfa`)
      - Experiment E3 / Proposal 1: NWD (`--nwd-loss`) or AGWD-OTA (`--agwd-ota`) or Squared-Dice (`--squared-dice`)
      - Proposal 4: ACA-SupCon (`--use-aca-supcon`, with training-only SupCon head stripped at checkpoint save)
    Supports AMP/FP16, activation gradient checkpointing (`--grad-checkpoint`), and multi-GPU DDP.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class PhysOTCompositeDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if weights is not None:
                model.load(weights, verbose=False)

            # 1. Layer-0 Optical Decomposition Stem (Proposal 2 IGR-Net, Weber Contrast Eq. 5, OR legacy Retinex-Edge)
            if args.use_igr_stem:
                from igr_stem import wrap_model_with_igr_stem
                wrap_model_with_igr_stem(model, use_checkpoint=args.grad_checkpoint)
                if verbose:
                    print(
                        f"[PhysOT Proposal 2: IGR-Net] Injected Dual-Stream Illumination-Gated Retinex "
                        f"& Differentiable Pixel-Wise Gamma into Layer 0 (grad_checkpoint={args.grad_checkpoint})"
                    )
            elif args.weber_stem:
                from weber_stem import wrap_model_with_weber_stem
                wrap_model_with_weber_stem(model)
                if verbose:
                    print("[Weber-Contrast Eq. 5] Injected Extended Weber Contrast Law 3x3 Edge Stem into Layer 0")
            elif args.use_icm:
                from icm_module import inject_icm_into_model
                inject_icm_into_model(model)
                if verbose:
                    print("[ICM-YOLO] Injected Illumination-Conditioned Modulation into Layer 0")
            elif args.retinex_stem:
                from retinex_stem import wrap_model_with_retinex_stem
                wrap_model_with_retinex_stem(model)
                if verbose:
                    print("[Retinex-Edge] Injected legacy Retinex-Edge stem into Layer 0")

            # 2. Quaternion Radiometric Stem + FR-4 Weave Suppressor + SPD/Deformable P2->P3 Neck (Proposal 3)
            if args.use_fg_csfa:
                from fg_csfa_neck import inject_fg_csfa_into_model
                priors_str = str(args.spectral_priors)
                inject_fg_csfa_into_model(
                    model,
                    inject_quaternion_stem=True,
                    priors_path=priors_str,
                )
                if verbose:
                    print(
                        f"[PhysOT Proposal 3: FG-CSFA] Injected Quaternion Radiometric Stem, "
                        f"FR-4 Weave Suppressor on P2, and SPD + Deformable P2->P3 Neck Alignment"
                    )

            # 3. Label Assignment & Box/Classification Loss Criterion (Experiment E3 / Proposal 1 / Squared-Dice)
            if args.agwd_ota or args.nwd_loss:
                from agwd_ota_loss import inject_agwd_criterion_into_model
                use_kld = bool(args.agwd_ota)
                dynamic_k = bool(args.agwd_ota)
                inject_agwd_criterion_into_model(
                    model,
                    kappa=args.agwd_tau,
                    constant_c=args.agwd_kappa,
                    tau_kl=args.agwd_alpha_kld,
                    tal_beta=args.tal_beta,
                    use_kld=use_kld,
                    dynamic_k=dynamic_k,
                    squared_dice=bool(args.squared_dice),
                    lambda_dice=args.lambda_dice,
                    smooth_dice=args.smooth_dice,
                    stage_weights=args.stage_weights,
                )
                mode_tag = "Proposal 1: AGWD-OTA (Anisotropic W2 + Jeffreys KLD + Dynamic-k SimOTA)" if use_kld else "Experiment E3: Base Isotropic NWD Loss & TAL"
                sdice_tag = f" + Multi-Stage Squared-Dice (lambda={args.lambda_dice}, smooth={args.smooth_dice}, weights={args.stage_weights})" if args.squared_dice else ""
                if verbose:
                    print(
                        f"[PhysOT {mode_tag}{sdice_tag}] constant_c={args.agwd_kappa}px, "
                        f"kappa={args.agwd_tau}, tal_beta={args.tal_beta}"
                    )
            elif args.squared_dice:
                from squared_dice_loss import squared_dice_init_criterion, squared_dice_loss_factory
                import functools
                loss_factory = functools.partial(
                    squared_dice_loss_factory,
                    lambda_dice=args.lambda_dice,
                    smooth=args.smooth_dice,
                    stage_weights=args.stage_weights,
                    squared_box=args.squared_box,
                )
                model.init_criterion = functools.partial(
                    squared_dice_init_criterion,
                    model,
                    loss_factory,
                )
                if verbose:
                    print(f"[Squared-Dice] Enabled BCE + {args.lambda_dice}*SDice (smooth={args.smooth_dice}, stage_weights={args.stage_weights}) criterion")

            # 4. Axial Manhattan Context + Auxiliary Supervised Contrastive Hard-Negative Head (Proposal 4)
            # Note: Injected AFTER step 3 so ACASupConCriterionWrapper wraps around AGWDDetectionLoss!
            if args.use_aca_supcon:
                from aca_supcon import inject_aca_supcon_into_model
                inject_aca_supcon_into_model(
                    model,
                    kernel_length=args.aca_kernel,
                    proj_dim=64,
                    temperature=args.supcon_temp,
                    lambda_supcon=args.supcon_lambda,
                )
                if verbose:
                    print(
                        f"[PhysOT Proposal 4: ACA-SupCon] Injected Manhattan Axial Context (1x{args.aca_kernel} + {args.aca_kernel}x1) "
                        f"& Prototype-Anchored SupCon-HN Head (lambda={args.supcon_lambda}, tau={args.supcon_temp}, stripped at inference)"
                    )

            return model

        def save_model(self):
            """
            Ensures clean checkpoint serialization across EMA and DDP models:
            - Strips training-only `supcon_head` and bound `init_criterion` methods
            - Clears transient feature caches (`cached_p2`, `last_p3_feat`)
            - Restores training hooks immediately after saving
            """
            from aca_supcon import strip_supcon_for_save, restore_supcon_after_save
            from fg_csfa_neck import P2FeatureTapWrapper

            def _clean_transient(m):
                if m is None:
                    return {}
                raw = unwrap_model(m)
                seq = getattr(raw, "model", None)
                if seq is not None:
                    for layer in seq:
                        if isinstance(layer, P2FeatureTapWrapper):
                            layer.cached_p2 = None
                return strip_supcon_for_save(raw)

            ema_raw = unwrap_model(self.ema.ema) if (hasattr(self, "ema") and self.ema is not None) else None
            mod_raw = unwrap_model(self.model) if hasattr(self, "model") else None
            b_ema = _clean_transient(ema_raw)
            b_mod = _clean_transient(mod_raw)
            try:
                return super().save_model()
            finally:
                if ema_raw is not None:
                    restore_supcon_after_save(ema_raw, b_ema)
                if mod_raw is not None:
                    restore_supcon_after_save(mod_raw, b_mod)

    return PhysOTCompositeDetectionTrainer


def load_model(weights_or_yaml):
    if "rtdetr" in str(weights_or_yaml).lower():
        from ultralytics import RTDETR
        return RTDETR(str(weights_or_yaml))
    return YOLO(str(weights_or_yaml))


def evaluate(weights_path, args, data_yaml, effective_classes=None):
    model = load_model(str(weights_path))

    val_kwargs = dict(
        data=str(args.eval_data or data_yaml),
        split=args.eval_split,
        conf=args.eval_conf,
        iou=args.eval_iou,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        half=False,
        project=str(args.results_dir / "evaluation_artifacts" / args.run_key),
        name=args.eval_split,
        exist_ok=True,
        workers=args.workers,
        max_det=args.max_det if args.max_det is not None else 1000,
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

    per_class_ap = {
        name: float(ap) for name, ap in zip(CLASS_NAMES, metrics.box.ap50)
    }

    summary = {
        "model": args.run_key,
        "model_dispatch": args.model,
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
        "pretrained_weights": args.pretrained_weights,
        "seed": args.seed,
        "max_det": args.max_det if args.max_det is not None else 1000,
        "hsv_v": args.hsv_v,
        "hsv_s": args.hsv_s,
        "nwd_loss": bool(args.nwd_loss),
        "agwd_ota": bool(args.agwd_ota),
        "agwd_kappa": args.agwd_kappa if (args.nwd_loss or args.agwd_ota) else None,
        "tal_beta": args.tal_beta if (args.nwd_loss or args.agwd_ota) else None,
        "use_igr_stem": bool(args.use_igr_stem),
        "use_fg_csfa": bool(args.use_fg_csfa),
        "use_aca_supcon": bool(args.use_aca_supcon),
        "retinex_stem": bool(args.retinex_stem),
        "weber_stem": bool(args.weber_stem),
        "squared_dice": bool(args.squared_dice),
        "lambda_dice": args.lambda_dice if args.squared_dice else None,
        "squared_box": bool(args.squared_box) if args.squared_dice else None,
        "data": str(args.data),
        "eval_data": str(args.eval_data or data_yaml),
        "smooth_dice": args.smooth_dice if args.squared_dice else None,
        "stage_weights": args.stage_weights,
        "use_icm": bool(args.use_icm),
        "architecture": args.architecture,
        "training_protocol": build_train_kwargs(args, data_yaml, effective_classes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary


def main():
    args = parse_args()

    data_yaml = args.data
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"

    # Auto-detect native 4-class vs legacy 23-class
    import yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg.get("nc", len(data_cfg.get("names", [])))

    if args.classes is not None:
        effective_classes = args.classes
    elif nc == 4:
        effective_classes = None  # Native 4-class: train on all classes
    else:
        effective_classes = LEGACY_23_CLASSES  # Legacy 23-class: filter [2, 4, 7, 9]

    run_dir = args.project_root / "runs" / args.run_key / "pcb-filtered"
    if args.skip_train and args.weights:
        wp = Path(args.weights)
        if wp.exists():
            weights_path = wp
        elif (args.project_root / wp).exists():
            weights_path = args.project_root / wp
        else:
            weights_path = wp
    else:
        weights_path = run_dir / "weights" / "best.pt"

    use_custom_trainer = any(
        [
            args.nwd_loss,
            args.agwd_ota,
            args.use_igr_stem,
            args.use_fg_csfa,
            args.use_aca_supcon,
            args.retinex_stem,
            args.weber_stem,
            args.use_icm,
            args.squared_dice,
        ]
    )

    if not args.skip_train:
        print("=" * 70)
        print(f"Training run: {args.run_key}")
        print(f"Model dispatch: {args.model or 'custom'} | Weights/config: {args.weights}")
        print(f"Data: {data_yaml} (nc={nc}, effective_classes={effective_classes}, max_det={args.max_det})")
        if args.architecture != "detection" and use_custom_trainer:
            raise ValueError("Standalone spectral/SuperYOLO architecture cannot be combined with composite trainer flags")
        if use_custom_trainer:
            print(
                f"Active Paradigms: "
                f"NWD={args.nwd_loss} | AGWD-OTA={args.agwd_ota} | "
                f"IGR-Stem={args.use_igr_stem} | FG-CSFA={args.use_fg_csfa} | "
                f"ACA-SupCon={args.use_aca_supcon}"
            )
        print("=" * 70)

        model = load_model(args.weights)
        if args.pretrained_weights:
            print(f"Loading/transferring pretrained weights from: {args.pretrained_weights}")
            model.load(args.pretrained_weights)
        train_kwargs = build_train_kwargs(args, data_yaml, effective_classes=effective_classes)
        if args.architecture == "superyolo":
            from superyolo_pcb import TrueSuperYOLOTrainer
            TrueSuperYOLOTrainer.lambda_sr = args.lambda_sr
            TrueSuperYOLOTrainer.sr_scale = args.sr_scale
            TrueSuperYOLOTrainer.edge_weight = args.edge_weight
            TrueSuperYOLOTrainer.downsample_input = False
            results = model.train(trainer=TrueSuperYOLOTrainer, **train_kwargs)
        elif args.architecture == "spectral":
            from physics_spectral_yolo26 import PhysicsSpectralDetectionTrainer
            PhysicsSpectralDetectionTrainer.priors_path = str(args.spectral_priors)
            PhysicsSpectralDetectionTrainer.alpha = args.spectral_alpha
            results = model.train(trainer=PhysicsSpectralDetectionTrainer, **train_kwargs)
        elif use_custom_trainer:
            custom_trainer = get_physot_composite_trainer(args)
            results = model.train(trainer=custom_trainer, **train_kwargs)
        else:
            results = model.train(**train_kwargs)
        print("Training finished. Run saved to:", results.save_dir)
    else:
        assert weights_path.exists(), (
            f"--skip-train given but no checkpoint found at {weights_path}"
        )
        print(f"Skipping training, evaluating existing checkpoint: {weights_path}")

    print("\n" + "=" * 70)
    print(f"Evaluating {args.run_key} on '{args.eval_split}' split (max_det={args.max_det})")
    print("=" * 70)
    summary = evaluate(weights_path, args, data_yaml, effective_classes=effective_classes)

    print("\n--- Overall Metrics ---")
    for k, v in summary.items():
        if k != "per_class_ap50":
            print(f"{k}: {v}")
    print("\n--- Per-Class AP@0.5 ---")
    for name, ap in summary["per_class_ap50"].items():
        print(f"{name}: {ap:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.results_dir / f"{args.run_key}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results JSON to: {result_path}")

    if args.backup_dir:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        dest = args.backup_dir / args.run_key
        shutil.copytree(run_dir.parent, dest, dirs_exist_ok=True)
        print(f"Backed up run directory to: {dest}")


if __name__ == "__main__":
    main()
