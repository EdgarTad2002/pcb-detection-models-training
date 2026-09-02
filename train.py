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
DEFAULT_CLASSES = [2, 4, 7, 9]  # Capacitor, Connector, Electrolytic Capacitor, IC
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate one YOLO PCB model.")

    # --- identity ---
    p.add_argument("--run-key", required=True, help="e.g. yolov26s_p2_only")
    p.add_argument(
        "--weights", required=True, help="e.g. yolo26s.pt or yolo26s-p2.yaml"
    )

    # --- paths ---
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data.yaml. Defaults to <project-root>/datasets/pcb-filtered-yolov8/data.yaml",
    )

    # --- core training hyperparameters ---
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--classes", type=int, nargs="+", default=DEFAULT_CLASSES
    )
    p.add_argument("--optimizer", default="SGD")
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

    # --- evaluation protocol -- keep these fixed across every run for a fair comparison ---
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.5)
    p.add_argument("--eval-split", default="test")

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

    return p.parse_args()


def build_train_kwargs(args, data_yaml):
    kwargs = dict(
        data=str(data_yaml),
        classes=args.classes,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project_root / "runs" / args.run_key),
        name="pcb-filtered",
        exist_ok=True,
        pretrained=not args.no_pretrained,
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
    )

    # Only pass augmentation and loss params the user explicitly set, so everything
    # else falls back to Ultralytics' own defaults rather than us silently
    # re-specifying (and potentially drifting from) them here.
    optional_overrides = {
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
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "label_smoothing": args.label_smoothing,
    }
    for key, val in optional_overrides.items():
        if val is not None:
            kwargs[key] = val

    return kwargs


def evaluate(weights_path, args, data_yaml):
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split=args.eval_split,
        classes=args.classes,
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary


def main():
    args = parse_args()

    data_yaml = args.data or (
        args.project_root / "datasets" / "pcb-filtered-yolov8" / "data.yaml"
    )
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"

    run_dir = args.project_root / "runs" / args.run_key / "pcb-filtered"
    weights_path = run_dir / "weights" / "best.pt"

    if not args.skip_train:
        print("=" * 70)
        print(f"Training run: {args.run_key}")
        print(f"Weights/config: {args.weights}")
        print(f"Data: {data_yaml}")
        print("=" * 70)

        model = YOLO(args.weights)
        train_kwargs = build_train_kwargs(args, data_yaml)
        results = model.train(**train_kwargs)
        print("Training finished. Run saved to:", results.save_dir)
    else:
        assert weights_path.exists(), (
            f"--skip-train given but no checkpoint found at {weights_path}"
        )
        print(f"Skipping training, evaluating existing checkpoint: {weights_path}")

    print("\n" + "=" * 70)
    print(f"Evaluating {args.run_key} on '{args.eval_split}' split")
    print("=" * 70)
    summary = evaluate(weights_path, args, data_yaml)

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
