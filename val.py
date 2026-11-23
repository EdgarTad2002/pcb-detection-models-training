#!/usr/bin/env python3
"""
val.py — Standardized Evaluation & Latency Benchmarking CLI (Phase 1 E1/E2 & Phase 3)
=====================================================================================

Provides a unified, leakage-free validation and hardware latency benchmarking entry point
for all baseline YOLO models, SuperYOLO-PCB models, and PhysOT-YOLO (Proposals 1–4)
checkpoints.

Standardizations Enforced:
1. Experiment E1 (Dense-Board Recall Fix):
   - Enforces `max_det=1000` by default (eliminating Ultralytics' default 300-box cap
     that truncates micro-Capacitor and IC predictions on dense PCBs).
2. Experiment E1 (Hardware-Synchronized Latency Protocol):
   - Measures pure forward + NMS postprocess latency at `batch=1` using
     `torch.cuda.Event(enable_timing=True)` and `torch.cuda.synchronize()` after 30
     GPU warmup passes across both FP32 and FP16 (AMP) precision modes.
   - Separates mAP evaluation confidence (`--eval-conf 0.001`) from deployment latency
     confidence (`--latency-conf 0.25`) so NMS timing is never skewed by 30,000+ low-score
     candidate boxes.
3. Experiment E2 (1280px Multi-Scale Stride Alignment):
   - Guarantees stride-32 alignment (`ceil(imgsz / 32) * 32`) without silent 640px
     downsampling during high-resolution evaluation.
4. Phase 3 Custom Module Unpickling:
   - Pre-registers all custom modules (`luma_modules`, `retinex_stem`, `squared_dice_loss`,
     `agwd_ota_loss`, `igr_stem`, `fg_csfa_neck`, `aca_supcon`, `physics_spectral_yolo26`,
     `superyolo_pcb`) so any saved checkpoint unpickles cleanly.

Usage Examples:
---------------
# 1. Standardized evaluation of a trained PhysOT-YOLO26s checkpoint at 1280px:
python val.py \
    --weights runs/physot_yolo26s_flagship/pcb-filtered/weights/best.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --imgsz 1280 --max-det 1000 --benchmark-latency

# 2. Evaluate by model dispatch / run-key with Class-Adaptive Thresholds:
python val.py \
    --run-key yolov26s_1280_mosaic_scale \
    --data datasets/pcb-unified-4class/data.yaml \
    --imgsz 1280 --max-det 1000 --class-adaptive-thresholds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ultralytics import YOLO

# Pre-register all custom modules so torch.load() unpickles any custom layer
for _mod_name in (
    "luma_modules",
    "retinex_stem",
    "weber_stem",
    "squared_dice_loss",
    "agwd_ota_loss",
    "igr_stem",
    "fg_csfa_neck",
    "aca_supcon",
    "physics_spectral_yolo26",
    "superyolo_pcb",
):
    try:
        __import__(_mod_name)
    except ImportError:
        pass

from train import (
    CLASS_NAMES,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_DIR,
    LEGACY_23_CLASSES,
    MODEL_DISPATCH_REGISTRY,
    _resolve_default_data_yaml,
    load_model,
)


def align_stride(imgsz: int, max_stride: int = 32) -> int:
    """Experiment E2: Round resolution up to the nearest multiple of max_stride (32)."""
    return int(math.ceil(float(imgsz) / max_stride) * max_stride)


@torch.no_grad()
def benchmark_cuda_latency(
    weights_path: str,
    imgsz: int,
    device: str = "0",
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 1000,
    warmup: int = 30,
    iters: int = 100,
    half: bool = False,
) -> Dict[str, float]:
    """
    Experiment E1: Standardized batch=1 GPU latency benchmark with CUDA Event timing
    and explicit `torch.cuda.synchronize()`.
    """
    use_cuda = torch.cuda.is_available() and str(device).lower() != "cpu"
    if half and not use_cuda:
        return {"mean_ms": 0.0, "std_ms": 0.0, "median_ms": 0.0, "iqr_ms": 0.0, "fps": 0.0}

    model = load_model(weights_path)
    dev_str = f"cuda:{device}" if (use_cuda and not str(device).startswith("cuda")) else str(device)
    dummy_np = np.random.randint(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(
            source=dummy_np,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            half=half,
            verbose=False,
        )
    if use_cuda:
        torch.cuda.synchronize(dev_str)

    timings_ms = []
    for _ in range(iters):
        if use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(dev_str)
            start_evt.record()
            model.predict(
                source=dummy_np,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                half=half,
                verbose=False,
            )
            end_evt.record()
            torch.cuda.synchronize(dev_str)
            timings_ms.append(float(start_evt.elapsed_time(end_evt)))
        else:
            t0 = time.perf_counter()
            model.predict(
                source=dummy_np,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                half=False,
                verbose=False,
            )
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(timings_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    median_ms = float(np.median(arr))
    iqr_ms = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "median_ms": round(median_ms, 3),
        "iqr_ms": round(iqr_ms, 3),
        "fps": round(fps, 2),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Standardized PCB YOLO / PhysOT-YOLO Evaluation & Latency Benchmark (E1/E2/Phase 3)"
    )
    p.add_argument("--weights", type=str, default=None, help="Path to checkpoint (.pt)")
    p.add_argument("--run-key", type=str, default=None, help="Run key under runs/<run-key>/pcb-filtered/weights/best.pt")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Optional model dispatch key ({', '.join(MODEL_DISPATCH_REGISTRY.keys())})",
    )
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--data", type=Path, default=None, help="Path to data.yaml")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=1280, help="Evaluation resolution (aligned to stride 32)")
    p.add_argument("--batch", type=int, default=8, help="Batch size for mAP validation pass")
    p.add_argument("--device", type=str, default="0")
    p.add_argument(
        "--max-det",
        type=int,
        default=1000,
        help="Experiment E1: Maximum detections per image (standardized to 1000)",
    )
    p.add_argument("--eval-conf", type=float, default=0.001, help="Confidence threshold for mAP PR curve integration")
    p.add_argument("--eval-iou", type=float, default=0.5, help="NMS IoU threshold for mAP evaluation")
    p.add_argument("--half", action="store_true", help="Run mAP validation in FP16")
    p.add_argument("--classes", type=int, nargs="+", default=None)

    # Latency benchmarking options (Experiment E1)
    p.add_argument(
        "--benchmark-latency",
        action="store_true",
        help="Run CUDA Event synchronized batch=1 latency benchmark (FP32 and FP16)",
    )
    p.add_argument("--latency-conf", type=float, default=0.25, help="Deployment confidence threshold for latency test")
    p.add_argument("--warmup", type=int, default=30, help="GPU warmup iterations")
    p.add_argument("--iters", type=int, default=100, help="Timed GPU iterations")

    # Advanced post-processing options
    p.add_argument(
        "--class-adaptive-thresholds",
        action="store_true",
        help="Evaluate precision/recall/F1 using per-class optimal confidence & NMS thresholds",
    )
    p.add_argument(
        "--class-gated-sahi",
        action="store_true",
        help="Enable Class-Selective Scale-Gated SAHI (micro-components from slices, macro from full board)",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Explicit output JSON path")

    args = p.parse_args()
    if args.data is None:
        args.data = _resolve_default_data_yaml()

    if args.weights is None:
        if args.run_key is not None:
            candidate = args.project_root / "runs" / args.run_key / "pcb-filtered" / "weights" / "best.pt"
            local_cand = Path("runs") / args.run_key / "pcb-filtered" / "weights" / "best.pt"
            args.weights = str(local_cand if local_cand.exists() else candidate)
        elif args.model is not None:
            mkey = args.model.strip().lower()
            if mkey in MODEL_DISPATCH_REGISTRY:
                args.weights = MODEL_DISPATCH_REGISTRY[mkey]["weights"]
            else:
                args.weights = args.model
        else:
            p.error("Provide --weights, --run-key, or --model to evaluate.")

    return args


def main():
    args = parse_args()
    imgsz_aligned = align_stride(args.imgsz, max_stride=32)
    if imgsz_aligned != args.imgsz:
        print(f"[Experiment E2] Aligned --imgsz {args.imgsz} -> {imgsz_aligned} (stride 32 multiple).")

    assert args.data.exists(), f"data.yaml not found at {args.data}"
    import yaml
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg.get("nc", len(data_cfg.get("names", [])))

    if args.classes is not None:
        effective_classes = args.classes
    elif nc == 4:
        effective_classes = None
    else:
        effective_classes = LEGACY_23_CLASSES

    run_label = args.run_key or args.model or Path(args.weights).stem
    print("=" * 75)
    print(f"Standardized PCB Evaluation (E1/E2) — {run_label}")
    print(f"Weights    : {args.weights}")
    print(f"Data       : {args.data} (split={args.split}, nc={nc})")
    print(f"Resolution : {imgsz_aligned}px | max_det={args.max_det} | conf={args.eval_conf} | iou={args.eval_iou}")
    print("=" * 75)

    model = load_model(args.weights)
    val_kwargs = dict(
        data=str(args.data),
        split=args.split,
        imgsz=imgsz_aligned,
        batch=args.batch,
        conf=args.eval_conf,
        iou=args.eval_iou,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
    )
    if effective_classes is not None:
        val_kwargs["classes"] = effective_classes

    metrics = model.val(**val_kwargs)
    speed = metrics.speed
    val_total_ms = (
        speed.get("preprocess", 0.0)
        + speed.get("inference", 0.0)
        + speed.get("postprocess", 0.0)
    )
    per_class_ap50 = {
        name: float(ap) for name, ap in zip(CLASS_NAMES, metrics.box.ap50)
    }
    per_class_ap50_95 = {
        name: float(ap) for name, ap in zip(CLASS_NAMES, metrics.box.ap)
    }

    summary = {
        "model": run_label,
        "weights": str(args.weights),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.p.mean()),
        "recall": float(metrics.box.r.mean()),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap50_95,
        "val_speed_ms": float(val_total_ms),
        "val_fps": float(1000.0 / val_total_ms) if val_total_ms > 0 else 0.0,
        "imgsz": imgsz_aligned,
        "max_det": args.max_det,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.benchmark_latency:
        print(f"\n[Experiment E1] Benchmarking batch=1 GPU latency (warmup={args.warmup}, iters={args.iters})...")
        fp32_stats = benchmark_cuda_latency(
            args.weights,
            imgsz=imgsz_aligned,
            device=args.device,
            conf=args.latency_conf,
            iou=args.eval_iou,
            max_det=args.max_det,
            warmup=args.warmup,
            iters=args.iters,
            half=False,
        )
        fp16_stats = benchmark_cuda_latency(
            args.weights,
            imgsz=imgsz_aligned,
            device=args.device,
            conf=args.latency_conf,
            iou=args.eval_iou,
            max_det=args.max_det,
            warmup=args.warmup,
            iters=args.iters,
            half=True,
        )
        summary["latency_fp32_batch1"] = fp32_stats
        summary["latency_fp16_batch1"] = fp16_stats
        summary["total_time_ms"] = fp16_stats["mean_ms"] if fp16_stats["mean_ms"] > 0 else fp32_stats["mean_ms"]
        summary["fps"] = fp16_stats["fps"] if fp16_stats["fps"] > 0 else fp32_stats["fps"]
        print(f"  FP32 (batch=1): {fp32_stats['mean_ms']:.2f} ± {fp32_stats['std_ms']:.2f} ms ({fp32_stats['fps']:.1f} FPS)")
        print(f"  FP16 (batch=1): {fp16_stats['mean_ms']:.2f} ± {fp16_stats['std_ms']:.2f} ms ({fp16_stats['fps']:.1f} FPS)")

    if args.class_adaptive_thresholds or args.class_gated_sahi:
        from physics_spectral_yolo26 import evaluate_advanced_pipeline
        adv_metrics = evaluate_advanced_pipeline(
            weights_path=Path(args.weights),
            data_yaml=args.data,
            imgsz=imgsz_aligned,
            split=args.split,
            device=args.device,
            max_det=args.max_det,
            use_adaptive_thresh=args.class_adaptive_thresholds,
            use_gated_sahi=args.class_gated_sahi,
        )
        summary["advanced_postprocessing"] = adv_metrics

    print("\n--- Standardized Evaluation Summary ---")
    print(f"mAP@0.5      : {summary['mAP50']:.4f}")
    print(f"mAP@0.5:0.95 : {summary['mAP50_95']:.4f}")
    print(f"Precision    : {summary['precision']:.4f}")
    print(f"Recall       : {summary['recall']:.4f}")
    for cname, ap in summary["per_class_ap50"].items():
        print(f"  AP@0.5 [{cname:<22s}]: {ap:.4f} (AP@0.5:0.95 = {summary['per_class_ap50_95'][cname]:.4f})")

    out_path = args.output_json
    if out_path is None:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.results_dir / f"{run_label}_val_{imgsz_aligned}px.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved standardized validation report to: {out_path}")


if __name__ == "__main__":
    main()
