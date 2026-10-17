#!/usr/bin/env python3
"""
Standardized Test Evaluation & Result Logger for PCB AOI Models.
================================================================
Evaluates a trained checkpoint on the 'test' split with:
  - max_det = 1000
  - conf = 0.001
  - iou = 0.50
  - imgsz = 1280
Logs per-class AP, overall mAP, latency, and writes results JSON to /mnt/weka/.../results/.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from ultralytics import YOLO

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def main():
    p = argparse.ArgumentParser(description="Standardized Test Evaluation & Result Logger")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt checkpoint")
    p.add_argument("--data", type=str, default="datasets/pcb-grand-unified-cappaste-1280/data.yaml", help="Dataset YAML")
    p.add_argument("--run-key", type=str, required=True, help="Run identifier for JSON results")
    p.add_argument("--split", type=str, default="test", help="Evaluation split (default: test)")
    p.add_argument("--imgsz", type=int, default=1280, help="Evaluation resolution (default: 1280)")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=1000, help="Maximum detections per board")
    p.add_argument("--device", type=str, default="0", help="GPU device index")
    p.add_argument("--results-dir", type=Path, default=Path("/mnt/weka/etadevosyan/pcb-yolo/results"),
                   help="Results directory for comparison tables")
    args = p.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Fallback check for nested paths
        alt_path = Path("runs/detect") / args.weights
        if alt_path.exists():
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

    device = args.device
    if str(device) not in ("cpu", "-1") and not torch.cuda.is_available():
        print("⚠️ CUDA GPU is not available on this node (login node). Automatically evaluating on CPU...")
        device = "cpu"

    print("=" * 75)
    print(f"📊 EVALUATING {args.run_key} ON '{args.split}' SPLIT")
    print(f"Weights:    {weights_path}")
    print(f"Dataset:    {args.data}")
    print(f"Resolution: {args.imgsz}px | Max Det: {args.max_det} | Device: {device}")
    print("=" * 75)

    model = YOLO(str(weights_path))
    t0 = time.time()
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=device,
        save_json=False,
        plots=True,
    )
    eval_time = time.time() - t0

    # Extract metrics
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    prec = float(metrics.box.mp)
    rec = float(metrics.box.mr)
    speed = metrics.speed
    total_time_ms = sum(speed.values())
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0

    # Per-class AP@50 and AP@50-95
    per_class_ap50 = {}
    per_class_ap50_95 = {}
    names = metrics.names if hasattr(metrics, "names") else {i: CLASS_NAMES[i] for i in range(4)}
    
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        for idx, ap in enumerate(metrics.box.ap50):
            c_name = names.get(idx, f"Class_{idx}")
            per_class_ap50[c_name] = round(float(ap), 4)
            
    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        for idx, ap in enumerate(metrics.box.maps):
            c_name = names.get(idx, f"Class_{idx}")
            per_class_ap50_95[c_name] = round(float(ap), 4)

    print("\n" + "=" * 75)
    print(f"🏆 EVALUATION RESULTS: {args.run_key}")
    print(f"  Overall mAP@50:    {map50:.4f} ({map50*100:.2f}%)")
    print(f"  Overall mAP@50-95: {map50_95:.4f} ({map50_95*100:.2f}%)")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"  Speed:             {total_time_ms:.1f} ms/board ({fps:.1f} FPS)")
    print("\n--- Per-Class AP@0.5 ---")
    for k, v in per_class_ap50.items():
        print(f"  {k:22s}: {v:.4f} (AP50-95: {per_class_ap50_95.get(k, 0.0):.4f})")
    print("=" * 75)

    # Save to JSON
    summary = {
        "model": args.run_key,
        "weights": str(weights_path),
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": prec,
        "recall": rec,
        "total_time_ms": total_time_ms,
        "fps": fps,
        "eval_conf": args.conf,
        "eval_iou": args.iou,
        "eval_split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "max_det": args.max_det,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap50_95,
    }
    # Add per-class top-level keys for comparison_table.csv compatibility
    for k, v in per_class_ap50.items():
        summary[f"{k}_ap50"] = v
    for k, v in per_class_ap50_95.items():
        summary[f"{k}_ap50_95"] = v

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"{args.run_key}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to: {out_json}")

    # Re-run aggregation
    agg_script = Path("aggregate_results.py")
    if agg_script.exists():
        os.system(f"python aggregate_results.py --results-dir {args.results_dir}")


if __name__ == "__main__":
    main()
