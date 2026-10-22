#!/usr/bin/env python3
"""
Train YOLO26s with Convolutional Grand Unified Learnable Stem
============================================================
Combines:
  - 15x15 Gaussian Conv for Illumination (L)
  - Differentiable Agaian AGC on L (IEEE Access Eq. 23)
  - Log-space Reflectance extraction: log(R) = log(I) - log(L)
  - Sargis Weber Contrast on Reflectance (3x3 local pooling, alpha=0.20)
  - Pure 2D Convolutional residual adapter into Layer 0 Conv
  - Champion recipe: Manual SGD, cls=1.5, box=5.0, dfl=2.0, ls=0.10, max_det=1000 @ 1280px
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from models.conv_grand_unified_stem import ConvGrandUnifiedTrainer, wrap_model_with_conv_gu_stem


def main():
    p = argparse.ArgumentParser(description="Train YOLO26s with Convolutional Grand Unified Stem")
    p.add_argument("--run-key", default="yolov26s_conv_grand_unified_stem_1280")
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--data", default="datasets/pcb-native-res-cleaned-cappaste-1280/data.yaml")
    p.add_argument("--eval-data", default=None, help="Optional canonical data.yaml for evaluation (e.g. pcb-native-res-cleaned-1280)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--max-det", type=int, default=1000)
    args = p.parse_args()

    project_dir = Path("runs") / args.run_key
    trainer = ConvGrandUnifiedTrainer(overrides={
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
        "optimizer": args.optimizer,
        "cls": 1.5,
        "box": 5.0,
        "dfl": 2.0,
        "label_smoothing": 0.10,
        "copy_paste": 0.30,
        "mosaic": 1.0,
        "mixup": 0.15,
    })
    trainer.train()

    best_pt = project_dir / "pcb-filtered" / "weights" / "best.pt"
    if best_pt.exists():
        eval_data_path = args.eval_data if args.eval_data else args.data
        print(f"\n[EVAL] Running evaluation on '{args.eval_split}' split with max_det={args.max_det} at {args.imgsz}px using {eval_data_path}...")
        eval_model = YOLO(str(best_pt))
        metrics = eval_model.val(
            data=eval_data_path,
            split=args.eval_split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.eval_conf,
            iou=args.eval_iou,
            max_det=args.max_det,
            device=args.device,
        )
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        prec = float(metrics.box.mp)
        rec = float(metrics.box.mr)
        print(f"Convolutional Grand Unified Evaluation mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")

        per_class_ap50 = {}
        names = metrics.names if hasattr(metrics, "names") else {0: "Capacitor", 1: "Connector", 2: "Electrolytic Capacitor", 3: "IC"}
        if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
            for idx, ap in enumerate(metrics.box.ap50):
                c_name = names.get(idx, f"Class_{idx}")
                per_class_ap50[c_name] = round(float(ap), 4)

        results_dir = Path("/mnt/weka/etadevosyan/pcb-yolo/results")
        if not results_dir.exists():
            results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": args.run_key,
            "weights": str(best_pt),
            "mAP50": map50,
            "mAP50_95": map50_95,
            "precision": prec,
            "recall": rec,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
            "eval_split": args.eval_split,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "per_class_ap50": per_class_ap50,
        }
        for k, v in per_class_ap50.items():
            summary[f"{k}_ap50"] = v
        out_json = results_dir / f"{args.run_key}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Results saved to: {out_json}")


if __name__ == "__main__":
    main()
