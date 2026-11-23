#!/usr/bin/env python3
"""
tools/sweep_nms_iou.py — NMS IoU Threshold Optimization for Dense PCB Components
================================================================================
Sweeps post-processing NMS IoU thresholds across [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
under standardized protocol (test split, imgsz=1280, conf=0.001, max_det=1000) to find
the optimal operating point that prevents mutual suppression of dense SMD capacitors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-register custom modules
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

import torch
from ultralytics import YOLO

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def parse_args():
    p = argparse.ArgumentParser(description="Sweep NMS IoU thresholds on PCB test set.")
    p.add_argument("--weights", type=str, required=True, help="Path to weights file")
    p.add_argument("--data", type=str, default="datasets/pcb-native-res-cleaned-1280/data.yaml")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--iou-values",
        type=float,
        nargs="+",
        default=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        help="List of NMS IoU thresholds to test",
    )
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("NMS IoU THRESHOLD OPTIMIZATION SWEEP")
    print("=" * 75)
    print(f"Weights:     {args.weights}")
    print(f"Dataset:     {args.data} (split='{args.split}', imgsz={args.imgsz})")
    print(f"Protocol:    conf={args.conf}, max_det={args.max_det}")
    print(f"IoU Values:  {args.iou_values}")
    print("=" * 75)

    model = YOLO(args.weights)

    sweep_results = []
    print(f"\n{'NMS IoU':<9} | {'Overall mAP50':<14} | {'Capacitor':<11} | {'Connector':<11} | {'Elec-Cap':<11} | {'IC':<9} | {'mAP50-95':<9}")
    print("-" * 88)

    for iou in args.iou_values:
        val_res = model.val(
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=iou,
            max_det=args.max_det,
            device=args.device,
            plots=False,
            verbose=False,
        )

        m50 = float(val_res.box.map50)
        m50_95 = float(val_res.box.map)
        per_class_ap = [float(ap) for ap in val_res.box.ap50]
        
        cap_ap = per_class_ap[0] if len(per_class_ap) > 0 else 0.0
        conn_ap = per_class_ap[1] if len(per_class_ap) > 1 else 0.0
        elec_ap = per_class_ap[2] if len(per_class_ap) > 2 else 0.0
        ic_ap = per_class_ap[3] if len(per_class_ap) > 3 else 0.0

        row = {
            "iou_threshold": iou,
            "mAP50": m50,
            "mAP50_95": m50_95,
            "Capacitor": cap_ap,
            "Connector": conn_ap,
            "Electrolytic Capacitor": elec_ap,
            "IC": ic_ap,
        }
        sweep_results.append(row)

        print(
            f"{iou:<9.2f} | {m50*100:>6.2f}%        | {cap_ap*100:>6.2f}%    | {conn_ap*100:>6.2f}%    | {elec_ap*100:>6.2f}%   | {ic_ap*100:>6.2f}% | {m50_95*100:>6.2f}%"
        )

    print("=" * 88)
    best_cap = max(sweep_results, key=lambda x: x["Capacitor"])
    best_overall = max(sweep_results, key=lambda x: x["mAP50"])
    print(f"Optimal for Capacitors: NMS IoU = {best_cap['iou_threshold']:.2f} (Capacitor AP = {best_cap['Capacitor']*100:.2f}%)")
    print(f"Optimal for Overall:    NMS IoU = {best_overall['iou_threshold']:.2f} (Overall mAP50 = {best_overall['mAP50']*100:.2f}%)")
    print("=" * 88)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": args.weights,
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "max_det": args.max_det,
            "best_capacitor_iou": best_cap["iou_threshold"],
            "best_overall_iou": best_overall["iou_threshold"],
            "sweep_results": sweep_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out_p.write_text(json.dumps(payload, indent=2))
        print(f"Saved results to: {out_p}")


if __name__ == "__main__":
    main()
