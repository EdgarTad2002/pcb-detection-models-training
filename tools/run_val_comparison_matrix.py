#!/usr/bin/env python3
"""
tools/run_val_comparison_matrix.py
==================================
Runs the complete controlled evaluation matrix on the validation split:
  1. Full-board baseline alone (--no-crops)
  2. Full-board baseline + baseline as crop specialist (NMS, merge_iou=0.50)
  3. Mixed-trained crop specialist alone (--no-crops)
  4. Dual-Branch: Full-board baseline + Mixed-trained crop specialist (NMS, merge_iou=0.50)
  5. Dual-Branch: Full-board baseline + Mixed-trained crop specialist (WBF, merge_iou=0.50)
  6. Dual-Branch: Sweep merge_iou (0.40, 0.60) for optimal NMS/WBF configuration

Zero Test Leakage Policy:
  This script evaluates EXCLUSIVELY on the validation split (80 boards).
  Once the optimal configuration is selected on validation, test split is evaluated separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_cmd(cmd: list[str]) -> dict:
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}:\n{res.stdout}")
    return {}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--global-weights", type=str, required=True, help="Baseline weights")
    p.add_argument("--crop-weights", type=str, required=True, help="Crop specialist weights")
    p.add_argument("--images-dir", type=str, required=True, help="Validation images dir")
    p.add_argument("--labels-dir", type=str, required=True, help="Validation labels dir")
    p.add_argument("--out-dir", type=str, default="runs/dual_branch_eval_val", help="Output directory for JSONs")
    p.add_argument("--device", default="0", help="CUDA device index")
    p.add_argument("--slice-size", type=int, default=1280)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf-thresh", type=float, default=0.001)
    p.add_argument("--max-det", type=int, default=1000)
    args = p.parse_args()

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "1_Baseline_FullBoard_Only",
            "desc": "Existing full-board baseline alone",
            "args": [
                "--global-weights", args.global_weights,
                "--no-crops",
            ]
        },
        {
            "name": "2_Baseline_With_BaselineCrops_NMS",
            "desc": "Existing checkpoint with capacitor-only tiled inference (Baseline on crops)",
            "args": [
                "--global-weights", args.global_weights,
                "--crop-weights", args.global_weights,
                "--fusion-method", "nms",
                "--merge-iou", "0.50",
            ]
        },
        {
            "name": "3_MixedSpecialist_FullBoard_Only",
            "desc": "Mixed-trained checkpoint with full-board inference only",
            "args": [
                "--global-weights", args.crop_weights,
                "--no-crops",
            ]
        },
        {
            "name": "4_DualBranch_MixedSpecialist_NMS_0.50",
            "desc": "Full-board baseline + Mixed crop specialist (NMS, merge=0.50)",
            "args": [
                "--global-weights", args.global_weights,
                "--crop-weights", args.crop_weights,
                "--fusion-method", "nms",
                "--merge-iou", "0.50",
            ]
        },
        {
            "name": "5_DualBranch_MixedSpecialist_WBF_0.50",
            "desc": "Full-board baseline + Mixed crop specialist (WBF, merge=0.50)",
            "args": [
                "--global-weights", args.global_weights,
                "--crop-weights", args.crop_weights,
                "--fusion-method", "wbf",
                "--merge-iou", "0.50",
            ]
        },
        {
            "name": "6_DualBranch_MixedSpecialist_NMS_0.40",
            "desc": "Full-board baseline + Mixed crop specialist (NMS, merge=0.40)",
            "args": [
                "--global-weights", args.global_weights,
                "--crop-weights", args.crop_weights,
                "--fusion-method", "nms",
                "--merge-iou", "0.40",
            ]
        },
        {
            "name": "7_DualBranch_MixedSpecialist_NMS_0.60",
            "desc": "Full-board baseline + Mixed crop specialist (NMS, merge=0.60)",
            "args": [
                "--global-weights", args.global_weights,
                "--crop-weights", args.crop_weights,
                "--fusion-method", "nms",
                "--merge-iou", "0.60",
            ]
        },
    ]

    results_summary = []

    for exp in experiments:
        json_path = out_p / f"{exp['name']}.json"
        cmd = [
            sys.executable,
            str(ROOT / "tools/eval_dual_branch_pipeline.py"),
            "--images-dir", args.images_dir,
            "--labels-dir", args.labels_dir,
            "--split-name", "val",
            "--device", args.device,
            "--slice-size", str(args.slice_size),
            "--imgsz", str(args.imgsz),
            "--overlap", str(args.overlap),
            "--conf-thresh", str(args.conf_thresh),
            "--max-det", str(args.max_det),
            "--output-json", str(json_path),
        ] + exp["args"]

        print(f"\n=======================================================")
        print(f"RUNNING EXPERIMENT: {exp['name']}")
        print(f"Description: {exp['desc']}")
        print(f"=======================================================")
        run_cmd(cmd)

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            data["experiment_name"] = exp["name"]
            data["description"] = exp["desc"]
            results_summary.append(data)

    summary_file = out_p / "validation_comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 105)
    print("CONTROLLED VALIDATION COMPARISON SUMMARY")
    print("=" * 105)
    header = f"{'Experiment':<38} | {'mAP@50':<8} | {'mAP50:95':<9} | {'Cap AP50':<9} | {'Cap Recall':<10} | {'Cap FP':<8} | {'Latency':<9}"
    print(header)
    print("-" * 105)
    for r in results_summary:
        exp_n = r["experiment_name"]
        map50 = r["mAP50"] * 100
        map50_95 = r["mAP50_95"] * 100
        cap_ap = r["per_class_ap50"]["Capacitor"] * 100
        cap_rec = r["detailed_metrics_iou50"]["Capacitor"]["recall"] * 100
        cap_fp = r["detailed_metrics_iou50"]["Capacitor"]["fp"]
        lat = f"{r['ms_per_board']:.1f}ms"
        print(f"{exp_n:<38} | {map50:6.2f}% | {map50_95:7.2f}% | {cap_ap:7.2f}% | {cap_rec:8.2f}% | {cap_fp:<8} | {lat:<9}")
    print("=" * 105)
    print(f"All summary details saved to: {summary_file}")

if __name__ == "__main__":
    main()
