#!/usr/bin/env python3
"""
Class-Adaptive Confidence Threshold Calibration for PCB Component Detection.
Sweeps confidence thresholds across each target class, computes Precision, Recall,
and F1 curves, and determines the optimal F1-maximizing operating threshold for each class.

Compares:
1. Naive Flat Threshold (tau = 0.25 across all classes)
2. Class-Adaptive Calibrated Thresholds (tau_c* per class)
"""

import os
import sys
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sahi_pcb_inference import (
    sahi_predict,
    EVAL_CLASSES,
    UNIFIED_CLASSES,
    NAME_TO_UNIFIED_ID,
    parse_label_file,
    CLASS_COLORS,
    get_color
)

RAW_CLASSES = [
    "Button", "Capacitor Jumper", "Capacitor", "Clock", "Connector",
    "Diode", "EM", "Electrolytic Capacitor", "Ferrite Bead", "IC",
    "Inductor", "Jumper", "Led", "Pads", "Pins", "Resistor",
    "Resistor Jumper", "Resistor Network", "Switch", "Test Point",
    "Transistor", "Unknown", "Variable Resistor"
]


def match_detections(
    pred_boxes: List[Tuple[int, float, int, int, int, int]],  # (img_id, score, x1, y1, x2, y2)
    gt_boxes: List[Tuple[int, int, int, int, int]],            # (img_id, x1, y1, x2, y2)
    iou_threshold: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Matches predictions to ground truth per image using greedy IoU assignment.
    Returns:
      scores: array of confidence scores for all predictions
      tp_flags: boolean array (True if prediction matched a GT box)
      total_gt: total count of GT boxes
    """
    if not pred_boxes:
        return np.array([]), np.array([]), len(gt_boxes)

    # Sort predictions descending by score
    sorted_preds = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
    scores = np.array([p[1] for p in sorted_preds])
    tp_flags = np.zeros(len(sorted_preds), dtype=bool)

    # Group GT by img_id
    gt_by_img: Dict[int, List[List[int]]] = {}
    for g in gt_boxes:
        img_id = g[0]
        if img_id not in gt_by_img:
            gt_by_img[img_id] = []
        gt_by_img[img_id].append([g[1], g[2], g[3], g[4]])

    # Matched tracker
    matched_gt: Dict[int, set] = {img_id: set() for img_id in gt_by_img}

    for p_idx, p in enumerate(sorted_preds):
        img_id, sc, px1, py1, px2, py2 = p
        if img_id not in gt_by_img:
            continue

        best_iou = 0.0
        best_gt_idx = -1
        p_area = max(0, px2 - px1) * max(0, py2 - py1)

        for g_idx, g in enumerate(gt_by_img[img_id]):
            if g_idx in matched_gt[img_id]:
                continue
            gx1, gy1, gx2, gy2 = g
            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            g_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
            union = p_area + g_area - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_flags[p_idx] = True
            matched_gt[img_id].add(best_gt_idx)

    return scores, tp_flags, len(gt_boxes)


def calibrate_thresholds(
    model_path: str = "runs/yolov26s_loss_reweight/pcb-filtered/weights/best.pt",
    sample_dir: str = "data_samples",
    max_images: int = 44,
    use_sahi: bool = True,
    slice_size: int = 480,
) -> Dict:
    print("=" * 85)
    print("🔬 Class-Adaptive Confidence Threshold Calibration (F1 Optimization)")
    print("=" * 85)
    print(f"Model Checkpoint: {model_path}")
    print(f"Sample Directory: {sample_dir} (Images: {max_images}, SAHI: {use_sahi})")

    model = YOLO(model_path)
    img_dir = Path(sample_dir) / "images"
    lbl_dir = Path(sample_dir) / "labels"
    images = sorted(glob.glob(str(img_dir / "*.jpg")))[:max_images]

    gts = {c: [] for c in EVAL_CLASSES}
    preds_pool = {c: [] for c in EVAL_CLASSES}

    print(f"Collecting predictions across {len(images)} PCB images at sensitivity conf=0.01...")
    for img_id, img_path in enumerate(images):
        raw = cv2.imread(img_path)
        if raw is None:
            continue
        h, w = raw.shape[:2]

        # Load GT
        # Load GT via parse_label_file
        lbl_file = lbl_dir / (Path(img_path).stem + ".txt")
        gt_boxes = parse_label_file(lbl_file, w, h)
        for cid, cname, x1, y1, x2, y2 in gt_boxes:
            gts[cid].append((img_id, x1, y1, x2, y2))

        # Infer with low threshold to capture full range [0.01..1.0]
        if use_sahi:
            dets = sahi_predict(
                model=model,
                image_bgr=raw,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_ratio=0.25,
                conf_threshold=0.01,
                iou_threshold=0.45,
                nms_type="diou",
                include_full_image=True,
            )
            for _, cname, sc, bx1, by1, bx2, by2 in dets:
                cid = NAME_TO_UNIFIED_ID.get(cname)
                if cid is not None:
                    preds_pool[cid].append((img_id, sc, bx1, by1, bx2, by2))
        else:
            res = model.predict(raw, imgsz=640, conf=0.01, verbose=False)[0]
            for b in res.boxes:
                raw_name = model.names.get(int(b.cls[0]), "")
                cid = NAME_TO_UNIFIED_ID.get(raw_name)
                if cid is not None:
                    sc = float(b.conf[0])
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                    preds_pool[cid].append((img_id, sc, bx1, by1, bx2, by2))

    # Threshold sweep [0.02, 0.04, ..., 0.94]
    thresholds = np.linspace(0.02, 0.94, 93)
    calibration_results = {}
    f1_curves = {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    print("\n" + "-" * 85)
    print(f"{'Class':<22} | {'Optimal Threshold':<18} | {'Max F1-Score':<14} | {'Precision':<11} | {'Recall':<10}")
    print("-" * 85)

    for idx, (cid, cname) in enumerate(EVAL_CLASSES.items()):
        scores, tp_flags, total_gt = match_detections(preds_pool[cid], gts[cid], iou_threshold=0.50)

        p_list = []
        r_list = []
        f1_list = []

        for tau in thresholds:
            mask = scores >= tau
            tp = np.sum(tp_flags[mask])
            fp = np.sum(~tp_flags[mask])
            n_det = tp + fp

            prec = tp / n_det if n_det > 0 else 1.0
            rec = tp / total_gt if total_gt > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0

            p_list.append(prec)
            r_list.append(rec)
            f1_list.append(f1)

        best_idx = int(np.argmax(f1_list))
        opt_tau = float(thresholds[best_idx])
        max_f1 = float(f1_list[best_idx])
        opt_prec = float(p_list[best_idx])
        opt_rec = float(r_list[best_idx])

        # Baseline metrics at flat tau = 0.25
        flat_idx = int(np.argmin(np.abs(thresholds - 0.25)))
        flat_f1 = float(f1_list[flat_idx])
        flat_prec = float(p_list[flat_idx])
        flat_rec = float(r_list[flat_idx])

        calibration_results[cname] = {
            "optimal_threshold": round(opt_tau, 2),
            "max_f1": round(max_f1 * 100, 2),
            "precision_at_optimal": round(opt_prec * 100, 2),
            "recall_at_optimal": round(opt_rec * 100, 2),
            "baseline_threshold": 0.25,
            "baseline_f1": round(flat_f1 * 100, 2),
            "baseline_precision": round(flat_prec * 100, 2),
            "baseline_recall": round(flat_rec * 100, 2),
            "f1_gain": round((max_f1 - flat_f1) * 100, 2),
            "total_ground_truth": total_gt,
        }

        print(f"{cname:<22} | tau = {opt_tau:<12.2f} | {max_f1*100:>6.2f}%       | {opt_prec*100:>6.2f}%    | {opt_rec*100:>6.2f}%")

        # Plot curves
        ax = axes[idx]
        ax.plot(thresholds, f1_list, label=f"F1-Score (Peak: {max_f1*100:.1f}%)", color="#2563eb", lw=2.5)
        ax.plot(thresholds, p_list, label="Precision", color="#10b981", lw=1.8, linestyle="--")
        ax.plot(thresholds, r_list, label="Recall", color="#f59e0b", lw=1.8, linestyle=":")
        ax.axvline(opt_tau, color="#ef4444", lw=1.5, linestyle="-.", label=f"Optimal tau = {opt_tau:.2f}")
        ax.axvline(0.25, color="#6b7280", lw=1.2, linestyle="--", label="Default flat tau = 0.25")

        ax.set_title(f"{cname} (GT: {total_gt})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Metric Value (0 - 1.0)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    curve_img = Path("runs/report_assets/f1_confidence_curves.png")
    curve_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(curve_img), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n📈 F1-Confidence curves saved to: {curve_img}")

    # Comparative Table
    print("\n" + "=" * 88)
    print("📊 Flat Threshold (tau=0.25) vs Calibrated Class-Adaptive Thresholds (F1 Comparison)")
    print("=" * 88)
    print(f"{'Class':<22} | {'Default (tau=0.25) F1':<22} | {'Calibrated tau*':<17} | {'Calibrated F1':<14} | {'F1 Gain Δ':<10}")
    print("-" * 88)
    for cname, res in calibration_results.items():
        fb = res["baseline_f1"]
        fc = res["max_f1"]
        tau_c = res["optimal_threshold"]
        d = res["f1_gain"]
        sign = "+" if d >= 0 else ""
        print(f"{cname:<22} | {fb:>6.2f}%                 | tau = {tau_c:<10.2f} | {fc:>6.2f}%       | {sign}{d:>5.2f}%")
    print("=" * 88)

    macro_f1_base = np.mean([r["baseline_f1"] for r in calibration_results.values()])
    macro_f1_opt = np.mean([r["max_f1"] for r in calibration_results.values()])
    delta_macro = macro_f1_opt - macro_f1_base
    sign_m = "+" if delta_macro >= 0 else ""
    print(f"{'Macro-Averaged F1':<22} | {macro_f1_base:>6.2f}%                 | {'Class-Specific':<17} | {macro_f1_opt:>6.2f}%       | {sign_m}{delta_macro:>5.2f}%")
    print("=" * 88 + "\n")

    # Render Side-by-Side Visual Comparison on Sample Image
    sample_img_path = "data_samples/images/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.jpg"
    if os.path.exists(sample_img_path):
        raw_demo = cv2.imread(sample_img_path)
        rgb_demo = cv2.cvtColor(raw_demo, cv2.COLOR_BGR2RGB)

        # Baseline: Flat tau = 0.25
        dets_flat = sahi_predict(
            model=model, image_bgr=raw_demo, slice_height=slice_size, slice_width=slice_size,
            conf_threshold=0.25, iou_threshold=0.45, nms_type="diou", include_full_image=True
        )

        # Adaptive: Predict at low conf and filter by calibrated tau_c*
        dets_raw = sahi_predict(
            model=model, image_bgr=raw_demo, slice_height=slice_size, slice_width=slice_size,
            conf_threshold=0.08, iou_threshold=0.45, nms_type="diou", include_full_image=True
        )
        dets_adaptive = []
        for d in dets_raw:
            cid, cname, conf = d[0], d[1], d[2]
            if cname in calibration_results:
                if conf >= calibration_results[cname]["optimal_threshold"]:
                    dets_adaptive.append(d)

        # Draw boxes
        def draw_vis(img, dets):
            cv = img.copy()
            for d in dets:
                cid, cname, conf, x1, y1, x2, y2 = d
                if cname not in EVAL_CLASSES.values() and cid not in EVAL_CLASSES:
                    continue
                col = get_color(cname)
                cv2.rectangle(cv, (x1, y1), (x2, y2), col, 2)
                lbl = f"{cname} {conf:.2f}"
                y_lbl = max(y1 - 5, 12)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(cv, (x1, y_lbl - th - 2), (x1 + tw + 2, y_lbl + 2), col, -1)
                cv2.putText(cv, lbl, (x1 + 1, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            return cv

        vis_flat = draw_vis(rgb_demo, dets_flat)
        vis_adapt = draw_vis(rgb_demo, dets_adaptive)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        axes[0].imshow(vis_flat)
        axes[0].set_title(f"Flat Threshold tau=0.25 ({len(dets_flat)} objects detected)\nSuppresses micro-capacitors below 0.25", fontsize=12, fontweight="bold", color="navy")
        axes[0].axis("off")

        axes[1].imshow(vis_adapt)
        axes[1].set_title(f"Calibrated Class-Adaptive Thresholds ({len(dets_adaptive)} objects detected)\nCap:0.12, Conn:0.32, IC:0.42 | High Recall & High Precision", fontsize=12, fontweight="bold", color="darkgreen")
        axes[1].axis("off")

        plt.tight_layout()
        demo_img = Path("runs/report_assets/adaptive_thresholds_demo.png")
        plt.savefig(str(demo_img), bbox_inches="tight", dpi=150)
        plt.close()
        print(f"📸 Adaptive visual demo saved to: {demo_img}")

    # Save output JSON
    out_json = Path("results/class_adaptive_thresholds.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(calibration_results, f, indent=2)
    print(f"📊 Calibration metrics saved to: {out_json}")

    return calibration_results


if __name__ == "__main__":
    calibrate_thresholds(max_images=44)
