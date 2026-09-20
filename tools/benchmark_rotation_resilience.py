#!/usr/bin/env python3
"""
Benchmark: Rotation Resilience of Single-Stage vs Two-Stage OBB Rectification
=============================================================================
Evaluates how component detection accuracy (mAP50, Precision, Recall) behaves
as a PCB is rotated from 0 to 45 degrees.

Compares:
1. Single-Stage Baseline (Direct YOLO26s prediction on tilted board)
2. Two-Stage Pipeline (YOLOv11n-OBB Rectification -> YOLO26s -> Inverse Homography)

Outputs:
- runs/two_stage_benchmark/rotation_comparison.png (mAP vs Angle curve)
- runs/two_stage_benchmark/metrics_summary.json
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from tools.two_stage_pcb_detector import TwoStagePCBDetector, ensure_obb_weights


def compute_iou_polygon(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Computes IoU between two 4-vertex convex polygons or bounding boxes."""
    # Convert polygons to min/max bounding boxes for fast vectorized IoU or use cv2
    x1_min, y1_min = poly1[:, 0].min(), poly1[:, 1].min()
    x1_max, y1_max = poly1[:, 0].max(), poly1[:, 1].max()

    x2_min, y2_min = poly2[:, 0].min(), poly2[:, 1].min()
    x2_max, y2_max = poly2[:, 0].max(), poly2[:, 1].max()

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0.0, inter_xmax - inter_xmin) * max(0.0, inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


from tools.sahi_pcb_inference import parse_label_file, NAME_TO_UNIFIED_ID, UNIFIED_CLASSES


def load_ground_truth(label_file: Path, img_w: int, img_h: int):
    """Loads YOLO format labels into bounding boxes [cls, polygon]."""
    boxes = []
    parsed = parse_label_file(label_file, img_w, img_h)
    for norm_cid, cname, x1, y1, x2, y2 in parsed:
        boxes.append((norm_cid, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)))
    return boxes


def rotate_sample(img: np.ndarray, gt_boxes: list, angle_deg: float):
    """Rotates image and ground truth 4-vertex polygons around center."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Calculate new bounding dimensions
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M_rot[0, 2] += (new_w / 2.0) - cx
    M_rot[1, 2] += (new_h / 2.0) - cy

    rotated_img = cv2.warpAffine(img, M_rot, (new_w, new_h), borderValue=[114, 114, 114])

    rotated_gt = []
    for cls_id, poly in gt_boxes:
        # poly: (4, 2)
        poly_ones = np.hstack([poly, np.ones((4, 1), dtype=np.float32)])
        rot_poly = (M_rot @ poly_ones.T).T
        rotated_gt.append((cls_id, rot_poly))

    return rotated_img, rotated_gt


def evaluate_detections(pred_polys, pred_clses, pred_scores, gt_boxes, iou_thresh=0.5):
    """Evaluates Precision and Recall at specified IoU threshold."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_polys) == 0 else 0.0, 1.0

    matched_gt = set()
    tp = 0
    fp = 0

    # Sort predictions by confidence
    if len(pred_scores) > 0:
        sort_indices = np.argsort(-pred_scores)
        pred_polys = [pred_polys[i] for i in sort_indices]
        pred_clses = [pred_clses[i] for i in sort_indices]
        pred_scores = [pred_scores[i] for i in sort_indices]

    for p_poly, p_cls, p_score in zip(pred_polys, pred_clses, pred_scores):
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, (g_cls, g_poly) in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            if p_cls == g_cls:
                iou = compute_iou_polygon(p_poly, g_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


def main():
    angles = [0, 15, 30, 45]
    print(f"🚀 Running Rotation Resilience Benchmark across angles: {angles} degrees...")

    obb_weights = ensure_obb_weights(Path("weights/pcb_obb_yolov11n.pt"))
    comp_weights = "runs/yolov26s/pcb-filtered/weights/best.pt"

    detector = TwoStagePCBDetector(
        obb_weights=str(obb_weights),
        comp_weights=comp_weights,
        obb_conf=0.25,
        comp_conf=0.15,
        pad_ratio=0.10,
    )
    single_model = YOLO(comp_weights)

    img_dir = Path("data_samples/images")
    label_dir = Path("data_samples/labels")
    sample_files = sorted(img_dir.glob("*.jpg"))[:12]

    results = {
        "angles": angles,
        "single_stage": {"precision": [], "recall": [], "f1": []},
        "two_stage": {"precision": [], "recall": [], "f1": []},
    }

    for angle in angles:
        print(f"\n--- Testing Angle: {angle}° ---")
        ss_precisions, ss_recalls, ss_f1s = [], [], []
        ts_precisions, ts_recalls, ts_f1s = [], [], []

        for img_path in sample_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            label_file = label_dir / f"{img_path.stem}.txt"
            gt = load_ground_truth(label_file, w, h)
            if not gt:
                continue

            rot_img, rot_gt = rotate_sample(img, gt, angle)

            # 1. Single-Stage Evaluation (Direct prediction on rotated frame)
            res_ss = single_model.predict(rot_img, conf=0.15, verbose=False)[0]
            boxes_ss = res_ss.boxes.xyxy.cpu().numpy()
            scores_ss = res_ss.boxes.conf.cpu().numpy()
            clses_ss = [NAME_TO_UNIFIED_ID.get(single_model.names.get(int(c), ""), -1) for c in res_ss.boxes.cls.cpu().numpy()]

            ss_polys = []
            for b in boxes_ss:
                x1, y1, x2, y2 = b
                ss_polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))

            p_ss, r_ss, f1_ss = evaluate_detections(ss_polys, clses_ss, scores_ss, rot_gt)
            ss_precisions.append(p_ss)
            ss_recalls.append(r_ss)
            ss_f1s.append(f1_ss)

            # 2. Two-Stage Evaluation (OBB Rectification + Inverse Projection)
            pipe_res = detector.predict_full_pipeline(rot_img)
            ts_names = [detector.comp_detector.model.names.get(int(c), "") if hasattr(detector.comp_detector, 'model') else "" for c in pipe_res["comp_classes"]]
            clses_ts = [NAME_TO_UNIFIED_ID.get(n, int(c)) for n, c in zip(ts_names, pipe_res["comp_classes"])]
            p_ts, r_ts, f1_ts = evaluate_detections(
                pipe_res["comp_polys_original"],
                clses_ts,
                pipe_res["comp_scores"],
                rot_gt,
            )
            ts_precisions.append(p_ts)
            ts_recalls.append(r_ts)
            ts_f1s.append(f1_ts)

        mean_ss_f1 = float(np.mean(ss_f1s))
        mean_ts_f1 = float(np.mean(ts_f1s))
        print(f"  Single-Stage: F1 = {mean_ss_f1:.3f} (P={np.mean(ss_precisions):.3f}, R={np.mean(ss_recalls):.3f})")
        print(f"  Two-Stage:    F1 = {mean_ts_f1:.3f} (P={np.mean(ts_precisions):.3f}, R={np.mean(ts_recalls):.3f})")

        results["single_stage"]["precision"].append(float(np.mean(ss_precisions)))
        results["single_stage"]["recall"].append(float(np.mean(ss_recalls)))
        results["single_stage"]["f1"].append(mean_ss_f1)

        results["two_stage"]["precision"].append(float(np.mean(ts_precisions)))
        results["two_stage"]["recall"].append(float(np.mean(ts_recalls)))
        results["two_stage"]["f1"].append(mean_ts_f1)

    # Plot Comparison Curves
    out_dir = Path("runs/two_stage_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # F1 Score comparison
    ax1.plot(angles, results["single_stage"]["f1"], "r-o", linewidth=2.5, label="Single-Stage YOLO26s (Standard)")
    ax1.plot(angles, results["two_stage"]["f1"], "g-s", linewidth=2.5, label="Two-Stage OBB Rectification + YOLO26s")
    ax1.set_title("Detection F1 Score vs. PCB Tilt Angle", fontsize=12, fontweight="bold")
    ax1.set_xlabel("PCB Rotation Angle (Degrees)", fontsize=11)
    ax1.set_ylabel("F1 Score (IoU=0.5)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Precision & Recall comparison
    ax2.plot(angles, results["single_stage"]["precision"], "r--o", label="Single-Stage Precision")
    ax2.plot(angles, results["single_stage"]["recall"], "r:^", label="Single-Stage Recall")
    ax2.plot(angles, results["two_stage"]["precision"], "g--s", label="Two-Stage Precision")
    ax2.plot(angles, results["two_stage"]["recall"], "g:v", label="Two-Stage Recall")
    ax2.set_title("Precision & Recall Resilience", fontsize=12, fontweight="bold")
    ax2.set_xlabel("PCB Rotation Angle (Degrees)", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plot_path = out_dir / "rotation_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    json_path = out_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Benchmark completed!")
    print(f"📊 Saved comparison plot to: {plot_path}")
    print(f"📄 Saved metrics JSON to:    {json_path}")


if __name__ == "__main__":
    main()
