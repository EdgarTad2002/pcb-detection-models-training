#!/usr/bin/env python3
"""
Evaluates a model trained on sliced tiles (see slice_dataset.py) using SAHI's
sliced inference on the ORIGINAL full-resolution test images -- this is what
real deployment would look like (slice the board, run the tile-trained model
on each slice, merge results back), and keeps the result comparable to every
other run in this project (same conf=0.25 / iou=0.5 / class-filter protocol,
same JSON schema consumed by aggregate_results.py).

Requires: pip install sahi

Usage:
    python eval_sahi.py \
        --run-key yolov26s_sahi_tiles \
        --weights runs/yolov26s_sahi_tiles/pcb-filtered/weights/best.pt \
        --test-images datasets/pcb-filtered-yolov8/test/images \
        --test-labels datasets/pcb-filtered-yolov8/test/labels \
        --slice-size 640 --overlap 0.2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

DEFAULT_CLASSES = [2, 4, 7, 9]  # raw dataset class ids: Capacitor, Connector, Electrolytic Capacitor, IC
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
RAW_ID_TO_NAME = dict(zip(DEFAULT_CLASSES, CLASS_NAMES))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-key", required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--test-images", type=Path, required=True)
    p.add_argument("--test-labels", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=Path("/mnt/weka/etadevosyan/pcb-yolo/results"))
    p.add_argument("--slice-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching predictions to ground truth")
    p.add_argument("--nms-iou", type=float, default=0.5, help="IoU threshold SAHI uses to merge overlapping slice predictions")
    p.add_argument("--device", default="0")
    return p.parse_args()


def load_yolo_labels(label_path, img_w, img_h):
    """Returns list of (class_id, x1, y1, x2, y2) in absolute pixel coords, filtered to DEFAULT_CLASSES."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls_id = int(parts[0])
        if cls_id not in DEFAULT_CLASSES:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap_and_pr(preds_by_class, gts_by_class, iou_thresh, conf_thresh):
    """
    preds_by_class[cls] = list of (image_id, score, x1,y1,x2,y2), any confidence
    gts_by_class[cls]   = list of (image_id, x1,y1,x2,y2)
    Returns dict: cls -> {ap50, precision_at_conf, recall_at_conf}
    """
    results = {}
    for cls_id in DEFAULT_CLASSES:
        preds = sorted(preds_by_class.get(cls_id, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cls_id, [])
        n_gt = len(gts)

        matched_gt = {}  # image_id -> set of matched gt indices
        gt_by_image = {}
        for idx, (img_id, *box) in enumerate(gts):
            gt_by_image.setdefault(img_id, []).append((idx, box))

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        # for fixed-threshold precision/recall at conf_thresh
        fixed_tp, fixed_fp, fixed_matched = 0, 0, set()

        for i, (img_id, score, *box) in enumerate(preds):
            candidates = gt_by_image.get(img_id, [])
            best_iou, best_idx = 0.0, -1
            for gidx, gbox in candidates:
                key = (img_id, gidx)
                if key in matched_gt:
                    continue
                iou = iou_xyxy(box, gbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, gidx

            is_tp = best_iou >= iou_thresh and best_idx != -1
            if is_tp:
                tp[i] = 1
                matched_gt[(img_id, best_idx)] = True
            else:
                fp[i] = 1

            if score >= conf_thresh:
                key = (img_id, score, tuple(box))
                if is_tp and (img_id, best_idx) not in fixed_matched:
                    fixed_tp += 1
                    fixed_matched.add((img_id, best_idx))
                elif not is_tp:
                    fixed_fp += 1

        # AP via precision-recall integration (all-points interpolation)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = 0.0
        if len(precision) > 0:
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        fixed_fn = n_gt - fixed_tp
        precision_fixed = fixed_tp / (fixed_tp + fixed_fp) if (fixed_tp + fixed_fp) > 0 else 0.0
        recall_fixed = fixed_tp / n_gt if n_gt > 0 else 0.0

        results[cls_id] = {
            "ap50": ap,
            "precision": precision_fixed,
            "recall": recall_fixed,
            "n_gt": n_gt,
        }
    return results


def main():
    args = parse_args()

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(args.weights),
        confidence_threshold=args.conf,
        device=f"cuda:{args.device}" if args.device != "cpu" else "cpu",
    )

    img_paths = sorted(list(args.test_images.glob("*.jpg")) + list(args.test_images.glob("*.png")))
    print(f"Running SAHI sliced inference on {len(img_paths)} test images...")

    preds_by_class = {c: [] for c in DEFAULT_CLASSES}
    gts_by_class = {c: [] for c in DEFAULT_CLASSES}

    total_time = 0.0
    for img_path in img_paths:
        import cv2
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        t0 = time.time()
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=args.nms_iou,
            verbose=0,
        )
        total_time += time.time() - t0

        for obj in result.object_prediction_list:
            cls_id = obj.category.id
            if cls_id not in DEFAULT_CLASSES:
                continue
            box = obj.bbox.to_xyxy()
            preds_by_class[cls_id].append((img_path.stem, obj.score.value, *box))

        label_path = args.test_labels / f"{img_path.stem}.txt"
        gts = load_yolo_labels(label_path, img_w, img_h)
        for cls_id, x1, y1, x2, y2 in gts:
            gts_by_class[cls_id].append((img_path.stem, x1, y1, x2, y2))

    per_class_results = compute_ap_and_pr(preds_by_class, gts_by_class, args.iou, args.conf)

    map50 = float(np.mean([r["ap50"] for r in per_class_results.values()]))
    mean_precision = float(np.mean([r["precision"] for r in per_class_results.values()]))
    mean_recall = float(np.mean([r["recall"] for r in per_class_results.values()]))
    avg_time_ms = (total_time / len(img_paths)) * 1000 if img_paths else 0.0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    per_class_ap50 = {RAW_ID_TO_NAME[c]: r["ap50"] for c, r in per_class_results.items()}

    summary = {
        "model": args.run_key,
        "weights": str(args.weights),
        "mAP50": map50,
        "mAP50_95": None,  # not computed -- would need multi-IoU sweep; see note below
        "precision": mean_precision,
        "recall": mean_recall,
        "total_time_ms": avg_time_ms,
        "fps": fps,
        "per_class_ap50": per_class_ap50,
        "eval_conf": args.conf,
        "eval_iou": args.iou,
        "eval_split": "test (SAHI sliced inference on original full-size images)",
        "slice_size": args.slice_size,
        "slice_overlap": args.overlap,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n--- Overall Metrics (SAHI sliced inference) ---")
    for k, v in summary.items():
        if k != "per_class_ap50":
            print(f"{k}: {v}")
    print("\n--- Per-Class AP@0.5 ---")
    for name, ap in per_class_ap50.items():
        print(f"{name}: {ap:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{args.run_key}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
