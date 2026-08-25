#!/usr/bin/env python3
"""
Context-based re-scoring (post-processing only, no retraining).

Boosts low-confidence Capacitor detections that sit near confidently-detected
anchor components (IC, Connector, Electrolytic Capacitor) -- PCB layouts
aren't random (e.g. decoupling capacitors sit near ICs), so a weak Capacitor
detection near a confident IC is more likely to be real than the same score
in open board space.

Runs inference ONCE on the given checkpoint, then evaluates both the raw
(baseline) and context-rescored predictions on the identical detections --
isolating the rescoring effect itself, not conflated with re-running the model.

Usage:
    python context_rescore.py \
        --weights runs/yolov26s_native_res/pcb-filtered/weights/best.pt \
        --run-key yolov26s_native_res_contextrescore
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

DEFAULT_CLASSES = [2, 4, 7, 9]
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
CAPACITOR_ID = 2
ANCHOR_IDS = [4, 7, 9]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--run-key", required=True)
    p.add_argument("--test-images", type=Path, default=Path("datasets/pcb-filtered-yolov8/test/images"))
    p.add_argument("--test-labels", type=Path, default=Path("datasets/pcb-filtered-yolov8/test/labels"))
    p.add_argument("--results-dir", type=Path, default=Path("/mnt/weka/etadevosyan/pcb-yolo/results"))
    p.add_argument("--inference-conf", type=float, default=0.01)
    p.add_argument("--anchor-conf", type=float, default=0.5)
    p.add_argument("--radius-frac", type=float, default=0.15, help="Search radius as a fraction of image diagonal")
    p.add_argument("--boost-factor", type=float, default=2.5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    return p.parse_args()


def load_ground_truth(img_paths, test_labels):
    import cv2
    gts_by_class = {c: [] for c in DEFAULT_CLASSES}
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_path = test_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id not in DEFAULT_CLASSES:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            gts_by_class[cls_id].append((img_path.stem, x1, y1, x2, y2))
    return gts_by_class


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
    results = {}
    for cls_id in DEFAULT_CLASSES:
        preds = sorted(preds_by_class.get(cls_id, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cls_id, [])
        n_gt = len(gts)

        gt_by_image = {}
        for idx, (img_id, *box) in enumerate(gts):
            gt_by_image.setdefault(img_id, []).append((idx, box))

        matched_gt = {}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
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
                if is_tp and (img_id, best_idx) not in fixed_matched:
                    fixed_tp += 1
                    fixed_matched.add((img_id, best_idx))
                elif not is_tp:
                    fixed_fp += 1

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
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

        precision_fixed = fixed_tp / (fixed_tp + fixed_fp) if (fixed_tp + fixed_fp) > 0 else 0.0
        recall_fixed = fixed_tp / n_gt if n_gt > 0 else 0.0
        results[cls_id] = {"ap50": ap, "precision": precision_fixed, "recall": recall_fixed}
    return results


def summarize(run_key, preds_by_class, gts_by_class, args):
    per_class = compute_ap_and_pr(preds_by_class, gts_by_class, args.iou, args.conf)
    return {
        "model": run_key,
        "mAP50": float(np.mean([r["ap50"] for r in per_class.values()])),
        "mAP50_95": None,
        "precision": float(np.mean([r["precision"] for r in per_class.values()])),
        "recall": float(np.mean([r["recall"] for r in per_class.values()])),
        "total_time_ms": None,
        "fps": None,
        "per_class_ap50": {CLASS_NAMES[DEFAULT_CLASSES.index(c)]: r["ap50"] for c, r in per_class.items()},
        "eval_conf": args.conf,
        "eval_iou": args.iou,
        "eval_split": "test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    args = parse_args()
    img_paths = sorted(list(args.test_images.glob("*.jpg")) + list(args.test_images.glob("*.png")))
    gts_by_class = load_ground_truth(img_paths, args.test_labels)

    model = YOLO(str(args.weights))
    preds_baseline = {c: [] for c in DEFAULT_CLASSES}
    preds_rescored = {c: [] for c in DEFAULT_CLASSES}

    for img_path in img_paths:
        img_id = img_path.stem
        results = model.predict(str(img_path), conf=args.inference_conf, verbose=False, device=args.device)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        w, h = r.orig_shape[1], r.orig_shape[0]
        diag = (w**2 + h**2) ** 0.5
        radius = diag * args.radius_frac

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        anchor_centers = []
        for box, score, cls in zip(boxes, scores, classes):
            if cls in ANCHOR_IDS and score >= args.anchor_conf:
                anchor_centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

        for box, score, cls in zip(boxes, scores, classes):
            if cls not in DEFAULT_CLASSES:
                continue
            x1, y1, x2, y2 = box
            preds_baseline[int(cls)].append((img_id, float(score), x1, y1, x2, y2))

            new_score = float(score)
            if cls == CAPACITOR_ID and anchor_centers:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                min_dist = min(((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 for ax, ay in anchor_centers)
                if min_dist <= radius:
                    new_score = min(1.0, new_score * args.boost_factor)
            preds_rescored[int(cls)].append((img_id, new_score, x1, y1, x2, y2))

    summary_baseline = summarize(f"{args.run_key}_baseline_rerun", preds_baseline, gts_by_class, args)
    summary_rescored = summarize(args.run_key, preds_rescored, gts_by_class, args)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / f"{args.run_key}_baseline_rerun.json", "w") as f:
        json.dump(summary_baseline, f, indent=2)
    with open(args.results_dir / f"{args.run_key}.json", "w") as f:
        json.dump(summary_rescored, f, indent=2)

    print("BASELINE (no rescoring):", json.dumps(summary_baseline, indent=2))
    print("\nCONTEXT-RESCORED:", json.dumps(summary_rescored, indent=2))


if __name__ == "__main__":
    main()
