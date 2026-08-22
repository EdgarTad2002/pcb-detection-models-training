#!/usr/bin/env python3
"""
Ensemble fusion evaluation: NMS, Affirmative/Consensus/Unanimous voting, and
WBF, run over your own trained checkpoints on the ORIGINAL test split, under
the same conf=0.25/iou=0.5/4-class protocol as every other run in this
project.

Ranks models by their already-computed test-set mAP50 (reads results/*.json
-- no need to re-run individual model eval), then builds Top-2/3/4/6
ensembles, mirroring the paper's own approach.

Requires: pip install ensemble-boxes

Usage:
    python ensemble_eval.py --top-n 2 3 4 6
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ensemble_boxes import nms as wbf_nms, weighted_boxes_fusion

DEFAULT_CLASSES = [2, 4, 7, 9]
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
RAW_ID_TO_LOCAL = {raw: i for i, raw in enumerate(DEFAULT_CLASSES)}  # 2->0, 4->1, 7->2, 9->3

# All candidate checkpoints -- ranking picks the actual Top-N automatically
# from whichever of these have a results/<key>.json already.
MODEL_CHECKPOINTS = {
    "yolov5s": "runs/yolov5s/pcb-filtered/weights/best.pt",
    "yolov8s": "runs/yolov8s/pcb-filtered/weights/best.pt",
    "yolov9s": "runs/yolov9s/pcb-filtered/weights/best.pt",
    "yolov10s": "runs/yolov10s/pcb-filtered/weights/best.pt",
    "yolov11s": "runs/yolov11s/pcb-filtered/weights/best.pt",
    "yolov12s": "runs/yolov12s/pcb-filtered/weights/best.pt",
    "yolov26s": "runs/yolov26s/pcb-filtered/weights/best.pt",
    "yolov26s_native_res": "runs/yolov26s_native_res/pcb-filtered/weights/best.pt",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--results-dir", type=Path, default=Path("/mnt/weka/etadevosyan/pcb-yolo/results"))
    p.add_argument("--test-images", type=Path, default=Path("datasets/pcb-filtered-yolov8/test/images"))
    p.add_argument("--test-labels", type=Path, default=Path("datasets/pcb-filtered-yolov8/test/labels"))
    p.add_argument("--top-n", type=int, nargs="+", default=[2, 3, 4, 6])
    p.add_argument("--conf", type=float, default=0.25, help="Fixed threshold for the reported precision/recall operating point")
    p.add_argument("--inference-conf", type=float, default=0.001, help="Low threshold used when collecting raw candidates for fusion")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for AP matching and NMS/WBF/voting grouping")
    p.add_argument("--device", default="0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model ranking (reuse existing results, don't recompute)
# ---------------------------------------------------------------------------
def rank_models(args):
    ranked = []
    for key in MODEL_CHECKPOINTS:
        result_path = args.results_dir / f"{key}.json"
        weight_path = args.project_root / MODEL_CHECKPOINTS[key]
        if not result_path.exists() or not weight_path.exists():
            continue
        with open(result_path) as f:
            data = json.load(f)
        ranked.append((key, data["mAP50"], weight_path))
    ranked.sort(key=lambda x: -x[1])
    print("Model ranking (by existing test-set mAP50):")
    for key, m, _ in ranked:
        print(f"  {key}: {m:.4f}")
    return ranked


# ---------------------------------------------------------------------------
# Inference collection (cached to disk per model, run once)
# ---------------------------------------------------------------------------
def collect_predictions(model_key, weight_path, img_paths, cache_dir, conf, device):
    cache_path = cache_dir / f"{model_key}_preds.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    print(f"Running inference: {model_key}")
    model = YOLO(str(weight_path))
    preds_by_image = {}
    for img_path in img_paths:
        results = model.predict(str(img_path), conf=conf, verbose=False, device=device)
        preds = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            w, h = r.orig_shape[1], r.orig_shape[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(boxes, scores, classes):
                if cls not in DEFAULT_CLASSES:
                    continue
                # normalize to [0,1] -- ensemble_boxes convention
                preds.append({
                    "box_norm": [box[0] / w, box[1] / h, box[2] / w, box[3] / h],
                    "score": float(score),
                    "class_local": RAW_ID_TO_LOCAL[int(cls)],
                    "w": w, "h": h,
                })
        preds_by_image[img_path.stem] = preds

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(preds_by_image, f, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    return preds_by_image


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------
def fuse_nms(boxes_list, scores_list, labels_list, iou_thr):
    return wbf_nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)


def fuse_wbf(boxes_list, scores_list, labels_list, iou_thr):
    return weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr)


def fuse_voting(boxes_list, scores_list, labels_list, iou_thr, strategy, num_models):
    """Affirmative (>=1), Consensus (>ceil(m/2)), Unanimous (==m) agreement voting."""
    all_boxes, all_scores, all_labels = [], [], []
    for m_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for b, s, l in zip(boxes, scores, labels):
            all_boxes.append(list(b))
            all_scores.append(s)
            all_labels.append(l)
            all_boxes[-1].append(m_idx)  # tag with source model index temporarily

    used = [False] * len(all_boxes)
    out_boxes, out_scores, out_labels = [], [], []

    def iou(a, b):
        ax1, ay1, ax2, ay2, _ = a
        bx1, by1, bx2, by2, _ = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
        inter = iw * ih
        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    for i in range(len(all_boxes)):
        if used[i]:
            continue
        group_idx = [i]
        used[i] = True
        for j in range(i + 1, len(all_boxes)):
            if used[j] or all_labels[j] != all_labels[i]:
                continue
            if iou(all_boxes[i], all_boxes[j]) >= iou_thr:
                group_idx.append(j)
                used[j] = True

        contributing_models = set(all_boxes[k][4] for k in group_idx)
        n_votes = len(contributing_models)

        if strategy == "affirmative":
            keep = n_votes >= 1
        elif strategy == "consensus":
            keep = n_votes >= math.ceil(num_models / 2)
        elif strategy == "unanimous":
            keep = n_votes == num_models
        else:
            raise ValueError(strategy)

        if not keep:
            continue

        total_conf = sum(all_scores[k] for k in group_idx)
        avg_box = [0.0, 0.0, 0.0, 0.0]
        for k in group_idx:
            w = all_scores[k] / total_conf
            for d in range(4):
                avg_box[d] += all_boxes[k][d] * w
        out_boxes.append(avg_box)
        out_scores.append(total_conf / len(group_idx))
        out_labels.append(all_labels[i])

    return np.array(out_boxes) if out_boxes else np.zeros((0, 4)), np.array(out_scores), np.array(out_labels)


# ---------------------------------------------------------------------------
# AP computation (same protocol as eval_sahi.py, for consistency project-wide)
# ---------------------------------------------------------------------------
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


def summarize_and_save(run_key, preds_by_class, gts_by_class, args, extra_meta=None):
    per_class = compute_ap_and_pr(preds_by_class, gts_by_class, args.iou, args.conf)
    summary = {
        "model": run_key,
        "mAP50": float(np.mean([r["ap50"] for r in per_class.values()])),
        "mAP50_95": None,
        "precision": float(np.mean([r["precision"] for r in per_class.values()])),
        "recall": float(np.mean([r["recall"] for r in per_class.values()])),
        "total_time_ms": None,
        "fps": None,
        "per_class_ap50": {CLASS_NAMES[RAW_ID_TO_LOCAL[c]]: r["ap50"] for c, r in per_class.items()},
        "eval_conf": args.conf,
        "eval_iou": args.iou,
        "eval_split": "test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        summary.update(extra_meta)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / f"{run_key}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{run_key}: mAP50={summary['mAP50']:.4f}  P={summary['precision']:.4f}  R={summary['recall']:.4f}")
    for name, ap in summary["per_class_ap50"].items():
        print(f"    {name}: {ap:.4f}")
    return summary


def main():
    args = parse_args()
    img_paths = sorted(list(args.test_images.glob("*.jpg")) + list(args.test_images.glob("*.png")))
    gts_by_class = load_ground_truth(img_paths, args.test_labels)

    ranked = rank_models(args)
    if len(ranked) < 2:
        print("Need at least 2 models with existing results to ensemble. Aborting.")
        return

    cache_dir = args.project_root / "ensemble_cache"
    all_preds = {}
    for key, _, weight_path in ranked:
        all_preds[key] = collect_predictions(key, weight_path, img_paths, cache_dir, args.inference_conf, args.device)

    for n in args.top_n:
        if n > len(ranked):
            continue
        model_keys = [k for k, _, _ in ranked[:n]]
        print(f"\n{'='*70}\nTop-{n}: {model_keys}\n{'='*70}")

        for strategy_name, fn_type in [
            ("nms", "nms"), ("wbf", "wbf"),
            ("affirmative", "vote"), ("consensus", "vote"), ("unanimous", "vote"),
        ]:
            preds_by_class = {c: [] for c in DEFAULT_CLASSES}
            for img_path in img_paths:
                img_id = img_path.stem
                boxes_list, scores_list, labels_list = [], [], []
                w = h = None
                for key in model_keys:
                    img_preds = all_preds[key].get(img_id, [])
                    if not img_preds:
                        boxes_list.append([]); scores_list.append([]); labels_list.append([])
                        continue
                    w, h = img_preds[0]["w"], img_preds[0]["h"]
                    boxes_list.append([p["box_norm"] for p in img_preds])
                    scores_list.append([p["score"] for p in img_preds])
                    labels_list.append([p["class_local"] for p in img_preds])

                if w is None or all(len(b) == 0 for b in boxes_list):
                    continue

                if fn_type == "nms":
                    boxes, scores, labels = fuse_nms(boxes_list, scores_list, labels_list, args.iou)
                elif fn_type == "wbf":
                    boxes, scores, labels = fuse_wbf(boxes_list, scores_list, labels_list, args.iou)
                else:
                    boxes, scores, labels = fuse_voting(boxes_list, scores_list, labels_list, args.iou, strategy_name, n)

                for b, s, l in zip(boxes, scores, labels):
                    raw_cls = DEFAULT_CLASSES[int(l)]
                    preds_by_class[raw_cls].append((img_id, float(s), b[0] * w, b[1] * h, b[2] * w, b[3] * h))

            run_key = f"ensemble_{strategy_name}_top{n}"
            summarize_and_save(run_key, preds_by_class, gts_by_class, args, extra_meta={"ensemble_models": model_keys})


if __name__ == "__main__":
    main()
