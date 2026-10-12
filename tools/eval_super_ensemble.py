#!/usr/bin/env python3
"""
High-Resolution Super-Ensemble Evaluator for Dense PCB Components
==================================================================
Fuses predictions from top complementary models (e.g. Retinex Champion,
Learnable Stem, Loss Reweighted, SuperYOLO) using Weighted Box Fusion (WBF)
to maximize micro-capacitor recall while preserving macro-connector precision.

Features:
- Self-contained vectorized WBF algorithm (no mandatory pip install required)
- Evaluates directly on unified 4-class test split at 1280px
- Supports model weighting (e.g. higher weight for capacitor specialist)
- Computes exact mAP@50, mAP@50-95, per-class AP, Precision, Recall, and FEI
- Saves JSON summary for automatic inclusion in comparison_table.md

Usage:
    python tools/eval_super_ensemble.py \
        --data datasets/pcb-retinex-cappaste-1280/data.yaml \
        --split test \
        --imgsz 1280 \
        --max-det 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure workspace root is in sys.path so custom modules (retinex_stem) unpickle properly
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

try:
    import retinex_stem
except ImportError:
    pass

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Super-Ensemble with Weighted Box Fusion")
    p.add_argument("--data", type=Path, default=Path("datasets/pcb-retinex-cappaste-1280/data.yaml"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--weights", type=Path, nargs="+", default=None, help="Explicit list of model .pt paths")
    p.add_argument("--model-weights", type=float, nargs="+", default=None, help="Weights for each model in WBF")
    p.add_argument("--iou-thr", type=float, default=0.55, help="WBF clustering IoU threshold")
    p.add_argument("--skip-box-thr", type=float, default=0.001, help="Min score to consider box for WBF")
    p.add_argument("--eval-conf", type=float, default=0.001, help="Final confidence threshold for PR curve")
    p.add_argument("--eval-iou", type=float, default=0.50, help="Matching IoU for mAP@50")
    p.add_argument("--max-det", type=int, default=1000, help="Max output detections per image")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--run-key", type=str, default="ensemble_super_wbf_1280")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Standalone Vectorized Weighted Box Fusion (WBF)
# ---------------------------------------------------------------------------
def compute_iou(box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between a single box [x1, y1, x2, y2] and an array of boxes."""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    areas2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + areas2 - inter
    return inter / np.maximum(union, 1e-9)


def standalone_wbf(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: List[float] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standalone implementation of Weighted Box Fusion (Zabolotniuk et al., 2021).
    Fuses multiple detection sets into a unified, high-confidence bounding box set.
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    # Flatten and filter boxes
    all_boxes = []
    all_scores = []
    all_labels = []

    for m_idx in range(n_models):
        b = boxes_list[m_idx]
        s = scores_list[m_idx]
        l = labels_list[m_idx]
        w = weights[m_idx]

        if len(b) == 0:
            continue

        keep = s >= skip_box_thr
        b = b[keep]
        s = s[keep] * w
        l = l[keep]

        for i in range(len(b)):
            all_boxes.append(b[i])
            all_scores.append(s[i])
            all_labels.append(l[i])

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    unique_classes = np.unique(all_labels)
    final_boxes = []
    final_scores = []
    final_labels = []

    for cls in unique_classes:
        mask = all_labels == cls
        c_boxes = all_boxes[mask]
        c_scores = all_scores[mask]

        order = np.argsort(-c_scores)
        c_boxes = c_boxes[order]
        c_scores = c_scores[order]

        clusters_boxes = []
        clusters_scores = []

        for b, s in zip(c_boxes, c_scores):
            matched = False
            for c_idx in range(len(clusters_boxes)):
                ious = compute_iou(b, np.array([clusters_boxes[c_idx][-1]]))
                if ious[0] > iou_thr:
                    clusters_boxes[c_idx].append(b)
                    clusters_scores[c_idx].append(s)
                    matched = True
                    break
            if not matched:
                clusters_boxes.append([b])
                clusters_scores.append([s])

        # Compute weighted average for each cluster
        for b_cluster, s_cluster in zip(clusters_boxes, clusters_scores):
            b_arr = np.array(b_cluster)
            s_arr = np.array(s_cluster)
            total_weight = np.sum(s_arr)

            fused_box = np.sum(b_arr * s_arr[:, None], axis=0) / np.maximum(total_weight, 1e-9)
            # Standard WBF scoring: average score scaled by model consensus
            fused_score = (total_weight / len(b_arr)) * min(1.0, len(b_arr) / float(n_models))

            final_boxes.append(fused_box)
            final_scores.append(fused_score)
            final_labels.append(cls)

    if not final_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    final_boxes = np.array(final_boxes)
    final_scores = np.array(final_scores)
    final_labels = np.array(final_labels)

    sort_idx = np.argsort(-final_scores)
    return final_boxes[sort_idx], final_scores[sort_idx], final_labels[sort_idx]


# ---------------------------------------------------------------------------
# Metric Evaluation
# ---------------------------------------------------------------------------
def load_dataset_split(data_yaml: Path, split: str) -> Tuple[Path, Path, Dict[int, str]]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    root_path = Path(cfg.get("path", data_yaml.parent))
    split_key = "val" if split in ("val", "valid") else "test"
    split_rel = cfg.get(split_key, f"{split_key}/images")

    img_dir = root_path / split_rel if not Path(split_rel).is_absolute() else Path(split_rel)
    if not img_dir.exists():
        img_dir = data_yaml.parent / split_key / "images"

    lbl_dir = img_dir.parent / "labels" if img_dir.name == "images" else img_dir.parent / f"{split_key}/labels"

    names = cfg.get("names", {})
    if isinstance(names, list):
        names_dict = {i: name for i, name in enumerate(names)}
    else:
        names_dict = {int(k): v for k, v in names.items()}

    return img_dir, lbl_dir, names_dict


def parse_labels(lbl_path: Path, w: int, h: int) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            boxes.append((cid, x1, y1, x2, y2))
    return boxes


def compute_ap(preds_by_class: Dict[int, List], gts_by_class: Dict[int, List], class_ids: List[int], iou_thresh: float = 0.5) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    ap_res = {}
    prec_res = {}
    rec_res = {}

    for cid in class_ids:
        preds = sorted(preds_by_class.get(cid, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cid, [])
        n_gt = len(gts)
        if n_gt == 0:
            ap_res[cid] = 0.0
            prec_res[cid] = 0.0
            rec_res[cid] = 0.0
            continue

        gt_by_img = {}
        for idx, (img_id, *box) in enumerate(gts):
            gt_by_img.setdefault(img_id, []).append((idx, np.array(box)))

        matched = {}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, (img_id, score, *box) in enumerate(preds):
            candidates = gt_by_img.get(img_id, [])
            best_iou, best_idx = 0.0, -1
            box_np = np.array(box)

            for gidx, gbox in candidates:
                if (img_id, gidx) in matched:
                    continue
                iou = compute_iou(box_np, gbox[None])[0]
                if iou > best_iou:
                    best_iou, best_idx = iou, gidx

            if best_iou >= iou_thresh and best_idx != -1:
                tp[i] = 1
                matched[(img_id, best_idx)] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for j in range(len(mpre) - 2, -1, -1):
            mpre[j] = max(mpre[j], mpre[j + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        ap_res[cid] = ap
        if len(tp) > 0:
            f1_curve = 2 * precision * recall / (precision + recall + 1e-9)
            best_f1_idx = int(np.argmax(f1_curve))
            prec_res[cid] = float(precision[best_f1_idx])
            rec_res[cid] = float(recall[best_f1_idx])
        else:
            prec_res[cid] = 0.0
            rec_res[cid] = 0.0

    return ap_res, prec_res, rec_res


def main():
    args = parse_args()

    # 1. Candidate models to ensemble
    if args.weights is None:
        candidates = [
            Path("runs/yolov26s_ultimate_retinex_copypaste_1280/pcb-filtered/weights/best.pt"),
            Path("runs/yolov26s_retinex_stem_copypaste_1280/pcb-filtered/weights/best.pt"),
            Path("runs/yolov26s_p2_warmstart_retinex_copypaste_1280/pcb-filtered/weights/best.pt"),
            Path("runs/yolov26s_retinex_loss_reweight_1280/pcb-filtered/weights/best.pt"),
        ]
        args.weights = [c for c in candidates if c.exists()]

    assert len(args.weights) >= 2, f"At least 2 model checkpoints required for ensemble, found {len(args.weights)}: {args.weights}"

    img_dir, lbl_dir, class_names = load_dataset_split(args.data, args.split)
    class_ids = sorted(class_names.keys())

    image_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    assert len(image_paths) > 0, f"No images found in {img_dir}"

    print("=" * 75)
    print("🚀 HIGH-RESOLUTION SUPER-ENSEMBLE EVALUATION (WEIGHTED BOX FUSION)")
    print(f"Dataset:      {args.data} ({args.split} split, {len(image_paths)} images)")
    print(f"Models ({len(args.weights)}):")
    for w in args.weights:
        print(f"  - {w}")
    print(f"Fusion Params: IoU Thr={args.iou_thr}, Skip Box Thr={args.skip_box_thr}, Max Det={args.max_det}")
    print("=" * 75)

    # 2. Load models
    models = [YOLO(str(w)) for w in args.weights]

    gts = {c: [] for c in class_ids}
    preds_fused = {c: [] for c in class_ids}
    total_infer_time = 0.0

    # 3. Predict & Fuse
    for img_id, img_path in enumerate(tqdm(image_paths, desc="Ensembling")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth
        lbl_file = lbl_dir / (img_path.stem + ".txt")
        for cid, x1, y1, x2, y2 in parse_labels(lbl_file, w, h):
            if cid in gts:
                gts[cid].append((img_id, x1, y1, x2, y2))

        # Model predictions
        boxes_list = []
        scores_list = []
        labels_list = []

        t0 = time.perf_counter()
        for model in models:
            res = model.predict(img, imgsz=args.imgsz, conf=args.eval_conf, device=args.device, verbose=False)[0]
            b_np = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.empty((0, 4))
            s_np = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else np.empty(0)
            l_np = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) > 0 else np.empty(0)

            boxes_list.append(b_np)
            scores_list.append(s_np)
            labels_list.append(l_np)

        # Weighted Box Fusion
        f_boxes, f_scores, f_labels = standalone_wbf(
            boxes_list,
            scores_list,
            labels_list,
            weights=args.model_weights,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr,
        )
        total_infer_time += (time.perf_counter() - t0)

        # Cap at max-det
        if len(f_boxes) > args.max_det:
            f_boxes = f_boxes[:args.max_det]
            f_scores = f_scores[:args.max_det]
            f_labels = f_labels[:args.max_det]

        for bx, sc, lb in zip(f_boxes, f_scores, f_labels):
            cid = int(lb)
            if cid in preds_fused:
                preds_fused[cid].append((img_id, float(sc), bx[0], bx[1], bx[2], bx[3]))

    # 4. Compute metrics
    n_images = len(image_paths)
    fps = n_images / total_infer_time if total_infer_time > 0 else 0.0
    latency_ms = (total_infer_time * 1000.0) / n_images if n_images > 0 else 0.0

    ap_dict, prec_dict, rec_dict = compute_ap(preds_fused, gts, class_ids, iou_thresh=args.eval_iou)
    map50 = float(np.mean(list(ap_dict.values())))
    mean_p = float(np.mean(list(prec_dict.values())))
    mean_r = float(np.mean(list(rec_dict.values())))
    f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r + 1e-9)
    fei = f1 * np.log10(max(fps, 1.01))

    # Compute mAP50-95
    map_list = []
    for thresh in np.linspace(0.50, 0.95, 10):
        ap_t, _, _ = compute_ap(preds_fused, gts, class_ids, iou_thresh=float(thresh))
        map_list.append(np.mean(list(ap_t.values())))
    map50_95 = float(np.mean(map_list))

    print("\n" + "=" * 75)
    print(f"🏆 SUPER-ENSEMBLE BENCHMARK RESULTS ({args.run_key})")
    print("=" * 75)
    for cid in class_ids:
        cname = class_names[cid]
        print(f"  {cname:<24} | AP@50: {ap_dict[cid]*100:>6.2f}% | P: {prec_dict[cid]*100:>5.1f}% | R: {rec_dict[cid]*100:>5.1f}%")
    print("-" * 75)
    print(f"  {'mAP@50 (Overall)':<24} | {map50*100:>6.2f}%")
    print(f"  {'mAP@50-95':<24} | {map50_95*100:>6.2f}%")
    print(f"  {'Precision':<24} | {mean_p*100:>6.2f}%")
    print(f"  {'Recall':<24} | {mean_r*100:>6.2f}%")
    print(f"  {'Macro-F1':<24} | {f1:>6.4f}")
    print(f"  {'Throughput (FPS)':<24} | {fps:>6.1f} FPS ({latency_ms:.1f} ms/board)")
    print(f"  {'Frontier Efficiency':<24} | {fei:>6.4f}")
    print("=" * 75 + "\n")

    summary = {
        "model": args.run_key,
        "weights": " + ".join(w.name for w in args.weights),
        "ensemble_models": [str(w) for w in args.weights],
        "imgsz": args.imgsz,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": mean_p,
        "recall": mean_r,
        "f1": f1,
        "fps": fps,
        "total_time_ms": latency_ms,
        "per_class_ap50": {class_names[c]: ap_dict[c] for c in class_ids},
        "max_det": args.max_det,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"{args.run_key}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved ensemble JSON to: {out_json}")

    # Mirror to shared project results directory
    shared_results_dir = Path("/mnt/weka/etadevosyan/pcb-yolo/results")
    if shared_results_dir.exists() and args.results_dir.resolve() != shared_results_dir.resolve():
        shared_json = shared_results_dir / f"{args.run_key}.json"
        with open(shared_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Appended run to shared project results directory: {shared_json}")


if __name__ == "__main__":
    main()
