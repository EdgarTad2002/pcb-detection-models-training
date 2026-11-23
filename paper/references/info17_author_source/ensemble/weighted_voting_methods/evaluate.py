"""
Phase 5: Evaluation Metrics

Compute AP, mAP, precision, recall, F1, per-class AP, and FEI
for the voting ensemble methods.

Works with normalized dict-based predictions from the voting pipeline.
Uses 11-point interpolation AP, consistent with evaluate_ensemble.py.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ensemble.voting_methods.box_utils import compute_iou
from ensemble.voting_methods.config import CLASS_NAMES


def compute_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision at a given IoU threshold (single class).

    Uses 11-point interpolation, matching evaluate_ensemble.py.

    Args:
        predictions: list of {"confidence": float, "box_norm": [x1,y1,x2,y2]}
        ground_truths: list of {"box_norm": [x1,y1,x2,y2]}
        iou_threshold: IoU threshold for a true positive

    Returns:
        AP value
    """
    if len(ground_truths) == 0:
        return 0.0 if len(predictions) > 0 else 1.0

    if len(predictions) == 0:
        return 0.0

    # Sort predictions by confidence descending
    preds_sorted = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    gt_matched = [False] * len(ground_truths)
    tp_list = []
    fp_list = []

    for pred in preds_sorted:
        best_iou = 0.0
        best_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou(pred["box_norm"], gt["box_norm"])
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx

        if best_iou >= iou_threshold and best_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[best_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recalls = tp_cum / len(ground_truths)
    precisions = tp_cum / (tp_cum + fp_cum)

    # Add sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precisions[mask]) / 11
    return ap


def compute_map(
    fused_preds: Dict[str, List[Dict]],
    ground_truth: Dict[str, List[Dict]],
    class_names: List[str] = None,
    iou_thresholds: List[float] = None,
) -> Dict:
    """
    Compute mAP@0.5 and mAP@0.5:0.95.

    Args:
        fused_preds: {image_name: [{"class": int, "confidence": float, "box_norm": [...]}]}
        ground_truth: {image_name: [{"class": int, "box_norm": [...]}]}
        class_names: list of class names
        iou_thresholds: IoU thresholds for mAP computation

    Returns:
        {"mAP50": float, "mAP50_95": float, "per_class_AP": {name: ap}}
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    num_classes = len(class_names)

    # Gather all predictions and GT by class across all images
    all_preds_by_class = defaultdict(list)
    all_gt_by_class = defaultdict(list)

    all_images = sorted(set(list(fused_preds.keys()) + list(ground_truth.keys())))

    for img_name in all_images:
        preds = fused_preds.get(img_name, [])
        gts = ground_truth.get(img_name, [])

        for p in preds:
            all_preds_by_class[p["class"]].append(p)
        for g in gts:
            all_gt_by_class[g["class"]].append(g)

    # Compute AP per class per IoU threshold
    ap_per_iou = {}
    for iou_thresh in iou_thresholds:
        ap_per_iou[iou_thresh] = {}
        for cls_id in range(num_classes):
            # Per-image matching for correct TP/FP computation
            cls_preds_all = []
            cls_gt_counts = 0

            # Collect per-image predictions with image tag for proper matching
            image_pred_groups = []
            image_gt_groups = []

            for img_name in all_images:
                preds = [p for p in fused_preds.get(img_name, []) if p["class"] == cls_id]
                gts = [g for g in ground_truth.get(img_name, []) if g["class"] == cls_id]
                image_pred_groups.append(preds)
                image_gt_groups.append(gts)

            # Flatten preds with image index for proper matching
            indexed_preds = []
            for img_idx, preds in enumerate(image_pred_groups):
                for p in preds:
                    indexed_preds.append((img_idx, p))

            # Sort all predictions by confidence descending
            indexed_preds.sort(key=lambda x: x[1]["confidence"], reverse=True)

            # Track matched GTs per image
            gt_matched = {
                img_idx: [False] * len(gts)
                for img_idx, gts in enumerate(image_gt_groups)
            }
            total_gt = sum(len(gts) for gts in image_gt_groups)

            if total_gt == 0:
                ap_per_iou[iou_thresh][cls_id] = 0.0 if indexed_preds else 1.0
                continue

            if not indexed_preds:
                ap_per_iou[iou_thresh][cls_id] = 0.0
                continue

            tp_list = []
            fp_list = []

            for img_idx, pred in indexed_preds:
                gts = image_gt_groups[img_idx]
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gts):
                    if gt_matched[img_idx][gt_idx]:
                        continue
                    iou = compute_iou(pred["box_norm"], gt["box_norm"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    tp_list.append(1)
                    fp_list.append(0)
                    gt_matched[img_idx][best_gt_idx] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

            tp_cum = np.cumsum(tp_list)
            fp_cum = np.cumsum(fp_list)
            recalls = tp_cum / total_gt
            precisions = tp_cum / (tp_cum + fp_cum)

            recalls = np.concatenate([[0], recalls, [1]])
            precisions = np.concatenate([[1], precisions, [0]])

            ap = 0.0
            for t in np.linspace(0, 1, 11):
                mask = recalls >= t
                if np.any(mask):
                    ap += np.max(precisions[mask]) / 11

            ap_per_iou[iou_thresh][cls_id] = ap

    # mAP@0.5
    map50 = np.mean([ap_per_iou[0.5].get(i, 0.0) for i in range(num_classes)])

    # mAP@0.5:0.95
    map50_95 = np.mean([
        np.mean([ap_per_iou[iou].get(i, 0.0) for i in range(num_classes)])
        for iou in iou_thresholds
    ])

    # Per-class AP at IoU=0.5
    per_class_ap = {}
    for cls_id in range(num_classes):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        per_class_ap[name] = ap_per_iou[0.5].get(cls_id, 0.0)

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class_AP": per_class_ap,
    }


def compute_precision_recall_f1(
    fused_preds: Dict[str, List[Dict]],
    ground_truth: Dict[str, List[Dict]],
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Compute mean per-class Precision, Recall, and F1 at each class's
    F1-optimal confidence threshold — matching YOLO val() / ap_per_class()
    reporting style.

    Algorithm (mirrors run_experiments_full.py ap_per_class):
      1. For each image: greedy confidence-sorted matching determines TP/FP
         per prediction (class-aware, IoU >= iou_threshold).
      2. All TP/conf/class labels are collected across images.
      3. For each class independently: sort by confidence, compute cumulative
         P/R curve, find the threshold that maximises F1.
      4. Return the mean of per-class Precision and Recall at those optima.

    This gives equal weight to every class regardless of GT box count,
    preventing rare classes (e.g. Capacitor with 66 % of GT boxes but low
    AP) from dominating the reported recall.

    Args:
        fused_preds: {image_name: [detections]}  — all confidence levels, unfiltered
        ground_truth: {image_name: [gt_dicts]}
        iou_threshold: IoU threshold for TP

    Returns:
        {"precision": float, "recall": float, "f1": float,
         "total_tp": int, "total_fp": int, "total_fn": int,
         "conf_threshold": float}   ← mean of per-class optimal conf values
    """
    all_images = sorted(set(list(fused_preds.keys()) + list(ground_truth.keys())))

    # --- Step 1: per-image greedy matching → flat TP/conf/class arrays -------
    all_tp: List[int] = []
    all_conf: List[float] = []
    all_pred_cls: List[int] = []
    all_gt_cls: List[int] = []

    for img in all_images:
        preds = sorted(fused_preds.get(img, []), key=lambda x: x["confidence"], reverse=True)
        gts = list(ground_truth.get(img, []))

        for g in gts:
            all_gt_cls.append(g["class"])

        if not preds:
            continue

        gt_matched = [False] * len(gts)
        for pred in preds:
            best_iou = 0.0
            best_idx = -1
            for gi, gt in enumerate(gts):
                if gt["class"] != pred["class"] or gt_matched[gi]:
                    continue
                iou = compute_iou(pred["box_norm"], gt["box_norm"])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi

            is_tp = int(best_iou >= iou_threshold and best_idx >= 0)
            all_tp.append(is_tp)
            all_conf.append(pred["confidence"])
            all_pred_cls.append(pred["class"])
            if is_tp:
                gt_matched[best_idx] = True

    total_gt = len(all_gt_cls)
    if total_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "total_tp": 0, "total_fp": 0, "total_fn": 0,
                "conf_threshold": 0.0}

    total_tp_raw = int(sum(all_tp))
    total_fp_raw = len(all_tp) - total_tp_raw
    total_fn_raw = total_gt - total_tp_raw

    if not all_tp:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "total_tp": 0, "total_fp": 0, "total_fn": total_gt,
                "conf_threshold": 0.0}

    # Sort all predictions globally by confidence (descending)
    tp_arr = np.array(all_tp, dtype=np.float32)
    conf_arr = np.array(all_conf, dtype=np.float32)
    pred_cls_arr = np.array(all_pred_cls, dtype=np.int32)
    gt_cls_arr = np.array(all_gt_cls, dtype=np.int32)

    sort_idx = np.argsort(-conf_arr)
    tp_sorted = tp_arr[sort_idx]
    conf_sorted = conf_arr[sort_idx]
    pred_cls_sorted = pred_cls_arr[sort_idx]

    # --- Step 2: per-class P/R curve → F1-optimal operating point -----------
    unique_classes = np.unique(gt_cls_arr)
    per_class_p: List[float] = []
    per_class_r: List[float] = []
    per_class_conf: List[float] = []

    for c in unique_classes:
        n_gt = int((gt_cls_arr == c).sum())
        cls_mask = pred_cls_sorted == c
        n_p = int(cls_mask.sum())

        if n_p == 0 or n_gt == 0:
            per_class_p.append(0.0)
            per_class_r.append(0.0)
            per_class_conf.append(0.0)
            continue

        tp_c = tp_sorted[cls_mask].cumsum()
        fp_c = (1.0 - tp_sorted[cls_mask]).cumsum()
        recall_c = tp_c / n_gt
        precision_c = tp_c / (tp_c + fp_c)
        f1_c = np.where(
            (precision_c + recall_c) > 0,
            2 * precision_c * recall_c / (precision_c + recall_c),
            0.0,
        )

        best = int(np.argmax(f1_c))
        per_class_p.append(float(precision_c[best]))
        per_class_r.append(float(recall_c[best]))
        per_class_conf.append(float(conf_sorted[cls_mask][best]))

    mean_p = float(np.mean(per_class_p)) if per_class_p else 0.0
    mean_r = float(np.mean(per_class_r)) if per_class_r else 0.0
    mean_f1 = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) > 0 else 0.0
    mean_conf = float(np.mean(per_class_conf)) if per_class_conf else 0.0

    return {
        "precision": mean_p,
        "recall": mean_r,
        "f1": mean_f1,
        "total_tp": total_tp_raw,
        "total_fp": total_fp_raw,
        "total_fn": total_fn_raw,
        "conf_threshold": mean_conf,
    }


def compute_per_class_ap(
    fused_preds: Dict[str, List[Dict]],
    ground_truth: Dict[str, List[Dict]],
    class_names: List[str] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute AP per class at given IoU threshold.

    Returns:
        {class_name: AP}
    """
    result = compute_map(
        fused_preds, ground_truth, class_names, iou_thresholds=[iou_threshold]
    )
    return result["per_class_AP"]


def compute_fei(performance_gain: float, time_ratio: float) -> float:
    """
    Compute Fusion Efficiency Index.

    FEI = performance_gain / time_ratio

    Args:
        performance_gain: (ensemble_metric - best_single_metric) / best_single_metric
        time_ratio: ensemble_time / best_single_time

    Returns:
        FEI value (higher is better)
    """
    if time_ratio <= 0:
        return 0.0
    return performance_gain / time_ratio


def evaluate_voting_predictions(
    fused_preds: Dict[str, List[Dict]],
    ground_truth: Dict[str, List[Dict]],
    class_names: List[str] = None,
) -> Dict:
    """
    Full evaluation for voting ensemble predictions.

    This is the eval_func to pass into grid_search / evaluate_weight_combination.

    Args:
        fused_preds: {image_name: [fused_detections]}
        ground_truth: {image_name: [gt_dicts]}
        class_names: list of class names

    Returns:
        Combined metrics dict with keys: mAP50, mAP50_95, precision, recall, f1,
        per_class_AP, total_tp, total_fp, total_fn
    """
    map_result = compute_map(fused_preds, ground_truth, class_names)
    prf_result = compute_precision_recall_f1(fused_preds, ground_truth)

    return {**map_result, **prf_result}


def compare_methods(results: Dict[str, Dict]) -> str:
    """
    Generate a formatted comparison table from results.

    Args:
        results: {method_name: metrics_dict}

    Returns:
        Formatted string table
    """
    header = f"{'Method':<35} {'mAP@0.5':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for name, metrics in results.items():
        line = (
            f"{name:<35} "
            f"{metrics.get('mAP50', 0.0):>8.4f} "
            f"{metrics.get('precision', 0.0):>10.4f} "
            f"{metrics.get('recall', 0.0):>8.4f} "
            f"{metrics.get('f1', 0.0):>8.4f}"
        )
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)
