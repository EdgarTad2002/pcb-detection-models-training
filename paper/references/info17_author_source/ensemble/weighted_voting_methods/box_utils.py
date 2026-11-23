"""
Phase 2: Box Utilities

Core functions for IoU computation, cross-model box matching,
model-weighted box fusion, and NMS.

All coordinates are normalized [x1, y1, x2, y2] in [0, 1].
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Works with both normalized and absolute coordinates.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def match_boxes_across_models(
    all_detections: Dict[str, List[Dict]],
    iou_threshold: float = 0.5,
) -> List[List[Dict]]:
    """
    Group overlapping detections from different models by class and IoU.

    Each detection dict has: {"class": int, "confidence": float, "box_norm": [x1,y1,x2,y2]}
    Each returned detection is augmented with "model" key.

    Args:
        all_detections: {model_name: [detection_dicts]}
        iou_threshold: minimum IoU to consider boxes as matching

    Returns:
        List of matched groups. Each group is a list of detection dicts
        (augmented with "model" key) that overlap across models.
    """
    # Flatten all detections with model source, grouped by class
    by_class = defaultdict(list)
    for model_name, dets in all_detections.items():
        for det in dets:
            entry = {
                "class": det["class"],
                "confidence": det["confidence"],
                "box_norm": det["box_norm"],
                "model": model_name,
            }
            by_class[det["class"]].append(entry)

    matched_groups = []

    for cls_id, dets in by_class.items():
        # Sort by confidence descending for greedy matching
        dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        used = [False] * len(dets_sorted)
        groups = []

        for i, det_i in enumerate(dets_sorted):
            if used[i]:
                continue

            group = [det_i]
            used[i] = True

            for j in range(i + 1, len(dets_sorted)):
                if used[j]:
                    continue

                # Check IoU against any box already in the group
                for g_det in group:
                    iou = compute_iou(g_det["box_norm"], dets_sorted[j]["box_norm"])
                    if iou >= iou_threshold:
                        group.append(dets_sorted[j])
                        used[j] = True
                        break

            groups.append(group)

        matched_groups.extend(groups)

    return matched_groups


def fuse_boxes_with_model_weights(
    matched_group: List[Dict],
    model_weights: Dict[str, float],
) -> Dict:
    """
    Fuse a group of matched detections using model-level weights.

    Formula:
        fused_conf = sum(w_i * conf_i) / sum(w_i)
        fused_coord = sum(w_i * conf_i * coord_i) / sum(w_i * conf_i)

    Args:
        matched_group: list of detection dicts with "model", "confidence", "box_norm", "class"
        model_weights: {model_name: weight}

    Returns:
        Fused detection dict with "class", "confidence", "box_norm", "num_models", "source_models"
    """
    if len(matched_group) == 1:
        det = matched_group[0]
        w = model_weights.get(det["model"], 1.0)
        return {
            "class": det["class"],
            "confidence": det["confidence"],
            "box_norm": det["box_norm"][:],
            "num_models": 1,
            "source_models": [det["model"]],
        }

    # Compute weighted confidence
    w_sum = 0.0
    wc_sum = 0.0
    fused_box = [0.0, 0.0, 0.0, 0.0]
    source_models = []

    for det in matched_group:
        w = model_weights.get(det["model"], 1.0)
        c = det["confidence"]
        wc = w * c

        w_sum += w
        wc_sum += wc

        for k in range(4):
            fused_box[k] += wc * det["box_norm"][k]

        if det["model"] not in source_models:
            source_models.append(det["model"])

    # Normalize
    fused_conf = wc_sum / w_sum if w_sum > 0 else 0.0

    if wc_sum > 0:
        for k in range(4):
            fused_box[k] /= wc_sum
    else:
        fused_box = matched_group[0]["box_norm"][:]

    return {
        "class": matched_group[0]["class"],
        "confidence": fused_conf,
        "box_norm": fused_box,
        "num_models": len(source_models),
        "source_models": source_models,
    }


def apply_nms(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> List[Dict]:
    """
    Apply per-class NMS to remove duplicate detections after fusion.

    Args:
        detections: list of dicts with "class", "confidence", "box_norm"
        iou_threshold: IoU threshold for suppression
        conf_threshold: minimum confidence to keep

    Returns:
        Filtered list of detections
    """
    # Filter by confidence
    dets = [d for d in detections if d["confidence"] >= conf_threshold]

    if not dets:
        return []

    # Group by class
    by_class = defaultdict(list)
    for d in dets:
        by_class[d["class"]].append(d)

    kept = []
    for cls_id, cls_dets in by_class.items():
        # Sort by confidence descending
        cls_dets.sort(key=lambda d: d["confidence"], reverse=True)

        selected = []
        for det in cls_dets:
            suppressed = False
            for sel in selected:
                if compute_iou(det["box_norm"], sel["box_norm"]) >= iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                selected.append(det)

        kept.extend(selected)

    return kept


def fuse_image_detections(
    model_detections: Dict[str, List[Dict]],
    model_weights: Dict[str, float],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> List[Dict]:
    """
    Full pipeline for a single image: match -> fuse -> NMS.

    Args:
        model_detections: {model_name: [detections]} for one image
        model_weights: {model_name: weight}
        iou_threshold: IoU threshold for matching and NMS
        conf_threshold: confidence threshold for final filtering

    Returns:
        List of fused detections
    """
    # Match boxes across models
    matched_groups = match_boxes_across_models(model_detections, iou_threshold)

    # Fuse each matched group
    fused = []
    for group in matched_groups:
        fused_det = fuse_boxes_with_model_weights(group, model_weights)
        fused.append(fused_det)

    # Apply NMS to remove remaining duplicates
    result = apply_nms(fused, iou_threshold, conf_threshold)

    return result
