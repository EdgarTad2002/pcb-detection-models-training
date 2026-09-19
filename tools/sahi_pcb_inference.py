#!/usr/bin/env python3
"""
SAHI: Slicing Aided Hyper Inference for PCB Component Detection
================================================================
Breaks large high-resolution PCB images into overlapping slices/tiles,
detects micro-components (especially 0402/0603 SMD chip capacitors) that are
lost during full-image downsampling, and merges detections using batched NMS.

Features:
- Dual-pass Hyper Inference: Full-frame (macro ICs/Connectors) + Slices (micro Capacitors)
- Overlapping tile slicing (customizable slice size and overlap ratio)
- Batched NMS coordinate reprojection back to original canvas
- Generates side-by-side [Standard YOLO vs SAHI Hyper-Inference] comparisons
- Zero retraining required (model-agnostic, works with any YOLO26 / YOLOv11 checkpoint)

Usage:
    python tools/sahi_pcb_inference.py \
        --source data_samples/images/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.jpg \
        --weights runs/yolov26s_loss_reweight/pcb-filtered/weights/best.pt \
        --slice-size 360 \
        --overlap 0.25 \
        --conf 0.20 \
        --out-dir runs/sahi_demo
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

EVAL_CLASSES = {2: "Capacitor", 4: "Connector", 7: "Electrolytic Capacitor", 9: "IC"}
CLASS_COLORS = {
    "Capacitor": (0, 220, 255),               # Cyan
    "Connector": (255, 190, 0),               # Gold
    "Electrolytic Capacitor": (255, 50, 180), # Magenta
    "IC": (0, 255, 100),                      # Bright Green
    "Other": (180, 180, 180),                 # Gray
}


def get_color(cls_name: str) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(cls_name, CLASS_COLORS["Other"])


def diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.50,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Distance-IoU Non-Maximum Suppression (DIoU-NMS, Zheng et al., AAAI 2020).
    Penalizes the normalized central distance between proposals:
        S_DIoU = IoU - (rho^2(b, b_cand) / c^2)^beta
    Touching adjacent components (e.g. side-by-side pin headers) have separated centers,
    lowering S_DIoU below iou_threshold and preserving both components.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        cand = order[1:]
        xx1 = torch.maximum(x1[i], x1[cand])
        yy1 = torch.maximum(y1[i], y1[cand])
        xx2 = torch.minimum(x2[i], x2[cand])
        yy2 = torch.minimum(y2[i], y2[cand])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        union = areas[i] + areas[cand] - inter
        iou = inter / union.clamp(min=1e-7)

        # Center distance rho^2
        cx_i = (x1[i] + x2[i]) / 2.0
        cy_i = (y1[i] + y2[i]) / 2.0
        cx_cand = (x1[cand] + x2[cand]) / 2.0
        cy_cand = (y1[cand] + y2[cand]) / 2.0
        rho2 = (cx_i - cx_cand) ** 2 + (cy_i - cy_cand) ** 2

        # Smallest enclosing box diagonal c^2
        c_x1 = torch.minimum(x1[i], x1[cand])
        c_y1 = torch.minimum(y1[i], y1[cand])
        c_x2 = torch.maximum(x2[i], x2[cand])
        c_y2 = torch.maximum(y2[i], y2[cand])
        c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

        # DIoU = IoU - (rho^2 / c^2)^beta
        diou = iou - (rho2 / c2.clamp(min=1e-7)) ** beta
        mask = diou <= iou_threshold
        order = cand[mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def batched_diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.50,
    beta: float = 1.0,
) -> torch.Tensor:
    """Class-aware (batched) Distance-IoU NMS."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coord = boxes.max() + 1.0
    offsets = classes.to(boxes.dtype) * max_coord
    boxes_shifted = boxes + offsets[:, None]
    return diou_nms(boxes_shifted, scores, iou_threshold=iou_threshold, beta=beta)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.50,
    sigma: float = 0.50,
    conf_threshold: float = 0.05,
    method: str = "gaussian",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS (Bodla et al., ICCV 2017).
    Decays confidence scores of overlapping proposals rather than completely removing them.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device), torch.empty((0,), device=scores.device)

    keep_indices, final_scores = [], []
    for c in torch.unique(classes):
        c_mask = (classes == c)
        c_indices = torch.where(c_mask)[0]
        c_boxes = boxes[c_indices].clone()
        c_scores = scores[c_indices].clone()
        N = c_boxes.size(0)
        areas = (c_boxes[:, 2] - c_boxes[:, 0]).clamp(min=0) * (c_boxes[:, 3] - c_boxes[:, 1]).clamp(min=0)

        for i in range(N):
            max_pos = int(torch.argmax(c_scores[i:]) + i)
            c_boxes[i], c_boxes[max_pos] = c_boxes[max_pos].clone(), c_boxes[i].clone()
            c_scores[i], c_scores[max_pos] = c_scores[max_pos].clone(), c_scores[i].clone()
            c_indices[i], c_indices[max_pos] = c_indices[max_pos].clone(), c_indices[i].clone()
            areas[i], areas[max_pos] = areas[max_pos].clone(), areas[i].clone()

            if c_scores[i] < conf_threshold:
                break
            if i + 1 < N:
                xx1 = torch.maximum(c_boxes[i, 0], c_boxes[i + 1:, 0])
                yy1 = torch.maximum(c_boxes[i, 1], c_boxes[i + 1:, 1])
                xx2 = torch.minimum(c_boxes[i, 2], c_boxes[i + 1:, 2])
                yy2 = torch.minimum(c_boxes[i, 3], c_boxes[i + 1:, 3])
                inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
                union = areas[i] + areas[i + 1:] - inter
                iou = inter / union.clamp(min=1e-7)

                if method == "gaussian":
                    weight = torch.exp(-(iou ** 2) / sigma)
                else:
                    weight = torch.where(iou >= iou_threshold, 1.0 - iou, torch.tensor(1.0, device=iou.device))
                c_scores[i + 1:] *= weight

        valid = c_scores >= conf_threshold
        keep_indices.append(c_indices[valid])
        final_scores.append(c_scores[valid])

    if not keep_indices:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device), torch.empty((0,), device=scores.device)

    all_keep = torch.cat(keep_indices)
    all_scores = torch.cat(final_scores)
    sort_idx = all_scores.argsort(descending=True)
    return all_keep[sort_idx], all_scores[sort_idx]


def sahi_predict(
    model: YOLO,
    image_bgr: np.ndarray,
    slice_height: int = 480,
    slice_width: int = 480,
    overlap_ratio: float = 0.25,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45,
    include_full_image: bool = True,
    nms_type: str = "diou",
    beta: float = 1.0,
    sigma: float = 0.50,
    device: str = "",
) -> List[Tuple[int, str, float, int, int, int, int]]:
    """
    Executes Slicing Aided Hyper Inference on an image with DIoU-NMS, Soft-NMS, or Hard-NMS.

    Args:
        model: YOLO detection model
        image_bgr: Input image in BGR format
        slice_height: Height of tile crops
        slice_width: Width of tile crops
        overlap_ratio: Overlap between adjacent tiles
        conf_threshold: Confidence filtering threshold
        iou_threshold: NMS overlap threshold
        include_full_image: If True, combines full-image pass with sliced passes
        nms_type: 'diou' (Distance-IoU NMS), 'soft' (Soft-NMS), or 'hard' (Standard NMS)
        beta: Power factor for DIoU-NMS center penalty (default 1.0)
        sigma: Variance for Soft-NMS Gaussian decay (default 0.50)
        device: Device to run inference on

    Returns:
        List of tuples: (cls_id, cls_name, confidence, x1, y1, x2, y2)
    """
    h, w = image_bgr.shape[:2]
    step_y = max(1, int(slice_height * (1.0 - overlap_ratio)))
    step_x = max(1, int(slice_width * (1.0 - overlap_ratio)))

    all_boxes = []
    all_scores = []
    all_clses = []

    # 1. Full-image pass: Captures large ICs, Connectors, and overall layout
    predict_kwargs = {"conf": conf_threshold, "verbose": False}
    if device:
        predict_kwargs["device"] = device

    if include_full_image:
        res_full = model.predict(image_bgr, **predict_kwargs)[0]
        for b in res_full.boxes:
            all_boxes.append(b.xyxy[0].tolist())
            all_scores.append(float(b.conf[0]))
            all_clses.append(int(b.cls[0]))

    # 2. Sliced passes: Uncovers micro-capacitors and dense components
    y_starts = list(range(0, max(1, h - slice_height + 1), step_y))
    if len(y_starts) == 0 or y_starts[-1] + slice_height < h:
        y_starts.append(max(0, h - slice_height))

    x_starts = list(range(0, max(1, w - slice_width + 1), step_x))
    if len(x_starts) == 0 or x_starts[-1] + slice_width < w:
        x_starts.append(max(0, w - slice_width))

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(w, x0 + slice_width)
            y1 = min(h, y0 + slice_height)
            slice_crop = image_bgr[y0:y1, x0:x1]

            res_slice = model.predict(slice_crop, **predict_kwargs)[0]
            for b in res_slice.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                # Shift box back to global image coordinates
                all_boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
                all_scores.append(float(b.conf[0]))
                all_clses.append(int(b.cls[0]))

    if len(all_boxes) == 0:
        return []

    t_boxes = torch.tensor(all_boxes, dtype=torch.float32)
    t_scores = torch.tensor(all_scores, dtype=torch.float32)
    t_clses = torch.tensor(all_clses, dtype=torch.int64)

    # NMS Selection: DIoU-NMS (recommended for touching connectors), Soft-NMS, or Hard-NMS
    if nms_type == "diou":
        keep = batched_diou_nms(t_boxes, t_scores, t_clses, iou_threshold=iou_threshold, beta=beta)
        kept_scores = t_scores[keep]
    elif nms_type == "soft":
        keep, kept_scores = soft_nms(t_boxes, t_scores, t_clses, iou_threshold=iou_threshold, sigma=sigma, conf_threshold=conf_threshold)
    else:
        keep = torchvision.ops.batched_nms(t_boxes, t_scores, t_clses, iou_threshold=iou_threshold)
        kept_scores = t_scores[keep]

    final_detections = []
    for idx, sc_val in zip(keep, kept_scores):
        cls_id = int(t_clses[idx])
        cls_name = model.names.get(cls_id, str(cls_id))
        score = float(sc_val)
        x1, y1, x2, y2 = map(int, t_boxes[idx].tolist())
        final_detections.append((cls_id, cls_name, score, x1, y1, x2, y2))

    return final_detections


def draw_detections(img_bgr: np.ndarray, detections: list, filter_eval_only: bool = True) -> np.ndarray:
    """Draws colored bounding boxes and labels on an image canvas."""
    canvas = img_bgr.copy()
    for d in detections:
        cls_id, cls_name, score, x1, y1, x2, y2 = d
        if filter_eval_only and cls_id not in EVAL_CLASSES and cls_name not in EVAL_CLASSES.values():
            continue

        color = get_color(cls_name)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label tag
        label = f"{cls_name[:3]}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        y_label = max(y1 - 4, th + 4)
        cv2.rectangle(canvas, (x1, y_label - th - 2), (x1 + tw + 2, y_label + 2), color, -1)
        cv2.putText(canvas, label, (x1 + 1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def visualize_sahi_comparison(
    img_bgr: np.ndarray,
    std_detections: list,
    sahi_detections: list,
    out_path: str,
    img_title: str = "PCB Component Detection",
):
    """Saves a high-resolution side-by-side comparison figure."""
    vis_std = draw_detections(img_bgr, std_detections)
    vis_sahi = draw_detections(img_bgr, sahi_detections)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(cv2.cvtColor(vis_std, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"Standard 640px Inference\nTotal Detections: {len(std_detections)}",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vis_sahi, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"SAHI Slicing Aided Hyper Inference\nTotal Detections: {len(sahi_detections)} (+{len(sahi_detections) - len(std_detections)} Micro-Objects)",
        fontsize=13,
        fontweight="bold",
        color="darkgreen",
    )
    axes[1].axis("off")

    plt.suptitle(img_title, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 Saved SAHI comparison visualization to: {out_path}")


def iou_xyxy(a, b):
    """Computes Intersection-over-Union between two [x1, y1, x2, y2] bounding boxes."""
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


def compute_ap(preds_by_class: dict, gts_by_class: dict, iou_thresh: float = 0.5):
    """Computes standard PASCAL VOC / COCO all-points interpolated AP at given IoU threshold."""
    res = {}
    for cls_id in EVAL_CLASSES:
        preds = sorted(preds_by_class.get(cls_id, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cls_id, [])
        n_gt = len(gts)
        if n_gt == 0:
            res[cls_id] = 0.0
            continue
        gt_by_img = {}
        for idx, (img_id, *box) in enumerate(gts):
            gt_by_img.setdefault(img_id, []).append((idx, box))

        matched_gt = {}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for i, (img_id, score, *box) in enumerate(preds):
            candidates = gt_by_img.get(img_id, [])
            best_iou, best_idx = 0.0, -1
            for gidx, gbox in candidates:
                if (img_id, gidx) in matched_gt:
                    continue
                iou = iou_xyxy(box, gbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, gidx
            if best_iou >= iou_thresh and best_idx != -1:
                tp[i] = 1
                matched_gt[(img_id, best_idx)] = True
            else:
                fp[i] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        res[cls_id] = ap
    return res


def evaluate_sahi_benchmark(
    model: YOLO,
    sample_dir: str = "data_samples",
    max_images: Optional[int] = 15,
    slice_size: int = 360,
    overlap: float = 0.25,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.50,
    imgsz: int = 640,
    plot: bool = True,
):
    """
    Computes and compares mAP50 for Standard YOLO vs SAHI Hyper-Inference.
    Displays formatted comparison table and bar chart.
    """
    import glob
    from pathlib import Path

    img_dir = Path(sample_dir) / "images"
    lbl_dir = Path(sample_dir) / "labels"
    images = sorted(glob.glob(str(img_dir / "*.jpg")))
    if max_images:
        images = images[:max_images]

    print(f"📊 Running mAP50 Benchmark Evaluation across {len(images)} PCB samples...")
    print(f"   Conf threshold: {conf_thresh}, IoU threshold: {iou_thresh}, Slice: {slice_size}px (overlap={int(overlap*100)}%)")

    gts = {c: [] for c in EVAL_CLASSES}
    preds_std = {c: [] for c in EVAL_CLASSES}
    preds_sahi = {c: [] for c in EVAL_CLASSES}

    for img_id, img_path in enumerate(images):
        raw = cv2.imread(img_path)
        if raw is None:
            continue
        h, w = raw.shape[:2]

        # 1. Load GT
        lbl_file = lbl_dir / (Path(img_path).stem + ".txt")
        if lbl_file.exists():
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        if cid in EVAL_CLASSES:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            gts[cid].append((img_id, x1, y1, x2, y2))

        # 2. Standard YOLO
        res_std = model.predict(raw, imgsz=imgsz, conf=conf_thresh, verbose=False)[0]
        for b in res_std.boxes:
            cid = int(b.cls[0])
            if cid in EVAL_CLASSES:
                sc = float(b.conf[0])
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                preds_std[cid].append((img_id, sc, bx1, by1, bx2, by2))

        # 3. SAHI
        s_dets = sahi_predict(
            model,
            raw,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_ratio=overlap,
            conf_threshold=conf_thresh,
            iou_threshold=0.45,
            include_full_image=True,
        )
        for cid, cname, sc, bx1, by1, bx2, by2 in s_dets:
            if cid in EVAL_CLASSES:
                preds_sahi[cid].append((img_id, sc, bx1, by1, bx2, by2))

    # Compute metrics
    ap_std = compute_ap(preds_std, gts, iou_thresh=iou_thresh)
    ap_sahi = compute_ap(preds_sahi, gts, iou_thresh=iou_thresh)

    map_std = float(np.mean(list(ap_std.values())))
    map_sahi = float(np.mean(list(ap_sahi.values())))
    delta_map = map_sahi - map_std

    # Print Table
    print("\n" + "=" * 70)
    print(f"{'Target Class':<24} | {'Standard YOLO':<14} | {'SAHI Hyper-Inf':<15} | {'Delta Gain':<10}")
    print("-" * 70)
    for cid, cname in EVAL_CLASSES.items():
        std_v = ap_std[cid] * 100
        sahi_v = ap_sahi[cid] * 100
        diff = sahi_v - std_v
        sign = "+" if diff >= 0 else ""
        print(f"{cname:<24} | {std_v:>6.2f}%        | {sahi_v:>6.2f}%         | {sign}{diff:>5.2f}%")
    print("=" * 70)
    sign_m = "+" if delta_map >= 0 else ""
    print(f"{'mAP@50 (Overall)':<24} | {map_std*100:>6.2f}%        | {map_sahi*100:>6.2f}%         | {sign_m}{delta_map*100:>5.2f}%")
    print("=" * 70 + "\n")

    # Plot Bar Chart
    if plot:
        import matplotlib.pyplot as plt
        classes = [EVAL_CLASSES[c] for c in EVAL_CLASSES] + ["mAP50 (Mean)"]
        std_scores = [ap_std[c] * 100 for c in EVAL_CLASSES] + [map_std * 100]
        sahi_scores = [ap_sahi[c] * 100 for c in EVAL_CLASSES] + [map_sahi * 100]

        x = np.arange(len(classes))
        w = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - w / 2, std_scores, w, label=f"Standard YOLO ({imgsz}px)", color="#3b82f6")
        bars2 = ax.bar(x + w / 2, sahi_scores, w, label=f"SAHI Hyper-Inference ({slice_size}px)", color="#10b981")

        for i in range(len(classes)):
            diff = sahi_scores[i] - std_scores[i]
            sign = "+" if diff >= 0 else ""
            color = "green" if diff >= 0 else "red"
            y_pos = max(std_scores[i], sahi_scores[i]) + 2
            ax.text(x[i] + w / 2, y_pos, f"{sign}{diff:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)

        ax.set_ylabel("AP@50 (%)", fontsize=12, fontweight="bold")
        ax.set_title("PCB Detection Benchmark: Standard YOLO vs SAHI Hyper-Inference", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(max(std_scores), max(sahi_scores)) + 15)
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return {
        "ap_std": ap_std,
        "ap_sahi": ap_sahi,
        "map_std": map_std,
        "map_sahi": map_sahi,
        "delta_map": delta_map,
    }


def main():
    p = argparse.ArgumentParser(description="SAHI: Slicing Aided Hyper Inference for PCB Components")
    p.add_argument("--source", type=str, required=True, help="Path to input image or directory")
    p.add_argument(
        "--weights",
        type=str,
        default="runs/yolov26s_loss_reweight/pcb-filtered/weights/best.pt",
        help="Path to trained YOLO weights",
    )
    p.add_argument("--slice-size", type=int, default=480, help="Tile slice size in pixels")
    p.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio (0.1 - 0.4)")
    p.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU merge threshold")
    p.add_argument(
        "--nms-type",
        type=str,
        choices=["diou", "soft", "hard"],
        default="diou",
        help="NMS suppression algorithm (diou: Distance-IoU NMS for touching components, soft: Soft-NMS, hard: Standard NMS)",
    )
    p.add_argument("--out-dir", type=str, default="runs/sahi_demo", help="Output directory")
    args = p.parse_args()

    model = YOLO(args.weights)
    source_path = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_file():
        img = cv2.imread(str(source_path))
        if img is None:
            print(f"Error loading image: {source_path}")
            return

        # 1. Standard full-image inference
        res_std = model.predict(img, conf=args.conf, imgsz=640, verbose=False)[0]
        std_dets = []
        for b in res_std.boxes:
            cls_id = int(b.cls[0])
            std_dets.append(
                (cls_id, model.names[cls_id], float(b.conf[0]), *map(int, b.xyxy[0].tolist()))
            )

        # 2. SAHI sliced hyper-inference with DIoU/Soft NMS
        sahi_dets = sahi_predict(
            model,
            img,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_ratio=args.overlap,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            nms_type=args.nms_type,
        )

        print(f"\nResults for {source_path.name}:")
        print(f"  Standard 640px Inference: {len(std_dets)} components detected")
        print(f"  SAHI Hyper-Inference [{args.nms_type.upper()}]: {len(sahi_dets)} components detected (+{len(sahi_dets) - len(std_dets)})")

        out_vis = out_dir / f"sahi_{args.nms_type}_{source_path.stem}.png"
        visualize_sahi_comparison(img, std_dets, sahi_dets, str(out_vis), source_path.name)
    else:
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        print(f"Processing {len(images)} images in {source_path}...")
        for img_path in images[:5]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            res_std = model.predict(img, conf=args.conf, imgsz=640, verbose=False)[0]
            std_dets = [
                (int(b.cls[0]), model.names[int(b.cls[0])], float(b.conf[0]), *map(int, b.xyxy[0].tolist()))
                for b in res_std.boxes
            ]
            sahi_dets = sahi_predict(
                model,
                img,
                slice_height=args.slice_size,
                slice_width=args.slice_size,
                overlap_ratio=args.overlap,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                nms_type=args.nms_type,
            )
            out_vis = out_dir / f"sahi_{args.nms_type}_{img_path.stem}.png"
            visualize_sahi_comparison(img, std_dets, sahi_dets, str(out_vis), img_path.name)


if __name__ == "__main__":
    main()
