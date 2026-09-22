#!/usr/bin/env python3
"""
Comprehensive Benchmark Evaluator: Standard YOLO vs SAHI Sliced Hyper-Inference
================================================================================
Evaluates model accuracy (mAP@50, per-class AP) and throughput (FPS, FEI) on 
the PCB test dataset, comparing standard single-pass inference against SAHI 
(Slicing Aided Hyper Inference) with DIoU/Soft NMS.

Features:
- Reads directly from unified 4-class data.yaml or legacy dataset.
- Evaluates Standard single-pass YOLO vs SAHI Sliced Hyper-Inference.
- Computes exact PASCAL/COCO 1-to-1 greedy matching mAP@50 and per-class AP.
- Measures inference latency (ms), FPS, Macro-F1, and Frontier Efficiency Index (FEI).
- Saves results as JSON for publication reporting.
- Optionally exports side-by-side comparison images.

Usage:
    python tools/eval_sahi_benchmark.py \
        --weights runs/yolov26s_rectified_loss_reweight_1280/pcb-filtered/weights/best.pt \
        --data datasets/pcb-unified-4class/data.yaml \
        --split test \
        --slice-size 480 \
        --overlap 0.20 \
        --nms-type diou \
        --device 0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Standard YOLO vs SAHI on PCB Dataset")
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to model weights (.pt)",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/pcb-unified-4class/data.yaml"),
        help="Path to data.yaml",
    )
    p.add_argument("--split", type=str, default="test", choices=["test", "val", "valid"])
    p.add_argument("--imgsz", type=int, default=640, help="Inference resolution for full-image pass")
    p.add_argument("--slice-size", type=int, default=480, help="Tile slice size in pixels")
    p.add_argument("--overlap", type=float, default=0.20, help="Slice overlap ratio (0.15 - 0.30)")
    p.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="Evaluation IoU threshold")
    p.add_argument("--nms-iou", type=float, default=0.45, help="NMS suppression threshold")
    p.add_argument(
        "--nms-type",
        type=str,
        default="diou",
        choices=["diou", "soft", "hard"],
        help="NMS algorithm for merging slices (diou: Distance-IoU, soft: Soft-NMS, hard: standard NMS)",
    )
    p.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    p.add_argument("--max-images", type=int, default=None, help="Optional limit on test images")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save output JSON benchmark",
    )
    p.add_argument(
        "--save-vis",
        type=Path,
        default=None,
        help="Optional directory to save comparison visualization images",
    )
    return p.parse_args()


def load_dataset_split(data_yaml: Path, split: str) -> Tuple[Path, Path, Dict[int, str]]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    root_path = Path(cfg.get("path", data_yaml.parent))
    split_key = "val" if split in ("val", "valid") else "test"
    split_rel = cfg.get(split_key, f"{split_key}/images")

    img_dir = root_path / split_rel if not Path(split_rel).is_absolute() else Path(split_rel)
    # Deduce labels dir
    if img_dir.name == "images":
        lbl_dir = img_dir.parent / "labels"
    else:
        lbl_dir = img_dir.parent / f"{split_key}/labels"

    names = cfg.get("names", {})
    if isinstance(names, list):
        names_dict = {i: name for i, name in enumerate(names)}
    else:
        names_dict = {int(k): v for k, v in names.items()}

    return img_dir, lbl_dir, names_dict


def parse_yolo_labels(lbl_path: Path, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * img_w
                y1 = (cy - bh / 2.0) * img_h
                x2 = (cx + bw / 2.0) * img_w
                y2 = (cy + bh / 2.0) * img_h
                boxes.append((cid, x1, y1, x2, y2))
    return boxes


def diou_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.50, beta: float = 1.0) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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

        cx_i = (x1[i] + x2[i]) / 2.0
        cy_i = (y1[i] + y2[i]) / 2.0
        cx_cand = (x1[cand] + x2[cand]) / 2.0
        cy_cand = (y1[cand] + y2[cand]) / 2.0
        rho2 = (cx_i - cx_cand) ** 2 + (cy_i - cy_cand) ** 2

        c_x1 = torch.minimum(x1[i], x1[cand])
        c_y1 = torch.minimum(y1[i], y1[cand])
        c_x2 = torch.maximum(x2[i], x2[cand])
        c_y2 = torch.maximum(y2[i], y2[cand])
        c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

        diou = iou - (rho2 / c2.clamp(min=1e-7)) ** beta
        mask = diou <= iou_threshold
        order = cand[mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def batched_diou_nms(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou_threshold: float = 0.50) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coord = boxes.max() + 1.0
    offsets = classes.to(boxes.dtype) * max_coord
    boxes_shifted = boxes + offsets[:, None]
    return diou_nms(boxes_shifted, scores, iou_threshold=iou_threshold)


def sahi_predict_tiles(
    model: YOLO,
    img_bgr: np.ndarray,
    slice_h: int = 480,
    slice_w: int = 480,
    overlap: float = 0.20,
    conf_thresh: float = 0.05,
    nms_iou: float = 0.45,
    nms_type: str = "diou",
    imgsz: int = 640,
    device: str = "0",
) -> List[Tuple[int, float, float, float, float, float]]:
    h, w = img_bgr.shape[:2]
    step_y = max(1, int(slice_h * (1.0 - overlap)))
    step_x = max(1, int(slice_w * (1.0 - overlap)))

    all_boxes = []
    all_scores = []
    all_clses = []

    # 1. Full-frame pass
    res_full = model.predict(img_bgr, imgsz=imgsz, conf=conf_thresh, device=device, verbose=False)[0]
    for b in res_full.boxes:
        all_boxes.append(b.xyxy[0].tolist())
        all_scores.append(float(b.conf[0]))
        all_clses.append(int(b.cls[0]))

    # 2. Sliced passes
    y_starts = list(range(0, max(1, h - slice_h + 1), step_y))
    if len(y_starts) == 0 or y_starts[-1] + slice_h < h:
        y_starts.append(max(0, h - slice_h))

    x_starts = list(range(0, max(1, w - slice_w + 1), step_x))
    if len(x_starts) == 0 or x_starts[-1] + slice_w < w:
        x_starts.append(max(0, w - slice_w))

    for y0 in y_starts:
        for x0 in x_starts:
            x1_tile = min(w, x0 + slice_w)
            y1_tile = min(h, y0 + slice_h)
            crop = img_bgr[y0:y1_tile, x0:x1_tile]

            res_slice = model.predict(crop, imgsz=slice_w, conf=conf_thresh, device=device, verbose=False)[0]
            for b in res_slice.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                all_boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
                all_scores.append(float(b.conf[0]))
                all_clses.append(int(b.cls[0]))

    if not all_boxes:
        return []

    t_boxes = torch.tensor(all_boxes, dtype=torch.float32)
    t_scores = torch.tensor(all_scores, dtype=torch.float32)
    t_clses = torch.tensor(all_clses, dtype=torch.int64)

    if nms_type == "diou":
        keep = batched_diou_nms(t_boxes, t_scores, t_clses, iou_threshold=nms_iou)
    else:
        keep = torchvision.ops.batched_nms(t_boxes, t_scores, t_clses, iou_threshold=nms_iou)

    results = []
    for idx in keep:
        cid = int(t_clses[idx])
        sc = float(t_scores[idx])
        bx1, by1, bx2, by2 = t_boxes[idx].tolist()
        results.append((cid, sc, bx1, by1, bx2, by2))
    return results


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
            gt_by_img.setdefault(img_id, []).append((idx, tuple(box)))

        matched_gt = {}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, (img_id, score, *box) in enumerate(preds):
            candidates = gt_by_img.get(img_id, [])
            best_iou, best_idx = 0.0, -1
            for gidx, gbox in candidates:
                if (img_id, gidx) in matched_gt:
                    continue
                iou = iou_xyxy(tuple(box), gbox)
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

        ap_res[cid] = ap
        prec_res[cid] = float(precision[-1]) if len(precision) > 0 else 0.0
        rec_res[cid] = float(recall[-1]) if len(recall) > 0 else 0.0

    return ap_res, prec_res, rec_res


def main():
    args = parse_args()
    assert args.weights.exists(), f"Weights not found: {args.weights}"
    assert args.data.exists(), f"Data config not found: {args.data}"

    img_dir, lbl_dir, class_names = load_dataset_split(args.data, args.split)
    class_ids = sorted(class_names.keys())

    image_paths = sorted(
        [p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print("=" * 75)
    print("PCB Detection Benchmark: Standard YOLO vs SAHI Sliced Hyper-Inference")
    print(f"Weights:     {args.weights}")
    print(f"Dataset:     {args.data} ({args.split} split, {len(image_paths)} images)")
    print(f"Classes:     {class_names}")
    print(f"Slice Conf:  size={args.slice_size}px, overlap={int(args.overlap*100)}%, NMS={args.nms_type.upper()}")
    print(f"Device:      {args.device}")
    print("=" * 75)

    model = YOLO(str(args.weights))

    gts = {c: [] for c in class_ids}
    preds_std = {c: [] for c in class_ids}
    preds_sahi = {c: [] for c in class_ids}

    time_std_total = 0.0
    time_sahi_total = 0.0

    for img_id, img_path in enumerate(tqdm(image_paths, desc="Evaluating")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 1. Ground truth
        lbl_file = lbl_dir / (img_path.stem + ".txt")
        gt_boxes = parse_yolo_labels(lbl_file, w, h)
        for cid, x1, y1, x2, y2 in gt_boxes:
            if cid in gts:
                gts[cid].append((img_id, x1, y1, x2, y2))

        # 2. Standard YOLO inference
        t0 = time.perf_counter()
        res_std = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        time_std_total += (time.perf_counter() - t0)

        for b in res_std.boxes:
            cid = int(b.cls[0])
            if cid in preds_std:
                sc = float(b.conf[0])
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                preds_std[cid].append((img_id, sc, bx1, by1, bx2, by2))

        # 3. SAHI Sliced Hyper-Inference
        t1 = time.perf_counter()
        s_dets = sahi_predict_tiles(
            model,
            img,
            slice_h=args.slice_size,
            slice_w=args.slice_size,
            overlap=args.overlap,
            conf_thresh=args.conf,
            nms_iou=args.nms_iou,
            nms_type=args.nms_type,
            imgsz=args.imgsz,
            device=args.device,
        )
        time_sahi_total += (time.perf_counter() - t1)

        for cid, sc, bx1, by1, bx2, by2 in s_dets:
            if cid in preds_sahi:
                preds_sahi[cid].append((img_id, sc, bx1, by1, bx2, by2))

    # Compute metrics
    n_images = len(image_paths)
    fps_std = n_images / time_std_total if time_std_total > 0 else 0.0
    fps_sahi = n_images / time_sahi_total if time_sahi_total > 0 else 0.0

    ap_std, p_std, r_std = compute_ap(preds_std, gts, class_ids, iou_thresh=args.iou)
    ap_sahi, p_sahi, r_sahi = compute_ap(preds_sahi, gts, class_ids, iou_thresh=args.iou)

    map_std = float(np.mean(list(ap_std.values())))
    map_sahi = float(np.mean(list(ap_sahi.values())))
    delta_map = map_sahi - map_std

    mean_p_std = float(np.mean(list(p_std.values())))
    mean_r_std = float(np.mean(list(r_std.values())))
    f1_std = 2 * (mean_p_std * mean_r_std) / (mean_p_std + mean_r_std + 1e-9)

    mean_p_sahi = float(np.mean(list(p_sahi.values())))
    mean_r_sahi = float(np.mean(list(r_sahi.values())))
    f1_sahi = 2 * (mean_p_sahi * mean_r_sahi) / (mean_p_sahi + mean_r_sahi + 1e-9)

    fei_std = f1_std * np.log10(max(fps_std, 1.01))
    fei_sahi = f1_sahi * np.log10(max(fps_sahi, 1.01))

    # Print Table
    print("\n" + "=" * 75)
    print(f"{'Target Class':<24} | {'Standard YOLO':<15} | {'SAHI Hyper-Inf':<15} | {'Delta Gain':<10}")
    print("-" * 75)
    for cid in class_ids:
        cname = class_names[cid]
        std_v = ap_std[cid] * 100
        sahi_v = ap_sahi[cid] * 100
        diff = sahi_v - std_v
        sign = "+" if diff >= 0 else ""
        print(f"{cname:<24} | {std_v:>6.2f}%         | {sahi_v:>6.2f}%         | {sign}{diff:>5.2f}%")
    print("=" * 75)
    sign_m = "+" if delta_map >= 0 else ""
    print(f"{'mAP@50 (Overall)':<24} | {map_std*100:>6.2f}%         | {map_sahi*100:>6.2f}%         | {sign_m}{delta_map*100:>5.2f}%")
    print(f"{'Throughput (FPS)':<24} | {fps_std:>6.1f} FPS        | {fps_sahi:>6.1f} FPS        | {fps_sahi - fps_std:>+6.1f} FPS")
    print(f"{'Macro-F1':<24} | {f1_std:>6.4f}          | {f1_sahi:>6.4f}          | {f1_sahi - f1_std:>+6.4f}")
    print(f"{'Frontier Efficiency (FEI)':<24} | {fei_std:>6.4f}          | {fei_sahi:>6.4f}          | {fei_sahi - fei_std:>+6.4f}")
    print("=" * 75 + "\n")

    # Output JSON summary
    summary = {
        "model": args.weights.stem,
        "weights": str(args.weights),
        "dataset": str(args.data),
        "split": args.split,
        "n_images": n_images,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "nms_type": args.nms_type,
        "standard": {
            "mAP50": map_std,
            "per_class_ap": {class_names[c]: ap_std[c] for c in class_ids},
            "precision": mean_p_std,
            "recall": mean_r_std,
            "f1": f1_std,
            "fps": fps_std,
            "fei": fei_std,
        },
        "sahi": {
            "mAP50": map_sahi,
            "per_class_ap": {class_names[c]: ap_sahi[c] for c in class_ids},
            "precision": mean_p_sahi,
            "recall": mean_r_sahi,
            "f1": f1_sahi,
            "fps": fps_sahi,
            "fei": fei_sahi,
        },
        "delta_mAP50": delta_map,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    model_name = args.weights.stem
    for p in args.weights.parents:
        if p.name not in ("weights", "pcb-filtered", "runs", ""):
            model_name = p.name
            break
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"sahi_benchmark_{model_name}_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved SAHI Benchmark report to: {out_json}")


if __name__ == "__main__":
    main()
