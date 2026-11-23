#!/usr/bin/env python3
"""
tools/eval_dual_branch_pipeline.py
==================================
Dual-Branch Hierarchical Inference Pipeline for PCB Inspection:
  Branch A (Full-Board Generalist):
    - Runs full-frame inference on the original board resized to 1280px.
    - Responsible for ALL 4 classes (Connectors, ICs, Electrolytics, and global Capacitors).
  Branch B (Capacitor Crop Specialist):
    - Runs tiled sliding window inference on native-resolution crops (e.g. 1280x1280, 20% overlap).
    - RESTRICTED AT INFERENCE TO PREDICT ONLY CAPACITORS (Class 0).
    - Discards any non-capacitor predictions from the crop specialist to prevent
      partial macro-component false positives.

Class-Aware Merging & Fusion:
  - For Non-Capacitor Classes (Connector, Electrolytic, IC):
      * Predictions from Branch A are 100% strictly preserved.
  - For Capacitor Class:
      * Predictions from Branch A and Branch B are combined.
      * Seam-crossing fragments are merged via IoS (Intersection-over-Smaller).
      * True physical duplicates are merged via either:
          (a) Class-aware NMS (batched_nms with configurable iou threshold)
          (b) Weighted Box Fusion (WBF via ensemble_boxes or custom weighted coordinate fusion)
      * Preserves close-proximity capacitor separation (tight thresholding prevents merging
        adjacent 0402/0201 capacitors separated by only 2-4 pixels).

Zero Leakage Policy:
  - Hyperparameters (tile size, overlap, merge threshold, fusion method) must be tuned
    on the VALIDATION split.
  - Test split is evaluated only under finalized, locked operating points.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-register custom modules
for _mod_name in (
    "luma_modules",
    "retinex_stem",
    "weber_stem",
    "squared_dice_loss",
    "agwd_ota_loss",
    "igr_stem",
    "fg_csfa_neck",
    "aca_supcon",
    "physics_spectral_yolo26",
    "superyolo_pcb",
    "icm_module",
):
    try:
        __import__(_mod_name)
    except ImportError:
        pass

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Dual-Branch Full-Board + Capacitor Crop Specialist Pipeline.")
    p.add_argument("--global-weights", type=str, required=True, help="Path to full-board detector weights")
    p.add_argument("--crop-weights", type=str, default=None, help="Path to crop specialist weights (if None, reuses global)")
    p.add_argument("--images-dir", type=str, required=True, help="Directory of test/val images")
    p.add_argument("--labels-dir", type=str, required=True, help="Directory of test/val YOLO label files")
    p.add_argument("--split-name", type=str, default="val", help="Split name for reporting ('val' or 'test')")
    p.add_argument("--slice-size", type=int, default=1280, help="Tile slice size in pixels (default: 1280)")
    p.add_argument("--imgsz", type=int, default=None, help="Detector input size, independent of native tile size")
    p.add_argument("--overlap", type=float, default=0.20, help="Tile overlap ratio (default: 0.20)")
    p.add_argument("--conf-thresh", type=float, default=0.001, help="Confidence threshold for predictions")
    p.add_argument("--eval-iou", type=float, default=0.50, help="IoU threshold for AP matching (default: 0.50)")
    p.add_argument("--merge-iou", type=float, default=0.50, help="IoU threshold for duplicate merging")
    p.add_argument("--prediction-iou", type=float, default=0.50, help="Fixed internal model suppression threshold")
    p.add_argument("--ios-thresh", type=float, default=0.65, help="IoS threshold for border-truncated seam merging")
    p.add_argument("--max-det", type=int, default=1000, help="Maximum detections per board")
    p.add_argument("--fusion-method", type=str, choices=["nms", "wbf"], default="nms", help="Merging algorithm for capacitors")
    p.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    p.add_argument("--no-crops", action="store_true", help="Ablation: run full-board branch only (no crop specialist)")
    p.add_argument("--no-global", action="store_true", help="Ablation: run crop specialist branch only")
    p.add_argument("--output-json", type=str, default=None, help="Path to save output evaluation metrics JSON")
    return p.parse_args()


def get_slice_boxes(img_w: int, img_h: int, slice_w: int, slice_h: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Generates sliding window coordinates covering the entire board canvas."""
    if img_w <= slice_w and img_h <= slice_h:
        return [(0, 0, img_w, img_h)]

    step_x = max(1, int(slice_w * (1.0 - overlap)))
    step_y = max(1, int(slice_h * (1.0 - overlap)))

    x_starts = list(range(0, max(1, img_w - slice_w + 1), step_x))
    if not x_starts or x_starts[-1] + slice_w < img_w:
        x_starts.append(max(0, img_w - slice_w))

    y_starts = list(range(0, max(1, img_h - slice_h + 1), step_y))
    if not y_starts or y_starts[-1] + slice_h < img_h:
        y_starts.append(max(0, img_h - slice_h))

    slices = []
    for y in y_starts:
        for x in x_starts:
            x2 = min(img_w, x + slice_w)
            y2 = min(img_h, y + slice_h)
            slices.append((x, y, x2, y2))
    return slices


def merge_border_truncated_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    ios_thresh: float = 0.65,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merges border-truncated fragments with more complete instances using IoS."""
    if len(boxes) <= 1:
        return boxes, scores, classes

    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    order = areas.argsort(descending=True)
    boxes = boxes[order]
    scores = scores[order]
    classes = classes[order]
    areas = areas[order]

    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 >= len(boxes):
            break

        c_match = (classes[i + 1:] == classes[i]) & (~suppressed[i + 1:])
        if not c_match.any():
            continue

        c_indices = torch.where(c_match)[0] + (i + 1)
        sub_boxes = boxes[c_indices]

        xx1 = torch.maximum(boxes[i, 0], sub_boxes[:, 0])
        yy1 = torch.maximum(boxes[i, 1], sub_boxes[:, 1])
        xx2 = torch.minimum(boxes[i, 2], sub_boxes[:, 2])
        yy2 = torch.minimum(boxes[i, 3], sub_boxes[:, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)

        sub_areas = areas[c_indices]
        smaller_areas = torch.minimum(areas[i], sub_areas).clamp(min=1e-7)
        ios = inter / smaller_areas

        drop_mask = ios >= ios_thresh
        suppressed[c_indices[drop_mask]] = True

    keep_t = torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    return boxes[keep_t], scores[keep_t], classes[keep_t]


def fuse_capacitor_predictions_wbf(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    canvas_w: int,
    canvas_h: int,
    iou_thr: float = 0.50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuses overlapping capacitor detections using Weighted Box Fusion."""
    if len(boxes) <= 1:
        return boxes, scores

    try:
        from ensemble_boxes import weighted_boxes_fusion
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("WBF requested but ensemble_boxes could not load; repair its dependencies. "
                           "Refusing to report NMS results as WBF.") from exc

    # Normalize to [0, 1] for ensemble_boxes
    norm_boxes = boxes.clone()
    norm_boxes[:, [0, 2]] /= max(1.0, float(canvas_w))
    norm_boxes[:, [1, 3]] /= max(1.0, float(canvas_h))
    norm_boxes = norm_boxes.clamp(0.0, 1.0)

    # Format for WBF: list of models (here we treat all candidate boxes as single ensemble pool)
    boxes_list = [norm_boxes.cpu().numpy().tolist()]
    scores_list = [scores.cpu().numpy().tolist()]
    labels_list = [[0] * len(scores)]

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=None, iou_thr=iou_thr, skip_box_thr=0.0001
    )

    t_fused_b = torch.tensor(fused_boxes, dtype=torch.float32, device=boxes.device)
    t_fused_b[:, [0, 2]] *= float(canvas_w)
    t_fused_b[:, [1, 3]] *= float(canvas_h)
    t_fused_s = torch.tensor(fused_scores, dtype=torch.float32, device=scores.device)
    return t_fused_b, t_fused_s


def run_dual_branch_inference(
    model_global: YOLO,
    model_crop: YOLO,
    img_bgr: np.ndarray,
    slice_size: int,
    overlap: float,
    conf_thresh: float,
    merge_iou: float,
    ios_thresh: float,
    max_det: int,
    fusion_method: str,
    device: str,
    run_global: bool = True,
    run_crops: bool = True,
    imgsz: Optional[int] = None,
    prediction_iou: float = 0.50,
) -> List[Tuple[int, float, float, float, float, float]]:
    """Runs Branch A (Full-Board) + Branch B (Capacitor Crop Specialist) and fuses."""
    ih, iw = img_bgr.shape[:2]

    # Containers
    non_cap_boxes = []
    non_cap_scores = []
    non_cap_classes = []

    cap_boxes = []
    cap_scores = []

    # -------------------------------------------------------------
    # Branch A: Full-Board Global Pass (All 4 classes)
    # -------------------------------------------------------------
    if run_global:
        res_full = model_global.predict(
            img_bgr,
            imgsz=imgsz or slice_size,
            conf=conf_thresh,
            iou=prediction_iou,
            max_det=max_det,
            device=device,
            verbose=False,
            half=False,
        )[0]

        if res_full.boxes is not None and len(res_full.boxes) > 0:
            for b in res_full.boxes:
                cid = int(b.cls[0].item())
                if 0 <= cid < len(CLASS_NAMES):
                    sc = float(b.conf[0].item())
                    bx = b.xyxy[0].tolist()
                    if cid == 0:
                        # Global capacitor prediction
                        cap_boxes.append(bx)
                        cap_scores.append(sc)
                    else:
                        # Non-capacitor prediction (Connector, Elec-Cap, IC)
                        non_cap_boxes.append(bx)
                        non_cap_scores.append(sc)
                        non_cap_classes.append(cid)

    # -------------------------------------------------------------
    # Branch B: Capacitor Crop Specialist (Capacitor-Only Output)
    # -------------------------------------------------------------
    num_tiles = 0
    if run_crops:
        slice_coords = get_slice_boxes(iw, ih, slice_size, slice_size, overlap)
        num_tiles = len(slice_coords)
        for (sx1, sy1, sx2, sy2) in slice_coords:
            tile = img_bgr[sy1:sy2, sx1:sx2]
            th, tw = tile.shape[:2]
            if th == 0 or tw == 0:
                continue

            res_tile = model_crop.predict(
                tile,
                imgsz=imgsz or max(tw, th),
                conf=conf_thresh,
                iou=prediction_iou,
                max_det=max_det,
                device=device,
                verbose=False,
                half=False,
            )[0]

            if res_tile.boxes is not None and len(res_tile.boxes) > 0:
                for b in res_tile.boxes:
                    cid = int(b.cls[0].item())
                    # STRICT FILTER: Retain ONLY Capacitor predictions (Class 0) from crops!
                    if cid == 0:
                        sc = float(b.conf[0].item())
                        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                        # Restore to original board pixel coordinates
                        gx1 = max(0.0, min(float(iw), bx1 + sx1))
                        gy1 = max(0.0, min(float(ih), by1 + sy1))
                        gx2 = max(0.0, min(float(iw), bx2 + sx1))
                        gy2 = max(0.0, min(float(ih), by2 + sy1))
                        cap_boxes.append([gx1, gy1, gx2, gy2])
                        cap_scores.append(sc)

    # -------------------------------------------------------------
    # Class-Aware Merging
    # -------------------------------------------------------------
    final_dets = []

    # 1. Non-capacitor classes: Preserved strictly from Branch A
    if non_cap_boxes:
        final_dets.extend((int(cid), float(score), *map(float, box))
                          for cid, score, box in zip(non_cap_classes, non_cap_scores, non_cap_boxes))

    # 2. Capacitor class: Merge seam cuts and fuse duplicates
    if cap_boxes:
        cb = torch.tensor(cap_boxes, dtype=torch.float32)
        cs = torch.tensor(cap_scores, dtype=torch.float32)
        cc = torch.zeros(len(cb), dtype=torch.int64)

        # Seam-crossing fragment consolidation (IoS)
        if run_crops and len(cb) > 1:
            cb, cs, cc = merge_border_truncated_boxes(cb, cs, cc, ios_thresh=ios_thresh)

        # Duplicate fusion (NMS or WBF)
        if fusion_method == "wbf":
            fused_cb, fused_cs = fuse_capacitor_predictions_wbf(cb, cs, iw, ih, iou_thr=merge_iou)
        else:
            keep_cap = torchvision.ops.nms(cb, cs, iou_threshold=merge_iou)
            fused_cb = cb[keep_cap]
            fused_cs = cs[keep_cap]

        for i in range(len(fused_cb)):
            final_dets.append((
                0,
                float(fused_cs[i].item()),
                float(fused_cb[i, 0].item()),
                float(fused_cb[i, 1].item()),
                float(fused_cb[i, 2].item()),
                float(fused_cb[i, 3].item()),
            ))

    # Reserve slots for global non-capacitor predictions so crop additions cannot evict them.
    macro = [d for d in final_dets if d[0] != 0]
    capacitors = sorted((d for d in final_dets if d[0] == 0), key=lambda d: -d[1])
    final_dets = macro + capacitors[:max(0, max_det - len(macro))]
    final_dets.sort(key=lambda x: -x[1])

    return final_dets, num_tiles


def compute_metrics_at_iou(
    preds_by_class: Dict[int, list],
    gts_by_class: Dict[int, list],
    iou_thresh: float = 0.50,
) -> Dict[str, dict]:
    """Computes exact continuous interpolated AP, TP, FP, FN, Precision, and Recall."""
    metrics_per_class = {}
    for cid in range(len(CLASS_NAMES)):
        cname = CLASS_NAMES[cid]
        preds = sorted(preds_by_class.get(cid, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cid, [])
        n_gt = len(gts)
        n_pred = len(preds)

        if n_gt == 0:
            metrics_per_class[cname] = {
                "ap": 0.0,
                "tp": 0,
                "fp": n_pred,
                "fn": 0,
                "n_gt": 0,
                "n_pred": n_pred,
                "precision": 0.0,
                "recall": 0.0,
            }
            continue

        gt_by_img = {}
        for idx, (img_id, *box) in enumerate(gts):
            gt_by_img.setdefault(img_id, []).append((idx, box))

        matched_gt = set()
        tp = np.zeros(n_pred)
        fp = np.zeros(n_pred)

        for i, (img_id, score, x1, y1, x2, y2) in enumerate(preds):
            pbox = (x1, y1, x2, y2)
            candidates = gt_by_img.get(img_id, [])
            best_iou, best_idx = 0.0, -1

            for gidx, gbox in candidates:
                if (img_id, gidx) in matched_gt:
                    continue
                ax1, ay1, ax2, ay2 = pbox
                bx1, by1, bx2, by2 = gbox
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gidx

            if best_iou >= iou_thresh and best_idx >= 0:
                matched_gt.add((img_id, best_idx))
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        total_tp = int(np.sum(tp))
        total_fp = int(np.sum(fp))
        total_fn = max(0, n_gt - total_tp)
        precision = total_tp / max(1, n_pred)
        recall = total_tp / n_gt

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / n_gt
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for j in range(len(mpre) - 2, -1, -1):
            mpre[j] = max(mpre[j], mpre[j + 1])
        i_change = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[i_change + 1] - mrec[i_change]) * mpre[i_change + 1]))

        metrics_per_class[cname] = {
            "ap": ap,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "precision": float(precision),
            "recall": float(recall),
        }

    return metrics_per_class


def main():
    args = parse_args()
    print("=" * 80)
    print("DUAL-BRANCH FULL-BOARD + CAPACITOR CROP SPECIALIST EVALUATION")
    print("=" * 80)
    print(f"Global Model:     {args.global_weights}")
    crop_w_path = args.crop_weights or args.global_weights
    print(f"Crop Specialist:  {crop_w_path}")
    print(f"Dataset Split:    {args.split_name} ({args.images_dir})")
    print(f"Pipeline Config:  slice_size={args.slice_size}, overlap={args.overlap}, fusion={args.fusion_method}")
    print(f"Thresholds:       conf={args.conf_thresh}, merge_iou={args.merge_iou}, ios={args.ios_thresh}")

    # Determine hardware info
    device_name = "CPU"
    if torch.cuda.is_available() and args.device != "cpu":
        dev_idx = int(args.device) if str(args.device).isdigit() else 0
        device_name = torch.cuda.get_device_name(dev_idx)
    print(f"Hardware Device:  {device_name} (Device: {args.device})")
    print(f"Precision:        FP32 (PyTorch/Ultralytics default inference)")
    print("=" * 80)

    model_global = YOLO(args.global_weights)
    model_crop = YOLO(crop_w_path)

    images_p = Path(args.images_dir)
    labels_p = Path(args.labels_dir)
    image_files = sorted([f for f in images_p.iterdir() if f.suffix in VALID_EXTS])
    print(f"Found {len(image_files)} images in {args.split_name} split.")

    # Load ground truths
    gts_by_class: Dict[int, list] = {c: [] for c in range(len(CLASS_NAMES))}
    for img_p in image_files:
        stem = img_p.stem
        lbl_p = labels_p / f"{stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if 0 <= cid < len(CLASS_NAMES):
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2.0) * iw
                        y1 = (cy - bh / 2.0) * ih
                        x2 = (cx + bw / 2.0) * iw
                        y2 = (cy + bh / 2.0) * ih
                        gts_by_class[cid].append((stem, x1, y1, x2, y2))

    # GPU warmup
    if torch.cuda.is_available() and args.device != "cpu":
        dummy = np.zeros((args.slice_size, args.slice_size, 3), dtype=np.uint8)
        for _ in range(5):
            model_global.predict(dummy, imgsz=args.imgsz or args.slice_size, verbose=False, device=args.device, half=False)
            model_crop.predict(dummy, imgsz=args.imgsz or args.slice_size, verbose=False, device=args.device, half=False)
        torch.cuda.synchronize()

    # Inference loop
    preds_by_class: Dict[int, list] = {c: [] for c in range(len(CLASS_NAMES))}
    total_tiles = 0
    t0 = time.time()

    for idx, img_p in enumerate(image_files):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        dets, n_tiles = run_dual_branch_inference(
            model_global=model_global,
            model_crop=model_crop,
            img_bgr=img_bgr,
            slice_size=args.slice_size,
            overlap=args.overlap,
            conf_thresh=args.conf_thresh,
            merge_iou=args.merge_iou,
            ios_thresh=args.ios_thresh,
            max_det=args.max_det,
            fusion_method=args.fusion_method,
            device=args.device,
            run_global=not args.no_global,
            run_crops=not args.no_crops,
            imgsz=args.imgsz,
            prediction_iou=args.prediction_iou,
        )
        total_tiles += n_tiles

        for cid, sc, x1, y1, x2, y2 in dets:
            preds_by_class[cid].append((img_p.stem, sc, x1, y1, x2, y2))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
            print(f"  Processed {idx + 1}/{len(image_files)} boards...")

    if torch.cuda.is_available() and args.device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    ms_per_board = (elapsed / len(image_files)) * 1000.0 if image_files else 0.0
    avg_tiles_per_board = total_tiles / len(image_files) if image_files else 0.0

    # Compute metrics at IoU 0.50
    metrics50 = compute_metrics_at_iou(preds_by_class, gts_by_class, iou_thresh=0.50)
    ap50_dict = {c: metrics50[c]["ap"] for c in CLASS_NAMES}
    mAP50 = float(np.mean(list(ap50_dict.values())))

    # 10-point COCO style mAP50:95 and per-class AP50:95
    iou_levels = [round(x, 2) for x in np.linspace(0.50, 0.95, 10)]
    per_class_all_levels: Dict[str, list] = {c: [] for c in CLASS_NAMES}
    for iou_thr in iou_levels:
        lvl_metrics = compute_metrics_at_iou(preds_by_class, gts_by_class, iou_thresh=iou_thr)
        for cname in CLASS_NAMES:
            per_class_all_levels[cname].append(lvl_metrics[cname]["ap"])

    ap50_95_dict = {c: float(np.mean(per_class_all_levels[c])) for c in CLASS_NAMES}
    mAP50_95 = float(np.mean(list(ap50_95_dict.values())))

    print("\n" + "=" * 90)
    print(f"EVALUATION RESULTS ({args.split_name.upper()} SET - {len(image_files)} BOARDS)")
    print("=" * 90)
    print(f"{'Class':<24} | {'AP@50':<8} | {'AP@50:95':<9} | {'Recall':<8} | {'Prec':<8} | {'TP':<6} | {'FP(Errors)':<10} | {'FN':<6}")
    print("-" * 90)
    for cname in CLASS_NAMES:
        m = metrics50[cname]
        ap50 = m["ap"] * 100
        ap50_95 = ap50_95_dict[cname] * 100
        rec = m["recall"] * 100
        prc = m["precision"] * 100
        print(f"{cname:<24} | {ap50:6.2f}% | {ap50_95:7.2f}% | {rec:6.2f}% | {prc:6.2f}% | {m['tp']:<6} | {m['fp']:<10} | {m['fn']:<6}")
    print("-" * 90)
    print(f"{'Overall Mean':<24} | {mAP50 * 100:6.2f}% | {mAP50_95 * 100:7.2f}% |")
    print(f"{'Latency per Board':<24} | {ms_per_board:6.1f} ms ({1000.0/max(1.0, ms_per_board):.1f} FPS)")
    print(f"{'Avg Tiles per Board':<24} | {avg_tiles_per_board:.1f} tiles (total {total_tiles} crops)")
    print(f"{'Hardware':<24} | {device_name}")
    print("=" * 90)

    # Save metrics JSON
    metrics = {
        "global_weights": args.global_weights,
        "crop_weights": crop_w_path,
        "split": args.split_name,
        "num_boards": len(image_files),
        "slice_size": args.slice_size,
        "imgsz": args.imgsz or args.slice_size,
        "prediction_iou": args.prediction_iou,
        "overlap": args.overlap,
        "fusion_method": args.fusion_method,
        "conf_thresh": args.conf_thresh,
        "merge_iou": args.merge_iou,
        "ios_thresh": args.ios_thresh,
        "max_det": args.max_det,
        "hardware_device": device_name,
        "precision": "FP32",
        "avg_tiles_per_board": avg_tiles_per_board,
        "total_tiles": total_tiles,
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "per_class_ap50": ap50_dict,
        "per_class_ap50_95": ap50_95_dict,
        "detailed_metrics_iou50": metrics50,
        "ms_per_board": ms_per_board,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved results to: {out_p}")


if __name__ == "__main__":
    main()

