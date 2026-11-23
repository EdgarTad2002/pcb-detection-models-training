#!/usr/bin/env python3
"""
tools/eval_sahi_native_test.py — Native-Resolution Sliced (SAHI) Benchmark Evaluator
===================================================================================
Evaluates trained PCB detection models on native-resolution PCB test images
using Slicing Aided Hyper Inference (SAHI) with Global-Local context preservation.

Key Capabilities & Fixes:
1. Strict parameter propagation:
   --max-det and --nms-iou are explicitly propagated to every model.predict() call
   (overriding Ultralytics defaults max_det=300 and iou=0.7).
2. Border-truncated box handling:
   Implements Intersection-over-Smaller (IoS) merging across tile boundaries
   so partial boxes sliced at tile edges are merged into complete detections
   instead of producing false positives.
3. Apple-to-apple baseline verification:
   Supports --no-slices to evaluate the un-sliced standard full-board model through
   the exact same evaluation protocol and AP matcher, guaranteeing fair comparison.
4. Robust image & annotation discovery:
   Case-insensitive globbing (.jpg, .jpeg, .png, .bmp, uppercase variants).
   Unlabeled images are retained with empty ground truth (gts=[]) so background
   false positives are strictly counted.
5. Class ID bounds guards:
   Silently ignores or filters out unexpected class indices outside [0, nc-1].
6. Hardware-synchronized latency measurement:
   Warms up CUDA across 5 iterations before starting the timer, and calls
   torch.cuda.synchronize() for honest latency reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path before custom model imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-register custom modules for unpickling
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
    p = argparse.ArgumentParser(description="Evaluate a model with Sliced / SAHI Inference on PCB Test Set.")
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    p.add_argument(
        "--test-images",
        type=str,
        default="datasets/pcb-native-res-cleaned-1280/test/images",
        help="Directory of native-res test images",
    )
    p.add_argument(
        "--test-labels",
        type=str,
        default="datasets/pcb-native-res-cleaned-1280/test/labels",
        help="Directory of native-res test labels",
    )
    p.add_argument("--slice-size", type=int, default=1280, help="Tile slice width and height in pixels")
    p.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio between adjacent slices (0.0 - 0.5)")
    p.add_argument("--conf", type=float, default=0.001, help="Evaluation confidence threshold (default 0.001)")
    p.add_argument("--iou", type=float, default=0.50, help="Evaluation IoU match threshold (default 0.50)")
    p.add_argument("--nms-iou", type=float, default=0.50, help="Tile merging NMS IoU threshold")
    p.add_argument("--ios-thresh", type=float, default=0.65, help="Intersection-over-Smaller threshold for edge seam merging")
    p.add_argument("--max-det", type=int, default=1000, help="Maximum detections per board (default 1000)")
    p.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    p.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    p.add_argument("--no-full-frame", action="store_true", help="Disable global full-frame context pass")
    p.add_argument("--no-slices", action="store_true", help="Disable slicing (runs standard un-sliced baseline through this evaluator)")
    return p.parse_args()


def discover_test_images(images_dir: Path) -> List[Path]:
    """Discovers all images regardless of extension case."""
    all_files = sorted(images_dir.iterdir())
    return [f for f in all_files if f.is_file() and f.suffix in VALID_EXTS]


def load_ground_truths(labels_dir: Path, image_paths: List[Path]) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """
    Loads all ground-truth bounding boxes converted to absolute [x1, y1, x2, y2].
    Crucial fix: Unlabeled images are NOT skipped; they receive an empty list []
    so any background false positives on those images are counted.
    """
    gts_by_image = {}
    for img_p in image_paths:
        stem = img_p.stem
        lbl_p = labels_dir / f"{stem}.txt"

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        boxes = []
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
                        boxes.append((cid, x1, y1, x2, y2))
        gts_by_image[stem] = boxes
    return gts_by_image


def get_slice_boxes(img_w: int, img_h: int, slice_w: int, slice_h: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Computes sliding window coordinates covering the entire canvas with guaranteed coverage."""
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
    """
    Merges border-truncated boxes using Intersection-over-Smaller (IoS).
    When a component is cut at a tile seam, one tile sees a partial sliver while an adjacent
    tile sees a larger portion or the full object. Plain IoU fails to suppress the partial box
    because (intersection / union) < 0.50, but (intersection / smaller_area) >= 0.65.
    """
    if len(boxes) <= 1:
        return boxes, scores, classes

    # Sort descending by area so larger/more complete boxes act as suppression targets
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

        # Compute intersection
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


def run_inference_on_board(
    model: YOLO,
    img_bgr: np.ndarray,
    slice_size: int,
    overlap: float,
    conf_thresh: float,
    nms_iou: float,
    ios_thresh: float,
    include_full_image: bool,
    run_slicing: bool,
    device: str,
    max_det: int,
) -> List[Tuple[int, float, float, float, float, float]]:
    """Runs global + sliced tile inference on a single image and merges detections."""
    ih, iw = img_bgr.shape[:2]
    all_boxes = []
    all_scores = []
    all_classes = []

    # 1. Global context pass (resized to slice_size, e.g. 1280px)
    if include_full_image:
        res_full = model.predict(
            img_bgr,
            imgsz=slice_size,
            conf=conf_thresh,
            iou=nms_iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )[0]
        if res_full.boxes is not None and len(res_full.boxes) > 0:
            for b in res_full.boxes:
                cid = int(b.cls[0].item())
                if 0 <= cid < len(CLASS_NAMES):
                    sc = float(b.conf[0].item())
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    all_boxes.append([bx1, by1, bx2, by2])
                    all_scores.append(sc)
                    all_classes.append(cid)

    # 2. Local high-resolution slices (1:1 optical scale)
    if run_slicing:
        slice_coords = get_slice_boxes(iw, ih, slice_size, slice_size, overlap)
        for (sx1, sy1, sx2, sy2) in slice_coords:
            tile = img_bgr[sy1:sy2, sx1:sx2]
            th, tw = tile.shape[:2]
            if th == 0 or tw == 0:
                continue

            res_tile = model.predict(
                tile,
                imgsz=max(tw, th),
                conf=conf_thresh,
                iou=nms_iou,
                max_det=max_det,
                device=device,
                verbose=False,
            )[0]
            if res_tile.boxes is not None and len(res_tile.boxes) > 0:
                for b in res_tile.boxes:
                    cid = int(b.cls[0].item())
                    if 0 <= cid < len(CLASS_NAMES):
                        sc = float(b.conf[0].item())
                        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                        # Reproject back to global canvas
                        all_boxes.append([bx1 + sx1, by1 + sy1, bx2 + sx1, by2 + sy1])
                        all_scores.append(sc)
                        all_classes.append(cid)

    if not all_boxes:
        return []

    t_boxes = torch.tensor(all_boxes, dtype=torch.float32)
    t_scores = torch.tensor(all_scores, dtype=torch.float32)
    t_classes = torch.tensor(all_classes, dtype=torch.int64)

    # 3. Resolve border-truncated boxes using Intersection-over-Smaller (IoS)
    if run_slicing and len(t_boxes) > 1:
        t_boxes, t_scores, t_classes = merge_border_truncated_boxes(
            t_boxes, t_scores, t_classes, ios_thresh=ios_thresh
        )

    # 4. Batched NMS across all proposals on the global canvas
    keep = torchvision.ops.batched_nms(t_boxes, t_scores, t_classes, iou_threshold=nms_iou)
    if len(keep) > max_det:
        keep = keep[:max_det]

    merged_dets = []
    for idx in keep:
        cid = int(t_classes[idx].item())
        sc = float(t_scores[idx].item())
        bx = t_boxes[idx].tolist()
        merged_dets.append((cid, sc, bx[0], bx[1], bx[2], bx[3]))
    return merged_dets


def compute_voc_ap(preds_by_class: Dict[int, list], gts_by_class: Dict[int, list], iou_thresh: float = 0.50) -> Dict[str, float]:
    """Computes exact VOC/COCO all-points continuous interpolated AP at iou_thresh."""
    per_class_ap = {}
    for cid in range(len(CLASS_NAMES)):
        cname = CLASS_NAMES[cid]
        preds = sorted(preds_by_class.get(cid, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cid, [])
        n_gt = len(gts)
        if n_gt == 0:
            per_class_ap[cname] = 0.0
            continue

        gt_by_img = {}
        for idx, (img_id, *box) in enumerate(gts):
            gt_by_img.setdefault(img_id, []).append((idx, box))

        matched_gt = set()
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

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

            if best_iou >= iou_thresh and best_idx != -1:
                tp[i] = 1.0
                matched_gt.add((img_id, best_idx))
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / float(n_gt)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for j in range(len(mpre) - 2, -1, -1):
            mpre[j] = max(mpre[j], mpre[j + 1])
        change_idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))
        per_class_ap[cname] = ap

    return per_class_ap


def main():
    args = parse_args()
    test_img_dir = Path(args.test_images)
    test_lbl_dir = Path(args.test_labels)

    mode_label = "STANDARD (UN-SLICED) BASELINE" if args.no_slices else "SLICED AIDED HYPER INFERENCE (SAHI)"
    print("=" * 70)
    print(f"{mode_label} EVALUATOR")
    print("=" * 70)
    print(f"Weights:        {args.weights}")
    print(f"Test Images:    {test_img_dir}")
    print(f"Tile Slicing:   {'Disabled' if args.no_slices else f'{args.slice_size}x{args.slice_size}px (overlap={int(args.overlap*100)}%)'}")
    print(f"Global Pass:    {'Enabled' if not args.no_full_frame else 'Disabled'}")
    print(f"Evaluation:     conf={args.conf}, iou={args.iou}, nms_iou={args.nms_iou}, max_det={args.max_det}")
    print(f"IoS Edge Seam:  {args.ios_thresh}")
    print("=" * 70)

    # 1. Load model
    print("Loading model...")
    model = YOLO(args.weights)

    # 2. Discover test images
    img_paths = discover_test_images(test_img_dir)
    assert len(img_paths) > 0, f"No valid images found in {test_img_dir}"
    print(f"Discovered {len(img_paths)} test images.")

    # 3. Load ground truths
    gt_dict = load_ground_truths(test_lbl_dir, img_paths)
    gts_by_class = {cid: [] for cid in range(len(CLASS_NAMES))}
    for img_id, img_p in enumerate(img_paths):
        boxes = gt_dict.get(img_p.stem, [])
        for cid, x1, y1, x2, y2 in boxes:
            gts_by_class[cid].append((img_id, x1, y1, x2, y2))

    total_gts = sum(len(v) for v in gts_by_class.values())
    print(f"Loaded ground-truth annotations: {total_gts} objects total (Capacitors: {len(gts_by_class[0])})")

    # 4. Hardware Warmup (5 iterations)
    if "cuda" in str(args.device) or str(args.device).isdigit():
        print("Performing GPU warmup (5 passes)...")
        dummy = np.zeros((args.slice_size, args.slice_size, 3), dtype=np.uint8)
        for _ in range(5):
            _ = model.predict(dummy, imgsz=args.slice_size, device=args.device, verbose=False)
        torch.cuda.synchronize()

    # 5. Run Inference
    print(f"\nRunning {mode_label} on {len(img_paths)} Test Images...")
    if torch.cuda.is_available() and ("cuda" in str(args.device) or str(args.device).isdigit()):
        torch.cuda.synchronize()
    t0 = time.time()

    preds_by_class = {cid: [] for cid in range(len(CLASS_NAMES))}
    total_slices_processed = 0

    for img_id, img_p in enumerate(img_paths):
        raw = cv2.imread(str(img_p))
        if raw is None:
            continue
        ih, iw = raw.shape[:2]

        if not args.no_slices:
            slices = get_slice_boxes(iw, ih, args.slice_size, args.slice_size, args.overlap)
            total_slices_processed += len(slices)

        dets = run_inference_on_board(
            model=model,
            img_bgr=raw,
            slice_size=args.slice_size,
            overlap=args.overlap,
            conf_thresh=args.conf,
            nms_iou=args.nms_iou,
            ios_thresh=args.ios_thresh,
            include_full_image=not args.no_full_frame,
            run_slicing=not args.no_slices,
            device=args.device,
            max_det=args.max_det,
        )

        for cid, sc, x1, y1, x2, y2 in dets:
            preds_by_class[cid].append((img_id, sc, x1, y1, x2, y2))

        if (img_id + 1) % 10 == 0 or (img_id + 1) == len(img_paths):
            slice_info = f" ({total_slices_processed} slices)" if not args.no_slices else ""
            print(f"  Processed {img_id + 1}/{len(img_paths)} images{slice_info}...")

    if torch.cuda.is_available() and ("cuda" in str(args.device) or str(args.device).isdigit()):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    fps = len(img_paths) / max(elapsed, 1e-4)
    print(f"\nInference completed in {elapsed:.2f}s ({fps:.2f} boards/sec, {elapsed*1000/len(img_paths):.1f} ms/board)")

    # 6. Compute exact metrics
    per_class_ap = compute_voc_ap(preds_by_class, gts_by_class, iou_thresh=args.iou)
    mAP50 = float(np.mean(list(per_class_ap.values())))

    print("\n" + "=" * 70)
    print(f"{mode_label} RESULTS (Test Set @ Native Resolution)")
    print("=" * 70)
    for cname in CLASS_NAMES:
        print(f"  {cname:<24}: {per_class_ap[cname]*100:6.2f}% AP@50")
    print("-" * 70)
    print(f"  {'mAP@50 (Overall)':<24}: {mAP50*100:6.2f}%")
    print("=" * 70)

    # 7. Save results
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res_payload = {
            "model_weights": str(args.weights),
            "mode": "standard_unsliced" if args.no_slices else "sahi_sliced",
            "slice_size": args.slice_size if not args.no_slices else None,
            "overlap": args.overlap if not args.no_slices else None,
            "global_pass": not args.no_full_frame,
            "mAP50": mAP50,
            "per_class_ap50": per_class_ap,
            "elapsed_seconds": elapsed,
            "ms_per_board": elapsed * 1000.0 / len(img_paths),
            "boards_per_second": fps,
            "total_slices": total_slices_processed if not args.no_slices else len(img_paths),
            "max_det": args.max_det,
            "nms_iou": args.nms_iou,
            "ios_thresh": args.ios_thresh,
            "conf": args.conf,
            "iou": args.iou,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out_p.write_text(json.dumps(res_payload, indent=2))
        print(f"Results saved to: {out_p}")


if __name__ == "__main__":
    main()
