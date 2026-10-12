#!/usr/bin/env python3
"""
tools/eval_adaptive_gamma_benchmark.py

Fast test-time evaluation of Adaptive Gamma Correction on PCB detection:
1. Evaluates Raw Baseline on the 44 held-out test images
2. Evaluates Adaptive Gamma Correction with Weighting Distribution (AGCWD) - Zero Canny!
3. Evaluates Local Bilateral Adaptive Gamma - Zero Canny!
4. Evaluates Dual-Exposure Weighted Box Fusion (Raw + AGCWD)

Outputs exact mAP@50, mAP@50-95, precision, recall, and per-class AP breakdown.
Saves visual comparison figures showing raw vs adaptive gamma PCB crops.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Adaptive Gamma Algorithms (Zero Canny Edge Filtering!)
# ---------------------------------------------------------------------------

def apply_agcwd(image_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Adaptive Gamma Correction with Weighting Distribution (AGCWD, Huang et al. 2013).
    Smooth non-linear intensity mapping based on normalized luminance PDF/CDF.
    No binary edge detection, no spatial artifacts.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    hist, _ = np.histogram(v, bins=256, range=(0, 256))
    pdf = hist / float(v.size)
    p_min, p_max = pdf.min(), pdf.max()

    if p_max > p_min:
        pdf_w = p_max * (((pdf - p_min) / (p_max - p_min)) ** alpha)
    else:
        pdf_w = pdf

    cdf_w = np.cumsum(pdf_w) / np.maximum(np.sum(pdf_w), 1e-9)

    l_norm = np.arange(256) / 255.0
    gamma = 1.0 - cdf_w
    lut = np.clip(255.0 * (l_norm ** gamma), 0, 255).astype(np.uint8)

    hsv[:, :, 2] = cv2.LUT(v, lut)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_bilateral_adaptive_gamma(image_bgr: np.ndarray) -> np.ndarray:
    """
    Local Bilateral Adaptive Gamma.
    Uses edge-preserving bilateral smoothing to estimate smooth illumination L,
    then adapts gamma per pixel without any Canny or high-frequency edge filtering:
        gamma(x,y) = 0.6 + 0.5 * (L(x,y) / 255.0)
    Dark shadow areas get gamma < 1 (brightened), bright solder pads get gamma ~ 1.1 (highlight protected).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # Fast downscaled bilateral estimate for high speed
    h, w = v.shape
    scale = 640.0 / max(h, w)
    small_v = cv2.resize(v, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_l = cv2.bilateralFilter(small_v, d=7, sigmaColor=50, sigmaSpace=50)
    l_map = cv2.resize(small_l, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    # Local gamma
    gamma_map = 0.65 + 0.45 * l_map  # Range: [0.65 (dark) to 1.10 (glare)]

    v_norm = v.astype(np.float32) / 255.0
    v_corr = np.clip(255.0 * np.power(v_norm, gamma_map), 0, 255).astype(np.uint8)

    hsv[:, :, 2] = v_corr
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Vectorized Weighted Box Fusion (WBF)
# ---------------------------------------------------------------------------

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def wbf_fusion(boxes_list: List[np.ndarray], scores_list: List[np.ndarray], labels_list: List[np.ndarray],
               weights: List[float] = None, iou_thr: float = 0.55, skip_thr: float = 0.001):
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    all_boxes, all_scores, all_labels = [], [], []
    for m in range(n_models):
        b, s, l = boxes_list[m], scores_list[m], labels_list[m]
        if len(b) == 0:
            continue
        keep = s >= skip_thr
        b, s, l = b[keep], s[keep] * weights[m], l[keep]
        for i in range(len(b)):
            all_boxes.append(b[i])
            all_scores.append(s[i])
            all_labels.append(l[i])

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    final_boxes, final_scores, final_labels = [], [], []
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        c_boxes, c_scores = all_boxes[mask], all_scores[mask]
        order = np.argsort(-c_scores)
        c_boxes, c_scores = c_boxes[order], c_scores[order]

        clusters_b, clusters_s = [], []
        for b, s in zip(c_boxes, c_scores):
            matched = False
            for idx in range(len(clusters_b)):
                if compute_iou(b, np.array([clusters_b[idx][-1]]))[0] > iou_thr:
                    clusters_b[idx].append(b)
                    clusters_s[idx].append(s)
                    matched = True
                    break
            if not matched:
                clusters_b.append([b])
                clusters_s.append([s])

        for b_clust, s_clust in zip(clusters_b, clusters_s):
            b_arr = np.array(b_clust)
            s_arr = np.array(s_clust)
            tot_w = np.sum(s_arr)
            fused_b = np.sum(b_arr * s_arr[:, None], axis=0) / max(tot_w, 1e-9)
            fused_s = (tot_w / len(b_arr)) * min(1.0, len(b_arr) / float(n_models))
            final_boxes.append(fused_b)
            final_scores.append(fused_s)
            final_labels.append(cls)

    if not final_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    final_boxes = np.array(final_boxes)
    final_scores = np.array(final_scores)
    final_labels = np.array(final_labels)
    order = np.argsort(-final_scores)
    return final_boxes[order], final_scores[order], final_labels[order]


# ---------------------------------------------------------------------------
# Label Parsing & AP Metric Calculation
# ---------------------------------------------------------------------------

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


def compute_ap(preds_by_class: Dict[int, List], gts_by_class: Dict[int, List],
               class_ids: List[int], iou_thresh: float = 0.5) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    ap_res, prec_res, rec_res = {}, {}, {}
    for cid in class_ids:
        preds = sorted(preds_by_class.get(cid, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cid, [])
        n_gt = len(gts)
        if n_gt == 0:
            ap_res[cid], prec_res[cid], rec_res[cid] = 0.0, 0.0, 0.0
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
            prec_res[cid], rec_res[cid] = 0.0, 0.0

    return ap_res, prec_res, rec_res


# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------

def run_evaluation():
    img_dir = Path("data_samples/native_images")
    lbl_dir = Path("data_samples/native_cleaned/labels")
    weights_path = Path("runs/rectified_yolov26s_unified_1280/pcb-filtered/weights/best.pt")

    class_names = {0: "Capacitor", 1: "Connector", 2: "Electrolytic Capacitor", 3: "IC"}
    class_ids = sorted(class_names.keys())

    images = sorted(list(img_dir.glob("*.jpg")))
    print("=" * 80)
    print("🔬 TEST-TIME ADAPTIVE GAMMA CORRECTION EVALUATION (NO CANNY EDGES)")
    print(f"Model:        {weights_path.name}")
    print(f"Test Set:     {len(images)} images ({lbl_dir})")
    print(f"Resolution:   1280px | max_det=1000 | conf=0.001")
    print("=" * 80)

    model = YOLO(str(weights_path))

    # Ground truth
    gts = {c: [] for c in class_ids}
    for img_id, img_path in enumerate(images):
        img_temp = cv2.imread(str(img_path))
        h, w = img_temp.shape[:2]
        lbl_file = lbl_dir / f"{img_path.stem}.txt"
        for cid, x1, y1, x2, y2 in parse_labels(lbl_file, w, h):
            if cid in gts:
                gts[cid].append((img_id, x1, y1, x2, y2))

    modes = ["raw", "agcwd_0.5", "bilateral_adaptive_gamma", "dual_exposure_wbf"]
    results = {}

    for mode in modes:
        print(f"\n🚀 Evaluating Mode: [{mode}]...")
        preds = {c: [] for c in class_ids}
        t0 = time.time()

        for img_id, img_path in enumerate(images):
            img_bgr = cv2.imread(str(img_path))

            if mode == "raw":
                res = model.predict(img_bgr, imgsz=1280, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
                b_np = res.boxes.xyxy.cpu().numpy()
                s_np = res.boxes.conf.cpu().numpy()
                l_np = res.boxes.cls.cpu().numpy().astype(int)

            elif mode == "agcwd_0.5":
                img_enh = apply_agcwd(img_bgr, alpha=0.5)
                res = model.predict(img_enh, imgsz=1280, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
                b_np = res.boxes.xyxy.cpu().numpy()
                s_np = res.boxes.conf.cpu().numpy()
                l_np = res.boxes.cls.cpu().numpy().astype(int)

            elif mode == "bilateral_adaptive_gamma":
                img_enh = apply_bilateral_adaptive_gamma(img_bgr)
                res = model.predict(img_enh, imgsz=1280, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
                b_np = res.boxes.xyxy.cpu().numpy()
                s_np = res.boxes.conf.cpu().numpy()
                l_np = res.boxes.cls.cpu().numpy().astype(int)

            elif mode == "dual_exposure_wbf":
                # Pass 1: Raw
                res_raw = model.predict(img_bgr, imgsz=1280, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
                b_raw = res_raw.boxes.xyxy.cpu().numpy()
                s_raw = res_raw.boxes.conf.cpu().numpy()
                l_raw = res_raw.boxes.cls.cpu().numpy().astype(int)

                # Pass 2: AGCWD
                img_enh = apply_agcwd(img_bgr, alpha=0.5)
                res_enh = model.predict(img_enh, imgsz=1280, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
                b_enh = res_enh.boxes.xyxy.cpu().numpy()
                s_enh = res_enh.boxes.conf.cpu().numpy()
                l_enh = res_enh.boxes.cls.cpu().numpy().astype(int)

                # WBF fusion of raw + agcwd
                b_np, s_np, l_np = wbf_fusion([b_raw, b_enh], [s_raw, s_enh], [l_raw, l_enh],
                                              weights=[1.0, 1.0], iou_thr=0.55, skip_thr=0.001)

            for bx, sc, lb in zip(b_np, s_np, l_np):
                cid = int(lb)
                if cid in preds:
                    preds[cid].append((img_id, float(sc), bx[0], bx[1], bx[2], bx[3]))

        dt = time.time() - t0
        ap_dict, prec_dict, rec_dict = compute_ap(preds, gts, class_ids, iou_thresh=0.5)
        map50 = float(np.mean(list(ap_dict.values())))

        # Compute mAP50-95
        map_list = []
        for thresh in np.linspace(0.50, 0.95, 10):
            ap_t, _, _ = compute_ap(preds, gts, class_ids, iou_thresh=float(thresh))
            map_list.append(np.mean(list(ap_t.values())))
        map50_95 = float(np.mean(map_list))

        mean_p = float(np.mean(list(prec_dict.values())))
        mean_r = float(np.mean(list(rec_dict.values())))
        f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r + 1e-9)

        results[mode] = {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "precision": mean_p,
            "recall": mean_r,
            "f1": f1,
            "ap_per_class": ap_dict,
            "time_s": dt
        }

        print(f"  -> mAP@50: {map50*100:.2f}% | mAP@50-95: {map50_95*100:.2f}% | Recall: {mean_r*100:.2f}% | Time: {dt:.1f}s")
        for cid in class_ids:
            print(f"     {class_names[cid]:<24}: {ap_dict[cid]*100:.2f}%")

    # Print comparative summary table
    print("\n" + "=" * 80)
    print("🏆 FINAL COMPARATIVE BENCHMARK: ADAPTIVE GAMMA VS RAW BASELINE")
    print("=" * 80)
    header = f"{'Configuration':<26} | {'mAP@50':<8} | {'mAP50-95':<8} | {'Recall':<8} | {'Capacitor':<10} | {'IC':<8} | {'Connector':<10}"
    print(header)
    print("-" * 80)
    for m, r in results.items():
        ap = r["ap_per_class"]
        row = f"{m:<26} | {r['mAP50']*100:>6.2f}% | {r['mAP50_95']*100:>6.2f}% | {r['recall']*100:>6.2f}% | {ap[0]*100:>8.2f}% | {ap[3]*100:>6.2f}% | {ap[1]*100:>8.2f}%"
        print(row)
    print("=" * 80)

    # Generate a publication-quality visual figure
    out_fig = Path("results/adaptive_gamma_inspection.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    demo_img_path = images[0]
    demo_bgr = cv2.imread(str(demo_img_path))
    demo_agcwd = apply_agcwd(demo_bgr, alpha=0.5)
    demo_bilat = apply_bilateral_adaptive_gamma(demo_bgr)

    # Crop a dense capacitor/IC region
    h, w = demo_bgr.shape[:2]
    crop_y1, crop_y2 = int(h * 0.25), int(h * 0.45)
    crop_x1, crop_x2 = int(w * 0.20), int(w * 0.45)

    c_raw = cv2.cvtColor(demo_bgr[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_BGR2RGB)
    c_agcwd = cv2.cvtColor(demo_agcwd[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_BGR2RGB)
    c_bilat = cv2.cvtColor(demo_bilat[crop_y1:crop_y2, crop_x1:crop_x2], cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    axes[0].imshow(c_raw)
    axes[0].set_title("1. Raw Optical Capture", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(c_agcwd)
    axes[1].set_title("2. Adaptive Gamma (AGCWD, α=0.5)\n[Smooth Intensity Tone Mapping - No Canny]", fontsize=12, fontweight="bold", color="darkgreen")
    axes[1].axis("off")

    axes[2].imshow(c_bilat)
    axes[2].set_title("3. Bilateral Adaptive Gamma\n[Local Smooth Illumination Guidance - No Canny]", fontsize=12, fontweight="bold", color="darkblue")
    axes[2].axis("off")

    plt.suptitle("Adaptive Gamma Visual Analysis (Zero Binary Edge Discontinuities)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_fig), bbox_inches="tight")
    plt.close()
    print(f"\n🖼️ Saved visual comparison figure to: {out_fig}")


if __name__ == "__main__":
    run_evaluation()
