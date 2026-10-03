#!/usr/bin/env python3
"""
tools/demo_matched_filter.py
============================
Demonstrates Matched Filtering (Normalized Cross-Correlation Template Matching)
for detecting surface-mount micro-capacitors on dense PCBs.

1. Extracts canonical horizontal & vertical micro-capacitor templates from ground-truth.
2. Performs multi-template Normalized Cross-Correlation (cv2.TM_CCOEFF_NORMED).
3. Computes the maximum correlation response map R_max(x, y).
4. Extracts peak detection proposals using local maxima / NMS.
5. Evaluates recall against actual ground-truth capacitors.
6. Saves a multi-panel visual comparison figure to the artifact directory.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path("/home/edgar/.gemini/antigravity-ide/brain/fa71b4ef-8789-4a6e-9038-7acf3fb2be58")


def load_image_and_capacitors(img_path: Path, lbl_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    capacitors = []
    other_objects = []

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int(round((cx - bw / 2.0) * w))
            y1 = int(round((cy - bh / 2.0) * h))
            x2 = int(round((cx + bw / 2.0) * w))
            y2 = int(round((cy + bh / 2.0) * h))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if cls_id == 0:  # Capacitor
                capacitors.append((x1, y1, x2, y2, (x2 - x1), (y2 - y1)))
            else:
                other_objects.append((cls_id, x1, y1, x2, y2))

    return img, capacitors, other_objects


def extract_prototypes(img, capacitors):
    """Finds representative horizontal and vertical capacitor prototypes."""
    horiz_caps = [c for c in capacitors if c[4] > c[5] * 1.15 and c[4] >= 8 and c[5] >= 6]
    vert_caps = [c for c in capacitors if c[5] > c[4] * 1.15 and c[5] >= 8 and c[4] >= 6]

    def get_representative_crop(caps):
        if not caps:
            return None
        areas = [c[4] * c[5] for c in caps]
        med_idx = np.argsort(areas)[len(areas) // 2]
        x1, y1, x2, y2, bw, bh = caps[med_idx]
        return img[y1:y2, x1:x2].copy()

    proto_h = get_representative_crop(horiz_caps)
    proto_v = get_representative_crop(vert_caps)
    return proto_h, proto_v


def run_matched_filtering(img_gray, template_gray):
    """Computes Normalized Cross-Correlation and pads to original image size."""
    th, tw = template_gray.shape[:2]
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    pad_top = th // 2
    pad_bottom = th - 1 - pad_top
    pad_left = tw // 2
    pad_right = tw - 1 - pad_left
    res_padded = cv2.copyMakeBorder(res, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0.0)
    return res_padded, tw, th


def extract_peaks(corr_map, threshold=0.65, min_dist=10):
    """Finds local maxima peaks in the correlation response map above a threshold."""
    peaks = []
    peaks_mask = (corr_map >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_dist, min_dist))
    dilated = cv2.dilate(corr_map, kernel)
    local_max = (corr_map == dilated) & (peaks_mask > 0)
    
    y_idxs, x_idxs = np.where(local_max)
    for y, x in zip(y_idxs, x_idxs):
        score = float(corr_map[y, x])
        peaks.append((x, y, score))
    return peaks


def evaluate_peaks(peaks, ground_truth_caps, tolerance=15):
    """Evaluates how many GT capacitors are detected by correlation peaks."""
    tp = 0
    matched_gt = set()
    peak_matches = []

    for px, py, score in peaks:
        is_tp = False
        for g_idx, (gx1, gy1, gx2, gy2, gbw, gbh) in enumerate(ground_truth_caps):
            gcx = (gx1 + gx2) / 2.0
            gcy = (gy1 + gy2) / 2.0
            dist = np.hypot(px - gcx, py - gcy)
            if dist <= tolerance:
                is_tp = True
                matched_gt.add(g_idx)
                break
        peak_matches.append((px, py, score, is_tp))
        if is_tp:
            tp += 1

    recall = len(matched_gt) / max(1, len(ground_truth_caps))
    precision = tp / max(1, len(peaks))
    return recall, precision, peak_matches


def main():
    # Test on Native 1280px image for superior spatial clarity
    native_img = Path("data_samples/native_images/ATTIOT_Bottom_jpg.rf.94cd89169043c7506cde7ced6de19680.jpg")
    native_lbl = Path("data_samples/native_labels/ATTIOT_Bottom_jpg.rf.94cd89169043c7506cde7ced6de19680.txt")

    if not native_img.exists():
        # Fallback to 640px sample
        native_img = Path("data_samples/images/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.jpg")
        native_lbl = Path("data_samples/labels/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.txt")

    img_bgr, gt_caps, other_objs = load_image_and_capacitors(native_img, native_lbl)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    proto_h, proto_v = extract_prototypes(img_bgr, gt_caps)

    if proto_h is None or proto_v is None:
        print("Error: Could not extract prototypes from sample image.")
        return

    proto_h_gray = cv2.cvtColor(proto_h, cv2.COLOR_BGR2GRAY)
    proto_v_gray = cv2.cvtColor(proto_v, cv2.COLOR_BGR2GRAY)

    print(f"Loaded image: {native_img.name} ({w}x{h})")
    print(f"Ground-truth capacitors: {len(gt_caps)}")
    print(f"Horizontal Template size: {proto_h.shape[1]}x{proto_h.shape[0]} px")
    print(f"Vertical Template size:   {proto_v.shape[1]}x{proto_v.shape[0]} px")

    # Run matched filtering for both orientations
    corr_h, th_w, th_h = run_matched_filtering(img_gray, proto_h_gray)
    corr_v, tv_w, tv_h = run_matched_filtering(img_gray, proto_v_gray)

    # Combined correlation map (maximum response across orientations)
    corr_max = np.maximum(corr_h, corr_v)
    corr_clipped = np.clip(corr_max, 0, 1.0)

    # Threshold sweep
    thresholds = [0.55, 0.65, 0.75]
    print("\n--- Correlation Threshold Sensitivity Sweep ---")
    sweep_results = {}
    for t in thresholds:
        pks = extract_peaks(corr_clipped, threshold=t, min_dist=12)
        rec, prec, _ = evaluate_peaks(pks, gt_caps, tolerance=18)
        sweep_results[t] = (rec, prec, len(pks))
        print(f"Threshold = {t:.2f} -> Peaks: {len(pks):3d} | Recall: {rec*100:5.1f}% | Precision: {prec*100:5.1f}%")

    # Primary demonstration threshold
    demo_thresh = 0.65
    peaks = extract_peaks(corr_clipped, threshold=demo_thresh, min_dist=12)
    recall, precision, peak_matches = evaluate_peaks(peaks, gt_caps, tolerance=18)

    # Visualization: 4-Panel High-Resolution Dashboard
    fig = plt.figure(figsize=(20, 14), dpi=150)

    # Panel 1: Original Image with Ground Truth
    ax1 = fig.add_subplot(2, 2, 1)
    disp1 = img_rgb.copy()
    for gx1, gy1, gx2, gy2, _, _ in gt_caps:
        cv2.rectangle(disp1, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    ax1.imshow(disp1)
    ax1.set_title(f"A. Ground Truth Capacitors ({len(gt_caps)} instances)", fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Prototype Kernels and Correlation Theory
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#1A202C")
    # Display the two templates centered on a dark canvas
    canvas_h, canvas_w = 280, 500
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 35

    h_rgb = cv2.cvtColor(proto_h, cv2.COLOR_BGR2RGB)
    h_zoom = cv2.resize(h_rgb, (proto_h.shape[1] * 4, proto_h.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    v_rgb = cv2.cvtColor(proto_v, cv2.COLOR_BGR2RGB)
    v_zoom = cv2.resize(v_rgb, (proto_v.shape[1] * 4, proto_v.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

    # Place horizontal template
    y1_h = 30
    canvas[y1_h:y1_h + h_zoom.shape[0], 40:40 + h_zoom.shape[1]] = h_zoom

    # Place vertical template
    y1_v = 30
    canvas[y1_v:y1_v + v_zoom.shape[0], 270:270 + v_zoom.shape[1]] = v_zoom

    ax2.imshow(canvas)
    ax2.text(40 + h_zoom.shape[1]//2, y1_h + h_zoom.shape[0] + 30, f"Horizontal T_h\n({proto_h.shape[1]}x{proto_h.shape[0]} px, 4x zoom)",
             color="white", fontsize=10, ha="center")
    ax2.text(270 + v_zoom.shape[1]//2, y1_v + v_zoom.shape[0] + 30, f"Vertical T_v\n({proto_v.shape[1]}x{proto_v.shape[0]} px, 4x zoom)",
             color="white", fontsize=10, ha="center")
    ax2.set_title("B. Matched Filter Prototype Kernels T(x,y)\nNormalized Cross-Correlation: R = (I * T) / (||I|| ||T||)", fontsize=13, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Correlation Heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(corr_clipped, cmap="inferno", vmin=0.0, vmax=1.0)
    ax3.set_title("C. Max Normalized Cross-Correlation Heatmap R_max(x,y)", fontsize=13, fontweight="bold")
    ax3.axis("off")
    cbar = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation Coefficient R in [0, 1]", fontsize=10)

    # Panel 4: Local Peak Detections vs Ground Truth
    ax4 = fig.add_subplot(2, 2, 4)
    disp4 = img_rgb.copy()
    for gx1, gy1, gx2, gy2, _, _ in gt_caps:
        cv2.rectangle(disp4, (gx1, gy1), (gx2, gy2), (40, 180, 40), 1)

    for px, py, sc, is_tp in peak_matches:
        color = (0, 255, 255) if is_tp else (255, 60, 60)  # Yellow = TP, Red = FP
        cv2.circle(disp4, (px, py), 7, color, 2)
        cv2.drawMarker(disp4, (px, py), color, cv2.MARKER_CROSS, 10, 1)

    ax4.imshow(disp4)
    ax4.set_title(f"D. Matched Filter Proposals (R >= {demo_thresh})\nRecall: {recall*100:.1f}% ({int(recall*len(gt_caps))}/{len(gt_caps)}) | Yellow=True Positive, Red=False Positive", fontsize=13, fontweight="bold")
    ax4.axis("off")

    plt.tight_layout()
    out_path = ARTIFACT_DIR / "matched_filter_demo.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n🎉 Demonstration figure successfully saved to:\n  {out_path}")


if __name__ == "__main__":
    main()
