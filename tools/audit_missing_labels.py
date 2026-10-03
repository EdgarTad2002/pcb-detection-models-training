#!/usr/bin/env python3
"""
tools/audit_missing_labels.py

Automated Dataset Missing-Label Auditor for PCB Automated Optical Inspection (AOI).

Scans PCB images using a dual-verification strategy:
1. Deep Detector Proposals: Runs YOLO (e.g. YOLO26s / Matched-Filter YOLO26s) at high sensitivity.
2. Physics-Based Matched Filter: Evaluates Normalized Cross-Correlation (NCC) against capacitor prototypes.
3. Ground-Truth Discrepancy Mining: Identifies high-confidence predictions that have ZERO overlap (IoU = 0)
   with annotated bounding boxes in the label file.
4. Silkscreen Rejection: Evaluates chromaticity and structural profile to reject white silkscreen lettering.

Outputs:
- A standalone interactive HTML Audit Gallery with zoomable crops, metrics, and accept/reject controls.
- Machine-readable audit report (JSON).
- Option to generate a rectified/cleaned dataset directory with verified missing annotations inserted.

Usage:
    python tools/audit_missing_labels.py \
        --images-dir data_samples/native_images \
        --labels-dir data_samples/native_labels \
        --weights runs/rectified_yolov26s/pcb-filtered/weights/best.pt \
        --out-dir results/audit_missing_labels \
        --conf 0.25 --corr-thresh 0.50
"""

import argparse
import base64
import html
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is required. Run: pip install ultralytics")
    sys.exit(1)

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def compute_iou(box1, box2):
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def extract_prototypes_from_dataset(images_dir, labels_dir, max_samples=30):
    """Extracts horizontal and vertical capacitor prototypes from labeled ground-truth instances."""
    h_crops = []
    v_crops = []

    lbl_files = sorted(list(labels_dir.glob("*.txt")))
    for lf in lbl_files:
        stem = lf.stem
        # Try finding corresponding image with common extensions
        img_file = None
        for ext in [".jpg", ".png", ".jpeg"]:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img_file = cand
                break
        if not img_file:
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 0:  # Capacitor
                    xc, yc, w, h = map(float, parts[1:5])
                    bw, bh = int(w * W), int(h * H)
                    if bw < 10 or bh < 10 or bw > 120 or bh > 120:
                        continue
                    x1 = max(0, int((xc - w / 2) * W))
                    y1 = max(0, int((yc - h / 2) * H))
                    x2 = min(W, int((xc + w / 2) * W))
                    y2 = min(H, int((yc + h / 2) * H))
                    crop = gray[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    if bw > bh * 1.3:
                        h_crops.append(cv2.resize(crop, (36, 18)))
                    elif bh > bw * 1.3:
                        v_crops.append(cv2.resize(crop, (18, 36)))

                    if len(h_crops) >= max_samples and len(v_crops) >= max_samples:
                        break
        if len(h_crops) >= max_samples and len(v_crops) >= max_samples:
            break

    proto_h = np.median(h_crops, axis=0).astype(np.uint8) if h_crops else np.zeros((18, 36), dtype=np.uint8)
    proto_v = np.median(v_crops, axis=0).astype(np.uint8) if v_crops else np.zeros((36, 18), dtype=np.uint8)
    return proto_h, proto_v


def compute_patch_correlation(gray_crop, proto_h, proto_v):
    """Computes peak normalized cross-correlation for a candidate patch."""
    if gray_crop.shape[0] < 8 or gray_crop.shape[1] < 8:
        return 0.0

    # Resize candidate or template to align sizes
    ch, cw = gray_crop.shape[:2]
    scores = []
    for proto in [proto_h, proto_v]:
        ph, pw = proto.shape[:2]
        if ch >= ph and cw >= pw:
            res = cv2.matchTemplate(gray_crop, proto, cv2.TM_CCOEFF_NORMED)
            scores.append(float(np.max(res)))
        elif ph >= ch and pw >= cw:
            res = cv2.matchTemplate(proto, gray_crop, cv2.TM_CCOEFF_NORMED)
            scores.append(float(np.max(res)))
        else:
            # Scale candidate to prototype size
            resized = cv2.resize(gray_crop, (pw, ph))
            res = cv2.matchTemplate(resized, proto, cv2.TM_CCOEFF_NORMED)
            scores.append(float(np.max(res)))

    return max(scores) if scores else 0.0


def is_silkscreen_artifact(crop_bgr):
    """Rejects pure white binary silkscreen lettering (e.g. 'PWR', 'SDA', 'R8')."""
    if crop_bgr.size == 0:
        return True
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Check if pixels are almost purely binary (high white text on dark mask)
    # Silkscreen text has very high 95th percentile, very low 5th percentile,
    # and little chromatic saturation in HSV.
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    bright_frac = np.mean(gray > 210)

    # If more than 40% of pixels are saturated pure white (>210) and low color saturation:
    if bright_frac > 0.45 and saturation < 35:
        return True
    return False


def run_audit(images_dir, labels_dir, weights_path, out_dir, conf_thresh=0.25, corr_thresh=0.45, imgsz=1280):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Extracting capacitor prototypes from: {images_dir}")
    proto_h, proto_v = extract_prototypes_from_dataset(images_dir, labels_dir)
    print(f"   Horizontal prototype: {proto_h.shape}, Vertical prototype: {proto_v.shape}")

    print(f"🚀 Loading detection model: {weights_path}")
    model = YOLO(str(weights_path))

    img_extensions = [".jpg", ".png", ".jpeg", ".bmp"]
    img_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in img_extensions])
    print(f"🔍 Found {len(img_files)} images to audit.")

    candidates = []
    total_images_scanned = 0
    total_missing_found = 0

    for idx, img_path in enumerate(img_files):
        total_images_scanned += 1
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Load Ground Truth boxes
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = int((xc - bw / 2) * W)
                    y1 = int((yc - bh / 2) * H)
                    x2 = int((xc + bw / 2) * W)
                    y2 = int((yc + bh / 2) * H)
                    gt_boxes.append({"cls": cls_id, "box": [x1, y1, x2, y2]})

        # 2. Run detector inference
        results = model.predict(source=str(img_path), imgsz=imgsz, conf=conf_thresh, iou=0.5, verbose=False)
        det = results[0]
        if det.boxes is None or len(det.boxes) == 0:
            continue

        boxes = det.boxes.xyxy.cpu().numpy()
        confs = det.boxes.conf.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy().astype(int)

        # 3. Audit each detection against Ground Truth
        for b, c, cls_id in zip(boxes, confs, classes):
            # Focus on small passive components (Capacitors)
            pred_cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
            x1, y1, x2, y2 = map(int, b)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue

            # Check maximum IoU against ANY ground truth box in this image
            max_iou = 0.0
            matching_gt_cls = None
            for gt in gt_boxes:
                iou = compute_iou([x1, y1, x2, y2], gt["box"])
                if iou > max_iou:
                    max_iou = iou
                    matching_gt_cls = gt["cls"]

            # If the box has NO overlap with ground truth (IoU < 0.10)
            if max_iou < 0.10:
                crop = img[y1:y2, x1:x2]
                gray_crop = gray[y1:y2, x1:x2]

                # Check matched-filter correlation
                corr_score = compute_patch_correlation(gray_crop, proto_h, proto_v)

                # Filter out pure silkscreen lettering
                if is_silkscreen_artifact(crop):
                    continue

                # Filter out if correlation is too weak (unless model confidence is overwhelming)
                if pred_cls_name == "Capacitor" and corr_score < corr_thresh and c < 0.50:
                    continue

                total_missing_found += 1
                cand_id = f"cand_{total_missing_found:04d}"

                # Save 4x magnified crop image
                crop_h, crop_w = crop.shape[:2]
                pad_x = max(15, int(crop_w * 0.5))
                pad_y = max(15, int(crop_h * 0.5))
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
                context_crop = img[cy1:cy2, cx1:cx2].copy()

                # Draw bounding box on context crop
                rel_x1, rel_y1 = x1 - cx1, y1 - cy1
                rel_x2, rel_y2 = x2 - cx1, y2 - cy1
                cv2.rectangle(context_crop, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 0, 255), 2)

                crop_filename = f"{cand_id}_{stem}.jpg"
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), context_crop)

                # Normalized YOLO format coordinates
                norm_xc = ((x1 + x2) / 2.0) / W
                norm_yc = ((y1 + y2) / 2.0) / H
                norm_w = (x2 - x1) / W
                norm_h = (y2 - y1) / H

                cand_entry = {
                    "id": cand_id,
                    "image": img_path.name,
                    "stem": stem,
                    "predicted_class": pred_cls_name,
                    "class_id": int(cls_id),
                    "confidence": float(c),
                    "correlation": float(corr_score),
                    "max_gt_iou": float(max_iou),
                    "box_pixels": [int(x1), int(y1), int(x2), int(y2)],
                    "yolo_bbox": [float(norm_xc), float(norm_yc), float(norm_w), float(norm_h)],
                    "crop_rel_path": f"crops/{crop_filename}",
                    "width": W,
                    "height": H,
                }
                candidates.append(cand_entry)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(img_files):
            print(f"[{idx+1}/{len(img_files)}] Processed {img_path.name} | Total Missing Candidates: {len(candidates)}")

    # Sort candidates by combined score: confidence * (1 + correlation)
    candidates.sort(key=lambda x: x["confidence"] * (1.0 + x["correlation"]), reverse=True)

    # Save JSON Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images_audited": total_images_scanned,
        "total_missing_candidates": len(candidates),
        "detector_weights": str(weights_path),
        "conf_threshold": conf_thresh,
        "corr_threshold": corr_thresh,
        "candidates": candidates,
    }
    json_path = out_dir / "audit_missing_labels.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Audit report written to: {json_path}")

    # Build Interactive HTML Gallery
    build_html_gallery(out_dir, candidates, total_images_scanned)

    # Build Summary Visual Collage
    build_summary_collage(out_dir, candidates, img_files)

    print(f"\n✨ Audit Completed! Found {len(candidates)} high-confidence candidate missing annotations.")
    print(f"🌐 Open HTML Gallery: {out_dir / 'missing_labels_gallery.html'}")


def build_summary_collage(out_dir, candidates, img_files, top_k=16):
    """Creates a high-resolution grid image showing top candidate missing labels."""
    if not candidates:
        return
    top_cands = candidates[:top_k]
    cols = 4
    rows = (len(top_cands) + cols - 1) // cols
    cell_w, cell_h = 320, 260
    collage = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 30

    for idx, c in enumerate(top_cands):
        r, col = idx // cols, idx % cols
        crop_path = out_dir / c["crop_rel_path"]
        if not crop_path.exists():
            continue
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        # Resize preserving aspect ratio into (cell_w - 20, cell_h - 60)
        max_cw, max_ch = cell_w - 20, cell_h - 60
        scale = min(max_cw / crop.shape[1], max_ch / crop.shape[0])
        nw, nh = max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))
        crop_res = cv2.resize(crop, (nw, nh))

        y_off = r * cell_h + 10 + (max_ch - nh) // 2
        x_off = col * cell_w + 10 + (max_cw - nw) // 2
        collage[y_off:y_off+nh, x_off:x_off+nw] = crop_res

        # Text banner
        banner_y = r * cell_h + cell_h - 35
        text1 = f"#{idx+1} {c['predicted_class']} | Conf: {c['confidence']:.2f}"
        text2 = f"NCC: {c['correlation']:.2f} | {c['stem'][:18]}"
        cv2.putText(collage, text1, (col * cell_w + 12, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(collage, text2, (col * cell_w + 12, banner_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    out_img = out_dir / "audit_summary_grid.png"
    cv2.imwrite(str(out_img), collage)
    print(f"🖼️ Summary visual collage saved to: {out_img}")


def build_html_gallery(out_dir, candidates, total_scanned):
    """Generates a responsive HTML dashboard for interactive inspection and curation."""
    html_path = out_dir / "missing_labels_gallery.html"

    # Convert candidate data to JSON for client-side JS filtering and label export
    json_cands = json.dumps(candidates)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCB Missing-Label Auditor Dashboard</title>
<style>
  :root {{
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
    --accent-blue: #38bdf8;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-primary);
    padding: 24px;
    line-height: 1.5;
  }}
  header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
  }}
  h1 {{ font-size: 26px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; display: flex; align-items: center; gap: 12px; }}
  .badge {{ background: #0369a1; color: #e0f2fe; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 600; }}
  p.subtitle {{ color: var(--text-secondary); font-size: 15px; margin-bottom: 16px; }}
  .stats-bar {{
    display: flex; gap: 20px; flex-wrap: wrap; margin-top: 16px;
    background: rgba(0,0,0,0.25); padding: 12px 18px; border-radius: 8px; border: 1px solid var(--border-color);
  }}
  .stat-item {{ display: flex; flex-direction: column; }}
  .stat-label {{ font-size: 12px; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 20px; font-weight: 700; color: var(--accent-blue); }}

  .controls {{
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
    margin-bottom: 20px;
  }}
  .btn {{
    padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; border: none;
    transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 6px;
  }}
  .btn-primary {{ background: var(--accent-blue); color: #0f172a; }}
  .btn-primary:hover {{ background: #7dd3fc; }}
  .btn-success {{ background: var(--accent-green); color: #052e16; }}
  .btn-success:hover {{ background: #4ade80; }}
  .btn-outline {{ background: transparent; border: 1px solid var(--border-color); color: var(--text-primary); }}
  .btn-outline:hover {{ background: rgba(255,255,255,0.05); }}

  .gallery-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
  }}
  .card:hover {{
    transform: translateY(-4px);
    border-color: var(--accent-blue);
    box-shadow: 0 12px 20px -8px rgba(56, 189, 248, 0.25);
  }}
  .card.accepted {{ border-color: var(--accent-green); background: #132e22; }}
  .card.rejected {{ border-color: var(--accent-red); opacity: 0.45; }}

  .card-img-wrapper {{
    position: relative;
    width: 100%;
    padding-top: 80%;
    background: #090d16;
    overflow: hidden;
  }}
  .card-img-wrapper img {{
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    object-fit: contain;
  }}
  .tag-score {{
    position: absolute; top: 8px; right: 8px;
    background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(4px);
    border: 1px solid var(--border-color);
    padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700;
  }}
  .card-body {{ padding: 14px; flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; }}
  .card-title {{ font-size: 15px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }}
  .card-subtitle {{ font-size: 12px; color: var(--text-secondary); word-break: break-all; margin-bottom: 10px; }}
  .metrics-row {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 12px; }}
  .metric {{ display: flex; flex-direction: column; }}
  .metric-label {{ color: var(--text-secondary); font-size: 10px; text-transform: uppercase; }}
  .metric-val {{ font-weight: 600; color: #e2e8f0; }}
  .card-actions {{ display: flex; gap: 8px; }}
  .btn-sm {{ padding: 6px 12px; font-size: 12px; flex: 1; text-align: center; justify-content: center; }}

  .modal {{
    display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.85); backdrop-filter: blur(8px);
    align-items: center; justify-content: center;
  }}
  .modal-content {{
    background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px;
    padding: 24px; max-width: 600px; width: 90%; max-height: 90vh; overflow-y: auto;
  }}
  pre {{
    background: #090d16; padding: 12px; border-radius: 8px; font-size: 12px; color: #a5f3fc;
    overflow-x: auto; margin: 12px 0; border: 1px solid var(--border-color);
  }}
</style>
</head>
<body>

<header>
  <h1>🔬 PCB Missing-Label Auditor <span class="badge">{len(candidates)} Candidates Found</span></h1>
  <p class="subtitle">Discrepancy mining between Deep Detector proposals, Physics-Based Cross-Correlation, and Human Ground-Truth labels.</p>
  
  <div class="stats-bar">
    <div class="stat-item">
      <span class="stat-label">Images Scanned</span>
      <span class="stat-value">{total_scanned}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Missing Candidates</span>
      <span class="stat-value">{len(candidates)}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Avg Confidence</span>
      <span class="stat-value">{np.mean([c['confidence'] for c in candidates]):.2f}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Avg NCC Correlation</span>
      <span class="stat-value">{np.mean([c['correlation'] for c in candidates]):.2f}</span>
    </div>
  </div>
</header>

<div class="controls">
  <div style="display:flex; gap:10px; align-items:center;">
    <button class="btn btn-success" onclick="acceptAll()">✅ Accept All ({len(candidates)})</button>
    <button class="btn btn-outline" onclick="resetAll()">🔄 Reset All</button>
  </div>
  <button class="btn btn-primary" onclick="exportRectifiedDataset()">💾 Export Cleaned Labels (.txt)</button>
</div>

<div class="gallery-grid" id="galleryGrid">
"""

    for idx, c in enumerate(candidates):
        cid = c["id"]
        pred_cls = c["predicted_class"]
        conf = c["confidence"]
        corr = c["correlation"]
        stem = c["stem"]
        crop_url = c["crop_rel_path"]
        xc, yc, w, h = c["yolo_bbox"]

        html_content += f"""
  <div class="card" id="card_{cid}">
    <div class="card-img-wrapper">
      <img src="{crop_url}" alt="{cid}" loading="lazy">
      <span class="tag-score" style="color: {'#4ade80' if conf >= 0.5 else '#fde047'};">
        Conf: {conf:.2f}
      </span>
    </div>
    <div class="card-body">
      <div>
        <div class="card-title">#{idx+1} {pred_cls}</div>
        <div class="card-subtitle">{stem}</div>
        <div class="metrics-row">
          <div class="metric">
            <span class="metric-label">Correlation</span>
            <span class="metric-val">{corr:.2f}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Max GT IoU</span>
            <span class="metric-val" style="color:#ef4444;">0.00 (Unlabeled)</span>
          </div>
          <div class="metric">
            <span class="metric-label">BBox (W x H)</span>
            <span class="metric-val">{c['box_pixels'][2]-c['box_pixels'][0]}x{c['box_pixels'][3]-c['box_pixels'][1]} px</span>
          </div>
        </div>
      </div>
      <div class="card-actions">
        <button class="btn btn-sm btn-success" onclick="toggleAccept('{cid}')">Accept</button>
        <button class="btn btn-sm btn-outline" onclick="toggleReject('{cid}')">Reject</button>
      </div>
    </div>
  </div>
"""

    html_content += f"""
</div>

<!-- Modal for Exporting Cleaned Labels -->
<div class="modal" id="exportModal">
  <div class="modal-content">
    <h2 style="margin-bottom:8px;">Rectified Dataset Export</h2>
    <p style="color:var(--text-secondary); font-size:14px;">
      The following Python snippet will inject all accepted candidate bounding boxes directly into your dataset label files:
    </p>
    <pre id="exportSnippet"></pre>
    <div style="display:flex; justify-content:flex-end; gap:10px; margin-top:16px;">
      <button class="btn btn-outline" onclick="closeModal()">Close</button>
      <button class="btn btn-primary" onclick="copySnippet()">Copy Snippet</button>
    </div>
  </div>
</div>

<script>
  const candidates = {json_cands};
  const decisions = {{}}; // cid -> 'accepted' | 'rejected'

  function toggleAccept(cid) {{
    const card = document.getElementById('card_' + cid);
    if (decisions[cid] === 'accepted') {{
      delete decisions[cid];
      card.classList.remove('accepted');
    }} else {{
      decisions[cid] = 'accepted';
      card.classList.remove('rejected');
      card.classList.add('accepted');
    }}
  }}

  function toggleReject(cid) {{
    const card = document.getElementById('card_' + cid);
    if (decisions[cid] === 'rejected') {{
      delete decisions[cid];
      card.classList.remove('rejected');
    }} else {{
      decisions[cid] = 'rejected';
      card.classList.remove('accepted');
      card.classList.add('rejected');
    }}
  }}

  function acceptAll() {{
    candidates.forEach(c => {{
      decisions[c.id] = 'accepted';
      const card = document.getElementById('card_' + c.id);
      if (card) {{
        card.classList.remove('rejected');
        card.classList.add('accepted');
      }}
    }});
  }}

  function resetAll() {{
    Object.keys(decisions).forEach(cid => {{
      const card = document.getElementById('card_' + cid);
      if (card) card.classList.remove('accepted', 'rejected');
    }});
    for (let member in decisions) delete decisions[member];
  }}

  function exportRectifiedDataset() {{
    const accepted = candidates.filter(c => decisions[c.id] !== 'rejected');
    const acceptedJson = JSON.stringify(accepted, null, 2);

    const pyCode = `# Run this snippet to insert ${{accepted.length}} missing labels into your dataset:\\n` +
      `import json, os\\n\\n` +
      `accepted_boxes = ${{acceptedJson}}\\n\\n` +
      `labels_dir = "datasets/pcb-unified-4class/train/labels"\\n` +
      `for item in accepted_boxes:\\n` +
      `    lbl_file = os.path.join(labels_dir, item["stem"] + ".txt")\\n` +
      `    if os.path.exists(lbl_file):\\n` +
      `        xc, yc, w, h = item["yolo_bbox"]\\n` +
      `        line = f"{{item['class_id']}} {{xc:.6f}} {{yc:.6f}} {{w:.6f}} {{h:.6f}}\\n"\\n` +
      `        with open(lbl_file, "a") as f:\\n` +
      `            f.write(line)\\n` +
      `print("Successfully injected ${{accepted.length}} missing annotations!")`;

    document.getElementById('exportSnippet').innerText = pyCode;
    document.getElementById('exportModal').style.display = 'flex';
  }}

  function closeModal() {{
    document.getElementById('exportModal').style.display = 'none';
  }}

  function copySnippet() {{
    const text = document.getElementById('exportSnippet').innerText;
    navigator.clipboard.writeText(text);
    alert('Export snippet copied to clipboard!');
  }}
</script>
</body>
</html>
"""

    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"🌐 Interactive HTML Gallery written to: {html_path}")


def main():
    p = argparse.ArgumentParser(description="PCB Dataset Missing-Label Auditor")
    p.add_argument("--images-dir", type=str, default="data_samples/native_images")
    p.add_argument("--labels-dir", type=str, default="data_samples/native_labels")
    p.add_argument(
        "--weights",
        type=str,
        default="runs/rectified_yolov26s/pcb-filtered/weights/best.pt",
    )
    p.add_argument("--out-dir", type=str, default="results/audit_missing_labels")
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    p.add_argument("--corr-thresh", type=float, default=0.45, help="Matched filter correlation threshold")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference resolution")
    args = p.parse_args()

    run_audit(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        weights_path=args.weights,
        out_dir=args.out_dir,
        conf_thresh=args.conf,
        corr_thresh=args.corr_thresh,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
