#!/usr/bin/env python3
"""
Generates 3-panel visual comparisons with Ground Truth:
Panel 1: Ground Truth Annotations (Human Expert Labels)
Panel 2: Baseline Model Detections
Panel 3: Enhanced / Unified Champion Pipeline Detections
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sahi_pcb_inference import EVAL_CLASSES, UNIFIED_CLASSES, NAME_TO_UNIFIED_ID, get_color, parse_label_file
from tools.eval_tta_benchmark import predict_with_tta, draw_bounding_boxes
from tools.eval_unified_pipeline import unified_predict

def generate_comparisons():
    img_path = "data_samples/images/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.jpg"
    lbl_path = "data_samples/labels/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.txt"
    model_path = "runs/yolov26s_loss_reweight/pcb-filtered/weights/best.pt"

    raw_bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 1. Load Ground Truth via universal parser
    gt_boxes = parse_label_file(lbl_path, w, h)
    print(f"Loaded {len(gt_boxes)} ground-truth annotations.")

    def draw_gt(img, boxes):
        cv = img.copy()
        for b in boxes:
            cid, cname, x1, y1, x2, y2 = b
            col = get_color(cname)
            cv2.rectangle(cv, (x1, y1), (x2, y2), col, 2)
            lbl = f"GT: {cname}"
            y_lbl = max(y1 - 5, 12)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(cv, (x1, y_lbl - th - 2), (x1 + tw + 2, y_lbl + 2), col, -1)
            cv2.putText(cv, lbl, (x1 + 1, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        return cv

    def draw_pred(img, boxes):
        cv = img.copy()
        for b in boxes:
            cid, cname, sc, x1, y1, x2, y2 = b
            col = get_color(cname)
            cv2.rectangle(cv, (x1, y1), (x2, y2), col, 2)
            lbl = f"{cname} {sc:.2f}"
            y_lbl = max(y1 - 5, 12)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(cv, (x1, y_lbl - th - 2), (x1 + tw + 2, y_lbl + 2), col, -1)
            cv2.putText(cv, lbl, (x1 + 1, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        return cv

    vis_gt = draw_gt(rgb, gt_boxes)

    # -------------------------------------------------------------
    # Figure 1: Ground Truth vs Baseline vs Multi-Scale TTA
    # -------------------------------------------------------------
    model = YOLO(model_path)
    res_base = model.predict(raw_bgr, imgsz=640, conf=0.15, verbose=False)[0]
    base_boxes_tta = []
    for b in res_base.boxes:
        raw_cid = int(b.cls[0])
        raw_name = model.names.get(raw_cid, f"Class_{raw_cid}")
        cid = NAME_TO_UNIFIED_ID.get(raw_name)
        if cid is not None:
            sc = float(b.conf[0])
            bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
            cname = UNIFIED_CLASSES[cid]
            base_boxes_tta.append((cid, cname, sc, bx1, by1, bx2, by2))

    tta_boxes = predict_with_tta(
        model=model,
        image_bgr=raw_bgr,
        scales=[0.85, 1.0, 1.15],
        enable_flip=True,
        conf_threshold=0.15,
        iou_threshold=0.45,
        nms_type="diou",
    )
    # Normalize TTA predictions
    norm_tta_boxes = []
    for b in tta_boxes:
        cid, cname, sc, bx1, by1, bx2, by2 = b
        u_cid = NAME_TO_UNIFIED_ID.get(cname)
        if u_cid is not None:
            norm_tta_boxes.append((u_cid, UNIFIED_CLASSES[u_cid], sc, bx1, by1, bx2, by2))
    tta_boxes = norm_tta_boxes

    vis_base_tta = draw_pred(rgb, base_boxes_tta)
    vis_tta = draw_pred(rgb, tta_boxes)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(vis_gt)
    axes[0].set_title(f"[1] Ground Truth (Human Expert Labels)\n{len(gt_boxes)} physical components", fontsize=12, fontweight="bold", color="darkred")
    axes[0].axis("off")

    axes[1].imshow(vis_base_tta)
    axes[1].set_title(f"[2] Baseline Single-Scale YOLO (640px)\n{len(base_boxes_tta)} objects detected (misses dense caps)", fontsize=12, fontweight="bold", color="navy")
    axes[1].axis("off")

    axes[2].imshow(vis_tta)
    axes[2].set_title(f"[3] Multi-Scale TTA + DIoU-NMS\n{len(tta_boxes)} objects detected (aligns closer to GT)", fontsize=12, fontweight="bold", color="darkgreen")
    axes[2].axis("off")

    plt.tight_layout()
    out_tta = Path("presentation_assets/tta_comparison_demo.png")
    plt.savefig(str(out_tta), bbox_inches="tight", dpi=160)
    plt.close()
    print(f"✅ Generated 3-panel TTA comparison with Ground Truth: {out_tta}")

    # -------------------------------------------------------------
    # Figure 2: Ground Truth vs Baseline vs Unified Champion Pipeline
    # -------------------------------------------------------------
    res_base_unified = model.predict(raw_bgr, imgsz=640, conf=0.25, verbose=False)[0]
    base_boxes_uni = []
    for b in res_base_unified.boxes:
        raw_cid = int(b.cls[0])
        raw_name = model.names.get(raw_cid, f"Class_{raw_cid}")
        cid = NAME_TO_UNIFIED_ID.get(raw_name)
        if cid is not None:
            sc = float(b.conf[0])
            bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
            cname = UNIFIED_CLASSES[cid]
            base_boxes_uni.append((cid, cname, sc, bx1, by1, bx2, by2))

    unified_boxes = unified_predict(
        model=model,
        image_bgr=raw_bgr,
        scales=[0.85, 1.0, 1.15],
        enable_flip=True,
        apply_calibrated_thresholds=True,
        min_sensitivity_conf=0.05,
        iou_threshold=0.45,
        imgsz=640,
    )
    norm_uni_boxes = []
    for b in unified_boxes:
        cid, cname, sc, bx1, by1, bx2, by2 = b
        u_cid = NAME_TO_UNIFIED_ID.get(cname)
        if u_cid is not None:
            norm_uni_boxes.append((u_cid, UNIFIED_CLASSES[u_cid], sc, bx1, by1, bx2, by2))
    unified_boxes = norm_uni_boxes

    vis_base_uni = draw_pred(rgb, base_boxes_uni)
    vis_uni = draw_pred(rgb, unified_boxes)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(vis_gt)
    axes[0].set_title(f"[1] Ground Truth (Human Expert Labels)\n{len(gt_boxes)} physical components", fontsize=12, fontweight="bold", color="darkred")
    axes[0].axis("off")

    axes[1].imshow(vis_base_uni)
    axes[1].set_title(f"[2] Baseline YOLO (Single-Scale, Flat tau=0.25)\n{len(base_boxes_uni)} objects detected", fontsize=12, fontweight="bold", color="navy")
    axes[1].axis("off")

    axes[2].imshow(vis_uni)
    axes[2].set_title(f"[3] Unified Champion Pipeline (TTA + DIoU + Calibrated tau*)\n{len(unified_boxes)} objects detected (recovers missing GT components)", fontsize=12, fontweight="bold", color="darkgreen")
    axes[2].axis("off")

    plt.tight_layout()
    out_uni = Path("presentation_assets/unified_pipeline_demo.png")
    plt.savefig(str(out_uni), bbox_inches="tight", dpi=160)
    plt.close()
    print(f"✅ Generated 3-panel Unified Pipeline comparison with Ground Truth: {out_uni}")

if __name__ == "__main__":
    generate_comparisons()
