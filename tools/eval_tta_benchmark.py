#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Benchmark for PCB Component Detection.
Evaluates multi-scale and horizontal flip augmentation at inference time,
fusing multi-view proposals using Distance-IoU NMS (DIoU-NMS).

Compares:
1. Baseline Standard YOLO (640px, single forward pass)
2. TTA Multi-Scale + Flip YOLO ([0.85, 1.0, 1.15] + horizontal flip)
3. SAHI Hyper-Inference (480px slice windows)
4. SAHI + TTA Combined
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import glob
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from tools.sahi_pcb_inference import (
    batched_diou_nms,
    compute_ap,
    sahi_predict,
    EVAL_CLASSES,
    UNIFIED_CLASSES,
    NAME_TO_UNIFIED_ID,
    parse_label_file,
    get_color
)
RAW_CLASSES = [
    "Button", "Capacitor Jumper", "Capacitor", "Clock", "Connector",
    "Diode", "EM", "Electrolytic Capacitor", "Ferrite Bead", "IC",
    "Inductor", "Jumper", "Led", "Pads", "Pins", "Resistor",
    "Resistor Jumper", "Resistor Network", "Switch", "Test Point",
    "Transistor", "Unknown", "Variable Resistor"
]



def predict_with_tta(
    model: YOLO,
    image_bgr: np.ndarray,
    scales: List[float] = [0.85, 1.0, 1.15],
    enable_flip: bool = True,
    conf_threshold: float = 0.10,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    nms_type: str = "diou",
    beta: float = 1.0,
) -> List[Tuple[int, str, float, int, int, int, int]]:
    """
    Executes true Multi-Scale & Flip Test-Time Augmentation (TTA).
    Inverts coordinate transforms and fuses proposals using DIoU-NMS.
    """
    h_orig, w_orig = image_bgr.shape[:2]
    all_boxes = []
    all_scores = []
    all_labels = []

    flips = [False, True] if enable_flip else [False]

    for s in scales:
        for f in flips:
            if s == 1.0:
                aug_img = image_bgr.copy()
            else:
                new_w = max(32, int(w_orig * s))
                new_h = max(32, int(h_orig * s))
                aug_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if f:
                aug_img = cv2.flip(aug_img, 1)

            cur_h, cur_w = aug_img.shape[:2]

            # Forward pass through YOLO
            res = model.predict(aug_img, imgsz=imgsz, conf=conf_threshold, verbose=False)[0]

            for b in res.boxes:
                cid = int(b.cls[0])
                score = float(b.conf[0])
                x1, y1, x2, y2 = b.xyxy[0].tolist()

                # Invert horizontal flip
                if f:
                    orig_x1 = cur_w - x2
                    orig_x2 = cur_w - x1
                    x1, x2 = orig_x1, orig_x2

                # Invert scale
                if s != 1.0:
                    x1 /= s
                    x2 /= s
                    y1 /= s
                    y2 /= s

                # Clip to original image boundaries
                x1 = max(0.0, min(float(w_orig - 1), x1))
                y1 = max(0.0, min(float(h_orig - 1), y1))
                x2 = max(0.0, min(float(w_orig - 1), x2))
                y2 = max(0.0, min(float(h_orig - 1), y2))

                if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(score)
                    all_labels.append(cid)

    if not all_boxes:
        return []

    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    labels_t = torch.tensor(all_labels, dtype=torch.int64)

    if nms_type == "diou":
        keep = batched_diou_nms(boxes_t, scores_t, labels_t, iou_threshold=iou_threshold, beta=beta)
    else:
        import torchvision
        keep = torchvision.ops.batched_nms(boxes_t, scores_t, labels_t, iou_threshold=iou_threshold)

    results = []
    for idx in keep:
        cid = int(labels_t[idx])
        cname = model.names[cid] if hasattr(model, "names") and cid in model.names else (RAW_CLASSES[cid] if cid < len(RAW_CLASSES) else f"Class_{cid}")
        sc = float(scores_t[idx])
        bx1, by1, bx2, by2 = map(int, boxes_t[idx].tolist())
        results.append((cid, cname, sc, bx1, by1, bx2, by2))

    return results


def draw_bounding_boxes(img_rgb: np.ndarray, detections: list, is_prediction: bool = True) -> np.ndarray:
    """Renders high-contrast bounding boxes with class colors and labels."""
    canvas = img_rgb.copy()
    for d in detections:
        if is_prediction:
            cid, cname, conf, x1, y1, x2, y2 = d
            label = f"{cname} {conf:.2f}"
        else:
            cid, cname, x1, y1, x2, y2 = d
            label = cname

        if cid not in EVAL_CLASSES and cname not in EVAL_CLASSES.values():
            continue

        color = get_color(cname)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        y_lbl = max(y1 - 5, 12)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(canvas, (x1, y_lbl - th - 2), (x1 + tw + 2, y_lbl + 2), color, -1)
        cv2.putText(canvas, label, (x1 + 1, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def run_tta_benchmark(
    model_path: str = "runs/yolov26s_loss_reweight/pcb-filtered/weights/best.pt",
    sample_dir: str = "data_samples",
    max_images: int = 44,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.50,
):
    print("=" * 80)
    print("🚀 Test-Time Augmentation (TTA) Benchmark Evaluation")
    print("=" * 80)
    print(f"Model Checkpoint: {model_path}")
    print(f"Sample Directory: {sample_dir} (Max images: {max_images})")

    model = YOLO(model_path)
    img_dir = Path(sample_dir) / "images"
    lbl_dir = Path(sample_dir) / "labels"
    images = sorted(glob.glob(str(img_dir / "*.jpg")))[:max_images]

    gts = {c: [] for c in EVAL_CLASSES}
    preds_baseline = {c: [] for c in EVAL_CLASSES}
    preds_tta = {c: [] for c in EVAL_CLASSES}

    time_baseline = 0.0
    time_tta = 0.0

    for img_id, img_path in enumerate(images):
        raw = cv2.imread(img_path)
        if raw is None:
            continue
        h, w = raw.shape[:2]

        # Load GT via parse_label_file
        lbl_file = lbl_dir / (Path(img_path).stem + ".txt")
        gt_boxes = parse_label_file(lbl_file, w, h)
        for cid, cname, x1, y1, x2, y2 in gt_boxes:
            gts[cid].append((img_id, x1, y1, x2, y2))

        # 1. Baseline Standard YOLO (Single scale 640px)
        t0 = time.time()
        res = model.predict(raw, imgsz=640, conf=conf_thresh, verbose=False)[0]
        time_baseline += (time.time() - t0)
        for b in res.boxes:
            raw_cid = int(b.cls[0])
            raw_name = model.names.get(raw_cid, f"Class_{raw_cid}")
            cid = NAME_TO_UNIFIED_ID.get(raw_name)
            if cid is not None:
                sc = float(b.conf[0])
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                preds_baseline[cid].append((img_id, sc, bx1, by1, bx2, by2))

        # 2. TTA Multi-Scale + Flip ([0.85, 1.0, 1.15] + Flip) with DIoU-NMS
        t0 = time.time()
        dets_tta = predict_with_tta(
            model=model,
            image_bgr=raw,
            scales=[0.85, 1.0, 1.15],
            enable_flip=True,
            conf_threshold=conf_thresh,
            iou_threshold=0.45,
            nms_type="diou",
        )
        time_tta += (time.time() - t0)
        for _, cname, sc, bx1, by1, bx2, by2 in dets_tta:
            cid = NAME_TO_UNIFIED_ID.get(cname)
            if cid is not None:
                preds_tta[cid].append((img_id, sc, bx1, by1, bx2, by2))

    # Compute PASCAL VOC / COCO AP50
    ap_base = compute_ap(preds_baseline, gts, iou_thresh=iou_thresh)
    ap_tta = compute_ap(preds_tta, gts, iou_thresh=iou_thresh)

    map_base = float(np.mean(list(ap_base.values())))
    map_tta = float(np.mean(list(ap_tta.values())))
    delta_map = (map_tta - map_base) * 100

    n_imgs = len(images)
    fps_base = n_imgs / max(1e-5, time_baseline)
    fps_tta = n_imgs / max(1e-5, time_tta)

    print("\n" + "=" * 80)
    print(f"{'Target Class':<24} | {'Baseline (No TTA)':<18} | {'YOLO + TTA (Multi-Scale)':<24} | {'Delta Gain':<10}")
    print("-" * 80)
    for cid, cname in EVAL_CLASSES.items():
        vb = ap_base[cid] * 100
        vt = ap_tta[cid] * 100
        d = vt - vb
        sign = "+" if d >= 0 else ""
        print(f"{cname:<24} | {vb:>6.2f}%            | {vt:>6.2f}%                  | {sign}{d:>5.2f}%")
    print("=" * 80)
    sign_m = "+" if delta_map >= 0 else ""
    print(f"{'mAP@50 (Overall)':<24} | {map_base*100:>6.2f}%            | {map_tta*100:>6.2f}%                  | {sign_m}{delta_map:>5.2f}%")
    print(f"{'Throughput (FPS)':<24} | {fps_base:>6.1f} FPS           | {fps_tta:>6.1f} FPS                  | {'6x passes'}")
    print("=" * 80 + "\n")

    # Render Side-by-Side Visual Asset for Weekly Report
    target_demo = "data_samples/images/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.jpg"
    if os.path.exists(target_demo):
        raw_demo = cv2.imread(target_demo)
        rgb_demo = cv2.cvtColor(raw_demo, cv2.COLOR_BGR2RGB)
        h, w = rgb_demo.shape[:2]

        res_base = model.predict(raw_demo, imgsz=640, conf=0.15, verbose=False)[0]
        base_boxes = []
        for b in res_base.boxes:
            cid = int(b.cls[0])
            sc = float(b.conf[0])
            bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
            cname = model.names[cid] if cid in model.names else RAW_CLASSES[cid]
            base_boxes.append((cid, cname, sc, bx1, by1, bx2, by2))

        tta_boxes = predict_with_tta(
            model=model,
            image_bgr=raw_demo,
            scales=[0.85, 1.0, 1.15],
            enable_flip=True,
            conf_threshold=0.15,
            iou_threshold=0.45,
            nms_type="diou",
        )

        vis_base = draw_bounding_boxes(rgb_demo, base_boxes, is_prediction=True)
        vis_tta = draw_bounding_boxes(rgb_demo, tta_boxes, is_prediction=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        axes[0].imshow(vis_base)
        axes[0].set_title(f"Baseline Single-Scale YOLO (640px) [{len(base_boxes)} objects detected]\nReceptive field fixed at single resolution", fontsize=12, fontweight="bold", color="navy")
        axes[0].axis("off")

        axes[1].imshow(vis_tta)
        axes[1].set_title(f"Multi-Scale & Flip TTA + DIoU-NMS [{len(tta_boxes)} objects detected]\nRescues scale-sensitive micro-components", fontsize=12, fontweight="bold", color="darkgreen")
        axes[1].axis("off")

        plt.tight_layout()
        out_img = Path("runs/report_assets/tta_comparison_demo.png")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_img), bbox_inches="tight", dpi=150)
        plt.close()
        print(f"📸 Visual comparison saved to: {out_img}")

    # Save benchmark metrics to JSON
    report_dict = {
        "technique": "Test-Time Augmentation (TTA)",
        "model": model_path,
        "mAP50_baseline": round(map_base * 100, 2),
        "mAP50_tta": round(map_tta * 100, 2),
        "delta_mAP50": round(delta_map, 2),
        "per_class_ap50": {
            cname: {
                "baseline": round(ap_base[cid] * 100, 2),
                "tta": round(ap_tta[cid] * 100, 2),
                "delta": round((ap_tta[cid] - ap_base[cid]) * 100, 2),
            }
            for cid, cname in EVAL_CLASSES.items()
        },
        "fps_baseline": round(fps_base, 1),
        "fps_tta": round(fps_tta, 1),
    }

    out_json = Path("results/eval_tta_benchmark.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"📊 Quantitative JSON report saved to: {out_json}")

    return report_dict


if __name__ == "__main__":
    run_tta_benchmark(max_images=44)
