#!/usr/bin/env python3
"""
3-Tier SOTA Evaluation & Class-Calibrated WBF + Native Quadrant Hyper-Inference
===============================================================================
Evaluates all R4-Final models on the canonical held-out Test Split
(`datasets/pcb-native-reviewed-r4-final/data.yaml`, 44 boards, 4,909 targets) across:

  - Tier 1: Standard Single-Model Full-Board Evaluation (Ultralytics `model.val(split='test')`)
            with true per-class AP@50 (`metrics.box.ap50`) and AP@50-95 (`metrics.box.maps`).
  - Tier 2: Per-Class Validation-Calibrated Multi-Model Weighted Boxes Fusion (WBF)
            where each model's weight on class `c` is proportional to `(val_AP50[m, c])^2`.
  - Tier 3: Scale-Conditioned Native-Quadrant + Full-Board WBF Hyper-Inference
            combining full-board context for macro components (IC, ElCap, Connector) with
            border-filtered 2x2 overlapping native-resolution quadrant passes for micro-capacitors.
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

for mod in [
    "apex_r4_model",
    "models.as_pgms_yolo",
    "agwd_ota_loss",
    "squared_dice_loss",
    "retinex_stem",
    "weber_stem",
    "train_sota_r4",
]:
    try:
        __import__(mod)
    except Exception:
        pass

DATA_YAML = "datasets/pcb-native-reviewed-r4-final/data.yaml"
OUT_JSON = Path("results/r4_final_apex_sota_summary.json")

# Candidate checkpoints: (Name, Path, Eval_Imgsz, Enable_Quadrant_MicroPass)
CANDIDATES = [
    ("APEX_YOLO11m_1536", "runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt", 1536, True),
    ("APEX_YOLO26s_1536", "runs/r4_apex_yolo26s_1536/pcb-filtered/weights/best.pt", 1536, True),
    ("APEX_YOLO26x_1280", "runs/r4_apex_yolo26x_1280/pcb-filtered/weights/best.pt", 1280, False),
    ("SOTA_YOLO11m_RetinexEdge_1536", "runs/r4_sota_yolo11m_retinex_agwd_1536/pcb-filtered/weights/best.pt", 1536, True),
    ("SOTA_YOLO26x_RetinexEdge_1280", "runs/r4_sota_yolo26x_retinex_agwd_1280/pcb-filtered/weights/best.pt", 1280, False),
    ("SOTA_YOLO26s_RetinexEdge_1536", "runs/r4_sota_yolo26s_retinex_agwd_1536/pcb-filtered/weights/best.pt", 1536, False),
    ("Baseline_YOLO11m_200ep_1280", "runs/detect/runs/r4_baseline_yolo11m_multiclass_200ep/weights/best.pt", 1280, False),
    ("Prev_YOLO26s_AGWD_SDice_1280", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, False),
]


def wbf_single_class(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    weights: List[float],
    iou_thr: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-specific Weighted Boxes Fusion with consensus confidence scaling."""
    all_b, all_s, all_w = [], [], []
    active_weights = [float(w) for b, w in zip(boxes_list, weights) if len(b) > 0]
    total_w = float(sum(weights)) if sum(weights) > 0 else 1.0

    for b, s, w in zip(boxes_list, scores_list, weights):
        if len(b) == 0 or w <= 0:
            continue
        all_b.append(b.astype(np.float32))
        all_s.append((s * float(w)).astype(np.float32))
        all_w.append(np.full(len(s), float(w), dtype=np.float32))

    if not all_b:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    all_b = np.concatenate(all_b, axis=0)
    all_s = np.concatenate(all_s, axis=0)
    all_w = np.concatenate(all_w, axis=0)
    order = np.argsort(-all_s)
    all_b, all_s, all_w = all_b[order], all_s[order], all_w[order]

    clusters_b, clusters_s, clusters_w, fused_b = [], [], [], []

    for b, s, w in zip(all_b, all_s, all_w):
        if not fused_b:
            clusters_b.append([b])
            clusters_s.append([s])
            clusters_w.append([w])
            fused_b.append(b.copy())
            continue

        fb = np.stack(fused_b, axis=0)
        ix1 = np.maximum(fb[:, 0], b[0])
        iy1 = np.maximum(fb[:, 1], b[1])
        ix2 = np.minimum(fb[:, 2], b[2])
        iy2 = np.minimum(fb[:, 3], b[3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = (fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        ious = inter / np.maximum(area_a + area_b - inter, 1e-7)
        best_idx = int(np.argmax(ious))

        if ious[best_idx] >= iou_thr:
            clusters_b[best_idx].append(b)
            clusters_s[best_idx].append(s)
            clusters_w[best_idx].append(w)
            cs = np.array(clusters_s[best_idx], dtype=np.float32)
            cb = np.stack(clusters_b[best_idx], axis=0)
            fused_b[best_idx] = np.sum(cb * cs[:, None], axis=0) / np.maximum(np.sum(cs), 1e-7)
        else:
            clusters_b.append([b])
            clusters_s.append([s])
            clusters_w.append([w])
            fused_b.append(b.copy())

    out_b = np.stack(fused_b, axis=0)
    out_s = np.array([np.sum(cs) / total_w for cs in clusters_s], dtype=np.float32)
    order = np.argsort(-out_s)
    return out_b[order], out_s[order]


def predict_quadrant_micro_capacitors(
    model: YOLO,
    img_bgr: np.ndarray,
    imgsz: int = 1280,
    overlap: float = 0.25,
    max_rel_side: float = 0.045,
    border_margin_px: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a 2x2 overlapping high-resolution quadrant pass on native-resolution test boards,
    strictly retaining ONLY interior micro-components (Class 0 Capacitor & tiny Class 1/3
    whose side <= 4.5% of board size and not touching any internal cut border).
    Prevents tile-cutting false positives on large ICs and Connectors!
    """
    H, W = img_bgr.shape[:2]
    if max(H, W) <= imgsz:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=int)

    quad_w = int(round(W * (0.5 + overlap * 0.5)))
    quad_h = int(round(H * (0.5 + overlap * 0.5)))
    xs = [0, max(0, W - quad_w)]
    ys = [0, max(0, H - quad_h)]

    boxes_all, scores_all, cls_all = [], [], []
    max_side_px = max_rel_side * float(max(W, H))

    for x0 in xs:
        for y0 in ys:
            x1_q, y1_q = x0, y0
            x2_q, y2_q = min(W, x0 + quad_w), min(H, y0 + quad_h)
            crop = img_bgr[y1_q:y2_q, x1_q:x2_q]
            r = model.predict(crop, imgsz=imgsz, conf=0.005, iou=0.55, max_det=600, verbose=False)[0]
            b_crop = r.boxes.xyxy.cpu().numpy()
            s_crop = r.boxes.conf.cpu().numpy()
            c_crop = r.boxes.cls.cpu().numpy().astype(int)

            ch, cw = crop.shape[:2]
            for (bx1, by1, bx2, by2), sc, cid in zip(b_crop, s_crop, c_crop):
                bw, bh = bx2 - bx1, by2 - by1
                # Keep only micro/small components (primarily Capacitors)
                if cid not in (0, 1) or max(bw, bh) > max_side_px:
                    continue
                # Reject boxes truncated by internal quadrant cut boundaries
                if x1_q > 0 and bx1 < border_margin_px:
                    continue
                if x2_q < W and bx2 > (cw - border_margin_px):
                    continue
                if y1_q > 0 and by1 < border_margin_px:
                    continue
                if y2_q < H and by2 > (ch - border_margin_px):
                    continue

                boxes_all.append([bx1 + x1_q, by1 + y1_q, bx2 + x1_q, by2 + y1_q])
                scores_all.append(sc)
                cls_all.append(cid)

    if not boxes_all:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=int)

    # Deduplicate overlapping quadrant detections via WBF
    b_arr = np.array(boxes_all, dtype=np.float32)
    s_arr = np.array(scores_all, dtype=np.float32)
    c_arr = np.array(cls_all, dtype=int)
    out_b, out_s, out_c = [], [], []
    for cid in np.unique(c_arr):
        m = c_arr == cid
        fb, fs = wbf_single_class([b_arr[m]], [s_arr[m]], [1.0], iou_thr=0.55)
        if len(fb) > 0:
            out_b.append(fb)
            out_s.append(fs)
            out_c.append(np.full(len(fs), cid, dtype=int))
    return np.concatenate(out_b, axis=0), np.concatenate(out_s, axis=0), np.concatenate(out_c, axis=0)


def compute_split_metrics_from_stats(stats: list, names: List[str], models_used: List[str]) -> dict:
    tp, conf, pred_cls, target_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    tp_r, fp_r, p, r, f1, ap, ap_class = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    ap50, ap50_95 = ap[:, 0], ap.mean(1)
    return {
        "models_fused": models_used,
        "mAP50": float(ap50.mean()),
        "mAP50_95": float(ap50_95.mean()),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
        "per_class_ap50": {names[int(c)]: round(float(ap50[i]), 4) for i, c in enumerate(ap_class)},
        "per_class_ap50_95": {names[int(c)]: round(float(ap50_95[i]), 4) for i, c in enumerate(ap_class)},
    }


def accumulate_image_stat(
    pred_b: torch.Tensor,
    pred_s: torch.Tensor,
    pred_c: torch.Tensor,
    gt_boxes_t: torch.Tensor,
    gt_cls_t: torch.Tensor,
    iouv: torch.Tensor,
    stats: list,
):
    if len(pred_b) == 0:
        if len(gt_cls_t) > 0:
            stats.append((torch.zeros((0, 10), dtype=torch.bool), torch.zeros((0,)), torch.zeros((0,)), gt_cls_t))
        return

    if len(gt_cls_t) > 0:
        ious = box_iou(gt_boxes_t, pred_b)
        correct = np.zeros((len(pred_b), len(iouv)), dtype=bool)
        correct_class = gt_cls_t[:, None] == pred_c
        for i, thr in enumerate(iouv):
            x = torch.where((ious >= thr) & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]][:, None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        tp = torch.tensor(correct, dtype=torch.bool)
    else:
        tp = torch.zeros((len(pred_b), 10), dtype=torch.bool)

    stats.append((tp, pred_s, pred_c, gt_cls_t))


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {"tier1_single_models": {}, "tier2_fullboard_wbf": {}, "tier3_quadrant_hyper_wbf": {}}
    loaded_models = []

    print("\n================ TIER 1: SINGLE-MODEL FULL-BOARD TEST EVALUATION ================")
    for name, w_path, imgsz, use_quad in CANDIDATES:
        p = Path(w_path)
        if not p.exists():
            print(f"  [SKIP] {name}: {w_path} not found")
            continue

        model = YOLO(str(p))
        # Evaluate on VAL first to obtain fair per-class validation weights for Tier 2/3 WBF
        val_metrics = model.val(
            data=DATA_YAML,
            split="val",
            imgsz=imgsz,
            conf=0.001,
            iou=0.50,
            max_det=1000,
            device=0,
            plots=False,
            verbose=False,
        )
        val_cls_weights = {}
        for idx, cls_idx in enumerate(val_metrics.box.ap_class_index):
            val_cls_weights[int(cls_idx)] = max(0.10, float(val_metrics.box.ap50[idx]) ** 2)

        # Evaluate on held-out TEST split
        test_metrics = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=imgsz,
            conf=0.001,
            iou=0.50,
            max_det=1000,
            device=0,
            plots=False,
            verbose=False,
        )
        per_class_ap50 = {}
        per_class_ap50_95 = {}
        for idx, cls_idx in enumerate(test_metrics.box.ap_class_index):
            cname = test_metrics.names[int(cls_idx)]
            per_class_ap50[cname] = round(float(test_metrics.box.ap50[idx]), 4)
            per_class_ap50_95[cname] = round(float(test_metrics.box.maps[idx]), 4)

        res = {
            "weights": str(p),
            "imgsz": imgsz,
            "val_mAP50": round(float(val_metrics.box.map50), 4),
            "mAP50": round(float(test_metrics.box.map50), 4),
            "mAP50_95": round(float(test_metrics.box.map), 4),
            "precision": round(float(test_metrics.box.mp), 4),
            "recall": round(float(test_metrics.box.mr), 4),
            "per_class_ap50": per_class_ap50,
            "per_class_ap50_95": per_class_ap50_95,
        }
        summary["tier1_single_models"][name] = res
        loaded_models.append((name, model, imgsz, use_quad, val_cls_weights, float(val_metrics.box.map50)))
        print(
            f"{name:34s} | Test mAP50={res['mAP50']:.4f} | mAP50-95={res['mAP50_95']:.4f} | "
            f"P={res['precision']:.4f} | R={res['recall']:.4f} | AP50={per_class_ap50}"
        )

    # Sort loaded models by validation mAP50 descending and keep top-5 for ensemble
    loaded_models.sort(key=lambda x: -x[5])
    top_models = loaded_models[:5]

    if len(top_models) >= 2:
        print("\n================ TIER 2 & 3: PER-CLASS CALIBRATED WBF & QUADRANT HYPER-ENSEMBLE ================")
        cfg = yaml.safe_load(open(DATA_YAML))
        root = Path(cfg["path"])
        test_imgs = sorted((root / cfg["test"]).glob("*"))
        names = cfg["names"]
        nc = len(names)
        iouv = torch.linspace(0.5, 0.95, 10)

        stats_tier2 = []
        stats_tier3 = []

        for im_path in test_imgs:
            img_bgr = cv2.imread(str(im_path))
            H, W = img_bgr.shape[:2]
            lb_path = root / "test" / "labels" / f"{im_path.stem}.txt"
            gt_cls, gt_boxes = [], []
            if lb_path.exists():
                for line in lb_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    c = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    gt_cls.append(c)
                    gt_boxes.append([
                        (cx - bw * 0.5) * W,
                        (cy - bh * 0.5) * H,
                        (cx + bw * 0.5) * W,
                        (cy + bh * 0.5) * H,
                    ])
            gt_cls_t = torch.tensor(gt_cls, dtype=torch.int64)
            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4))

            full_preds = []
            quad_preds = []
            for m_name, model, imgsz, use_quad, val_cw, _ in top_models:
                r = model.predict(img_bgr, imgsz=imgsz, conf=0.001, iou=0.55, max_det=1000, verbose=False)[0]
                full_preds.append((
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int),
                    val_cw,
                ))
                if use_quad:
                    qb, qs, qc = predict_quadrant_micro_capacitors(model, img_bgr, imgsz=1280)
                    if len(qb) > 0:
                        quad_preds.append((qb, qs, qc, val_cw))

            # --- Tier 2: Full-Board Per-Class WBF ---
            t2_xyxy, t2_conf, t2_cls = [], [], []
            for c in range(nc):
                b_list = [p[0][p[2] == c] for p in full_preds]
                s_list = [p[1][p[2] == c] for p in full_preds]
                w_list = [p[3].get(c, 0.25) for p in full_preds]
                fb, fs = wbf_single_class(b_list, s_list, w_list, iou_thr=0.55)
                if len(fb) > 0:
                    t2_xyxy.append(fb)
                    t2_conf.append(fs)
                    t2_cls.append(np.full(len(fs), c, dtype=np.int64))

            if t2_xyxy:
                pb2 = torch.from_numpy(np.concatenate(t2_xyxy, axis=0)).float()
                ps2 = torch.from_numpy(np.concatenate(t2_conf, axis=0)).float()
                pc2 = torch.from_numpy(np.concatenate(t2_cls, axis=0)).long()
                ord2 = torch.argsort(ps2, descending=True)[:1000]
                pb2, ps2, pc2 = pb2[ord2], ps2[ord2], pc2[ord2]
            else:
                pb2, ps2, pc2 = torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64)
            accumulate_image_stat(pb2, ps2, pc2, gt_boxes_t, gt_cls_t, iouv, stats_tier2)

            # --- Tier 3: Scale-Conditioned Quadrant + Full-Board WBF ---
            t3_xyxy, t3_conf, t3_cls = [], [], []
            for c in range(nc):
                b_list = [p[0][p[2] == c] for p in full_preds]
                s_list = [p[1][p[2] == c] for p in full_preds]
                w_list = [p[3].get(c, 0.25) for p in full_preds]
                if c in (0, 1) and quad_preds:
                    for qp in quad_preds:
                        b_list.append(qp[0][qp[2] == c])
                        s_list.append(qp[1][qp[2] == c])
                        w_list.append(qp[3].get(c, 0.25) * (0.85 if c == 0 else 0.50))
                fb, fs = wbf_single_class(b_list, s_list, w_list, iou_thr=0.55)
                if len(fb) > 0:
                    t3_xyxy.append(fb)
                    t3_conf.append(fs)
                    t3_cls.append(np.full(len(fs), c, dtype=np.int64))

            if t3_xyxy:
                pb3 = torch.from_numpy(np.concatenate(t3_xyxy, axis=0)).float()
                ps3 = torch.from_numpy(np.concatenate(t3_conf, axis=0)).float()
                pc3 = torch.from_numpy(np.concatenate(t3_cls, axis=0)).long()
                ord3 = torch.argsort(ps3, descending=True)[:1200]
                pb3, ps3, pc3 = pb3[ord3], ps3[ord3], pc3[ord3]
            else:
                pb3, ps3, pc3 = torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64)
            accumulate_image_stat(pb3, ps3, pc3, gt_boxes_t, gt_cls_t, iouv, stats_tier3)

        used_names = [m[0] for m in top_models]
        t2_res = compute_split_metrics_from_stats(stats_tier2, names, used_names)
        t3_res = compute_split_metrics_from_stats(stats_tier3, names, used_names)
        summary["tier2_fullboard_wbf"] = t2_res
        summary["tier3_quadrant_hyper_wbf"] = t3_res

        print(
            f"TIER 2 (Class-Calibrated Full-Board WBF) | Test mAP50={t2_res['mAP50']:.4f} | "
            f"mAP50-95={t2_res['mAP50_95']:.4f} | P={t2_res['precision']:.4f} | "
            f"R={t2_res['recall']:.4f} | AP50={t2_res['per_class_ap50']}"
        )
        print(
            f"TIER 3 (Quadrant Micro + Full-Board WBF) | Test mAP50={t3_res['mAP50']:.4f} | "
            f"mAP50-95={t3_res['mAP50_95']:.4f} | P={t3_res['precision']:.4f} | "
            f"R={t3_res['recall']:.4f} | AP50={t3_res['per_class_ap50']}"
        )

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    for rdir in (Path("/mnt/weka/etadevosyan/pcb-yolo/results"), Path("results")):
        try:
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "r4_final_apex_sota_summary.json").write_text(json.dumps(summary, indent=2))
        except Exception:
            pass
    print(f"\nSaved complete 3-Tier SOTA evaluation summary to {OUT_JSON}")


if __name__ == "__main__":
    main()
