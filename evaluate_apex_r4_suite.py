#!/usr/bin/env python3
"""
3-Tier SOTA Evaluation: Multi-Model + 4 Single-Class Specialists + TTA + Native Slicing WBF
===========================================================================================
Evaluates all R4-Final models on the canonical held-out Test Split
(`datasets/pcb-native-reviewed-r4-final/data.yaml`, 44 boards, 4,909 targets) across:

  - Tier 1: Standard Single-Model Full-Board Evaluation (`model.val(split='test')`)
            PLUS dedicated evaluation of the 4 Single-Class Specialists (`c0..c3`).
  - Tier 2: Per-Class Validation-Calibrated Full-Board WBF combining:
            * Top Multi-Class Models (`YOLO11m`, `YOLO26s Crop`, `P2 Crop`, `AGWD+SDice`, `PhysOT`, `YOLOv12s`, `APEX`)
            * High-Res `1536px` + Test-Time Augmentation (`augment=True`) passes
            * The 4 Dedicated 160-Epoch Single-Class Specialists (`c0_capacitor`, `c1_connector`, `c2_electrolytic`, `c3_ic`)
            with per-class `(iou_thr, beta, gamma, tta_mult, spec_mult)` calibrated on `valid` (78 boards).
  - Tier 3: Scale-Conditioned Native High-Res Slicing (`2x2` Quadrants + `3x3` Native Tiles) + Full-Board WBF
            using `R4_S_YOLO26s_CropSpec_200ep`, `R4_S_YOLO26s_P2_CropSpec_200ep`, and `Spec_C0_Capacitor`
            to recover sub-8px micro-capacitors lost in 1280px downsampling, calibrated per-class on `valid`.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml
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

# Multi-class candidate checkpoints: (Name, Path, Eval_Imgsz, Enable_Tile_MicroPass)
CANDIDATES = [
    ("APEX_YOLO11m_1536", "runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt", 1536, True),
    ("APEX_YOLO26s_1536", "runs/r4_apex_yolo26s_1536/pcb-filtered/weights/best.pt", 1536, True),
    ("APEX_YOLO26x_1280", "runs/r4_apex_yolo26x_1280/pcb-filtered/weights/best.pt", 1280, False),
    ("APEX_YOLOv12s_1536", "runs/r4_apex_yolov12s_1536/pcb-filtered/weights/best.pt", 1536, True),
    ("SOTA_YOLO11m_RetinexEdge_1536", "runs/r4_sota_yolo11m_retinex_agwd_1536/pcb-filtered/weights/best.pt", 1536, True),
    ("SOTA_YOLO26x_RetinexEdge_1280", "runs/r4_sota_yolo26x_retinex_agwd_1280/pcb-filtered/weights/best.pt", 1280, False),
    ("SOTA_YOLO26s_RetinexEdge_1536", "runs/r4_sota_yolo26s_retinex_agwd_1536/pcb-filtered/weights/best.pt", 1536, False),
    ("Baseline_YOLO11m_200ep_1280", "runs/detect/r4_baseline_yolo11m_multiclass_200ep/weights/best.pt", 1280, True),
    ("R4_S_YOLO26s_CropSpec_200ep", "runs/r4_s_yolo26s_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, True),
    ("R4_S_YOLO26s_P2_CropSpec_200ep", "runs/r4_s_yolo26s_p2_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, True),
    ("R4_S_PhysOT_YOLO26s_Mixed_200ep", "runs/r4_s_physot_yolo26s_mixed_200ep/pcb-filtered/weights/best.pt", 1280, False),
    ("Prev_YOLO26s_AGWD_SDice_1280", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, False),
    ("Prev_PhysOT_YOLO26s_1280", "runs/r4_final_physot_yolo26s_1280/pcb-filtered/weights/best.pt", 1280, False),
    ("Prev_YOLOv12s_Champion_1280", "runs/r4_final_yolov12s_champion_1280/pcb-filtered/weights/best.pt", 1280, False),
    ("Prev_YOLO26s_CropSpec_100ep", "runs/r4_final_yolo26s_crop_specialist_1280/pcb-filtered/weights/best.pt", 1280, False),
]

# Dedicated 1-Class Specialist checkpoints: (Name, Path, Target_Class_ID, Eval_Imgsz, Enable_Tile_Pass)
SPECIALIST_CANDIDATES = [
    ("Spec_C0_Capacitor", "runs/detect/r4_specialist_c0_capacitor/weights/best.pt", 0, 1280, True),
    ("Spec_C1_Connector", "runs/detect/r4_specialist_c1_connector/weights/best.pt", 1, 1280, False),
    ("Spec_C2_Electrolytic", "runs/detect/r4_specialist_c2_electrolytic/weights/best.pt", 2, 1280, False),
    ("Spec_C3_IC", "runs/detect/r4_specialist_c3_ic/weights/best.pt", 3, 1280, False),
]


def resolve_ckpt_path(w_path: str) -> Optional[Path]:
    for cand in [
        Path(w_path),
        Path("runs/detect") / w_path,
        Path(w_path.replace("runs/detect/", "runs/")),
        Path(w_path.replace("best.pt", "last.pt")),
    ]:
        if cand.exists():
            return cand
    # Also search parent run directory for any .pt file if best.pt was renamed
    run_dir = Path(w_path).parent.parent
    if run_dir.exists():
        pts = sorted(run_dir.rglob("*.pt"))
        if pts:
            return pts[0]
    return None


def wbf_single_class(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    weights: List[float],
    iou_thr: float = 0.55,
    beta: float = 0.65,
    gamma: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft-Consensus Weighted Boxes Fusion for a single class.
    When beta=1.0 and gamma=1.0, reduces to classic WBF: score = sum(w_i * s_i) / sum_all(w).
    When 0 < beta < 1 or gamma < 1, preserves high-confidence true positives found only by
    high-resolution tile passes or class specialists while still ranking multi-model consensus highest:
        mean_s = sum_{i in C} (w_i * s_i) / sum_{i in C} w_i
        frac_w = min(1.0, sum_{i in C} w_i / sum_all(w))
        final_s = mean_s * ((1 - beta) + beta * (frac_w ** gamma))
    """
    all_b, all_ws, all_w = [], [], []
    valid_weights = [float(w) for w in weights if w > 0]
    total_w = float(sum(valid_weights)) if valid_weights else 1.0

    for b, s, w in zip(boxes_list, scores_list, weights):
        if len(b) == 0 or w <= 0:
            continue
        fw = float(w)
        all_b.append(b.astype(np.float32))
        all_ws.append((s * fw).astype(np.float32))
        all_w.append(np.full(len(s), fw, dtype=np.float32))

    if not all_b:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    all_b = np.concatenate(all_b, axis=0)
    all_ws = np.concatenate(all_ws, axis=0)
    all_w = np.concatenate(all_w, axis=0)
    order = np.argsort(-all_ws)
    all_b, all_ws, all_w = all_b[order], all_ws[order], all_w[order]

    clusters_b, clusters_ws, clusters_w, fused_b = [], [], [], []

    for b, ws, w in zip(all_b, all_ws, all_w):
        if not fused_b:
            clusters_b.append([b])
            clusters_ws.append([ws])
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
            clusters_ws[best_idx].append(ws)
            clusters_w[best_idx].append(w)
            cws = np.array(clusters_ws[best_idx], dtype=np.float32)
            cb = np.stack(clusters_b[best_idx], axis=0)
            fused_b[best_idx] = np.sum(cb * cws[:, None], axis=0) / np.maximum(np.sum(cws), 1e-7)
        else:
            clusters_b.append([b])
            clusters_ws.append([ws])
            clusters_w.append([w])
            fused_b.append(b.copy())

    out_b = np.stack(fused_b, axis=0)
    out_s = np.empty(len(fused_b), dtype=np.float32)
    for k, (cws, cw) in enumerate(zip(clusters_ws, clusters_w)):
        sum_ws = float(np.sum(cws))
        sum_w = max(float(np.sum(cw)), 1e-7)
        mean_s = sum_ws / sum_w
        frac_w = min(1.0, sum_w / max(total_w, 1e-7))
        if gamma <= 0.0:
            out_s[k] = mean_s
        else:
            out_s[k] = mean_s * ((1.0 - beta) + beta * (frac_w ** gamma))

    order = np.argsort(-out_s)
    return out_b[order], out_s[order]


def predict_native_grid_tiles(
    model: YOLO,
    img_bgr: np.ndarray,
    grid_n: int = 2,
    imgsz: int = 1280,
    overlap: float = 0.22,
    border_margin_px: int = 12,
    specialist_target_cid: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a `grid_n x grid_n` (e.g. 2x2 or 3x3) overlapping high-resolution tile pass on native
    4112x3008 boards, rejecting boxes truncated by internal cut boundaries and returning:
        (boxes_xyxy, scores, class_ids, rel_sides)
    where `rel_sides = max(bw, bh) / max(W, H)` allows fast in-RAM filtering by scale!
    """
    H, W = img_bgr.shape[:2]
    if max(H, W) <= imgsz:
        z4 = np.zeros((0, 4), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        zi = np.zeros((0,), dtype=int)
        return z4, z1, zi, z1

    tile_w = int(round(W / float(grid_n) * (1.0 + overlap)))
    tile_h = int(round(H / float(grid_n) * (1.0 + overlap)))
    tile_w = min(W, max(imgsz, tile_w))
    tile_h = min(H, max(int(imgsz * 0.75), tile_h))

    xs = np.linspace(0, max(0, W - tile_w), grid_n).round().astype(int).tolist()
    ys = np.linspace(0, max(0, H - tile_h), grid_n).round().astype(int).tolist()

    boxes_all, scores_all, cls_all = [], [], []
    max_board_dim = float(max(W, H))
    nc_model = len(getattr(model, "names", {0: "0"}))

    for x0 in xs:
        for y0 in ys:
            x1_q, y1_q = x0, y0
            x2_q, y2_q = min(W, x0 + tile_w), min(H, y0 + tile_h)
            crop = img_bgr[y1_q:y2_q, x1_q:x2_q]
            r = model.predict(crop, imgsz=imgsz, conf=0.004, iou=0.55, max_det=600, verbose=False)[0]
            b_crop = r.boxes.xyxy.cpu().numpy()
            s_crop = r.boxes.conf.cpu().numpy()
            c_crop = r.boxes.cls.cpu().numpy().astype(int)

            ch, cw = crop.shape[:2]
            for (bx1, by1, bx2, by2), sc, cid in zip(b_crop, s_crop, c_crop):
                if specialist_target_cid is not None:
                    if nc_model == 1:
                        cid = specialist_target_cid
                    elif cid != specialist_target_cid:
                        continue
                # Reject boxes touching internal cut boundaries
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
        z4 = np.zeros((0, 4), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        zi = np.zeros((0,), dtype=int)
        return z4, z1, zi, z1

    b_arr = np.array(boxes_all, dtype=np.float32)
    s_arr = np.array(scores_all, dtype=np.float32)
    c_arr = np.array(cls_all, dtype=int)

    # Deduplicate overlapping tile detections per class
    out_b, out_s, out_c = [], [], []
    for cid in np.unique(c_arr):
        m = c_arr == cid
        fb, fs = wbf_single_class([b_arr[m]], [s_arr[m]], [1.0], iou_thr=0.52, beta=1.0, gamma=0.0)
        if len(fb) > 0:
            out_b.append(fb)
            out_s.append(fs)
            out_c.append(np.full(len(fs), cid, dtype=int))

    fb_all = np.concatenate(out_b, axis=0)
    fs_all = np.concatenate(out_s, axis=0)
    fc_all = np.concatenate(out_c, axis=0)
    rel_sides = np.maximum(fb_all[:, 2] - fb_all[:, 0], fb_all[:, 3] - fb_all[:, 1]) / max_board_dim
    return fb_all, fs_all, fc_all, rel_sides.astype(np.float32)


def load_split_gt(root: Path, split_subpath: str) -> List[Tuple[Path, np.ndarray, int, int, torch.Tensor, torch.Tensor]]:
    """Loads images and ground-truth boxes for a split (`valid` or `test`)."""
    img_dir = root / split_subpath
    lbl_dir = img_dir.parent / "labels"
    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    records = []
    for im_path in img_paths:
        img_bgr = cv2.imread(str(im_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        lb_path = lbl_dir / f"{im_path.stem}.txt"
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
        records.append((im_path, img_bgr, H, W, gt_boxes_t, gt_cls_t))
    return records


def compute_single_class_ap50(
    per_img_preds: List[Tuple[np.ndarray, np.ndarray]],
    split_records: List[Tuple[Path, np.ndarray, int, int, torch.Tensor, torch.Tensor]],
    target_cid: int,
) -> Tuple[float, float]:
    """Computes exact Ultralytics AP@50 and AP@50-95 for a single class `target_cid` across a split."""
    iouv = torch.linspace(0.5, 0.95, 10)
    stats = []
    for (pb_np, ps_np), (_, _, _, _, gt_boxes_t, gt_cls_t) in zip(per_img_preds, split_records):
        gt_mask = gt_cls_t == target_cid
        gt_b = gt_boxes_t[gt_mask]
        gt_c = gt_cls_t[gt_mask]

        if len(pb_np) > 1000:
            pb_np = pb_np[:1000]
            ps_np = ps_np[:1000]

        if len(pb_np) == 0:
            if len(gt_c) > 0:
                stats.append((torch.zeros((0, 10), dtype=torch.bool), torch.zeros((0,)), torch.zeros((0,)), gt_c))
            continue

        pred_b = torch.from_numpy(pb_np).float()
        pred_s = torch.from_numpy(ps_np).float()
        pred_c = torch.full((len(pb_np),), target_cid, dtype=torch.int64)

        if len(gt_c) > 0:
            ious = box_iou(gt_b, pred_b)
            correct = np.zeros((len(pred_b), len(iouv)), dtype=bool)
            for i, thr in enumerate(iouv):
                x = torch.where(ious >= thr)
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
        stats.append((tp, pred_s, pred_c, gt_c))

    if not stats:
        return 0.0, 0.0
    tp, conf, pred_cls, target_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(target_cls) == 0 or len(tp) == 0:
        return 0.0, 0.0
    _, _, _, _, _, ap, _, *_ = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    return float(ap[0, 0]), float(ap[0].mean())


def compute_split_metrics_from_stats(stats: list, names: List[str], models_used: List[str]) -> dict:
    tp, conf, pred_cls, target_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    tp_r, fp_r, p, r, f1, ap, ap_class, *_ = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
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


def extract_split_predictions_for_streams(
    split_records: List[Tuple[Path, np.ndarray, int, int, torch.Tensor, torch.Tensor]],
    top_models: list,
    specialists: dict,
    nc: int = 4,
) -> Dict[int, List[Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]]]]:
    """
    Runs inference on `split_records` and organizes per-image predictions per class `c` into
    4 streams:
      - 'base': Full-board multi-class models (`imgsz=1280/1536`)
      - 'tta':  High-res `1536px` + Test-Time Augmentation (`augment=True`) passes for top-3 models
      - 'spec': Dedicated 1-class specialist for class `c` (`Spec_C0..C3`) + its TTA pass
      - 'tile': Native `2x2` & `3x3` high-res overlapping tiles (with `rel_sides` stored for scale gating)
    """
    n_imgs = len(split_records)
    per_cls_data = {
        c: [{"base": [], "tta": [], "spec": [], "tile": []} for _ in range(n_imgs)]
        for c in range(nc)
    }

    for img_idx, (im_path, img_bgr, H, W, _, _) in enumerate(split_records):
        # 1. Multi-class full-board models + TTA on top-3 + Tiles on tile-enabled models
        for rank_idx, (m_name, model, imgsz, use_tile, val_cw, _) in enumerate(top_models):
            r = model.predict(img_bgr, imgsz=imgsz, conf=0.001, iou=0.55, max_det=1000, verbose=False)[0]
            b_np = r.boxes.xyxy.cpu().numpy()
            s_np = r.boxes.conf.cpu().numpy()
            c_np = r.boxes.cls.cpu().numpy().astype(int)
            for c in range(nc):
                m = c_np == c
                per_cls_data[c][img_idx]["base"].append((b_np[m], s_np[m], float(val_cw.get(c, 0.20))))

            # Run Multi-Scale Flip TTA (`augment=True`) and `1536px` high-res pass on top 3 models
            if rank_idx < 3:
                r_tta = model.predict(
                    img_bgr, imgsz=max(imgsz, 1536), conf=0.001, iou=0.55, max_det=1000, augment=True, verbose=False
                )[0]
                bt_np = r_tta.boxes.xyxy.cpu().numpy()
                st_np = r_tta.boxes.conf.cpu().numpy()
                ct_np = r_tta.boxes.cls.cpu().numpy().astype(int)
                for c in range(nc):
                    m = ct_np == c
                    per_cls_data[c][img_idx]["tta"].append((bt_np[m], st_np[m], float(val_cw.get(c, 0.20))))

            # Run 2x2 and 3x3 native high-res tile passes on top 2 tile-enabled models
            if use_tile and rank_idx < 4:
                for grid_n, g_weight in ((2, 1.0), (3, 0.85)):
                    tb, ts, tc, trs = predict_native_grid_tiles(model, img_bgr, grid_n=grid_n, imgsz=1280)
                    if len(tb) > 0:
                        for c in (0, 1):
                            m = tc == c
                            if np.any(m):
                                per_cls_data[c][img_idx]["tile"].append(
                                    (tb[m], ts[m], float(val_cw.get(c, 0.20)) * g_weight, trs[m])
                                )

        # 2. Dedicated Single-Class Specialists (c0..c3)
        for c, spec_info in specialists.items():
            s_name, s_model, s_imgsz, s_use_tile, s_val_w = spec_info
            nc_s = len(getattr(s_model, "names", {0: "0"}))
            for is_tta, w_factor in ((False, 1.0), (True, 0.75)):
                r_s = s_model.predict(
                    img_bgr,
                    imgsz=max(s_imgsz, 1536) if is_tta else s_imgsz,
                    conf=0.001,
                    iou=0.55,
                    max_det=1000,
                    augment=is_tta,
                    verbose=False,
                )[0]
                bs_np = r_s.boxes.xyxy.cpu().numpy()
                ss_np = r_s.boxes.conf.cpu().numpy()
                cs_np = r_s.boxes.cls.cpu().numpy().astype(int)
                if nc_s == 1:
                    per_cls_data[c][img_idx]["spec"].append((bs_np, ss_np, float(s_val_w) * w_factor))
                else:
                    m = cs_np == c
                    per_cls_data[c][img_idx]["spec"].append((bs_np[m], ss_np[m], float(s_val_w) * w_factor))

            if s_use_tile and c == 0:
                tb, ts, tc, trs = predict_native_grid_tiles(
                    s_model, img_bgr, grid_n=2, imgsz=1280, specialist_target_cid=c
                )
                if len(tb) > 0:
                    per_cls_data[c][img_idx]["tile"].append((tb, ts, float(s_val_w) * 0.90, trs))

    return per_cls_data


def run_stream_wbf_for_class(
    cls_img_streams: List[ dict ],
    iou_thr: float,
    beta: float,
    gamma: float,
    tta_mult: float,
    spec_mult: float,
    tile_mult: float = 0.0,
    max_rel_side: float = 0.055,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Fuses cached streams for a single class across all images in a split."""
    preds = []
    for item in cls_img_streams:
        b_list, s_list, w_list = [], [], []
        for b, s, w in item["base"]:
            b_list.append(b)
            s_list.append(s)
            w_list.append(w)
        if tta_mult > 0.0:
            for b, s, w in item["tta"]:
                b_list.append(b)
                s_list.append(s)
                w_list.append(w * tta_mult)
        if spec_mult > 0.0:
            for b, s, w in item["spec"]:
                b_list.append(b)
                s_list.append(s)
                w_list.append(w * spec_mult)
        if tile_mult > 0.0:
            for b, s, w, rs in item["tile"]:
                m = rs <= max_rel_side
                if np.any(m):
                    b_list.append(b[m])
                    s_list.append(s[m])
                    w_list.append(w * tile_mult)

        fb, fs = wbf_single_class(b_list, s_list, w_list, iou_thr=iou_thr, beta=beta, gamma=gamma)
        preds.append((fb, fs))
    return preds


def calibrate_class_hyperparams_on_val(
    c: int,
    cname: str,
    val_cls_streams: List[dict],
    val_records: list,
    allow_tiles: bool = False,
    init_cfg: Optional[dict] = None,
) -> Tuple[dict, float]:
    """Coordinate-ascent hyperparameter calibration on `valid` (78 boards) for class `c`."""
    cfg = dict(
        init_cfg
        or {
            "iou_thr": 0.55,
            "beta": 0.65,
            "gamma": 0.65,
            "tta_mult": 0.60,
            "spec_mult": 0.65,
            "tile_mult": 0.0,
            "max_rel_side": 0.055,
        }
    )

    def eval_cfg(cand_cfg: dict) -> float:
        preds = run_stream_wbf_for_class(val_cls_streams, **cand_cfg)
        ap50, _ = compute_single_class_ap50(preds, val_records, c)
        return ap50

    best_ap50 = eval_cfg(cfg)

    if not allow_tiles:
        search_grid = [
            ("iou_thr", [0.46, 0.52, 0.56, 0.62, 0.68]),
            ("tta_mult", [0.0, 0.40, 0.75, 1.10]),
            ("spec_mult", [0.0, 0.25, 0.55, 0.90, 1.35, 1.80]),
            ("beta", [0.35, 0.65, 0.85, 1.0]),
            ("gamma", [0.0, 0.35, 0.65, 1.0]),
        ]
    else:
        search_grid = [
            ("tile_mult", [0.0, 0.20, 0.45, 0.75, 1.10]),
            ("max_rel_side", [0.035, 0.050, 0.075]),
            ("iou_thr", [0.46, 0.52, 0.56, 0.62]),
            ("beta", [0.35, 0.55, 0.75, 1.0]),
            ("gamma", [0.25, 0.50, 0.75, 1.0]),
        ]

    for param_name, values in search_grid:
        for val in values:
            trial = dict(cfg)
            trial[param_name] = val
            score = eval_cfg(trial)
            if score > best_ap50 + 1e-5:
                best_ap50 = score
                cfg = trial

    return cfg, best_ap50


def evaluate_calibrated_ensemble_on_split(
    per_cls_streams: Dict[int, List[dict]],
    split_records: list,
    per_cls_cfg: Dict[int, dict],
    names: List[str],
    models_used: List[str],
) -> dict:
    nc = len(names)
    n_imgs = len(split_records)
    iouv = torch.linspace(0.5, 0.95, 10)

    per_cls_preds = {}
    for c in range(nc):
        per_cls_preds[c] = run_stream_wbf_for_class(per_cls_streams[c], **per_cls_cfg[c])

    stats = []
    for img_idx in range(n_imgs):
        _, _, _, _, gt_boxes_t, gt_cls_t = split_records[img_idx]
        img_b, img_s, img_c = [], [], []
        for c in range(nc):
            fb, fs = per_cls_preds[c][img_idx]
            if len(fb) > 0:
                img_b.append(fb)
                img_s.append(fs)
                img_c.append(np.full(len(fs), c, dtype=np.int64))
        if img_b:
            pb = torch.from_numpy(np.concatenate(img_b, axis=0)).float()
            ps = torch.from_numpy(np.concatenate(img_s, axis=0)).float()
            pc = torch.from_numpy(np.concatenate(img_c, axis=0)).long()
            ord_idx = torch.argsort(ps, descending=True)[:1200]
            pb, ps, pc = pb[ord_idx], ps[ord_idx], pc[ord_idx]
        else:
            pb, ps, pc = torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64)
        accumulate_image_stat(pb, ps, pc, gt_boxes_t, gt_cls_t, iouv, stats)

    res = compute_split_metrics_from_stats(stats, names, models_used)
    res["calibrated_class_configs"] = {names[c]: per_cls_cfg[c] for c in range(nc)}
    return res


def main():
    ap = argparse.ArgumentParser(description="3-Tier APEX & Specialist Benchmark on R4-Final Test Split")
    ap.add_argument("--data", default=DATA_YAML)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--top-k", type=int, default=6, help="Number of top multi-class models to fuse")
    ap.add_argument("--include-legacy", action="store_true")
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    args, _ = ap.parse_known_args()

    data_yaml = args.data
    out_json = args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "tier1_single_models": {},
        "tier1_single_class_specialists": {},
        "tier2_fullboard_wbf": {},
        "tier3_quadrant_hyper_wbf": {},
    }
    loaded_models = []

    cfg = yaml.safe_load(open(data_yaml))
    root = Path(cfg["path"])
    names = cfg["names"]
    nc = len(names)

    print("\n================ TIER 1A: MULTI-CLASS SINGLE-MODEL EVALUATION ================")
    for name, w_path, imgsz, use_tile in CANDIDATES:
        p = resolve_ckpt_path(w_path)
        if p is None:
            print(f"  [SKIP] {name}: {w_path} not found")
            continue

        model = YOLO(str(p))
        val_metrics = model.val(
            data=data_yaml,
            split="val",
            imgsz=imgsz,
            conf=0.001,
            iou=0.50,
            max_det=1000,
            device=args.device,
            plots=False,
            verbose=False,
        )
        val_cls_weights = {}
        for idx, cls_idx in enumerate(val_metrics.box.ap_class_index):
            val_cls_weights[int(cls_idx)] = max(0.08, float(val_metrics.box.ap50[idx]) ** 2)

        test_metrics = model.val(
            data=data_yaml,
            split="test",
            imgsz=imgsz,
            conf=0.001,
            iou=0.50,
            max_det=1000,
            device=args.device,
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
        loaded_models.append((name, model, imgsz, use_tile, val_cls_weights, float(val_metrics.box.map50)))
        print(
            f"{name:34s} | Val={res['val_mAP50']:.4f} | Test mAP50={res['mAP50']:.4f} | "
            f"mAP50-95={res['mAP50_95']:.4f} | P={res['precision']:.4f} | R={res['recall']:.4f} | AP50={per_class_ap50}"
        )

    # Load validation & test splits for 1-class specialists and WBF calibration
    val_records = load_split_gt(root, cfg["val"])
    test_records = load_split_gt(root, cfg["test"])

    print("\n================ TIER 1B: DEDICATED 1-CLASS SPECIALISTS EVALUATION ================")
    loaded_specialists = {}
    for s_name, w_path, target_cid, s_imgsz, s_use_tile in SPECIALIST_CANDIDATES:
        p = resolve_ckpt_path(w_path)
        if p is None:
            print(f"  [SKIP] {s_name}: {w_path} not found")
            continue
        s_model = YOLO(str(p))
        nc_s = len(getattr(s_model, "names", {0: "0"}))

        val_preds, test_preds = [], []
        for _, img_bgr, _, _, _, _ in val_records:
            r = s_model.predict(img_bgr, imgsz=s_imgsz, conf=0.001, iou=0.55, max_det=1000, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            val_preds.append((b, s) if nc_s == 1 else (b[c == target_cid], s[c == target_cid]))

        for _, img_bgr, _, _, _, _ in test_records:
            r = s_model.predict(img_bgr, imgsz=s_imgsz, conf=0.001, iou=0.55, max_det=1000, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            test_preds.append((b, s) if nc_s == 1 else (b[c == target_cid], s[c == target_cid]))

        val_ap50, val_ap50_95 = compute_single_class_ap50(val_preds, val_records, target_cid)
        test_ap50, test_ap50_95 = compute_single_class_ap50(test_preds, test_records, target_cid)
        cname = names[target_cid]
        summary["tier1_single_class_specialists"][s_name] = {
            "weights": str(p),
            "target_class": cname,
            "val_AP50": round(val_ap50, 4),
            "test_AP50": round(test_ap50, 4),
            "test_AP50_95": round(test_ap50_95, 4),
        }
        s_weight = max(0.10, val_ap50 ** 2)
        loaded_specialists[target_cid] = (s_name, s_model, s_imgsz, s_use_tile, s_weight)
        print(
            f"{s_name:34s} | Class={cname:22s} | Val AP50={val_ap50:.4f} | "
            f"Test AP50={test_ap50:.4f} | Test AP50-95={test_ap50_95:.4f}"
        )

    # Sort multi-class models by validation mAP50 descending and keep top-K
    loaded_models.sort(key=lambda x: -x[5])
    top_models = loaded_models[: args.top_k]

    if len(top_models) >= 2:
        print("\n================ TIER 2 & 3: VALIDATION-CALIBRATED SUPER-ENSEMBLE WBF ================")
        used_names = [m[0] for m in top_models] + [v[0] for v in loaded_specialists.values()]
        print(f"Extracting multi-stream predictions on VAL ({len(val_records)} boards) & TEST ({len(test_records)} boards)...")
        val_streams = extract_split_predictions_for_streams(val_records, top_models, loaded_specialists, nc=nc)
        test_streams = extract_split_predictions_for_streams(test_records, top_models, loaded_specialists, nc=nc)

        tier2_cfgs = {}
        tier3_cfgs = {}
        for c in range(nc):
            cname = names[c]
            cfg2, val_ap2 = calibrate_class_hyperparams_on_val(
                c, cname, val_streams[c], val_records, allow_tiles=False
            )
            tier2_cfgs[c] = cfg2

            cfg3, val_ap3 = calibrate_class_hyperparams_on_val(
                c, cname, val_streams[c], val_records, allow_tiles=(c in (0, 1)), init_cfg=cfg2
            )
            tier3_cfgs[c] = cfg3
            print(
                f"  [Val Calib] {cname:22s}: Tier2 Val AP50={val_ap2:.4f} -> Tier3 Val AP50={val_ap3:.4f} "
                f"(iou={cfg3['iou_thr']}, spec={cfg3['spec_mult']}, tta={cfg3['tta_mult']}, tile={cfg3['tile_mult']})"
            )

        t2_res = evaluate_calibrated_ensemble_on_split(test_streams, test_records, tier2_cfgs, names, used_names)
        t3_res = evaluate_calibrated_ensemble_on_split(test_streams, test_records, tier3_cfgs, names, used_names)
        summary["tier2_fullboard_wbf"] = t2_res
        summary["tier3_quadrant_hyper_wbf"] = t3_res

        print(
            f"\nTIER 2 (Calibrated Multi-Model + TTA + 4-Specialist WBF) | Test mAP50={t2_res['mAP50']:.4f} | "
            f"mAP50-95={t2_res['mAP50_95']:.4f} | P={t2_res['precision']:.4f} | "
            f"R={t2_res['recall']:.4f} | AP50={t2_res['per_class_ap50']}"
        )
        print(
            f"TIER 3 (Native 2x2/3x3 Tile + Specialist + Multi-Model WBF) | Test mAP50={t3_res['mAP50']:.4f} | "
            f"mAP50-95={t3_res['mAP50_95']:.4f} | P={t3_res['precision']:.4f} | "
            f"R={t3_res['recall']:.4f} | AP50={t3_res['per_class_ap50']}"
        )

    out_json.write_text(json.dumps(summary, indent=2))
    for rdir in (Path("/mnt/weka/etadevosyan/pcb-yolo/results"), Path("results")):
        try:
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "r4_final_apex_sota_summary.json").write_text(json.dumps(summary, indent=2))
        except Exception:
            pass
    print(f"\nSaved complete 3-Tier SOTA evaluation summary to {out_json}")


if __name__ == "__main__":
    main()
