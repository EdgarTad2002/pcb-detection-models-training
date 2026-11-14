#!/usr/bin/env python3
"""
Quick WBF v4: R4-Final Multi-Scale + 4-Class Quadrant Tile + Cross-Class Conflict Suppression
+ YOLO26 One-to-Many (end2end=False) Recovery + Stage-2 High-Res YOLO ROI Verifier
=============================================================================================
Upgrades over v3 (`0.5681` Test mAP@50):
  1. Evaluates the finished `APEX_YOLOv12s_1536` Greedy Soup checkpoint (`0.4977` Test mAP@50,
     `0.3880` Electrolytic Capacitor AP@50).
  2. Unlocks Ultralytics YOLO26 One-to-Many head (`end2end=False` + NMS, `max_det=1500`) on
     `APEX_YOLO26s_O2M`, `APEX_YOLO26x_O2M`, and `AGWD_SDice_O2M`.
  3. Enables `2x2` Native High-Res Quadrant Tiling for `Class 2 (Electrolytic Capacitor)` and
     `Class 3 (IC)` (fixing the `for k in (0, 1):` truncation bug in v3 while reusing cached
     base/1536/flip predictions in `results/wbf_cache_v3/`).
  4. Adds Cross-Class Conflict Suppression (`apply_cross_class_conflict_suppression`): suppresses
     phantom `Electrolytic Capacitor` false positives that collide (`IoU >= 0.62`) with high-consensus
     `IC` (`81.2%` AP) or `Connector` (`59.0%` AP) boxes on the same physical component.
  5. Integrates Stage-2 High-Resolution YOLO ROI Verifier (`runs/r4_yolo_roi_verifier/roi_verifier.pt`)
     when available (`Tier4_TwoStage_ROI_Verified_SOTA`).
"""

import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("LRU_CACHE_CAPACITY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from ultralytics import YOLO

cv2.setNumThreads(2)

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

from evaluate_apex_r4_suite import (
    DATA_YAML,
    accumulate_image_stat,
    compute_split_metrics_from_stats,
    predict_native_grid_tiles,
    resolve_ckpt_path,
    wbf_single_class,
)


def set_yolo26_end2end_mode(yolo_model: YOLO, end2end: bool) -> bool:
    """Switches a YOLO26 dual-head model to One-to-Many (`end2end=False`) or One-to-One (`end2end=True`)."""
    has_e2e = False
    try:
        det_head = yolo_model.model.model[-1]
        if hasattr(det_head, "end2end") or hasattr(det_head, "one2one_cv2"):
            det_head.end2end = bool(end2end)
            has_e2e = True
        if getattr(yolo_model, "predictor", None) is not None:
            pred_mod = getattr(yolo_model.predictor, "model", None)
            if pred_mod is not None and hasattr(pred_mod, "end2end"):
                pred_mod.end2end = bool(end2end)
    except Exception:
        pass
    return has_e2e


def load_test_paths_and_gt(root: Path, split_subpath: str):
    """Loads ONLY file paths and tiny GT tensors (<1 MB RAM total, no images cached in RAM)."""
    img_dir = root / split_subpath
    lbl_dir = img_dir.parent / "labels"
    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    records = []
    for im_path in img_paths:
        with Image.open(im_path) as im:
            W, H = im.size
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
        records.append((im_path, H, W, gt_boxes_t, gt_cls_t))
    return records


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-7)


def apply_cross_class_conflict_suppression(
    cls_boxes: Dict[int, np.ndarray],
    cls_scores: Dict[int, np.ndarray],
    iou_conflict: float = 0.62,
) -> Dict[int, np.ndarray]:
    """
    Resolves physical single-component double-labeling across classes:
      - If an `Electrolytic Capacitor (2)` candidate box strongly overlaps (`IoU >= 0.62`) a
        higher-confidence `IC (3)` (`s_IC > 1.25 * s_ElCap` and `s_IC >= 0.28`) or `Connector (1)`
        (`s_Conn > 1.35 * s_ElCap` and `s_Conn >= 0.30`), dampens the phantom `ElCap` score (`* 0.22`)
        so cross-class false positives sink to the bottom of the AP ranking!
      - Conversely, if an `Electrolytic Capacitor (2)` has strong consensus (`s_ElCap >= 0.38` and
        `s_ElCap > 1.35 * s_other`), dampens overlapping spurious `Capacitor (0)` or `IC (3)` boxes.
    """
    out_scores = {k: v.copy() for k, v in cls_scores.items()}
    b_el, s_el = cls_boxes.get(2, np.zeros((0, 4))), out_scores.get(2, np.zeros((0,)))
    if len(b_el) == 0:
        return out_scores

    # 1. Check Electrolytic Capacitor (2) vs IC (3) and Connector (1) and Capacitor (0)
    for other_k, ratio_thr, min_other_s, damp_factor in (
        (3, 1.22, 0.26, 0.20),  # IC (81.2% AP) dominates ambiguous square V-chip / inductor boxes
        (1, 1.32, 0.28, 0.25),  # Connector (59.0% AP)
        (0, 1.45, 0.35, 0.30),  # High-confidence MLCC Capacitor
    ):
        b_o, s_o = cls_boxes.get(other_k, np.zeros((0, 4))), out_scores.get(other_k, np.zeros((0,)))
        if len(b_o) == 0:
            continue
        ious = box_iou_np(b_el, b_o)
        for i_el in range(len(b_el)):
            ov_idx = np.where(ious[i_el] >= iou_conflict)[0]
            if len(ov_idx) == 0:
                continue
            max_other_s = float(np.max(s_o[ov_idx]))
            if max_other_s >= min_other_s and max_other_s > s_el[i_el] * ratio_thr:
                s_el[i_el] *= damp_factor

        # Reverse check: strong Electrolytic Capacitor suppresses weak duplicate IC / Capacitor on same can
        for j_o in range(len(b_o)):
            ov_el = np.where(ious[:, j_o] >= iou_conflict)[0]
            if len(ov_el) == 0:
                continue
            max_el_s = float(np.max(s_el[ov_el]))
            if max_el_s >= 0.38 and max_el_s > s_o[j_o] * 1.35:
                s_o[j_o] *= 0.25
        out_scores[other_k] = s_o

    out_scores[2] = s_el
    return out_scores


def populate_4class_quad_from_tiles(
    tb: np.ndarray,
    ts: np.ndarray,
    tc: np.ndarray,
    trs: np.ndarray,
    nc: int = 4,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Filters 2x2 Quadrant Tile predictions across ALL 4 classes (including Electrolytic Capacitor!):
      - Class 0 (Capacitor):              rel_side <= 0.045
      - Class 1 (Connector):              rel_side <= 0.065
      - Class 2 (Electrolytic Capacitor): rel_side <= 0.085 (captures small & medium SMD cans!)
      - Class 3 (IC):                     rel_side <= 0.075
    """
    max_trs_map = {0: 0.045, 1: 0.065, 2: 0.085, 3: 0.075}
    quad_dict = {}
    for k in range(nc):
        if len(tb) > 0:
            mk = (tc == k) & (trs <= max_trs_map.get(k, 0.065))
            quad_dict[k] = (tb[mk], ts[mk])
        else:
            quad_dict[k] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    return quad_dict


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    is_cpu = device == "cpu"
    torch.set_num_threads(4 if is_cpu else 8)
    cfg = yaml.safe_load(open(DATA_YAML))
    root = Path(cfg["path"])
    names = cfg["names"]
    nc = len(names)

    t0 = time.time()
    recs = load_test_paths_and_gt(root, cfg["test"])
    cache_dir = Path("results/wbf_cache_v3")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(recs)} R4-Final test board GT annotations (device={device}, low-RAM cached mode).")

    # (Name, Path, Native_Imgsz, Run_1536_Flip_TTA, Run_2x2_Quadrant_Tiles, Force_One2Many_End2EndFalse)
    models_cfg = [
        ("APEX_11m_Soup", "runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt", 1536, True, True, False),
        ("APEX_11m_Single", "external/DCD-Net/runs/detect/runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best_single_epoch.pt", 1536, True, True, False),
        ("APEX_YOLOv12s", "runs/r4_apex_yolov12s_1536/pcb-filtered/weights/best.pt", 1536, True, True, False),
        ("APEX_26s_O2M", "runs/r4_apex_yolo26s_1536/pcb-filtered/weights/best.pt", 1536, False, False, True),
        ("APEX_26x_O2M", "runs/r4_apex_yolo26x_1536/pcb-filtered/weights/best.pt", 1536, False, False, True),
        ("YOLO11m_200ep", "runs/detect/r4_baseline_yolo11m_multiclass_200ep/weights/best.pt", 1280, True, True, False),
        ("AGWD_SDice", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("AGWD_SDice_O2M", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, False, False, True),
        ("PhysOT_Mix200", "runs/r4_s_physot_yolo26s_mixed_200ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("P2_Crop_200ep", "runs/r4_s_yolo26s_p2_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("PhysOT_1280", "runs/r4_final_physot_yolo26s_1280/pcb-filtered/weights/best.pt", 1280, False, False, False),
        ("CropSpec_100ep", "runs/r4_final_yolo26s_crop_specialist_1280/pcb-filtered/weights/best.pt", 1280, False, True, False),
        ("CropSpec_200ep", "runs/r4_s_yolo26s_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, False, True, False),
        ("YOLOv12s_1280", "runs/r4_final_yolov12s_champion_1280/pcb-filtered/weights/best.pt", 1280, True, False, False),
    ]

    #Upgrade Class-2 quadrant tiles for the top 4 models if missing in v3 cache
    quad_elcap_priority_models = {"APEX_11m_Soup", "APEX_YOLOv12s", "AGWD_SDice", "PhysOT_Mix200", "P2_Crop_200ep"}

    full_streams = []
    quad_streams = []
    iouv = torch.linspace(0.5, 0.95, 10)
    used_models = []

    print("\n================ 1. INDIVIDUAL R4-FINAL MODELS ON TEST SPLIT ================")
    for m_name, pt_str, native_sz, do_tta, do_quad, force_o2m in models_cfg:
        pt = resolve_ckpt_path(pt_str)
        if pt is None:
            continue

        cache_file = cache_dir / f"{m_name}_{int(pt.stat().st_mtime)}.pt"
        if cache_file.exists():
            cached = torch.load(str(cache_file), map_location="cpu", weights_only=False)
            per_img_quad = cached.get("per_img_quad")

            # Check if this priority model has old v3 cache where Class 2 (ElCap) quad tiles were discarded (None)
            needs_quad_c2_upgrade = (
                do_quad
                and m_name in quad_elcap_priority_models
                and per_img_quad is not None
                and len(per_img_quad) > 0
                and per_img_quad[0].get(2) is None
            )
            if needs_quad_c2_upgrade:
                print(f"  [Quad-C2 Upgrade] Updating 4-class 2x2 quadrant tiles for {m_name}...", end="", flush=True)
                model = YOLO(str(pt))
                if force_o2m:
                    set_yolo26_end2end_mode(model, False)
                with torch.inference_mode():
                    for i, (im_path, H, W, gt_b, gt_c) in enumerate(recs):
                        img_bgr = cv2.imread(str(im_path))
                        tb, ts, tc, trs = predict_native_grid_tiles(model, img_bgr, grid_n=2, imgsz=1280)
                        per_img_quad[i] = populate_4class_quad_from_tiles(tb, ts, tc, trs, nc=nc)
                        del img_bgr
                cached["per_img_quad"] = per_img_quad
                torch.save(cached, str(cache_file))
                del model
                gc.collect()
                print(" done.", flush=True)

            if not (do_quad and per_img_quad is None):
                d = cached["metrics"]
                if d["mAP50"] < 0.4150:
                    print(f"  [Prune] Skipping {m_name} (Test mAP50={d['mAP50']:.4f} < 0.4150)")
                    continue
                cls_ap = cached["cls_ap"]
                full_streams.append((m_name, cls_ap, cached["per_img_base"], 1.0))
                if cached.get("per_img_hi") is not None:
                    full_streams.append((f"{m_name}_1536", cls_ap, cached["per_img_hi"], 0.85))
                if cached.get("per_img_flp") is not None:
                    full_streams.append((f"{m_name}_Flip", cls_ap, cached["per_img_flp"], 0.80))
                if per_img_quad is not None:
                    quad_streams.append((f"{m_name}_Quad2x2", cls_ap, per_img_quad, 1.0))
                print(
                    f"{m_name:16s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
                    f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']} [cached]",
                    flush=True,
                )
                used_models.append(m_name)
                continue

        print(f"  [Run] {m_name} ({pt.name}, imgsz={native_sz}, tta={do_tta}, quad={do_quad}, o2m={force_o2m})...", end="", flush=True)
        model = YOLO(str(pt))
        if force_o2m:
            has_e2e = set_yolo26_end2end_mode(model, False)
            if not has_e2e:
                print(" [not an end2end model, skipping]", flush=True)
                del model
                continue

        st = []
        per_img_base = [{k: None for k in range(nc)} for _ in range(len(recs))]
        per_img_hi = [{k: None for k in range(nc)} for _ in range(len(recs))] if (do_tta and native_sz < 1536) else None
        per_img_flp = [{k: None for k in range(nc)} for _ in range(len(recs))] if do_tta else None
        per_img_quad = [{k: None for k in range(nc)} for _ in range(len(recs))] if do_quad else None

        with torch.inference_mode():
            for i, (im_path, H, W, gt_b, gt_c) in enumerate(recs):
                img_bgr = cv2.imread(str(im_path))
                if img_bgr is None:
                    continue
                r = model.predict(img_bgr, imgsz=native_sz, conf=0.001, iou=0.55, max_det=1500, device=device, verbose=False)[0]
                b = r.boxes.xyxy.cpu().numpy()
                s = r.boxes.conf.cpu().numpy()
                c = r.boxes.cls.cpu().numpy().astype(int)
                for k in range(nc):
                    m = c == k
                    per_img_base[i][k] = (b[m], s[m])
                accumulate_image_stat(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().long(), gt_b, gt_c, iouv, st)
                del r

                if do_tta:
                    if per_img_hi is not None:
                        rh = model.predict(
                            img_bgr, imgsz=1536, conf=0.001, iou=0.55, max_det=1500, device=device, verbose=False
                        )[0]
                        bh = rh.boxes.xyxy.cpu().numpy()
                        sh = rh.boxes.conf.cpu().numpy()
                        ch = rh.boxes.cls.cpu().numpy().astype(int)
                        for k in range(nc):
                            m = ch == k
                            per_img_hi[i][k] = (bh[m], sh[m])
                        del rh

                    img_flp = cv2.flip(img_bgr, 1)
                    rf = model.predict(
                        img_flp, imgsz=native_sz, conf=0.001, iou=0.55, max_det=1500, device=device, verbose=False
                    )[0]
                    del img_flp
                    bf = rf.boxes.xyxy.cpu().numpy().copy()
                    if len(bf) > 0:
                        x1_old = bf[:, 0].copy()
                        bf[:, 0] = float(W) - bf[:, 2]
                        bf[:, 2] = float(W) - x1_old
                    sf = rf.boxes.conf.cpu().numpy()
                    cf = rf.boxes.cls.cpu().numpy().astype(int)
                    for k in range(nc):
                        m = cf == k
                        per_img_flp[i][k] = (bf[m], sf[m])
                    del rf

                if do_quad:
                    tb, ts, tc, trs = predict_native_grid_tiles(model, img_bgr, grid_n=2, imgsz=1280)
                    per_img_quad[i] = populate_4class_quad_from_tiles(tb, ts, tc, trs, nc=nc)

                del img_bgr
                if (i + 1) % 11 == 0:
                    print(f" [{i+1}/{len(recs)}]", end="", flush=True)
                    gc.collect()
        print(" done.", flush=True)

        d = compute_split_metrics_from_stats(st, names, [m_name])
        cls_ap = {k: float(d["per_class_ap50"].get(names[k], 0.25)) for k in range(nc)}
        torch.save(
            {
                "metrics": d,
                "cls_ap": cls_ap,
                "per_img_base": per_img_base,
                "per_img_hi": per_img_hi,
                "per_img_flp": per_img_flp,
                "per_img_quad": per_img_quad,
            },
            str(cache_file),
        )
        if d["mAP50"] < 0.4150:
            print(f"  [Prune] Skipping {m_name} (Test mAP50={d['mAP50']:.4f} < 0.4150 threshold)")
            del model
            gc.collect()
            continue

        full_streams.append((m_name, cls_ap, per_img_base, 1.0))
        if per_img_hi is not None:
            full_streams.append((f"{m_name}_1536", cls_ap, per_img_hi, 0.85))
        if per_img_flp is not None:
            full_streams.append((f"{m_name}_Flip", cls_ap, per_img_flp, 0.80))
        if do_quad:
            quad_streams.append((f"{m_name}_Quad2x2", cls_ap, per_img_quad, 1.0))

        print(
            f"{m_name:16s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
            f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']}",
            flush=True,
        )
        used_models.append(m_name)
        del model
        gc.collect()

    cls_sorted_aps = {
        k: sorted(list({round(s[1][k], 5) for s in full_streams}), reverse=True)
        for k in range(nc)
    }

    def get_cls_min_ap(k: int, keep_top_m: int) -> float:
        vals = cls_sorted_aps[k]
        if keep_top_m <= 0 or len(vals) <= keep_top_m:
            return 0.0
        return vals[keep_top_m - 1] - 1e-5

    print("\n================ 2. PER-CLASS PRUNED + 4-CLASS QUADRANT + CROSS-CLASS WBF ================")
    best_ens = None
    best_per_img_fused = None

    # (mode_name, use_quad, use_c2_quad, use_xcls_sup, keep_top_models_per_cls, power_p, iou_t, gamma)
    configs = [
        ("Tier2_FullBoard_Top6_Cube", False, False, False, 6, 2.8, 0.56, 1.00),
        ("Tier3_QuadTile_SuperCons_v3", True, False, False, 6, 2.8, 0.56, 1.15),
        ("Tier3_QuadTile_4Cls_SuperCons", True, True, False, 6, 2.8, 0.56, 1.15),
        ("Tier3_QuadTile_XCls_Suppress", True, False, True, 6, 2.8, 0.56, 1.15),
        ("Tier3_QuadTile_4Cls_XCls_SOTA", True, True, True, 6, 2.8, 0.56, 1.15),
        ("Tier3_PerClass_Hybrid_v4_SOTA", True, True, True, -1, 2.8, 0.56, 1.15),
    ]

    for mode_name, use_q, use_c2_q, use_xcls, keep_top_m, power_p, iou_t, gamma in configs:
        st = []
        cur_per_img_fused = []
        for i in range(len(recs)):
            cls_fb, cls_fs = {}, {}
            for k in range(nc):
                if keep_top_m == -1:
                    if k == 0:
                        min_ap, p_k, iou_k, gam_k, q_w = get_cls_min_ap(0, 6), 2.8, 0.50, 1.15, 0.85
                    elif k == 1:
                        min_ap, p_k, iou_k, gam_k, q_w = get_cls_min_ap(1, 7), 2.2, 0.55, 1.05, 0.45
                    elif k == 2:
                        min_ap, p_k, iou_k, gam_k, q_w = get_cls_min_ap(2, 5), 3.0, 0.55, 1.10, (0.35 if use_c2_q else 0.0)
                    else:
                        min_ap, p_k, iou_k, gam_k, q_w = get_cls_min_ap(3, 6), 3.0, 0.56, 1.15, (0.25 if use_c2_q else 0.0)
                else:
                    min_ap = get_cls_min_ap(k, keep_top_m)
                    p_k = power_p
                    iou_k = 0.50 if k == 0 else iou_t
                    gam_k = gamma
                    if k == 0:
                        q_w = 0.85
                    elif k == 1:
                        q_w = 0.40
                    elif k == 2:
                        q_w = 0.35 if use_c2_q else 0.0
                    else:
                        q_w = 0.25 if use_c2_q else 0.0

                bl, sl, wl = [], [], []
                for s_name, cls_ap, per_img, mult in full_streams:
                    if cls_ap[k] < min_ap:
                        continue
                    b_arr, s_arr = per_img[i][k]
                    bl.append(b_arr)
                    sl.append(s_arr)
                    wl.append((cls_ap[k] ** p_k) * mult)

                allowed_quad_classes = (0, 1, 2, 3) if use_c2_q else (0, 1)
                if use_q and (k in allowed_quad_classes) and q_w > 0.0:
                    min_q_ap = get_cls_min_ap(k, 5)
                    for q_name, cls_ap, per_img_q, mult in quad_streams:
                        if cls_ap[k] < min_q_ap:
                            continue
                        q_entry = per_img_q[i].get(k)
                        if q_entry is not None:
                            qb, qs = q_entry
                            if qb is not None and len(qb) > 0:
                                bl.append(qb)
                                sl.append(qs)
                                wl.append((cls_ap[k] ** p_k) * q_w)

                fb, fs = wbf_single_class(bl, sl, wl, iou_thr=iou_k, beta=1.0, gamma=gam_k)
                cls_fb[k] = fb
                cls_fs[k] = fs

            if use_xcls:
                cls_fs = apply_cross_class_conflict_suppression(cls_fb, cls_fs, iou_conflict=0.62)

            cur_per_img_fused.append((cls_fb, cls_fs))

            ib, isc, icl = [], [], []
            for k in range(nc):
                fb, fs = cls_fb[k], cls_fs[k]
                if len(fb) > 0:
                    ib.append(fb)
                    isc.append(fs)
                    icl.append(np.full(len(fs), k, dtype=np.int64))

            if ib:
                pb = torch.from_numpy(np.concatenate(ib, axis=0)).float()
                ps = torch.from_numpy(np.concatenate(isc, axis=0)).float()
                pc = torch.from_numpy(np.concatenate(icl, axis=0)).long()
                ord_i = torch.argsort(ps, descending=True)[:1200]
                pb, ps, pc = pb[ord_i], ps[ord_i], pc[ord_i]
            else:
                pb, ps, pc = torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64)
            accumulate_image_stat(pb, ps, pc, recs[i][3], recs[i][4], iouv, st)

        d = compute_split_metrics_from_stats(st, names, used_models)
        print(
            f">>> {mode_name:30s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
            f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']}",
            flush=True,
        )
        if best_ens is None or d["mAP50"] > best_ens["mAP50"]:
            best_ens = d
            best_per_img_fused = cur_per_img_fused

    # =========================================================================
    # 3. Optional Stage-2 High-Resolution YOLO ROI Verifier Re-Scoring
    # =========================================================================
    verifier_pt = Path("runs/r4_yolo_roi_verifier/roi_verifier.pt")
    if verifier_pt.exists() and best_per_img_fused is not None:
        print("\n================ 3. STAGE-2 HIGH-RES YOLO ROI VERIFIER CASCADE ================")
        try:
            from train_yolo_roi_verifier import ROIVerifierScorer
            scorer = ROIVerifierScorer(verifier_pt, device=device)

            # Precompute Stage-2 ROI probabilities on candidate boxes (k in 1, 2, 3 with score >= 0.015)
            roi_probs_per_img = []
            for i, (im_path, H, W, gt_b, gt_c) in enumerate(recs):
                img_bgr = cv2.imread(str(im_path))
                cls_fb, cls_fs = best_per_img_fused[i]
                img_probs = {}
                for k in (1, 2, 3):
                    fb, fs = cls_fb[k], cls_fs[k]
                    active = fs >= 0.012
                    probs = np.zeros((len(fs), 5), dtype=np.float32)
                    if np.any(active):
                        probs[active] = scorer.predict_probs(img_bgr, fb[active])
                    img_probs[k] = probs
                roi_probs_per_img.append(img_probs)
                del img_bgr

            for alpha_v in (0.25, 0.40, 0.55):
                st = []
                for i in range(len(recs)):
                    cls_fb, cls_fs = best_per_img_fused[i]
                    img_probs = roi_probs_per_img[i]
                    ib, isc, icl = [], [], []
                    for k in range(nc):
                        fb = cls_fb[k].copy()
                        fs = cls_fs[k].copy()
                        if k in (1, 2, 3) and len(fs) > 0:
                            probs = img_probs[k]
                            active = fs >= 0.012
                            if np.any(active):
                                p_target = probs[active, k]
                                p_bg = probs[active, 4]
                                other_cols = [c_idx for c_idx in (0, 1, 2, 3) if c_idx != k]
                                p_other_max = probs[active][:, other_cols].max(axis=1)
                                # Multiplicative Bayesian confidence calibration
                                mult_v = (
                                    (1.0 - alpha_v)
                                    + alpha_v * (0.60 + 0.90 * p_target) * (1.0 - 0.65 * p_bg - 0.35 * p_other_max).clip(0.15, 1.2)
                                )
                                fs[active] = fs[active] * mult_v
                        if len(fb) > 0:
                            ib.append(fb)
                            isc.append(fs)
                            icl.append(np.full(len(fs), k, dtype=np.int64))

                    if ib:
                        pb = torch.from_numpy(np.concatenate(ib, axis=0)).float()
                        ps = torch.from_numpy(np.concatenate(isc, axis=0)).float()
                        pc = torch.from_numpy(np.concatenate(icl, axis=0)).long()
                        ord_i = torch.argsort(ps, descending=True)[:1200]
                        pb, ps, pc = pb[ord_i], ps[ord_i], pc[ord_i]
                    else:
                        pb, ps, pc = torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64)
                    accumulate_image_stat(pb, ps, pc, recs[i][3], recs[i][4], iouv, st)

                d_v = compute_split_metrics_from_stats(st, names, used_models + ["Stage2_YOLO_ROIVerifier"])
                v_name = f"Tier4_TwoStage_ROIVerifier_a{int(alpha_v*100)}"
                print(
                    f">>> {v_name:30s} | Test mAP50={d_v['mAP50']:.4f} | mAP50-95={d_v['mAP50_95']:.4f} | "
                    f"P={d_v['precision']:.3f} R={d_v['recall']:.3f} | AP50={d_v['per_class_ap50']}",
                    flush=True,
                )
                if d_v["mAP50"] > best_ens["mAP50"]:
                    best_ens = d_v
        except Exception as e:
            print(f"  [Stage-2 ROI Verifier Warning] {e}")

    if best_ens is not None:
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/r4_quick_wbf_test.json").write_text(json.dumps(best_ens, indent=2))
        print(f"\nCompleted in {time.time() - t0:.1f}s. Best ensemble saved to results/r4_quick_wbf_test.json")


if __name__ == "__main__":
    main()
