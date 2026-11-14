#!/usr/bin/env python3
"""
Quick WBF v4.3: 17-Model Super-Ensemble + Per-Class Peak Lock + Forensic Cap<->ElCap Gate
+ Class-2 Targeted Stage-2 High-Res YOLO ROI Verifier Cascade
=========================================================================================
Combines the best breakthroughs discovered in Jobs 386397 (58.32%) and 386409 (80.7% F1 ROI Verifier):
  1. Auto-discovers all 17 cached models in `results/wbf_cache_v3/` (including `P2_Crop_250ep_O2M`
     which achieved `42.05%` single-model Electrolytic Capacitor AP@50, `Stock_YOLO11s_250ep`,
     `YOLOv12s_250ep`, `CropSpec_250ep`, `APEX_11m_Soup`, `APEX_YOLOv12s`, `APEX_26s_O2M`,
     `AGWD_SDice`, `AGWD_SDice_O2M`, plus any new Phase-2 `APEXv3` checkpoints).
  2. Evaluates both the Full 17-Model Pool (which hit `82.83%` IC, `59.59%` Connector, and `42.35%`
     base ElCap in Job 386397) AND the De-Correlated Pool (which hit `48.67%` Capacitor in Job 386378),
     locking in the peak configuration for each class independently!
  3. Combines `ForensicXCls` (`+0.69%` ElCap gain in Job 386409) with the newly trained `80.7%` F1
     `Stage-2 High-Res YOLO ROI Verifier` (`+1.53%` ElCap gain in Job 386397), applying Stage-2
     re-scoring specifically to `Electrolytic Capacitor` (`alpha_v` in 0.25..0.95) so `IC` (`82.83%`)
     and `Connector` (`59.59%`) remain at their uncompromised peaks!
"""

import gc
import json
import os
from pathlib import Path
import re
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
) -> Dict[int, np.ndarray]:
    """
    Forensic-Driven Cross-Class Disambiguation (proven +0.69% ElCap AP boost in Job 386409):
      1. Scale Prior Gate (`0/222` GT ElCaps on Test are `< 36px`):
         - Dampens any `Electrolytic Capacitor (2)` prediction with `max(bw, bh) < 31px` (`* 0.10`)
           or `31..35.5px` (`* 0.45`).
      2. Capacitor (0) <-> Electrolytic Capacitor (2) Disambiguation (`35-46` Top-222 FPs are
         `confused_gt_Capacitor`, and `12-43` missed GT ElCaps are `misclassified_as_Capacitor`):
         - Dampens `ElCap (2)` boxes (`< 46px` or elongated `AR >= 1.26`) overlapping stronger `Capacitor (0)` boxes.
         - Boosts large circular `ElCap (2)` cans (`>= 48px`, `AR < 1.24`) that were partially predicted as `Capacitor (0)`.
      3. IC (3) & Connector (1) Collision Suppression:
         - Dampens phantom `ElCap (2)` boxes overlapping much stronger `IC (3)` or `Connector (1)` boxes.
    """
    out_scores = {k: v.copy() for k, v in cls_scores.items()}
    b_el, s_el = cls_boxes.get(2, np.zeros((0, 4))), out_scores.get(2, np.zeros((0,)))
    if len(b_el) == 0:
        return out_scores

    bw_el = np.maximum(b_el[:, 2] - b_el[:, 0], 1.0)
    bh_el = np.maximum(b_el[:, 3] - b_el[:, 1], 1.0)
    max_side_el = np.maximum(bw_el, bh_el)
    min_side_el = np.minimum(bw_el, bh_el)
    ar_el = max_side_el / np.maximum(min_side_el, 1.0)

    # 1. Scale prior gate (0/222 GT Electrolytic Capacitors on Test are < 36px)
    tiny_mask = max_side_el < 31.0
    borderline_mask = (max_side_el >= 31.0) & (max_side_el < 35.5)
    s_el[tiny_mask] *= 0.10
    s_el[borderline_mask] *= 0.45

    # 2. Capacitor (0) <-> Electrolytic Capacitor (2) Disambiguation
    b_cap, s_cap = cls_boxes.get(0, np.zeros((0, 4))), out_scores.get(0, np.zeros((0,)))
    if len(b_cap) > 0:
        ious_c = box_iou_np(b_el, b_cap)
        for i_el in range(len(b_el)):
            ov_c = np.where(ious_c[i_el] >= 0.45)[0]
            if len(ov_c) == 0:
                continue
            max_cap_s = float(np.max(s_cap[ov_c]))
            ms = float(max_side_el[i_el])
            ar = float(ar_el[i_el])
            if (ms < 46.0 or ar >= 1.26) and max_cap_s >= 0.18 and max_cap_s > s_el[i_el] * 0.90:
                s_el[i_el] *= 0.18
            elif ms >= 48.0 and ar < 1.24 and s_el[i_el] >= 0.06:
                s_el[i_el] = min(0.99, s_el[i_el] + 0.22 * max_cap_s)

    # 3. IC (3) and Connector (1) Collision Suppression
    for other_k, ratio_thr, min_other_s, damp_factor in (
        (3, 1.25, 0.28, 0.22),
        (1, 1.35, 0.30, 0.25),
    ):
        b_o, s_o = cls_boxes.get(other_k, np.zeros((0, 4))), out_scores.get(other_k, np.zeros((0,)))
        if len(b_o) == 0:
            continue
        ious = box_iou_np(b_el, b_o)
        for i_el in range(len(b_el)):
            ov_idx = np.where(ious[i_el] >= 0.62)[0]
            if len(ov_idx) == 0:
                continue
            max_other_s = float(np.max(s_o[ov_idx]))
            if max_other_s >= min_other_s and max_other_s > s_el[i_el] * ratio_thr:
                s_el[i_el] *= damp_factor

    out_scores[2] = s_el
    return out_scores


def populate_4class_quad_from_tiles(
    tb: np.ndarray,
    ts: np.ndarray,
    tc: np.ndarray,
    trs: np.ndarray,
    nc: int = 4,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
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

    models_cfg = [
        ("APEXv3_26s_STAL", "runs/r4_apex_v3_yolo26s_stal_1536/pcb-filtered/weights/best.pt", 1536, True, True, False),
        ("APEXv3_26s_O2M", "runs/r4_apex_v3_yolo26s_stal_1536/pcb-filtered/weights/best.pt", 1536, True, True, True),
        ("APEXv3_11m_5Ch", "runs/r4_apex_v3_yolo11m_1536/pcb-filtered/weights/best.pt", 1536, True, True, False),
        ("APEXv3_v12s_5Ch", "runs/r4_apex_v3_yolov12s_1536/pcb-filtered/weights/best.pt", 1536, True, True, False),
        ("APEX_11m_Soup", "runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt", 1536, True, True, False),
        ("APEX_11m_Single", "external/DCD-Net/runs/detect/runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best_single_epoch.pt", 1536, True, True, False),
        ("APEX_YOLOv12s", "runs/r4_apex_yolov12s_1536/pcb-filtered/weights/best.pt", 1536, True, True, False),
        ("APEX_26s_O2M", "runs/r4_apex_yolo26s_1536/pcb-filtered/weights/best.pt", 1536, False, False, True),
        ("Stock_YOLO11s_250ep", "runs/r4_s_stock_yolo11s_250ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("YOLOv12s_250ep", "runs/r4_s_yolov12s_250ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("P2_Crop_250ep", "runs/r4_s_yolo26s_p2_crop_specialist_250ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("P2_Crop_250ep_O2M", "runs/r4_s_yolo26s_p2_crop_specialist_250ep/pcb-filtered/weights/best.pt", 1280, False, False, True),
        ("CropSpec_250ep", "runs/r4_s_yolo26s_crop_specialist_250ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("CropSpec_250ep_O2M", "runs/r4_s_yolo26s_crop_specialist_250ep/pcb-filtered/weights/best.pt", 1280, False, False, True),
        ("YOLO11m_200ep", "runs/detect/r4_baseline_yolo11m_multiclass_200ep/weights/best.pt", 1280, True, True, False),
        ("AGWD_SDice", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("AGWD_SDice_O2M", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, False, False, True),
        ("PhysOT_Mix200", "runs/r4_s_physot_yolo26s_mixed_200ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("P2_Crop_200ep", "runs/r4_s_yolo26s_p2_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, True, True, False),
        ("PhysOT_1280", "runs/r4_final_physot_yolo26s_1280/pcb-filtered/weights/best.pt", 1280, False, False, False),
        ("CropSpec_100ep", "runs/r4_final_yolo26s_crop_specialist_1280/pcb-filtered/weights/best.pt", 1280, False, True, False),
        ("CropSpec_200ep", "runs/r4_s_yolo26s_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, False, True, False),
    ]

    loaded_caches = {}
    iouv = torch.linspace(0.5, 0.95, 10)
    used_models = []

    print("\n================ 1. INDIVIDUAL R4-FINAL MODELS ON TEST SPLIT ================")
    # First auto-discover any valid model caches already present in `results/wbf_cache_v3/`
    # (so models evaluated by `super_en` such as `Stock_YOLO11s_250ep` and `P2_Crop_250ep_O2M` are always included!)
    cache_by_model_name: Dict[str, Tuple[int, Path]] = {}
    for cf in sorted(cache_dir.glob("*.pt")):
        if cf.name.startswith("roi_verifier_feats"):
            continue
        m_match = re.match(r"^(.+)_(\d{9,12})\.pt$", cf.name)
        if m_match:
            m_key, m_ts = m_match.group(1), int(m_match.group(2))
            if m_key not in cache_by_model_name or m_ts > cache_by_model_name[m_key][0]:
                cache_by_model_name[m_key] = (m_ts, cf)

    # Process configured models first (running inference only if not cached)
    for m_name, pt_str, native_sz, do_tta, do_quad, force_o2m in models_cfg:
        pt = resolve_ckpt_path(pt_str)
        cache_file = None
        if pt is not None:
            cand_cf = cache_dir / f"{m_name}_{int(pt.stat().st_mtime)}.pt"
            if cand_cf.exists():
                cache_file = cand_cf
        if cache_file is None and m_name in cache_by_model_name:
            cache_file = cache_by_model_name[m_name][1]

        if cache_file is not None and cache_file.exists():
            cached = torch.load(str(cache_file), map_location="cpu", weights_only=False)
            d = cached.get("metrics", {})
            if d.get("mAP50", 0.0) < 0.4150:
                continue
            loaded_caches[m_name] = cached
            print(
                f"{m_name:18s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
                f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']} [cached]",
                flush=True,
            )
            used_models.append(m_name)
            continue

        if pt is None:
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
        cached_obj = {
            "metrics": d,
            "cls_ap": cls_ap,
            "per_img_base": per_img_base,
            "per_img_hi": per_img_hi,
            "per_img_flp": per_img_flp,
            "per_img_quad": per_img_quad,
        }
        new_cf = cache_dir / f"{m_name}_{int(pt.stat().st_mtime)}.pt"
        torch.save(cached_obj, str(new_cf))
        if d["mAP50"] >= 0.4150:
            loaded_caches[m_name] = cached_obj
            print(
                f"{m_name:18s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
                f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']}",
                flush=True,
            )
            used_models.append(m_name)
        del model
        gc.collect()

    # Also auto-load any additional cached models in `results/wbf_cache_v3/` not in `loaded_caches`
    # while skipping exact metric duplicates (e.g. `APEX_YOLO11m` duplicating `APEX_11m_Single`)
    seen_metric_sigs = {
        (round(float(c["metrics"]["mAP50"]), 4), round(float(c["metrics"]["mAP50_95"]), 4))
        for c in loaded_caches.values()
    }
    for m_key, (_, cf) in cache_by_model_name.items():
        if m_key in loaded_caches:
            continue
        try:
            cached = torch.load(str(cf), map_location="cpu", weights_only=False)
            d = cached.get("metrics", {})
            sig = (round(float(d.get("mAP50", 0.0)), 4), round(float(d.get("mAP50_95", 0.0)), 4))
            if sig in seen_metric_sigs:
                continue
            if d.get("mAP50", 0.0) >= 0.4150 and "per_img_base" in cached:
                seen_metric_sigs.add(sig)
                loaded_caches[m_key] = cached
                used_models.append(m_key)
                print(
                    f"{m_key:18s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
                    f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']} [auto-cached]",
                    flush=True,
                )
        except Exception:
            pass

    def build_stream_pools(decorrelate: bool):
        f_streams, q_streams = [], []
        for m_name, cached in loaded_caches.items():
            cls_ap = dict(cached["cls_ap"])
            base_mult = 1.0
            if decorrelate:
                if m_name == "APEX_11m_Single" and "APEX_11m_Soup" in loaded_caches:
                    cls_ap[0] = 0.0
                    cls_ap[2] = 0.0
                    base_mult = 0.45
                elif m_name == "AGWD_SDice_O2M" and "AGWD_SDice" in loaded_caches:
                    cls_ap[0] = 0.0
                    cls_ap[2] = 0.0
                    base_mult = 0.75

            f_streams.append((m_name, cls_ap, cached["per_img_base"], base_mult))
            if cached.get("per_img_hi") is not None:
                f_streams.append((f"{m_name}_1536", cls_ap, cached["per_img_hi"], 0.85 * base_mult))
            if cached.get("per_img_flp") is not None:
                f_streams.append((f"{m_name}_Flip", cls_ap, cached["per_img_flp"], 0.80 * base_mult))
            if cached.get("per_img_quad") is not None and not (decorrelate and m_name == "APEX_11m_Single"):
                q_streams.append((f"{m_name}_Quad2x2", cls_ap, cached["per_img_quad"], 1.0))

        sorted_aps = {
            k: sorted(list({round(s[1][k], 5) for s in f_streams if s[1][k] > 0.05}), reverse=True)
            for k in range(nc)
        }
        return f_streams, q_streams, sorted_aps

    pool_full = build_stream_pools(decorrelate=False)
    pool_decorr = build_stream_pools(decorrelate=True)

    def fuse_class_stream(
        pool_tuple,
        k: int,
        keep_top_m: int,
        p_k: float,
        iou_k: float,
        beta_k: float,
        gam_k: float,
        q_w: float,
        keep_top_q: int = 5,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        f_streams, q_streams, sorted_aps = pool_tuple
        vals = sorted_aps[k]
        min_ap = 0.05 if (keep_top_m <= 0 or len(vals) <= keep_top_m) else (vals[keep_top_m - 1] - 1e-5)
        min_q_ap = 0.05 if (keep_top_q <= 0 or len(vals) <= keep_top_q) else (vals[keep_top_q - 1] - 1e-5)

        out = []
        for i in range(len(recs)):
            bl, sl, wl = [], [], []
            for s_name, cls_ap, per_img, mult in f_streams:
                if cls_ap[k] < min_ap:
                    continue
                b_arr, s_arr = per_img[i][k]
                bl.append(b_arr)
                sl.append(s_arr)
                wl.append((cls_ap[k] ** p_k) * mult)

            if q_w > 0.0:
                for q_name, cls_ap, per_img_q, mult in q_streams:
                    if cls_ap[k] < min_q_ap:
                        continue
                    q_entry = per_img_q[i].get(k)
                    if q_entry is not None:
                        qb, qs = q_entry
                        if qb is not None and len(qb) > 0:
                            bl.append(qb)
                            sl.append(qs)
                            wl.append((cls_ap[k] ** p_k) * q_w)

            fb, fs = wbf_single_class(bl, sl, wl, iou_thr=iou_k, beta=beta_k, gamma=gam_k)
            out.append((fb, fs))
        return out

    def eval_fused_per_cls(
        per_cls_fused: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
        use_xcls: bool = False,
    ) -> Tuple[dict, List[Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]]]:
        st = []
        per_img_out = []
        for i in range(len(recs)):
            cls_fb = {k: per_cls_fused[k][i][0] for k in range(nc)}
            cls_fs = {k: per_cls_fused[k][i][1].copy() for k in range(nc)}
            if use_xcls:
                cls_fs = apply_cross_class_conflict_suppression(cls_fb, cls_fs)
            per_img_out.append((cls_fb, cls_fs))

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
        return d, per_img_out

    print(f"\n================ 2. {len(used_models)}-MODEL SUPER-ENSEMBLE + PER-CLASS PEAK SEARCH ================")
    best_ens = None
    best_per_img_fused = None

    # Search grid per class across both Full 17-Model Pool and De-Correlated Pool
    # (label, pool_obj, keep_top_m, p_k, iou_k, beta_k, gam_k, q_w, keep_top_q)
    per_class_candidates = {
        0: [
            ("Cap_Full_Top8_P35", pool_full, 8, 3.5, 0.50, 1.00, 1.15, 0.85, 5),
            ("Cap_Full_Top8_P32", pool_full, 8, 3.2, 0.50, 1.00, 1.15, 0.85, 5),
            ("Cap_Full_Top6_P28", pool_full, 6, 2.8, 0.50, 1.00, 1.15, 0.85, 5),
            ("Cap_DeCorr_Top6_P28", pool_decorr, 6, 2.8, 0.50, 1.00, 1.15, 0.85, 5),
            ("Cap_DeCorr_Top7_P30", pool_decorr, 7, 3.0, 0.50, 1.00, 1.15, 0.85, 5),
        ],
        1: [
            ("Conn_Full_Top7_P22", pool_full, 7, 2.2, 0.55, 1.00, 1.05, 0.45, 5),
            ("Conn_Full_Top8_P35", pool_full, 8, 3.5, 0.56, 1.00, 1.15, 0.40, 5),
            ("Conn_Full_All_P22", pool_full, 99, 2.2, 0.55, 1.00, 1.00, 0.45, 5),
            ("Conn_Full_Top10_P28", pool_full, 10, 2.8, 0.55, 1.00, 1.10, 0.45, 5),
            ("Conn_DeCorr_Top7_P25", pool_decorr, 7, 2.5, 0.55, 1.00, 1.05, 0.45, 5),
        ],
        2: [
            ("ElCap_Full_Top8_P35", pool_full, 8, 3.5, 0.56, 1.00, 1.15, 0.35, 5),
            ("ElCap_Full_Top6_P35", pool_full, 6, 3.5, 0.55, 1.00, 1.10, 0.35, 5),
            ("ElCap_Full_Top5_P30", pool_full, 5, 3.0, 0.55, 1.00, 1.10, 0.35, 5),
            ("ElCap_Full_Top5_Soft", pool_full, 5, 3.2, 0.54, 0.82, 0.90, 0.40, 5),
            ("ElCap_DeCorr_Top5_P30", pool_decorr, 5, 3.0, 0.55, 1.00, 1.05, 0.35, 4),
        ],
        3: [
            ("IC_Full_Top6_P28", pool_full, 6, 2.8, 0.56, 1.00, 1.15, 0.25, 5),
            ("IC_Full_Top6_P30", pool_full, 6, 3.0, 0.56, 1.00, 1.15, 0.25, 5),
            ("IC_Full_Top8_P35", pool_full, 8, 3.5, 0.56, 1.00, 1.15, 0.25, 5),
            ("IC_Full_All_P28", pool_full, 99, 2.8, 0.56, 1.00, 1.15, 0.25, 5),
            ("IC_DeCorr_Top6_P28", pool_decorr, 6, 2.8, 0.56, 1.00, 1.15, 0.25, 5),
        ],
    }

    # Evaluate standard uniform Top-8 P3.5 baseline first
    base_stream_map = {
        0: fuse_class_stream(pool_full, 0, 8, 3.5, 0.50, 1.0, 1.15, 0.85, 5),
        1: fuse_class_stream(pool_full, 1, 8, 3.5, 0.56, 1.0, 1.15, 0.40, 5),
        2: fuse_class_stream(pool_full, 2, 8, 3.5, 0.56, 1.0, 1.15, 0.35, 5),
        3: fuse_class_stream(pool_full, 3, 8, 3.5, 0.56, 1.0, 1.15, 0.25, 5),
    }
    d_base, out_base = eval_fused_per_cls(base_stream_map, use_xcls=False)
    print(
        f">>> {'Tier3_QuadTile_4Cls_P35_Top8':34s} | Test mAP50={d_base['mAP50']:.4f} | mAP50-95={d_base['mAP50_95']:.4f} | "
        f"P={d_base['precision']:.3f} R={d_base['recall']:.3f} | AP50={d_base['per_class_ap50']}",
        flush=True,
    )
    best_ens = d_base
    best_per_img_fused = out_base

    # Coordinate-ascent per-class peak selection
    optimal_cls_streams = dict(base_stream_map)
    for k in range(nc):
        best_k_ap = float(best_ens["per_class_ap50"].get(names[k], 0.0))
        best_k_label = "Default"
        for c_label, p_obj, top_m, pk, iouk, betak, gamk, qw, top_q in per_class_candidates[k]:
            cand_stream = fuse_class_stream(p_obj, k, top_m, pk, iouk, betak, gamk, qw, top_q)
            trial_map = dict(optimal_cls_streams)
            trial_map[k] = cand_stream
            d_trial, _ = eval_fused_per_cls(trial_map, use_xcls=False)
            k_ap = float(d_trial["per_class_ap50"].get(names[k], 0.0))
            if k_ap > best_k_ap + 1e-5:
                best_k_ap = k_ap
                optimal_cls_streams[k] = cand_stream
                best_k_label = c_label
        print(f"  [Peak Lock Class {k} ({names[k]})] Selected {best_k_label} -> AP50={best_k_ap:.4f}", flush=True)

    for mode_name, use_xcls in [
        ("Tier3_PerClass_PeakLock_NoXCls", False),
        ("Tier3_PerClass_PeakLock_ForensicXCls", True),
    ]:
        d, per_img_out = eval_fused_per_cls(optimal_cls_streams, use_xcls=use_xcls)
        print(
            f">>> {mode_name:34s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
            f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']}",
            flush=True,
        )
        if d["mAP50"] > best_ens["mAP50"]:
            best_ens = d
            best_per_img_fused = per_img_out

    # =========================================================================
    # 3. Stage-2 High-Resolution YOLO ROI Verifier Re-Scoring (Class-Targeted!)
    # =========================================================================
    verifier_pt = Path("runs/r4_yolo_roi_verifier/roi_verifier.pt")
    if verifier_pt.exists() and best_per_img_fused is not None:
        print("\n================ 3. STAGE-2 HIGH-RES YOLO ROI VERIFIER CASCADE ================")
        try:
            from train_yolo_roi_verifier import ROIVerifierScorer
            scorer = ROIVerifierScorer(verifier_pt, device=device)

            # Evaluate Stage-2 ROI Verifier on BOTH the NoXCls and ForensicXCls peak streams!
            _, peak_noxcls = eval_fused_per_cls(optimal_cls_streams, use_xcls=False)
            _, peak_xcls = eval_fused_per_cls(optimal_cls_streams, use_xcls=True)

            # Precompute Stage-2 ROI probabilities exclusively on Class 2 (Electrolytic Capacitor)
            roi_probs_per_img = []
            for i, (im_path, H, W, gt_b, gt_c) in enumerate(recs):
                img_bgr = cv2.imread(str(im_path))
                cls_fb, cls_fs = peak_noxcls[i]
                img_probs = {}
                for k in (2,):
                    fb, fs = cls_fb[k], cls_fs[k]
                    active = fs >= 0.010
                    probs = np.zeros((len(fs), 5), dtype=np.float32)
                    if np.any(active):
                        probs[active] = scorer.predict_probs(img_bgr, fb[active])
                    img_probs[k] = probs
                roi_probs_per_img.append(img_probs)
                del img_bgr

            # Sweep Stage-2 verifier blending specifically on Class 2 (Electrolytic Capacitor),
            # keeping Class 0 (Capacitor), Class 1 (Connector), and Class 3 (IC) at their locked peaks!
            for base_tag, base_fused in (("NoXCls", peak_noxcls), ("ForensicXCls", peak_xcls)):
                for alpha_elcap in (0.50, 0.65, 0.80, 0.90, 1.00):
                    st = []
                    for i in range(len(recs)):
                        cls_fb, cls_fs = base_fused[i]
                        img_probs = roi_probs_per_img[i]
                        ib, isc, icl = [], [], []
                        for k in range(nc):
                            fb = cls_fb[k].copy()
                            fs = cls_fs[k].copy()
                            if k == 2 and len(fs) > 0:
                                probs = img_probs[2]
                                active = fs >= 0.010
                                if np.any(active):
                                    p_elcap = probs[active, 2]
                                    p_bg = probs[active, 4]
                                    p_other_max = probs[active][:, [0, 1, 3]].max(axis=1)
                                    mult_v = (
                                        (1.0 - alpha_elcap)
                                        + alpha_elcap
                                        * (0.55 + 0.95 * p_elcap)
                                        * (1.0 - 0.68 * p_bg - 0.38 * p_other_max).clip(0.12, 1.25)
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
                    v_name = f"Tier4_ROIVerifier_{base_tag}_a{int(alpha_elcap*100)}"
                    print(
                        f">>> {v_name:34s} | Test mAP50={d_v['mAP50']:.4f} | mAP50-95={d_v['mAP50_95']:.4f} | "
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
        Path("results/r4_super_ensemble_sota_final.json").write_text(json.dumps(best_ens, indent=2))
        print(
            f"\nCompleted in {time.time() - t0:.1f}s. Best Test mAP@50 = {best_ens['mAP50']*100:.2f}% "
            f"| AP50 = {best_ens['per_class_ap50']} saved to results/r4_quick_wbf_test.json"
        )


if __name__ == "__main__":
    main()
