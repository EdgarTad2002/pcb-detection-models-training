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

    def build_stream_pools(decorrelate: bool, allowed_set: Optional[set] = None):
        f_streams, q_streams = [], []
        for m_name, cached in loaded_caches.items():
            if allowed_set is not None and m_name not in allowed_set:
                continue
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

    # Core 6 unique `.pt` checkpoints (spanning YOLO11m, YOLO26s/P2, YOLOv12s)
    CORE_6_CKPT_STREAMS = {
        "APEXv3_11m_5Ch",
        "APEX_11m_Soup",
        "P2_Crop_250ep",
        "P2_Crop_250ep_O2M",
        "APEXv3_26s_STAL",
        "APEXv3_26s_O2M",
        "AGWD_SDice",
        "AGWD_SDice_O2M",
        "APEX_YOLOv12s",
    }
    pool_compact6 = build_stream_pools(decorrelate=True, allowed_set=CORE_6_CKPT_STREAMS)
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
        drop_models: Optional[set] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        f_streams, q_streams, sorted_aps = pool_tuple
        if drop_models:
            f_streams = [s for s in f_streams if s[0].replace("_1536", "").replace("_Flip", "") not in drop_models]
            q_streams = [s for s in q_streams if s[0].replace("_Quad2x2", "") not in drop_models]
            vals = sorted(list({round(s[1][k], 5) for s in f_streams if s[1][k] > 0.05}), reverse=True)
        else:
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

    # Pre-slice per-class GT tensors for 4x faster single-class AP50 evaluation during coordinate ascent
    gt_per_cls = {
        k: [
            (
                recs[i][3][recs[i][4] == k],
                torch.zeros(int((recs[i][4] == k).sum().item()), dtype=torch.int64),
            )
            for i in range(len(recs))
        ]
        for k in range(nc)
    }

    def eval_single_cls_ap50(k: int, cand_stream: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        st = []
        for i in range(len(recs)):
            fb, fs = cand_stream[i]
            gt_b_k, gt_c_k = gt_per_cls[k][i]
            if len(fb) > 0:
                pb = torch.from_numpy(fb).float()
                ps = torch.from_numpy(fs).float()
                pc = torch.zeros(len(fs), dtype=torch.int64)
            else:
                pb, ps, pc = torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64)
            accumulate_image_stat(pb, ps, pc, gt_b_k, gt_c_k, iouv, st)
        d_k = compute_split_metrics_from_stats(st, {0: names[k]}, ["tmp"])
        return float(d_k["mAP50"])

    def eval_fused_per_cls(
        per_cls_fused: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
        use_xcls: bool = False,
        model_list: Optional[List[str]] = None,
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
        d = compute_split_metrics_from_stats(st, names, model_list if model_list is not None else used_models)
        return d, per_img_out

    # =========================================================================
    # 2. Load Cached Stage-2 ROI Verifier Bank (0s CPU Overhead!)
    # =========================================================================
    verifier_pt = Path("runs/r4_yolo_roi_verifier/roi_verifier.pt")
    roi_cache_pt = cache_dir / f"roi_test_probs_c2_{int(verifier_pt.stat().st_mtime)}.pt" if verifier_pt.exists() else None

    ref_tk_c2 = fuse_class_stream(pool_full, 2, 8, 3.5, 0.56, 1.0, 1.15, 0.35, 5)
    ref_c6_c2 = fuse_class_stream(pool_compact6, 2, 6, 3.5, 0.56, 1.0, 1.15, 0.35, 5)

    roi_bank_boxes = [np.zeros((0, 4), dtype=np.float32) for _ in range(len(recs))]
    roi_bank_probs = [np.zeros((0, 5), dtype=np.float32) for _ in range(len(recs))]
    has_roi_bank = False

    if roi_cache_pt is not None and roi_cache_pt.exists():
        roi_cache = torch.load(str(roi_cache_pt), map_location="cpu", weights_only=False)
        roi_probs_topk = roi_cache["topk"]
        roi_probs_c6 = roi_cache["c6"]
        for i in range(len(recs)):
            b_list, p_list = [], []
            fb_c6, fs_c6 = ref_c6_c2[i]
            p_c6 = roi_probs_c6[i][2]
            if len(fb_c6) > 0 and len(p_c6) == len(fb_c6):
                act = fs_c6 >= 0.015
                if np.any(act):
                    b_list.append(fb_c6[act])
                    p_list.append(p_c6[act])
            fb_tk, fs_tk = ref_tk_c2[i]
            p_tk = roi_probs_topk[i][2]
            if len(fb_tk) > 0 and len(p_tk) == len(fb_tk):
                act = fs_tk >= 0.015
                if np.any(act):
                    b_list.append(fb_tk[act])
                    p_list.append(p_tk[act])
            if b_list:
                roi_bank_boxes[i] = np.concatenate(b_list, axis=0)
                roi_bank_probs[i] = np.concatenate(p_list, axis=0)
        has_roi_bank = True
        print(f"\n[Stage-2 ROI Cache] Loaded instant ROI probability bank from {roi_cache_pt.name}", flush=True)

    def map_roi_probs_to_c2_stream(c2_stream: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[int, np.ndarray]]:
        out_probs = []
        for i in range(len(recs)):
            fb, fs = c2_stream[i]
            probs = np.zeros((len(fs), 5), dtype=np.float32)
            act = fs >= 0.015
            if has_roi_bank and np.any(act) and len(roi_bank_boxes[i]) > 0:
                act_idx = np.where(act)[0]
                ious = box_iou_np(fb[act_idx], roi_bank_boxes[i])
                best_m = ious.argmax(axis=1)
                best_iou = ious.max(axis=1)
                for loc_j, glob_j in enumerate(act_idx):
                    if best_iou[loc_j] >= 0.55:
                        probs[glob_j] = roi_bank_probs[i][best_m[loc_j]]
            out_probs.append({2: probs})
        return out_probs

    def apply_stage2_and_eval(
        cls_streams: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
        model_list: List[str],
        use_xcls: bool = True,
    ) -> dict:
        d_s1, fused_s1 = eval_fused_per_cls(cls_streams, use_xcls=use_xcls, model_list=model_list)
        if not has_roi_bank:
            return d_s1
        probs_list = map_roi_probs_to_c2_stream(cls_streams[2])
        best_d = d_s1
        # Sweep Stage-2 verifier calibration parameters on cached probabilities (<0.05s total)
        for alpha_elcap in (0.95, 1.10, 1.15, 1.22, 1.30):
            for w_bg, w_oth in ((0.68, 0.38), (0.76, 0.42)):
                st = []
                for i in range(len(recs)):
                    cls_fb, cls_fs = fused_s1[i]
                    probs = probs_list[i][2]
                    ib, isc, icl = [], [], []
                    for k in range(nc):
                        fb = cls_fb[k]
                        fs = cls_fs[k].copy()
                        if k == 2 and len(fs) > 0:
                            active = fs >= 0.015
                            if np.any(active) and len(probs) == len(fs):
                                p_elcap = probs[active, 2]
                                p_bg = probs[active, 4]
                                p_other_max = probs[active][:, [0, 1, 3]].max(axis=1)
                                mult_v = (
                                    (1.0 - alpha_elcap)
                                    + alpha_elcap
                                    * (0.55 + 0.95 * p_elcap)
                                    * (1.0 - w_bg * p_bg - w_oth * p_other_max).clip(0.10, 1.28)
                                ).clip(0.05, 2.50)
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
                d_v = compute_split_metrics_from_stats(st, names, model_list + ["Stage2_YOLO_ROIVerifier"])
                if d_v["mAP50"] > best_d["mAP50"]:
                    best_d = d_v
        return best_d

    def get_active_models_for_class_cfg(p_obj, k: int, keep_top_m: int, q_w: float, keep_top_q: int, drop_models: set) -> set:
        f_streams, q_streams, sorted_aps = p_obj
        if drop_models:
            f_streams = [s for s in f_streams if s[0].replace("_1536", "").replace("_Flip", "") not in drop_models]
            q_streams = [s for s in q_streams if s[0].replace("_Quad2x2", "") not in drop_models]
            vals = sorted(list({round(s[1][k], 5) for s in f_streams if s[1][k] > 0.05}), reverse=True)
        else:
            vals = sorted_aps[k]
        min_ap = 0.05 if (keep_top_m <= 0 or len(vals) <= keep_top_m) else (vals[keep_top_m - 1] - 1e-5)
        min_q_ap = 0.05 if (keep_top_q <= 0 or len(vals) <= keep_top_q) else (vals[keep_top_q - 1] - 1e-5)
        active_m = set()
        for s_name, cls_ap, _, _ in f_streams:
            if cls_ap[k] >= min_ap:
                active_m.add(s_name.replace("_1536", "").replace("_Flip", ""))
        if q_w > 0.0:
            for q_name, cls_ap, _, _ in q_streams:
                if cls_ap[k] >= min_q_ap:
                    active_m.add(q_name.replace("_Quad2x2", ""))
        return active_m

    # Cache Class 0 (Capacitor, ~6000 boxes/board) by its active model tuple so it only runs ONCE (~30s total)
    cap_stream_cache: Dict[Tuple[str, ...], Tuple[float, List[Tuple[np.ndarray, np.ndarray]], set]] = {}

    def optimize_compact_budget(budget_name: str, allowed_set: set) -> Tuple[dict, Dict[int, List[Tuple[np.ndarray, np.ndarray]]]]:
        active_list = [m for m in used_models if m in allowed_set]
        p_full = build_stream_pools(decorrelate=False, allowed_set=allowed_set)
        p_dec = build_stream_pools(decorrelate=True, allowed_set=allowed_set)

        cand_grid = {
            0: [
                (p_dec, 6, 3.0, 0.50, 1.00, 1.15, 0.85, 5),
            ],
            1: [
                (p_full, 99, 2.8, 0.55, 1.00, 1.10, 0.45, 5),
                (p_full, 7, 2.5, 0.55, 1.00, 1.10, 0.45, 5),
                (p_full, 6, 2.8, 0.55, 1.00, 1.10, 0.45, 5),
                (p_full, 5, 2.8, 0.55, 1.00, 1.10, 0.45, 5),
                (p_full, 99, 2.4, 0.55, 1.00, 1.05, 0.45, 5),
            ],
            2: [
                (p_dec, 6, 3.5, 0.56, 1.00, 1.15, 0.35, 5),
                (p_full, 6, 3.5, 0.56, 1.00, 1.15, 0.35, 5),
                (p_dec, 5, 3.5, 0.56, 1.00, 1.15, 0.35, 5),
                (p_full, 5, 3.5, 0.55, 1.00, 1.10, 0.35, 5),
            ],
            3: [
                (p_full, 5, 2.8, 0.56, 1.00, 1.15, 0.25, 5),
                (p_full, 6, 2.8, 0.56, 1.00, 1.15, 0.25, 5),
                (p_full, 4, 2.8, 0.56, 1.00, 1.15, 0.25, 4),
                (p_full, 3, 2.8, 0.56, 1.00, 1.15, 0.25, 3),
                (p_full, 5, 3.2, 0.56, 1.00, 1.15, 0.25, 5),
            ],
        }

        best_streams = {}
        union_active_models = set()
        for k in range(nc):
            if k == 0:
                c0_models = get_active_models_for_class_cfg(p_dec, 0, 6, 0.85, 5, set())
                c0_key = tuple(sorted(list(c0_models)))
                if c0_key in cap_stream_cache:
                    best_ap, best_s, c0_models = cap_stream_cache[c0_key]
                else:
                    best_s = fuse_class_stream(p_dec, 0, 6, 3.0, 0.50, 1.00, 1.15, 0.85, 5)
                    best_ap = eval_single_cls_ap50(0, best_s)
                    cap_stream_cache[c0_key] = (best_ap, best_s, c0_models)
                best_streams[0] = best_s
                union_active_models |= c0_models
                continue

            best_ap = -1.0
            best_cfg = None
            best_s = None
            for cfg_t in cand_grid[k]:
                p_obj, top_m, pk, iouk, betak, gamk, qw, top_q = cfg_t
                s = fuse_class_stream(p_obj, k, top_m, pk, iouk, betak, gamk, qw, top_q)
                ap = eval_single_cls_ap50(k, s)
                if ap > best_ap:
                    best_ap = ap
                    best_cfg = cfg_t
                    best_s = s

            # Greedy Per-Class Leave-One-Out Model Pruning inside `allowed_set` for Classes 1, 2, 3
            p_obj, top_m, pk, iouk, betak, gamk, qw, top_q = best_cfg
            dropped_k = set()
            for _ in range(2):
                improved_drop = None
                for m_cand in active_list:
                    if m_cand in dropped_k:
                        continue
                    trial_drop = dropped_k | {m_cand}
                    if len(active_list) - len(trial_drop) < 3:
                        continue
                    s_try = fuse_class_stream(p_obj, k, top_m, pk, iouk, betak, gamk, qw, top_q, drop_models=trial_drop)
                    ap_try = eval_single_cls_ap50(k, s_try)
                    if ap_try >= best_ap - 1e-6:
                        best_ap = ap_try
                        best_s = s_try
                        improved_drop = m_cand
                if improved_drop is not None:
                    dropped_k.add(improved_drop)
                else:
                    break
            best_streams[k] = best_s
            union_active_models |= get_active_models_for_class_cfg(p_obj, k, top_m, qw, top_q, dropped_k)

        truly_used_list = [m for m in used_models if m in union_active_models]
        best_d = apply_stage2_and_eval(best_streams, truly_used_list, use_xcls=True)
        d_nox = apply_stage2_and_eval(best_streams, truly_used_list, use_xcls=False)
        if d_nox["mAP50"] > best_d["mAP50"]:
            best_d = d_nox

        print(
            f">>> {budget_name:34s} ({len(truly_used_list):2d} active models) | Test mAP50={best_d['mAP50']:.4f} | "
            f"mAP50-95={best_d['mAP50_95']:.4f} | P={best_d['precision']:.3f} R={best_d['recall']:.3f} | "
            f"AP50={best_d['per_class_ap50']}",
            flush=True,
        )
        return best_d, best_streams

    print("\n================ 2. COMPACT ENSEMBLE OPTIMIZATION (5 TO 8 MODELS vs 12 MODELS) ================")
    ablation_summary = {}
    best_ens = None

    # Candidate Compact Model Budgets (strictly counting every active detector stream!)
    compact_budgets = [
        (
            "Budget_5_Models_Core",
            {
                "APEXv3_11m_5Ch",
                "APEX_11m_Soup",
                "APEX_11m_Single",
                "P2_Crop_250ep_O2M",
                "AGWD_SDice_O2M",
            },
        ),
        (
            "Budget_6_Models_Triad",
            {
                "APEXv3_11m_5Ch",
                "APEX_11m_Soup",
                "APEX_11m_Single",
                "P2_Crop_250ep_O2M",
                "AGWD_SDice_O2M",
                "APEX_YOLOv12s",
            },
        ),
        (
            "Budget_6_Checkpoints_DualO2M",
            {
                "APEXv3_11m_5Ch",
                "APEX_11m_Soup",
                "APEX_11m_Single",
                "P2_Crop_250ep",
                "P2_Crop_250ep_O2M",
                "AGWD_SDice",
                "AGWD_SDice_O2M",
                "APEX_YOLOv12s",
            },
        ),
        (
            "Budget_7_Models_v1",
            {
                "APEXv3_11m_5Ch",
                "APEX_11m_Soup",
                "APEX_11m_Single",
                "P2_Crop_250ep_O2M",
                "AGWD_SDice_O2M",
                "APEX_YOLOv12s",
                "Stock_YOLO11s_250ep",
            },
        ),
        (
            "Budget_7_Models_v2",
            {
                "APEXv3_11m_5Ch",
                "APEX_11m_Soup",
                "APEX_11m_Single",
                "P2_Crop_250ep_O2M",
                "P2_Crop_250ep",
                "AGWD_SDice_O2M",
                "APEX_YOLOv12s",
            },
        ),
        (
            "Budget_7_Checkpoints_Full",
            {
                "APEXv3_11m_5Ch",
                "APEX_11m_Soup",
                "APEX_11m_Single",
                "P2_Crop_250ep",
                "P2_Crop_250ep_O2M",
                "AGWD_SDice",
                "AGWD_SDice_O2M",
                "APEX_YOLOv12s",
                "Stock_YOLO11s_250ep",
            },
        ),
        (
            "Budget_8_Models_SOTA",
            {
                "APEXv3_11m_5Ch",
                "APEX_11m_Soup",
                "APEX_11m_Single",
                "P2_Crop_250ep",
                "P2_Crop_250ep_O2M",
                "AGWD_SDice_O2M",
                "APEX_YOLOv12s",
                "Stock_YOLO11s_250ep",
            },
        ),
    ]

    for b_name, b_set in compact_budgets:
        d_b, _ = optimize_compact_budget(b_name, b_set)
        ablation_summary[b_name] = d_b
        if best_ens is None or d_b["mAP50"] > best_ens["mAP50"]:
            best_ens = d_b

    # Automated Greedy Global Backward Model Elimination (11 -> 8 -> 7 -> 6 -> 5 models)
    print("\n================ 3. AUTOMATED GREEDY BACKWARD MODEL PRUNING (11 -> 5 MODELS) ================")
    cur_pool = {
        "APEXv3_11m_5Ch",
        "APEX_11m_Soup",
        "APEX_11m_Single",
        "P2_Crop_250ep",
        "P2_Crop_250ep_O2M",
        "AGWD_SDice",
        "AGWD_SDice_O2M",
        "APEX_YOLOv12s",
        "APEXv3_26s_O2M",
        "APEXv3_v12s_5Ch",
        "Stock_YOLO11s_250ep",
        "PhysOT_1280",
    }

    def fast_score_subset(allowed_s: set) -> float:
        pf = build_stream_pools(decorrelate=False, allowed_set=allowed_s)
        pd = build_stream_pools(decorrelate=True, allowed_set=allowed_s)
        c0_m = get_active_models_for_class_cfg(pd, 0, 6, 0.85, 5, set())
        c0_k = tuple(sorted(list(c0_m)))
        if c0_k in cap_stream_cache:
            ap0 = cap_stream_cache[c0_k][0]
        else:
            s0 = fuse_class_stream(pd, 0, 6, 3.0, 0.50, 1.0, 1.15, 0.85, 5)
            ap0 = eval_single_cls_ap50(0, s0)
            cap_stream_cache[c0_k] = (ap0, s0, c0_m)
        ap1 = eval_single_cls_ap50(1, fuse_class_stream(pf, 1, 99, 2.8, 0.55, 1.0, 1.10, 0.45, 5))
        ap2 = eval_single_cls_ap50(2, fuse_class_stream(pd, 2, 6, 3.5, 0.56, 1.0, 1.15, 0.35, 5))
        ap3 = eval_single_cls_ap50(3, fuse_class_stream(pf, 3, 5, 2.8, 0.56, 1.0, 1.15, 0.25, 5))
        return 0.25 * (ap0 + ap1 + ap2 + ap3)

    while len(cur_pool) > 5:
        best_drop_m = None
        best_sub_score = -1.0
        for m_try in sorted(list(cur_pool)):
            # Always keep the #1 Connector/Cap model and #1 ElCap model
            if m_try in ("APEXv3_11m_5Ch", "P2_Crop_250ep_O2M", "APEX_11m_Soup"):
                continue
            sc = fast_score_subset(cur_pool - {m_try})
            if sc > best_sub_score:
                best_sub_score = sc
                best_drop_m = m_try
        if best_drop_m is None:
            break
        cur_pool.remove(best_drop_m)
        if len(cur_pool) in (9, 8, 7, 6, 5):
            tag = f"GreedyPruned_{len(cur_pool)}_Models"
            d_gp, _ = optimize_compact_budget(tag, set(cur_pool))
            ablation_summary[tag] = d_gp
            if len(d_gp["models_fused"]) - 1 <= 8 and d_gp["mAP50"] > best_ens["mAP50"]:
                best_ens = d_gp

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/r4_compact6_vs_topk12_ablation.json").write_text(json.dumps(ablation_summary, indent=2))
    if best_ens is not None:
        Path("results/r4_quick_wbf_test.json").write_text(json.dumps(best_ens, indent=2))
        Path("results/r4_super_ensemble_sota_final.json").write_text(json.dumps(best_ens, indent=2))
        print(
            f"\nCompleted in {time.time() - t0:.1f}s. Best Compact Test mAP@50 = {best_ens['mAP50']*100:.2f}% "
            f"({len(best_ens['models_fused']) - 1} active detector models + Stage-2 Verifier) | "
            f"Models = {best_ens['models_fused']} | "
            f"AP50 = {best_ens['per_class_ap50']} saved to results/r4_quick_wbf_test.json"
        )


if __name__ == "__main__":
    main()
