#!/usr/bin/env python3
"""
Phase 0 Forensic Diagnostic & YOLO26 One-to-Many (end2end=False) Recovery Evaluator
===================================================================================
Reads cached predictions from `results/wbf_cache_v3/` (or runs inference if uncached) to:
  1. Compare Ultralytics YOLO26 `end2end=True` (One-to-One head, 300-box cap, corrupted by
     topk=12 on `one2one` in APEX v2) vs `end2end=False` (One-to-Many `cv2`/`cv3` head + NMS)
     across `APEX_YOLO26s_1536`, `APEX_YOLO26x_1536`, and `Prev_YOLO26s_AGWD_SDice`.
  2. Perform a complete Forensic Error Taxonomy on `Electrolytic Capacitor` (Class 2) across
     all top single models and WBF ensembles:
       - True Positives (IoU >= 0.50) & Recall by Scale (<36px, 36-80px, >80px) and Aspect Ratio
       - False Positive Breakdown (Cross-Class Confusion with IC/Connector/Capacitor,
         Localization Error 0.10<=IoU<0.50, Duplicate Box on same GT, Pure Background FP,
         and Same-Image Collision with higher-confidence predicted IC/Connector)
       - Missed GT Breakdown (Detected as wrong class vs Low-Confidence vs Localization Error vs FN)
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
    resolve_ckpt_path,
)
from quick_wbf import (
    box_iou_np,
    load_test_paths_and_gt,
    set_yolo26_end2end_mode,
)


def analyze_elcap_forensics(
    recs,
    per_img_preds: List[Dict[int, Tuple[np.ndarray, np.ndarray]]],
    names: Dict[int, str],
    target_cid: int = 2,
    conf_thresh: float = 0.20,
) -> dict:
    gt_total = 0
    scale_counts = {"small_<36px": [0, 0], "med_36_80px": [0, 0], "large_>80px": [0, 0]}
    ar_counts = {"circular_<1.25": [0, 0], "elongated_>=1.25": [0, 0]}

    missed_gt_causes = {
        "detected_at_low_conf_<0.20": 0,
        "localization_err_0.1_to_0.5": 0,
        "misclassified_as_IC": 0,
        "misclassified_as_Connector": 0,
        "misclassified_as_Capacitor": 0,
        "completely_missed_fn": 0,
    }

    fp_taxonomy = {
        "true_positive": 0,
        "duplicate_elcap_box": 0,
        "loc_error_on_gt_elcap_0.1_0.5": 0,
        "confused_gt_IC": 0,
        "confused_gt_Connector": 0,
        "confused_gt_Capacitor": 0,
        "pure_background_fp": 0,
        "fp_overlapping_stronger_pred_IC_or_Conn": 0,
    }

    all_ranked_elcap = []

    for i, (im_path, H, W, gt_b_t, gt_c_t) in enumerate(recs):
        gt_b = gt_b_t.numpy()
        gt_c = gt_c_t.numpy()

        gt_t_mask = gt_c == target_cid
        gt_t_boxes = gt_b[gt_t_mask]
        gt_other_boxes = {k: gt_b[gt_c == k] for k in range(len(names)) if k != target_cid}

        pred_t_boxes, pred_t_scores = per_img_preds[i][target_cid]
        pred_other = {k: per_img_preds[i][k] for k in range(len(names)) if k != target_cid}

        if len(pred_t_scores) > 0:
            order = np.argsort(-pred_t_scores)
            pred_t_boxes = pred_t_boxes[order]
            pred_t_scores = pred_t_scores[order]

        hi_mask = pred_t_scores >= conf_thresh
        hi_boxes = pred_t_boxes[hi_mask]
        hi_scores = pred_t_scores[hi_mask]

        matched_gt_hi = np.zeros(len(gt_t_boxes), dtype=bool)
        iou_hi_gt = box_iou_np(hi_boxes, gt_t_boxes)

        for p_idx in range(len(hi_boxes)):
            pb = hi_boxes[p_idx : p_idx + 1]
            ps = float(hi_scores[p_idx])

            overlaps_stronger_other = False
            for other_k in (3, 1, 0):
                ob, os_arr = pred_other[other_k]
                if len(ob) > 0:
                    ious_o = box_iou_np(pb, ob)[0]
                    strong_mask = (ious_o >= 0.60) & (os_arr > ps * 1.22) & (os_arr >= 0.25)
                    if np.any(strong_mask):
                        overlaps_stronger_other = True
                        break

            best_gt_iou = 0.0
            best_gt_idx = -1
            if len(gt_t_boxes) > 0:
                best_gt_idx = int(np.argmax(iou_hi_gt[p_idx]))
                best_gt_iou = float(iou_hi_gt[p_idx, best_gt_idx])

            if best_gt_iou >= 0.50:
                if not matched_gt_hi[best_gt_idx]:
                    matched_gt_hi[best_gt_idx] = True
                    fp_taxonomy["true_positive"] += 1
                    all_ranked_elcap.append((ps, 1, "TP", overlaps_stronger_other))
                else:
                    fp_taxonomy["duplicate_elcap_box"] += 1
                    if overlaps_stronger_other:
                        fp_taxonomy["fp_overlapping_stronger_pred_IC_or_Conn"] += 1
                    all_ranked_elcap.append((ps, 0, "duplicate_elcap", overlaps_stronger_other))
            else:
                cat = "pure_background_fp"
                if best_gt_iou >= 0.10:
                    cat = "loc_error_on_gt_elcap_0.1_0.5"
                else:
                    for other_k, other_name in (
                        (3, "confused_gt_IC"),
                        (1, "confused_gt_Connector"),
                        (0, "confused_gt_Capacitor"),
                    ):
                        ogb = gt_other_boxes[other_k]
                        if len(ogb) > 0 and float(np.max(box_iou_np(pb, ogb))) >= 0.45:
                            cat = other_name
                            break
                fp_taxonomy[cat] += 1
                if overlaps_stronger_other:
                    fp_taxonomy["fp_overlapping_stronger_pred_IC_or_Conn"] += 1
                all_ranked_elcap.append((ps, 0, cat, overlaps_stronger_other))

        iou_all_gt = box_iou_np(gt_t_boxes, pred_t_boxes)
        for g_idx, gb in enumerate(gt_t_boxes):
            gt_total += 1
            bw = float(gb[2] - gb[0])
            bh = float(gb[3] - gb[1])
            max_side = max(bw, bh)
            min_side = max(min(bw, bh), 1.0)
            ar = max_side / min_side

            s_key = "small_<36px" if max_side < 36.0 else ("med_36_80px" if max_side <= 80.0 else "large_>80px")
            a_key = "circular_<1.25" if ar < 1.25 else "elongated_>=1.25"
            scale_counts[s_key][1] += 1
            ar_counts[a_key][1] += 1

            if matched_gt_hi[g_idx]:
                scale_counts[s_key][0] += 1
                ar_counts[a_key][0] += 1
            else:
                max_iou_any_conf = float(np.max(iou_all_gt[g_idx])) if len(pred_t_boxes) > 0 else 0.0
                if max_iou_any_conf >= 0.50:
                    missed_gt_causes["detected_at_low_conf_<0.20"] += 1
                elif max_iou_any_conf >= 0.15:
                    missed_gt_causes["localization_err_0.1_to_0.5"] += 1
                else:
                    gb_2d = gb[None, :]
                    found_other = False
                    for other_k, cause_key in (
                        (3, "misclassified_as_IC"),
                        (1, "misclassified_as_Connector"),
                        (0, "misclassified_as_Capacitor"),
                    ):
                        ob, os_arr = pred_other[other_k]
                        ob_hi = ob[os_arr >= 0.15]
                        if len(ob_hi) > 0 and float(np.max(box_iou_np(gb_2d, ob_hi))) >= 0.50:
                            missed_gt_causes[cause_key] += 1
                            found_other = True
                            break
                    if not found_other:
                        missed_gt_causes["completely_missed_fn"] += 1

    all_ranked_elcap.sort(key=lambda x: -x[0])
    top_k_slice = all_ranked_elcap[: max(gt_total, 1)]
    top_k_fps = {}
    top_k_overlap_stronger = 0
    for sc, is_tp, cat, ov_str in top_k_slice:
        if not is_tp:
            top_k_fps[cat] = top_k_fps.get(cat, 0) + 1
            if ov_str:
                top_k_overlap_stronger += 1

    return {
        "gt_total": gt_total,
        "conf_thresh": conf_thresh,
        "recall_at_conf": round(fp_taxonomy["true_positive"] / max(gt_total, 1), 4),
        "precision_at_conf": round(
            fp_taxonomy["true_positive"]
            / max(sum(v for k, v in fp_taxonomy.items() if k != "fp_overlapping_stronger_pred_IC_or_Conn"), 1),
            4,
        ),
        "recall_by_scale": {
            k: f"{v[0]}/{v[1]} ({100.0 * v[0] / max(v[1], 1):.1f}%)" for k, v in scale_counts.items()
        },
        "recall_by_aspect_ratio": {
            k: f"{v[0]}/{v[1]} ({100.0 * v[0] / max(v[1], 1):.1f}%)" for k, v in ar_counts.items()
        },
        "missed_gt_breakdown": missed_gt_causes,
        "fp_taxonomy_at_conf": fp_taxonomy,
        "top_N_ranking_contaminants": {
            "top_N": gt_total,
            "tp_in_top_N": sum(x[1] for x in top_k_slice),
            "fp_in_top_N": len(top_k_slice) - sum(x[1] for x in top_k_slice),
            "fp_causes_in_top_N": top_k_fps,
            "fp_in_top_N_overlapping_stronger_IC_or_Conn": top_k_overlap_stronger,
        },
    }


def load_or_run_test_cache(
    m_name: str,
    pt: Path,
    sz: int,
    force_o2m: bool,
    test_recs,
    names: Dict[int, str],
    device: str,
    cache_dir: Path,
):
    cache_file = cache_dir / f"{m_name}_{int(pt.stat().st_mtime)}.pt"
    if cache_file.exists():
        cached = torch.load(str(cache_file), map_location="cpu", weights_only=False)
        return cached["metrics"], cached["per_img_base"]

    nc = len(names)
    iouv = torch.linspace(0.5, 0.95, 10)
    st = []
    per_img_base = [{k: None for k in range(nc)} for _ in range(len(test_recs))]
    model = YOLO(str(pt))
    if force_o2m:
        set_yolo26_end2end_mode(model, False)

    with torch.inference_mode():
        for i, (im_path, H, W, gt_b, gt_c) in enumerate(test_recs):
            img_bgr = cv2.imread(str(im_path))
            r = model.predict(img_bgr, imgsz=sz, conf=0.001, iou=0.55, max_det=1500, device=device, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            for k in range(nc):
                m = c == k
                per_img_base[i][k] = (b[m], s[m])
            accumulate_image_stat(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().long(), gt_b, gt_c, iouv, st)
            del r, img_bgr

    d = compute_split_metrics_from_stats(st, names, [m_name])
    cls_ap = {k: float(d["per_class_ap50"].get(names[k], 0.25)) for k in range(nc)}
    torch.save(
        {
            "metrics": d,
            "cls_ap": cls_ap,
            "per_img_base": per_img_base,
            "per_img_hi": None,
            "per_img_flp": None,
            "per_img_quad": None,
        },
        str(cache_file),
    )
    del model
    gc.collect()
    return d, per_img_base


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(DATA_YAML))
    root = Path(cfg["path"])
    names = cfg["names"]

    test_recs = load_test_paths_and_gt(root, cfg["test"])
    cache_dir = Path("results/wbf_cache_v3")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded R4-Final Test Split ({len(test_recs)} boards, device={device}).")

    report = {"yolo26_end2end_ablation": {}, "forensics_test": {}}

    # =========================================================================
    # 1. YOLO26 One-to-One (end2end=True) vs One-to-Many (end2end=False) Check
    # =========================================================================
    print("\n================ 1. YOLO26 DUAL-HEAD ABLATION: end2end=True (O2O) vs end2end=False (O2M + NMS) ================")
    yolo26_pairs = [
        (
            "APEX_YOLO26s_1536",
            "APEX_26s_O2M",
            "runs/r4_apex_yolo26s_1536/pcb-filtered/weights/best.pt",
            "results/r4_apex_yolo26s_1536.json",
            1536,
        ),
        (
            "APEX_YOLO26x_1536",
            "APEX_26x_O2M",
            "runs/r4_apex_yolo26x_1536/pcb-filtered/weights/best.pt",
            "results/r4_apex_yolo26x_1536.json",
            1536,
        ),
        (
            "AGWD_SDice",
            "AGWD_SDice_O2M",
            "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt",
            None,
            1280,
        ),
    ]
    for o2o_name, o2m_name, pt_str, o2o_json_str, sz in yolo26_pairs:
        pt = resolve_ckpt_path(pt_str)
        if pt is None:
            continue

        d_o2o = None
        if o2o_json_str and Path(o2o_json_str).exists():
            d_o2o = json.loads(Path(o2o_json_str).read_text())
        else:
            d_o2o, _ = load_or_run_test_cache(o2o_name, pt, sz, False, test_recs, names, device, cache_dir)

        d_o2m, preds_o2m = load_or_run_test_cache(o2m_name, pt, sz, True, test_recs, names, device, cache_dir)
        delta = (d_o2m["mAP50"] - d_o2o["mAP50"]) * 100.0
        print(
            f"  {o2o_name:18s} [end2end=True  O2O] | Test mAP50={d_o2o['mAP50']:.4f} | AP50={d_o2o['per_class_ap50']}\n"
            f"  {o2m_name:18s} [end2end=False O2M] | Test mAP50={d_o2m['mAP50']:.4f} ({delta:+.2f}%) | AP50={d_o2m['per_class_ap50']}",
            flush=True,
        )
        report["yolo26_end2end_ablation"][o2o_name] = {
            "end2end_true_o2o": d_o2o,
            "end2end_false_o2m": d_o2m,
            "delta_map50_pct": round(delta, 2),
        }

    # =========================================================================
    # 2. Forensic Error Breakdown on Electrolytic Capacitor (Test Split)
    # =========================================================================
    print("\n================ 2. ELECTROLYTIC CAPACITOR FORENSIC ERROR AUDIT (TEST SPLIT) ================")
    audit_models = [
        ("APEX_11m_Soup", "runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt", 1536, False),
        ("APEX_11m_Single", "external/DCD-Net/runs/detect/runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best_single_epoch.pt", 1536, False),
        ("APEX_YOLOv12s", "runs/r4_apex_yolov12s_1536/pcb-filtered/weights/best.pt", 1536, False),
        ("AGWD_SDice", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, False),
    ]
    for m_name, pt_str, sz, force_o2m in audit_models:
        pt = resolve_ckpt_path(pt_str)
        if pt is None:
            continue
        d_m, test_preds = load_or_run_test_cache(m_name, pt, sz, force_o2m, test_recs, names, device, cache_dir)
        f_test = analyze_elcap_forensics(test_recs, test_preds, names, target_cid=2, conf_thresh=0.20)
        report["forensics_test"][m_name] = {"metrics": d_m, "elcap_audit": f_test}

        print(f"\n--- {m_name} (Test mAP50={d_m['mAP50']:.4f}, ElCap AP50={d_m['per_class_ap50'].get('Electrolytic Capacitor', 0):.4f}) ---")
        print(f"  Recall by Scale:        {f_test['recall_by_scale']}")
        print(f"  Recall by Aspect Ratio: {f_test['recall_by_aspect_ratio']}")
        print(f"  Missed GT Breakdown:    {f_test['missed_gt_breakdown']}")
        print(f"  FP Taxonomy (conf>=.2): {f_test['fp_taxonomy_at_conf']}")
        print(f"  Top-167 Contaminants:   {f_test['top_N_ranking_contaminants']}")

    Path("results").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/r4_elcap_forensic_diagnosis.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved complete forensic report to {out_path}")


if __name__ == "__main__":
    main()
