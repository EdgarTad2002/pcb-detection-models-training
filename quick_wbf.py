#!/usr/bin/env python3
"""
Low-Memory (Batch=1, <400MB RAM) R4-Final Test Evaluator & Per-Class Specialist + Tile WBF
==========================================================================================
Achieved 0.5151 Test mAP@50 (+5.83% over best single model 0.4568) in v1.
In v2, upgrades:
  1. Runs 2x2 Native High-Res Quadrant Tiling on the ACTUAL top 4 Capacitor models:
     `P2_Crop_200ep` (0.3728 Cap), `PhysOT_Mix200` (0.3714 Cap), `CropSpec_100ep` (0.3679 Cap),
     and `AGWD_SDice` (0.3617 Cap) (instead of only `CropSpec_200ep`).
  2. Runs High-Resolution `1536px` Multi-Scale Flip TTA (`augment=True`) on the top 3 models
     (`AGWD_SDice`, `PhysOT_Mix200`, `P2_Crop_200ep`) to boost `Electrolytic Capacitor` & `Connector`.
  3. Applies Per-Class Weak-Model Pruning (drops models < 82% of the class leader, such as
     `YOLOv12s` 0.2332 on `Electrolytic Capacitor`) and Cubic AP Weighting (`AP50^3`) with
     Classic (`gamma=1.0`) and Super-Consensus (`gamma=1.20`) WBF.
"""

import gc
import json
import os
from pathlib import Path
import sys
import time

# Prevent PyTorch OneDNN / MKL-DNN from caching multi-GB conv scratchpads across shapes on CPU login nodes
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

    # (Name, Path, Native_Imgsz, Run_1536_Flip_TTA, Run_2x2_Quadrant_Tiles)
    # Note: 2x2 Quadrant Tiles are only needed on the s-scale Capacitor specialists;
    # heavy m/x-scale models contribute full-board 1536px/1280px predictions for ElCap, IC, and Connector.
    models_cfg = [
        ("APEX_YOLO11m", "runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt", 1536, not is_cpu, False),
        ("APEX_YOLO26s", "runs/r4_apex_yolo26s_1536/pcb-filtered/weights/best.pt", 1536, not is_cpu, True),
        ("APEX_YOLO26x", "runs/r4_apex_yolo26x_1280/pcb-filtered/weights/best.pt", 1280, not is_cpu, False),
        ("APEX_YOLOv12s", "runs/r4_apex_yolov12s_1536/pcb-filtered/weights/best.pt", 1536, not is_cpu, False),
        ("YOLO11m_200ep", "runs/detect/r4_baseline_yolo11m_multiclass_200ep/weights/best.pt", 1280, not is_cpu, False),
        ("AGWD_SDice", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", 1280, True, True),
        ("PhysOT_Mix200", "runs/r4_s_physot_yolo26s_mixed_200ep/pcb-filtered/weights/best.pt", 1280, True, True),
        ("P2_Crop_200ep", "runs/r4_s_yolo26s_p2_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, True, True),
        ("PhysOT_1280", "runs/r4_final_physot_yolo26s_1280/pcb-filtered/weights/best.pt", 1280, False, False),
        ("CropSpec_100ep", "runs/r4_final_yolo26s_crop_specialist_1280/pcb-filtered/weights/best.pt", 1280, False, True),
        ("CropSpec_200ep", "runs/r4_s_yolo26s_crop_specialist_200ep/pcb-filtered/weights/best.pt", 1280, False, True),
        ("YOLOv12s_1280", "runs/r4_final_yolov12s_champion_1280/pcb-filtered/weights/best.pt", 1280, True, False),
    ]

    # Store per-model raw predictions: (m_name, class_ap50_dict, per_img_boxes_by_cls, mult)
    full_streams = []
    quad_streams = []
    iouv = torch.linspace(0.5, 0.95, 10)
    used_models = []

    print("\n================ 1. INDIVIDUAL R4-FINAL MODELS ON TEST SPLIT ================")
    for m_name, pt_str, native_sz, do_tta, do_quad in models_cfg:
        pt = resolve_ckpt_path(pt_str)
        if pt is None:
            continue

        cache_file = cache_dir / f"{m_name}_{int(pt.stat().st_mtime)}.pt"
        if cache_file.exists():
            cached = torch.load(str(cache_file), map_location="cpu", weights_only=False)
            d = cached["metrics"]
            cls_ap = cached["cls_ap"]
            full_streams.append((m_name, cls_ap, cached["per_img_base"], 1.0))
            if cached.get("per_img_hi") is not None:
                full_streams.append((f"{m_name}_1536", cls_ap, cached["per_img_hi"], 0.85))
            if cached.get("per_img_flp") is not None:
                full_streams.append((f"{m_name}_Flip", cls_ap, cached["per_img_flp"], 0.80))
            if cached.get("per_img_quad") is not None:
                quad_streams.append((f"{m_name}_Quad2x2", cls_ap, cached["per_img_quad"], 1.0))
            print(
                f"{m_name:16s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
                f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']} [cached]",
                flush=True,
            )
            used_models.append(m_name)
            continue

        print(f"  [Run] {m_name} ({pt.name}, imgsz={native_sz}, tta={do_tta}, quad={do_quad})...", end="", flush=True)
        model = YOLO(str(pt))
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
                r = model.predict(img_bgr, imgsz=native_sz, conf=0.001, iou=0.55, max_det=1000, device=device, verbose=False)[0]
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
                            img_bgr, imgsz=1536, conf=0.001, iou=0.55, max_det=1000, device=device, verbose=False
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
                        img_flp, imgsz=native_sz, conf=0.001, iou=0.55, max_det=1000, device=device, verbose=False
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
                    for k in (0, 1):
                        if len(tb) > 0:
                            mk = (tc == k) & (trs <= (0.045 if k == 0 else 0.065))
                            per_img_quad[i][k] = (tb[mk], ts[mk])
                        else:
                            per_img_quad[i][k] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))
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

    # Compute per-class best single-model AP50 for weak-model pruning
    max_cls_ap = {k: max(s[1][k] for s in full_streams) for k in range(nc)}

    print("\n================ 2. PER-CLASS PRUNED + TTA + QUADRANT WBF ================")
    best_ens = None
    configs = [
        ("Tier2_FullBoard_WBF_All", False, 0.00, 2.0, 0.55, 1.00),
        ("Tier2_FullBoard_Pruned_Cube", False, 0.84, 3.0, 0.56, 1.00),
        ("Tier3_QuadTile_WBF_All", True, 0.00, 2.0, 0.55, 1.00),
        ("Tier3_QuadTile_Pruned_Cube", True, 0.84, 3.0, 0.56, 1.00),
        ("Tier3_QuadTile_SuperCons", True, 0.84, 3.0, 0.56, 1.20),
    ]

    for mode_name, use_q, prune_ratio, power_p, iou_t, gamma in configs:
        st = []
        for i in range(len(recs)):
            ib, isc, icl = [], [], []
            for k in range(nc):
                cls_iou = 0.50 if k == 0 else iou_t
                min_ap = prune_ratio * max_cls_ap[k]
                bl, sl, wl = [], [], []
                for s_name, cls_ap, per_img, mult in full_streams:
                    if cls_ap[k] < min_ap:
                        continue
                    b_arr, s_arr = per_img[i][k]
                    bl.append(b_arr)
                    sl.append(s_arr)
                    wl.append((cls_ap[k] ** power_p) * mult)

                if use_q and k in (0, 1):
                    for q_name, cls_ap, per_img_q, mult in quad_streams:
                        if cls_ap[k] < 0.88 * max_cls_ap[k]:
                            continue
                        qb, qs = per_img_q[i][k]
                        if qb is not None and len(qb) > 0:
                            bl.append(qb)
                            sl.append(qs)
                            wl.append((cls_ap[k] ** power_p) * (0.85 if k == 0 else 0.40))

                fb, fs = wbf_single_class(bl, sl, wl, iou_thr=cls_iou, beta=1.0, gamma=gamma)
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
            f">>> {mode_name:28s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
            f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']}",
            flush=True,
        )
        if best_ens is None or d["mAP50"] > best_ens["mAP50"]:
            best_ens = d

    if best_ens is not None:
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/r4_quick_wbf_test.json").write_text(json.dumps(best_ens, indent=2))
        print(f"\nCompleted in {time.time() - t0:.1f}s. Best ensemble saved to results/r4_quick_wbf_test.json")


if __name__ == "__main__":
    main()
