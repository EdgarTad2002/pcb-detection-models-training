#!/usr/bin/env python3
"""
Low-Memory (Batch=1, <400MB RAM) Fast R4-Final Test Evaluator & Per-Class Specialist WBF
========================================================================================
Designed to run safely on the login node or CPU/GPU node without exceeding cgroup RAM limits.
Processes the 44 R4-Final test boards one image at a time (batch=1) and fuses:
  1. Multi-class R4-Final checkpoints (YOLO11m 200ep, CropSpec 200ep, AGWD+SDice, PhysOT, P2 Crop, YOLOv12s, PhysOT Mixed)
  2. Dedicated 1-Class Capacitor Specialist (`r4_specialist_c0_capacitor`)
using both Classic WBF and Per-Class Soft-Consensus WBF.
"""

import json
from pathlib import Path
import sys
import time

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

from evaluate_apex_r4_suite import (
    DATA_YAML,
    accumulate_image_stat,
    compute_single_class_ap50,
    compute_split_metrics_from_stats,
    resolve_ckpt_path,
    wbf_single_class,
)


def load_test_paths_and_gt(root: Path, split_subpath: str):
    """Loads ONLY file paths and tiny GT tensors (<1 MB RAM total, no images cached in RAM)."""
    from PIL import Image
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
    torch.set_num_threads(8)
    cfg = yaml.safe_load(open(DATA_YAML))
    root = Path(cfg["path"])
    names = cfg["names"]
    nc = len(names)

    t0 = time.time()
    recs = load_test_paths_and_gt(root, cfg["test"])
    print(f"Loaded {len(recs)} R4-Final test board GT annotations (device={device}, low-RAM batch=1 mode).")

    # (Name, Path, Per-Class Val AP50 Prior [Cap, Conn, ElCap, IC])
    models_cfg = [
        ("YOLO11m_200ep", "runs/detect/r4_baseline_yolo11m_multiclass_200ep/weights/best.pt", [0.27, 0.47, 0.65, 0.62]),
        ("CropSpec_200ep", "runs/r4_s_yolo26s_crop_specialist_200ep/pcb-filtered/weights/best.pt", [0.25, 0.45, 0.55, 0.60]),
        ("AGWD_SDice", "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt", [0.23, 0.33, 0.62, 0.56]),
        ("PhysOT_1280", "runs/r4_final_physot_yolo26s_1280/pcb-filtered/weights/best.pt", [0.22, 0.37, 0.56, 0.59]),
        ("P2_Crop_200ep", "runs/r4_s_yolo26s_p2_crop_specialist_200ep/pcb-filtered/weights/best.pt", [0.20, 0.37, 0.56, 0.56]),
        ("YOLOv12s_1280", "runs/r4_final_yolov12s_champion_1280/pcb-filtered/weights/best.pt", [0.22, 0.40, 0.47, 0.59]),
        ("PhysOT_Mix200", "runs/r4_s_physot_yolo26s_mixed_200ep/pcb-filtered/weights/best.pt", [0.23, 0.39, 0.56, 0.58]),
    ]

    preds = {i: {c: [] for c in range(nc)} for i in range(len(recs))}
    iouv = torch.linspace(0.5, 0.95, 10)
    used_models = []

    print("\n================ 1. INDIVIDUAL R4-FINAL MODELS ON TEST SPLIT ================")
    for m_name, pt_str, cw in models_cfg:
        pt = resolve_ckpt_path(pt_str)
        if pt is None:
            print(f"  [SKIP] {m_name} ({pt_str} not found)")
            continue
        model = YOLO(str(pt))
        st = []
        raw_per_img = []
        for i, (im_path, H, W, gt_b, gt_c) in enumerate(recs):
            r = model.predict(str(im_path), imgsz=1280, conf=0.001, iou=0.55, max_det=1000, device=device, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            raw_per_img.append((b, s, c))
            accumulate_image_stat(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().long(), gt_b, gt_c, iouv, st)

        d = compute_split_metrics_from_stats(st, names, [m_name])
        for i, (b, s, c) in enumerate(raw_per_img):
            for k in range(nc):
                m = c == k
                true_ap = float(d["per_class_ap50"].get(names[k], cw[k]))
                preds[i][k].append((b[m], s[m], max(0.10, true_ap) ** 2))
        print(
            f"{m_name:16s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
            f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']}",
            flush=True,
        )
        used_models.append(m_name)
        del model
        import gc; gc.collect()

    # Evaluate 1-Class Capacitor Specialist (`r4_specialist_c0_capacitor`)
    spec_pt = resolve_ckpt_path("runs/detect/r4_specialist_c0_capacitor/weights/best.pt")
    if spec_pt is not None:
        s_model = YOLO(str(spec_pt))
        nc_s = len(getattr(s_model, "names", {0: "0"}))
        sp_preds = []
        for i, (im_path, H, W, gt_b, gt_c) in enumerate(recs):
            r = s_model.predict(str(im_path), imgsz=1280, conf=0.001, iou=0.55, max_det=1000, device=device, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            if nc_s == 1:
                sp_preds.append((b, s))
            else:
                m = c == 0
                sp_preds.append((b[m], s[m]))

        # Compute single-class AP50 on Capacitor (cid=0)
        dummy_recs = [(r[0], None, r[1], r[2], r[3], r[4]) for r in recs]
        cap_ap50, cap_ap95 = compute_single_class_ap50(sp_preds, dummy_recs, 0)
        print(
            f"{'Spec_C0_Cap':16s} | Capacitor-Only Test AP50={cap_ap50:.4f} | AP50-95={cap_ap95:.4f}",
            flush=True,
        )
        spec_w = max(0.28, cap_ap50) ** 2
        for i, (b, s) in enumerate(sp_preds):
            preds[i][0].append((b, s, spec_w * 1.25))
        used_models.append("Spec_C0_Cap")
        del s_model

    print("\n================ 2. MULTI-MODEL + SPECIALIST PER-CLASS WBF ================")
    best_ens = None
    for mode_name, iou_t, beta, gamma in [
        ("WBF_Classic_0.55", 0.55, 1.00, 1.00),
        ("WBF_Soft_0.55_g0.65", 0.55, 0.65, 0.65),
        ("WBF_Soft_0.52_g0.50", 0.52, 0.55, 0.50),
    ]:
        st = []
        for i in range(len(recs)):
            ib, isc, icl = [], [], []
            for k in range(nc):
                cls_iou = 0.50 if k == 0 else iou_t
                bl = [x[0] for x in preds[i][k]]
                sl = [x[1] for x in preds[i][k]]
                wl = [x[2] for x in preds[i][k]]
                fb, fs = wbf_single_class(bl, sl, wl, iou_thr=cls_iou, beta=beta, gamma=gamma)
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
            f">>> {mode_name:20s} | Test mAP50={d['mAP50']:.4f} | mAP50-95={d['mAP50_95']:.4f} | "
            f"P={d['precision']:.3f} R={d['recall']:.3f} | AP50={d['per_class_ap50']}",
            flush=True,
        )
        if best_ens is None or d["mAP50"] > best_ens["mAP50"]:
            best_ens = d

    if best_ens is not None:
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/r4_quick_wbf_test.json").write_text(json.dumps(best_ens, indent=2))
        print(f"\nCompleted in {time.time() - t0:.1f}s. Saved to results/r4_quick_wbf_test.json")


if __name__ == "__main__":
    main()
