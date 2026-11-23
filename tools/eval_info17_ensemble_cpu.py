#!/usr/bin/env python3
"""Evaluate 640-trained multi-seed WBF and 1280-trained champions on info17-original-fourclass-v1/test.

Runs purely on CPU (cluster research_cpu partition) and scores both Ultralytics 101-point
metrics (common_ultralytics101) and Information 2026 public compute_map metrics.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_champion_audit import score  # noqa: E402
from tools.evaluate_info17_protocol import (  # noqa: E402
    AUTHOR_NAMES,
    author_inputs,
    load_metric_functions,
)

CUSTOM_MODULES = (
    "luma_modules", "retinex_stem", "weber_stem", "icm_module",
    "physics_spectral_yolo26", "superyolo_pcb", "squared_dice_loss",
    "agwd_ota_loss", "igr_stem", "fg_csfa_neck", "aca_supcon",
)


def register_modules():
    for name in CUSTOM_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            pass


def load_wbf():
    try:
        from ensemble_boxes import weighted_boxes_fusion
        return weighted_boxes_fusion
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "ensemble_boxes_wbf",
            Path(sys.modules["ensemble_boxes"].__file__).parent / "ensemble_boxes_wbf.py"
            if "ensemble_boxes" in sys.modules
            else next(Path(p) / "ensemble_boxes/ensemble_boxes_wbf.py" for p in sys.path if (Path(p) / "ensemble_boxes/ensemble_boxes_wbf.py").exists()),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.weighted_boxes_fusion


def wbf_fuse_batch(pred_lists, shapes, weights, iou_thr=0.50, skip=0.001, max_det=1000, wbf_fn=None):
    fused = []
    for k, (h, w) in enumerate(shapes):
        scale = np.array([w, h, w, h], dtype=float)
        bl, sl, cl = [], [], []
        for member_preds in pred_lists:
            a = np.asarray(member_preds[k], dtype=float).reshape(-1, 6)
            bl.append((a[:, :4] / scale).clip(0, 1).tolist() if len(a) else [])
            sl.append(a[:, 4].tolist() if len(a) else [])
            cl.append(a[:, 5].astype(int).tolist() if len(a) else [])
        b, s, c = wbf_fn(bl, sl, cl, weights=weights, iou_thr=iou_thr, skip_box_thr=skip)
        if len(s):
            res = np.column_stack((np.asarray(b).reshape(-1, 4) * scale, s, c))
            res = res[np.argsort(-res[:, 4], kind="stable")][:max_det].tolist()
        else:
            res = []
        fused.append(res)
    return fused


def eval_both(preds, records, shapes, env):
    ultra = score(preds, records, shapes, {"recall_conf": 0.25})
    p, g, pn, gn = author_inputs(preds, records)
    pub_nms = env["evaluate_predictions"](p, g)
    pub_wbf = env["compute_map"](pn, gn, AUTHOR_NAMES)
    return {
        "common_ultralytics101": {
            "mAP50": ultra["mAP50"],
            "mAP50_95": ultra["mAP50_95"],
            "per_class_ap50": {k: v["ap50"] for k, v in ultra["per_class"].items()},
        },
        "public_nms101_code": {
            "mAP50": float(pub_nms["mAP50"]),
        },
        "public_wbf11_code_raw001": {
            "mAP50": float(pub_wbf["mAP50"]),
            "mAP50_95": float(pub_wbf["mAP50_95"]),
            "per_class_ap50": pub_wbf.get("per_class_ap50", {}),
        },
    }


def main():
    torch.set_num_threads(16)
    register_modules()
    from models import as_pgms_yolo  # noqa: F401
    env = load_metric_functions()
    wbf_fn = load_wbf()

    data_root = ROOT / "datasets/info17-original-fourclass-v1/test"
    img_paths = sorted((data_root / "images").glob("*"))
    images = [cv2.imread(str(p)) for p in img_paths]
    shapes = [list(im.shape[:2]) for im in images]
    records = []
    for p in img_paths:
        lbl = data_root / "labels" / (p.stem + ".txt")
        rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()]
        records.append({"name": p.name, "image": str(p), "label": str(lbl), "boxes_cxywh": rows})
    print(f"Loaded {len(records)} info17 test images", flush=True)

    output = {"test_images": len(records), "track_640_3seed_wbf": {}, "track_1280_champions_on_info17": {}}

    # ---- TRACK 1: 640-trained 3-seed cached predictions + Exp7 640 ----
    pred_dir = ROOT / "results/info17_protocol_v1_20261031/predictions"
    model_jobs = {
        "yolov8s": "info17_final_v1_56f756d1_main_yolov8s_paper",
        "yolov12s": "info17_final_v1_56f756d1_main_yolov12s_paper",
        "yolo26s": "info17_final_v1_56f756d1_main_yolo26s_paper",
        "agwd_sdice": "info17_final_v1_56f756d1_main_yolo26s_agwd_sdice_paper",
        "physot": "info17_final_v1_56f756d1_main_physot_yolo26s_paper",
        "crop": "info17_final_v1_56f756d1_budget_yolo26s_crop_specialist_mixed",
    }

    seed_preds = {0: {}, 1: {}, 2: {}}
    for seed in (0, 1, 2):
        for mname, prefix in model_jobs.items():
            cache = json.loads((pred_dir / f"{prefix}_s{seed}_test_common.json").read_text())
            if cache["image_names"] != [r["name"] for r in records]:
                raise ValueError(f"Image order mismatch for {prefix}_s{seed}")
            seed_preds[seed][mname] = cache["predictions"]

        exp7_pt = ROOT / f"runs/original_top2_20261105_exp7_as_pgms_640_s{seed}/pcb-filtered/weights/best.pt"
        if exp7_pt.exists():
            m = YOLO(str(exp7_pt))
            seed_preds[seed]["exp7_640"] = [
                m.predict(im, imgsz=640, rect=False, batch=1, half=False, conf=0.001, iou=0.5, max_det=1000, device="cpu", verbose=False, augment=False)[0].boxes.data.detach().cpu().tolist()
                for im in images
            ]
            del m

    combos_640 = [
        ("exp7_640_single", ["exp7_640"], [1.0]),
        ("wbf2_crop_v12", ["crop", "yolov12s"], [1.2, 1.0]),
        ("wbf3_crop_v12_agwd", ["crop", "yolov12s", "agwd_sdice"], [1.3, 1.2, 1.1]),
        ("wbf4a_crop_v12_agwd_physot", ["crop", "yolov12s", "agwd_sdice", "physot"], [1.3, 1.2, 1.1, 1.0]),
        ("wbf4b_crop_v12_agwd_v8", ["crop", "yolov12s", "agwd_sdice", "yolov8s"], [1.3, 1.2, 1.1, 1.0]),
        ("wbf5_plus_exp7_640", ["crop", "yolov12s", "agwd_sdice", "yolov8s", "exp7_640"], [1.3, 1.2, 1.1, 1.0, 1.25]),
        ("wbf6_all_plus_exp7_640", ["crop", "yolov12s", "agwd_sdice", "yolov8s", "physot", "exp7_640"], [1.3, 1.2, 1.1, 1.0, 1.0, 1.25]),
    ]
    for cname, mems, ws in combos_640:
        if any(m not in seed_preds[0] for m in mems):
            continue
        per_seed = []
        for seed in (0, 1, 2):
            if len(mems) == 1:
                preds = seed_preds[seed][mems[0]]
            else:
                preds = wbf_fuse_batch([seed_preds[seed][m] for m in mems], shapes, ws, wbf_fn=wbf_fn)
            per_seed.append(eval_both(preds, records, shapes, env))
        u50 = [r["common_ultralytics101"]["mAP50"] for r in per_seed]
        u95 = [r["common_ultralytics101"]["mAP50_95"] for r in per_seed]
        p50 = [r["public_wbf11_code_raw001"]["mAP50"] for r in per_seed]
        p95 = [r["public_wbf11_code_raw001"]["mAP50_95"] for r in per_seed]
        pnms = [r["public_nms101_code"]["mAP50"] for r in per_seed]
        summary_entry = {
            "members": mems,
            "weights": ws,
            "common_ultralytics101_mAP50_mean": statistics.mean(u50),
            "common_ultralytics101_mAP50_std": statistics.stdev(u50),
            "common_ultralytics101_mAP50_95_mean": statistics.mean(u95),
            "common_ultralytics101_mAP50_95_std": statistics.stdev(u95),
            "public_nms101_mAP50_mean": statistics.mean(pnms),
            "public_nms101_mAP50_std": statistics.stdev(pnms),
            "public_wbf11_mAP50_mean": statistics.mean(p50),
            "public_wbf11_mAP50_std": statistics.stdev(p50),
            "public_wbf11_mAP50_95_mean": statistics.mean(p95),
            "public_wbf11_mAP50_95_std": statistics.stdev(p95),
            "seeds": per_seed,
        }
        output["track_640_3seed_wbf"][cname] = summary_entry
        print(
            f"[640-3seed] {cname:26s} | Ultra mAP50={summary_entry['common_ultralytics101_mAP50_mean']*100:.2f}+/-{summary_entry['common_ultralytics101_mAP50_std']*100:.2f} "
            f"| mAP50:95={summary_entry['common_ultralytics101_mAP50_95_mean']*100:.2f} "
            f"| Info26-WBF11={summary_entry['public_wbf11_mAP50_mean']*100:.2f}+/-{summary_entry['public_wbf11_mAP50_std']*100:.2f}",
            flush=True,
        )

    # ---- TRACK 2: 1280-trained champions tested on info17/test ----
    champions = [
        ("m1_yolo12", ROOT / "ensemble_champions/1_yolov12s_champion.pt"),
        ("m2_crop", ROOT / "ensemble_champions/2_yolo26s_crop_specialist.pt"),
        ("m3_agwd", ROOT / "ensemble_champions/3_yolo26s_agwd_sdice.pt"),
        ("m4_physot", ROOT / "ensemble_champions/4_physot_yolo26s.pt"),
        ("exp7_1280", Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen/runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt")),
    ]
    for imgsz in (640, 1280):
        res_key = f"imgsz_{imgsz}"
        output["track_1280_champions_on_info17"][res_key] = {}
        cached = {}
        for mid, path in champions:
            if not path.exists():
                print(f"Skipping missing {mid}: {path}", flush=True)
                continue
            t0 = time.time()
            m = YOLO(str(path))
            preds = [
                m.predict(im, imgsz=imgsz, rect=False, batch=1, half=False, conf=0.001, iou=0.45, max_det=1000, device="cpu", verbose=False, save=False, augment=False)[0].boxes.data.detach().cpu().tolist()
                for im in images
            ]
            cached[mid] = preds
            ev = eval_both(preds, records, shapes, env)
            output["track_1280_champions_on_info17"][res_key][mid] = ev
            print(
                f"[1280-champ@{imgsz}] {mid:12s} ({time.time()-t0:.1f}s) "
                f"| Ultra mAP50={ev['common_ultralytics101']['mAP50']*100:.2f} "
                f"| mAP50:95={ev['common_ultralytics101']['mAP50_95']*100:.2f} "
                f"| Info26-WBF11={ev['public_wbf11_code_raw001']['mAP50']*100:.2f}",
                flush=True,
            )
            del m

        ens_1280 = [
            ("wbf2_m1_m2", ["m1_yolo12", "m2_crop"], [1.3, 1.2]),
            ("wbf4_m1_to_m4", ["m1_yolo12", "m2_crop", "m3_agwd", "m4_physot"], [1.3, 1.2, 1.1, 1.0]),
            ("wbf5_plus_exp7", ["m1_yolo12", "m2_crop", "m3_agwd", "m4_physot", "exp7_1280"], [1.3, 1.2, 1.1, 1.0, 1.25]),
            ("wbf3_m1_m2_m3", ["m1_yolo12", "m2_crop", "m3_agwd"], [1.3, 1.2, 1.1]),
        ]
        for ename, mems, ws in ens_1280:
            if any(m not in cached for m in mems):
                continue
            fused = wbf_fuse_batch([cached[m] for m in mems], shapes, ws, wbf_fn=wbf_fn)
            ev = eval_both(fused, records, shapes, env)
            output["track_1280_champions_on_info17"][res_key][ename] = ev
            print(
                f"[1280-champ@{imgsz}] {ename:16s} "
                f"| Ultra mAP50={ev['common_ultralytics101']['mAP50']*100:.2f} "
                f"| mAP50:95={ev['common_ultralytics101']['mAP50_95']*100:.2f} "
                f"| Info26-WBF11={ev['public_wbf11_code_raw001']['mAP50']*100:.2f}",
                flush=True,
            )

    out_path = ROOT / "results/info17_ensemble_cpu_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved audit results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
