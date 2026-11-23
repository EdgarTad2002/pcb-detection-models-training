#!/usr/bin/env python3
"""Evaluate Information 2026 voting strategies on our 1280-trained champions tested on info17/test."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_info17_ensemble_cpu import eval_both, load_wbf, register_modules, wbf_fuse_batch  # noqa: E402
from tools.eval_voting_on_our_models import cluster_boxes_image, fuse_clusters_image  # noqa: E402
from tools.evaluate_info17_protocol import load_metric_functions  # noqa: E402


def main():
    t0_all = time.time()
    register_modules()
    env = load_metric_functions()
    wbf_fn = load_wbf()

    data_root = ROOT / "datasets/info17-original-fourclass-v1/test"
    images = sorted((data_root / "images").glob("*"))
    shapes = []
    records = []
    for p in images:
        im = cv2.imread(str(p))
        shapes.append([im.shape[0], im.shape[1]])
        lbl = data_root / "labels" / (p.stem + ".txt")
        rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()]
        records.append({"name": p.name, "image": str(p), "label": str(lbl), "boxes_cxywh": rows})

    champions = [
        ("m1_yolo12", ROOT / "ensemble_champions/1_yolov12s_champion.pt"),
        ("m2_crop", ROOT / "ensemble_champions/2_yolo26s_crop_specialist.pt"),
        ("m3_agwd", ROOT / "ensemble_champions/3_yolo26s_agwd_sdice.pt"),
        ("m4_physot", ROOT / "ensemble_champions/4_physot_yolo26s.pt"),
        ("exp7_1280", Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen/runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt")),
    ]

    sets = {
        "4champ_M1_to_M4": (["m1_yolo12", "m2_crop", "m3_agwd", "m4_physot"], {"m1_yolo12": 1.3, "m2_crop": 1.2, "m3_agwd": 1.1, "m4_physot": 1.0}),
        "5champ_plus_exp7": (["m1_yolo12", "m2_crop", "m3_agwd", "m4_physot", "exp7_1280"], {"m1_yolo12": 1.3, "m2_crop": 1.2, "m3_agwd": 1.1, "m4_physot": 1.0, "exp7_1280": 1.25}),
    }

    results = {}
    for imgsz in (640, 1280):
        res_key = f"imgsz_{imgsz}"
        results[res_key] = {}
        cached = {}
        for mid, path in champions:
            if not path.exists():
                print(f"Skipping missing {mid}: {path}", flush=True)
                continue
            t0 = time.time()
            m = YOLO(str(path))
            cached[mid] = [
                m.predict(
                    im, imgsz=imgsz, rect=False, batch=1, half=False,
                    conf=0.001, iou=0.45, max_det=1000, device="cpu",
                    verbose=False, save=False, augment=False,
                )[0].boxes.data.detach().cpu().tolist()
                for im in images
            ]
            del m
            print(f"[predict @{imgsz}] {mid:12s} done in {time.time() - t0:.1f}s", flush=True)

        for sname, (mems, wdict) in sets.items():
            if any(m not in cached for m in mems):
                continue
            # WBF baseline for direct side-by-side comparison
            ws = [wdict[m] for m in mems]
            wbf_preds = wbf_fuse_batch([cached[m] for m in mems], shapes, ws, wbf_fn=wbf_fn)
            wbf_ev = eval_both(wbf_preds, records, shapes, env)
            results[res_key][f"{sname}__WBF"] = wbf_ev
            print(
                f"[{res_key}] {sname:18s} WBF                        | "
                f"Ultra={wbf_ev['common_ultralytics101']['mAP50']*100:.2f} "
                f"(50:95={wbf_ev['common_ultralytics101']['mAP50_95']*100:.2f}) | "
                f"WBF11={wbf_ev['public_wbf11_code_raw001']['mAP50']*100:.2f} "
                f"NMS={wbf_ev['public_nms101_code']['mAP50']*100:.2f}",
                flush=True,
            )

            clusters = [
                cluster_boxes_image({m: cached[m][i] for m in mems}, iou_thr=0.5, conf_thr=0.001)
                for i in range(len(records))
            ]
            for strategy in ("affirmative", "consensus", "strict_majority", "unanimous"):
                for use_w, wlabel in ((False, "unweighted"), (True, "weighted")):
                    fused = [
                        fuse_clusters_image(
                            clusters[i],
                            num_models=len(mems),
                            strategy=strategy,
                            unique_model_votes=True,
                            weights=wdict if use_w else None,
                        )
                        for i in range(len(records))
                    ]
                    ev = eval_both(fused, records, shapes, env)
                    key = f"{sname}__{strategy}__{wlabel}"
                    results[res_key][key] = ev
                    print(
                        f"[{res_key}] {sname:18s} {strategy:15s} {wlabel:10s} | "
                        f"Ultra={ev['common_ultralytics101']['mAP50']*100:.2f} "
                        f"(50:95={ev['common_ultralytics101']['mAP50_95']*100:.2f}) | "
                        f"WBF11={ev['public_wbf11_code_raw001']['mAP50']*100:.2f} "
                        f"NMS={ev['public_nms101_code']['mAP50']*100:.2f}",
                        flush=True,
                    )

    out = ROOT / "results/voting_1280_on_info17.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Completed in {time.time() - t0_all:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
