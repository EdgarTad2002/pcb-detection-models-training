#!/usr/bin/env python3
"""Evaluate Option B 1280-trained info17 models (single, WBF, and voting) on info17 valid & test on CPU."""
from __future__ import annotations

import argparse
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


def eval_split(split_name: str, split_dir: Path, trained: dict[str, Path], env, wbf_fn):
    images = sorted((split_dir / "images").glob("*"))
    shapes = []
    records = []
    for p in images:
        im = cv2.imread(str(p))
        shapes.append([im.shape[0], im.shape[1]])
        lbl = split_dir / "labels" / (p.stem + ".txt")
        rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()]
        records.append({"name": p.name, "image": str(p), "label": str(lbl), "boxes_cxywh": rows})

    out = {"num_images": len(images), "singles": {}, "wbf": {}, "voting": {}}
    cached = {}
    for model_name, weight_path in trained.items():
        if not weight_path.is_file():
            print(f"[{split_name}] Missing checkpoint for {model_name}: {weight_path}", flush=True)
            continue
        t0 = time.time()
        m = YOLO(str(weight_path))
        preds = [
            m.predict(
                im, imgsz=1280, rect=False, batch=1, half=False,
                conf=0.001, iou=0.45, max_det=1000, device="cpu",
                verbose=False, save=False, augment=False,
            )[0].boxes.data.detach().cpu().tolist()
            for im in images
        ]
        del m
        cached[model_name] = preds
        ev = eval_both(preds, records, shapes, env)
        out["singles"][model_name] = ev
        u = ev["common_ultralytics101"]
        w11 = ev["public_wbf11_code_raw001"]
        print(
            f"[{split_name:5s}] single {model_name:24s} ({time.time()-t0:.1f}s) | "
            f"Ultra={u['mAP50']*100:.2f} (50:95={u['mAP50_95']*100:.2f}) | WBF11={w11['mAP50']*100:.2f}",
            flush=True,
        )

    combos = [
        ("wbf4_champs", ["yolov12s", "yolo26s_crop_specialist", "yolo26s_agwd_sdice", "physot_yolo26s"], [1.3, 1.2, 1.1, 1.0]),
        ("wbf4_best_v9", ["yolov12s", "yolo26s_crop_specialist", "yolo26s_agwd_sdice", "yolov9s"], [1.3, 1.2, 1.1, 1.15]),
        ("wbf4_v12_agwd_v9_v8", ["yolov12s", "yolo26s_agwd_sdice", "yolov9s", "yolov8s"], [1.2, 1.1, 1.2, 1.0]),
        ("wbf5_champs_v9", ["yolov12s", "yolo26s_crop_specialist", "yolo26s_agwd_sdice", "physot_yolo26s", "yolov9s"], [1.3, 1.2, 1.1, 1.0, 1.15]),
        ("wbf6_all", ["yolov12s", "yolo26s_crop_specialist", "yolo26s_agwd_sdice", "physot_yolo26s", "yolov9s", "yolov8s"], [1.3, 1.2, 1.1, 1.0, 1.15, 1.0]),
    ]
    for cname, mems, ws in combos:
        if any(m not in cached for m in mems):
            continue
        fused = wbf_fuse_batch([cached[m] for m in mems], shapes, ws, wbf_fn=wbf_fn)
        ev = eval_both(fused, records, shapes, env)
        out["wbf"][cname] = ev
        u = ev["common_ultralytics101"]
        w11 = ev["public_wbf11_code_raw001"]
        print(
            f"[{split_name:5s}] WBF    {cname:24s} | "
            f"Ultra={u['mAP50']*100:.2f} (50:95={u['mAP50_95']*100:.2f}) | WBF11={w11['mAP50']*100:.2f}",
            flush=True,
        )

    for cname, mems, ws in combos[:2]:
        if any(m not in cached for m in mems):
            continue
        wdict = dict(zip(mems, ws))
        clusters = [cluster_boxes_image({m: cached[m][i] for m in mems}, iou_thr=0.5, conf_thr=0.001) for i in range(len(records))]
        for strat in ("affirmative", "consensus", "unanimous"):
            vfused = [fuse_clusters_image(clusters[i], len(mems), strat, True, wdict) for i in range(len(records))]
            ev = eval_both(vfused, records, shapes, env)
            key = f"{cname}__{strat}"
            out["voting"][key] = ev
            u = ev["common_ultralytics101"]
            print(
                f"[{split_name:5s}] vote   {key:24s} | "
                f"Ultra={u['mAP50']*100:.2f} (50:95={u['mAP50_95']*100:.2f})",
                flush=True,
            )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    register_modules()
    env = load_metric_functions()
    wbf_fn = load_wbf()

    plan = json.loads(args.plan.read_text())
    trained = {}
    for job in plan["jobs"]:
        ckpt = Path(plan["root"]) / "runs" / job["key"] / "pcb-filtered/weights/best.pt"
        trained[job["model"]] = ckpt

    ds_root = ROOT / "datasets/info17-original-fourclass-v1"
    results = {
        "plan": str(args.plan),
        "checkpoints": {k: str(v) for k, v in trained.items()},
        "test": eval_split("test", ds_root / "test", trained, env, wbf_fn),
        "valid": eval_split("valid", ds_root / "valid", trained, env, wbf_fn),
    }
    out_path = ROOT / "results/info17_1280_optionb_audit.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Completed Option B audit in {time.time() - t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
