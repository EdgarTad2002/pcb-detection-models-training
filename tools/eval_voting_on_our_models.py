#!/usr/bin/env python3
"""Evaluate Information 2026 voting strategies (Affirmative, Consensus, Unanimous,
and Author Weighted Voting) on both:
  Track A: Our 640-trained 4 best models (3 seeds) on info17/test (41 images)
  Track B: Our 1280-trained 4 champions (+ optional Exp7) on native-1280/test (44 images)
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_champion_audit import score  # noqa: E402
from tools.evaluate_info17_protocol import (  # noqa: E402
    AUTHOR_NAMES,
    author_inputs,
    load_metric_functions,
)


def cluster_boxes_image(
    member_boxes: dict[str, list[list[float]]],
    iou_thr: float = 0.5,
    conf_thr: float = 0.001,
) -> dict[int, list[list[tuple[str, np.ndarray, float]]]]:
    """Cluster overlapping same-class detections once per image using vectorized NumPy IoU."""
    by_cls = defaultdict(list)
    for mname, boxes in member_boxes.items():
        for row in boxes:
            x1, y1, x2, y2, conf, cls = row[:6]
            if conf >= conf_thr:
                by_cls[int(cls)].append((mname, [float(x1), float(y1), float(x2), float(y2)], float(conf)))

    clusters_by_cls: dict[int, list[list[tuple[str, np.ndarray, float]]]] = {}
    for cls_id, dets in by_cls.items():
        dets.sort(key=lambda x: x[2], reverse=True)
        n = len(dets)
        coords = np.array([d[1] for d in dets], dtype=np.float32)  # (n, 4)
        areas = np.maximum(0.0, coords[:, 2] - coords[:, 0]) * np.maximum(0.0, coords[:, 3] - coords[:, 1])
        x1 = np.maximum(coords[:, 0:1], coords[:, 0:1].T)
        y1 = np.maximum(coords[:, 1:2], coords[:, 1:2].T)
        x2 = np.minimum(coords[:, 2:3], coords[:, 2:3].T)
        y2 = np.minimum(coords[:, 3:4], coords[:, 3:4].T)
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        union = areas[:, None] + areas[None, :] - inter
        adj = (inter > 0.0) & (inter > iou_thr * union)

        used = np.zeros(n, dtype=bool)
        groups = []
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            group_idx = [i]
            # Exact Info26 greedy matching: merge any unused j overlapping any box already in group
            k = 0
            while k < len(group_idx):
                cur = group_idx[k]
                nbrs = np.flatnonzero(adj[cur] & (~used))
                if len(nbrs):
                    used[nbrs] = True
                    group_idx.extend(nbrs.tolist())
                k += 1
            groups.append([(dets[idx][0], coords[idx], dets[idx][2]) for idx in group_idx])
        clusters_by_cls[cls_id] = groups
    return clusters_by_cls


def fuse_clusters_image(
    clusters_by_cls: dict[int, list[list[tuple[str, np.ndarray, float]]]],
    num_models: int,
    strategy: str,
    unique_model_votes: bool = True,
    weights: dict[str, float] | None = None,
    max_det: int = 1000,
) -> list[list[float]]:
    threshold = 1 if strategy == "affirmative" else (
        math.ceil(num_models / 2) if strategy == "consensus" else (
            (num_models // 2 + 1) if strategy == "strict_majority" else num_models
        )
    )
    out = []
    for cls_id, groups in clusters_by_cls.items():
        for group in groups:
            votes = len({g[0] for g in group}) if unique_model_votes else len(group)
            if votes < threshold:
                continue
            if weights is None:
                tot_c = sum(g[2] for g in group)
                box = sum(g[1] * (g[2] / tot_c) for g in group)
                conf = tot_c / len(group)
            else:
                w_sum = sum(weights[g[0]] for g in group)
                wc_sum = sum(weights[g[0]] * g[2] for g in group)
                box = sum(g[1] * (weights[g[0]] * g[2] / wc_sum) for g in group)
                conf = wc_sum / w_sum
            out.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf), float(cls_id)])
    out.sort(key=lambda r: r[4], reverse=True)
    return out[:max_det]


def evaluate_640_track(env):
    data_root = ROOT / "datasets/info17-original-fourclass-v1/test"
    img_paths = sorted((data_root / "images").glob("*"))
    shapes = [[640, 640] for _ in img_paths]
    records = []
    for p in img_paths:
        lbl = data_root / "labels" / (p.stem + ".txt")
        rows = [list(map(float, ln.split())) for ln in lbl.read_text().splitlines() if ln.strip()]
        records.append({"name": p.name, "image": str(p), "label": str(lbl), "boxes_cxywh": rows})

    pred_dir = ROOT / "results/info17_protocol_v1_20261031/predictions"
    jobs = {
        "crop": "info17_final_v1_56f756d1_budget_yolo26s_crop_specialist_mixed",
        "yolov12s": "info17_final_v1_56f756d1_main_yolov12s_paper",
        "agwd_sdice": "info17_final_v1_56f756d1_main_yolo26s_agwd_sdice_paper",
        "physot": "info17_final_v1_56f756d1_main_physot_yolo26s_paper",
        "yolov8s": "info17_final_v1_56f756d1_main_yolov8s_paper",
    }

    sets = {
        "4best_crop_v12_agwd_v8": (["crop", "yolov12s", "agwd_sdice", "yolov8s"], {"crop": 1.3, "yolov12s": 1.2, "agwd_sdice": 1.1, "yolov8s": 1.0}),
        "4paper_crop_v12_agwd_physot": (["crop", "yolov12s", "agwd_sdice", "physot"], {"crop": 1.3, "yolov12s": 1.2, "agwd_sdice": 1.1, "physot": 1.0}),
    }

    results = {}
    for profile in ("common", "public_predict"):
        seed_cache = {s: {} for s in (0, 1, 2)}
        for s in (0, 1, 2):
            for m, pref in jobs.items():
                d = json.loads((pred_dir / f"{pref}_s{s}_test_{profile}.json").read_text())
                seed_cache[s][m] = d["predictions"]

        for sname, (mems, wdict) in sets.items():
            clusters_per_seed = {
                s: [cluster_boxes_image({m: seed_cache[s][m][i] for m in mems}, iou_thr=0.5, conf_thr=0.001) for i in range(len(records))]
                for s in (0, 1, 2)
            }
            for strategy in ("affirmative", "consensus", "strict_majority", "unanimous"):
                for use_w, wlabel in ((False, "unweighted"), (True, "weighted")):
                    key = f"{profile}__{sname}__{strategy}__{wlabel}"
                    per_seed = []
                    for s in (0, 1, 2):
                        fused = [
                            fuse_clusters_image(
                                clusters_per_seed[s][i],
                                num_models=len(mems),
                                strategy=strategy,
                                unique_model_votes=True,
                                weights=wdict if use_w else None,
                            )
                            for i in range(len(records))
                        ]
                        u = score(fused, records, shapes, {"recall_conf": 0.25})
                        p, g, pn, gn = author_inputs(fused, records)
                        pnms = env["evaluate_predictions"](p, g)
                        pwbf = env["compute_map"](pn, gn, AUTHOR_NAMES)
                        per_seed.append({
                            "ultra_mAP50": u["mAP50"],
                            "ultra_mAP50_95": u["mAP50_95"],
                            "ultra_per_class": {k: v["ap50"] for k, v in u["per_class"].items()},
                            "info26_nms_mAP50": float(pnms["mAP50"]),
                            "info26_wbf11_mAP50": float(pwbf["mAP50"]),
                        })
                    entry = {
                        "ultra_mAP50_mean": statistics.mean(x["ultra_mAP50"] for x in per_seed),
                        "ultra_mAP50_std": statistics.stdev(x["ultra_mAP50"] for x in per_seed),
                        "ultra_mAP50_95_mean": statistics.mean(x["ultra_mAP50_95"] for x in per_seed),
                        "ultra_mAP50_95_std": statistics.stdev(x["ultra_mAP50_95"] for x in per_seed),
                        "info26_nms_mean": statistics.mean(x["info26_nms_mAP50"] for x in per_seed),
                        "info26_wbf11_mean": statistics.mean(x["info26_wbf11_mAP50"] for x in per_seed),
                        "seeds": per_seed,
                    }
                    results[key] = entry
                    print(
                        f"[640 {profile:14s}] {sname[:14]:14s} {strategy:15s} {wlabel:10s} | "
                        f"Ultra mAP50={entry['ultra_mAP50_mean']*100:.2f}+/-{entry['ultra_mAP50_std']*100:.2f} "
                        f"mAP50:95={entry['ultra_mAP50_95_mean']*100:.2f} | "
                        f"Info26-NMS={entry['info26_nms_mean']*100:.2f} Info26-WBF11={entry['info26_wbf11_mean']*100:.2f}",
                        flush=True,
                    )
    return results


def evaluate_1280_track():
    manifest = json.loads((ROOT / "results/champion_audit_v1_20261030/manifest.json").read_text())
    records = manifest["records"]
    pred_dir = ROOT / "results/champion_audit_v1_20261030/predictions"
    m_preds = {}
    shapes = None
    for mid in ("yolo12", "crop", "agwd_sdice", "physot"):
        d = json.loads((pred_dir / f"{mid}.json").read_text())
        m_preds[mid] = d["predictions"]
        if shapes is None:
            shapes = d["shapes"]
    exp7_path = ROOT / "results/guide_on_native1280_test_v1_20261109/predictions/historical_exp7.json"
    if exp7_path.exists():
        m_preds["exp7"] = json.loads(exp7_path.read_text())["predictions"]

    sets = {
        "4champ_M1_to_M4": (["yolo12", "crop", "agwd_sdice", "physot"], {"yolo12": 1.3, "crop": 1.2, "agwd_sdice": 1.1, "physot": 1.0}),
    }
    if "exp7" in m_preds:
        sets["5champ_plus_exp7"] = (["yolo12", "crop", "agwd_sdice", "physot", "exp7"], {"yolo12": 1.3, "crop": 1.2, "agwd_sdice": 1.1, "physot": 1.0, "exp7": 1.25})

    results = {}
    for sname, (mems, wdict) in sets.items():
        clusters = [cluster_boxes_image({m: m_preds[m][i] for m in mems}, iou_thr=0.5, conf_thr=0.001) for i in range(len(records))]
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
                u = score(fused, records, shapes, {"recall_conf": 0.25})
                key = f"{sname}__{strategy}__{wlabel}"
                results[key] = {
                    "mAP50": u["mAP50"],
                    "mAP50_95": u["mAP50_95"],
                    "per_class": u["per_class"],
                }
                pc = {c: round(v["ap50"] * 100, 2) for c, v in u["per_class"].items()}
                print(
                    f"[1280 native44] {sname:18s} {strategy:15s} {wlabel:10s} | "
                    f"mAP50={u['mAP50']*100:.2f} mAP50:95={u['mAP50_95']*100:.2f} | {pc}",
                    flush=True,
                )
    return results


def main():
    t0 = time.time()
    env = load_metric_functions()
    res640 = evaluate_640_track(env)
    res1280 = evaluate_1280_track()
    out = ROOT / "results/voting_ensemble_audit.json"
    out.write_text(json.dumps({"track_640_info17": res640, "track_1280_native44": res1280}, indent=2) + "\n")
    print(f"Completed in {time.time() - t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
