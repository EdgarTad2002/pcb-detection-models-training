#!/usr/bin/env python3
"""Audit Information 17 data and evaluate frozen checkpoints without training.

Primary results use one Ultralytics protocol for every model. Public-code metric
views are diagnostics, not certificates of equivalence to the published tables.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import typing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
AUTHOR_NAMES = ["IC", "Capacitor", "Connector", "Electrolytic Capacitor"]
TO_AUTHOR = {0: 1, 1: 2, 2: 3, 3: 0}
COMMIT = "21044102c2d73e516d45e17a7fc06e894e841104"
AUTHOR_SOURCE = ROOT / "paper/references/info17_author_source"
PROFILES = {
    "common": {"conf": .001, "iou": .5, "max_det": 1000},
    # nms.py explicitly sets conf/max_det, and leaves predict IoU at default .7.
    "public_predict": {"conf": .001, "iou": .7, "max_det": 300},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def checked_author_source():
    manifest = json.loads((AUTHOR_SOURCE / "source_manifest.json").read_text())
    if manifest["commit"] != COMMIT:
        raise ValueError("Wrong author source revision")
    for entry in manifest["files"]:
        if sha(AUTHOR_SOURCE / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Modified author source: {entry['path']}")
    return manifest


def load_metric_functions():
    """Execute only audited metric function definitions, with explicit globals.

    The upstream weighted evaluator has broken package import paths. Avoiding
    module execution also prevents loading external training code or globals.
    """
    import numpy as np
    import torch
    checked_author_source()
    env = dict(np=np, torch=torch, defaultdict=defaultdict,
               CLASS_NAMES=AUTHOR_NAMES, Dict=typing.Dict, List=typing.List,
               Optional=typing.Optional, Tuple=typing.Tuple)
    aliases = []
    if not hasattr(np, "trapz"):
        # NumPy 2.4 removed the old name; same trapezoidal operation.
        class NumpyCompat:
            def __getattr__(self, name):
                return np.trapezoid if name == "trapz" else getattr(np, name)
        env["np"] = NumpyCompat()
        aliases.append("np.trapz -> np.trapezoid (same integration operation)")
    selection = {
        "ensemble/nms.py": {"box_iou", "compute_ap", "ap_per_class", "process_batch", "evaluate_predictions"},
        "ensemble/weighted_voting_methods/box_utils.py": {"compute_iou"},
        "ensemble/weighted_voting_methods/evaluate.py": {"compute_map"},
    }
    for rel, names in selection.items():
        source = ast.parse((AUTHOR_SOURCE / rel).read_text())
        nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError(f"Missing upstream metric functions: {rel}")
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(AUTHOR_SOURCE / rel), "exec"), env)
    env["compatibility_aliases"] = aliases
    return env


def checkpoint_entries(plan, frozen):
    jobs = {job["key"]: job for job in plan["jobs"]}
    entries = []
    for entry in frozen["checkpoints"]:
        job = jobs[entry["job"]]
        if sha(entry["weights"]) != entry["sha256"]:
            raise ValueError(f"Checkpoint changed: {entry['job']}")
        entries.append({**entry, "model": job["model"], "seed": job["seed"]})
    expected = {(m, seed) for m in plan["config"]["models"] for seed in (0, 1, 2)}
    if len(entries) != len(expected) or {(x["model"], x["seed"]) for x in entries} != expected:
        raise ValueError("Expected all six methods and all three frozen seeds")
    return entries


def audit_dataset(data_root):
    from PIL import Image
    from tools.prepare_info17_final import convert_label
    manifest_path = data_root / "dataset_manifest.json"
    original = json.loads(manifest_path.read_text())
    splits = {}
    image_hashes = defaultdict(set)
    for split, expected_count in (("train", 543), ("valid", 77), ("test", 41)):
        records = []
        expected = [r for r in original["records"] if r["split"] == split]
        for source in sorted(expected, key=lambda r: Path(r["image"]).name):
            image = data_root / split / "images" / Path(source["image"]).name
            label = data_root / split / "labels" / (image.stem + ".txt")
            for path, recorded in ((image, "image_sha256"), (label, "derived_label_sha256")):
                if sha(path) != source[recorded]:
                    raise ValueError(f"Derived data changed: {path}")
            if sha(source["image"]) != source["image_sha256"] or sha(source["label"]) != source["source_label_sha256"]:
                raise ValueError("Original source data changed")
            rows, polygons = convert_label(Path(source["label"]).read_text())
            if label.read_text().splitlines() != rows:
                raise ValueError("Derived labels do not preserve original four-class boxes")
            with Image.open(image) as im:
                if im.size != (640, 640):
                    raise ValueError(f"Unexpected image dimensions: {image}")
            boxes = [list(map(float, row.split())) for row in rows]
            records.append({"name": image.name, "image": str(image), "label": str(label),
                            "image_sha256": source["image_sha256"], "label_sha256": source["derived_label_sha256"],
                            "boxes_cxywh": boxes, "polygon_lines_converted": polygons})
            image_hashes[source["image_sha256"]].add(split)
        if len(records) != expected_count:
            raise ValueError(f"Wrong {split} image count")
        actual_names = {p.name for p in (data_root / split / "images").iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if actual_names != {r["name"] for r in records}:
            raise ValueError("Extra/missing images relative to frozen dataset manifest")
        counts = Counter(NAMES[int(b[0])] for r in records for b in r["boxes_cxywh"])
        splits[split] = {"images": len(records), "box_counts": dict(counts), "records": records}
    overlaps = {h: sorted(s) for h, s in image_hashes.items() if len(s) > 1}
    return {"source_manifest_sha256": sha(manifest_path), "metadata": original.get("original_metadata"),
            "splits": splits, "identical_image_hash_cross_split_overlaps": overlaps,
            "author_image_annotation_identity": "unverified: author repository supplies no image list, labels or content hashes",
            "known_validation_discrepancy": {"our_IC": 1032, "paper_table7_IC": 1026,
                                             "our_total": 5542, "paper_class_counts_total": 5536},
            "independent_unseen_external_benchmark": False}


def author_inputs(predictions, records):
    """Map class semantics, never assume the authors use our numeric class IDs."""
    preds, targets, normalized_preds, normalized_gt = [], [], {}, {}
    for raw, record in zip(predictions, records, strict=True):
        name = record["name"]
        normalized_preds[name] = []
        normalized_gt[name] = []
        for row in raw:
            x1, y1, x2, y2, confidence, cls = row
            cls = TO_AUTHOR[int(cls)]
            preds.append(dict(image_id=name, class_id=cls, box_xyxy=[x1, y1, x2, y2], score=confidence))
            normalized_preds[name].append({"class": cls, "confidence": confidence,
                                                "box_norm": [x1 / 640, y1 / 640, x2 / 640, y2 / 640]})
        for cls, cx, cy, width, height in record["boxes_cxywh"]:
            cls = TO_AUTHOR[int(cls)]
            box = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
            targets.append(dict(image_id=name, class_id=cls, box_xyxy=[x * 640 for x in box]))
            normalized_gt[name].append({"class": cls, "box_norm": box})
    return preds, targets, normalized_preds, normalized_gt


def aggregate(rows):
    summary = []
    keys = sorted({(r["view"], r["split"], r["model"]) for r in rows})
    for view, split, model in keys:
        group = [r for r in rows if (r["view"], r["split"], r["model"]) == (view, split, model)]
        if sorted(r["seed"] for r in group) != [0, 1, 2]:
            raise ValueError("Incomplete seed group")
        values = [r["mAP50"] for r in group]
        result = dict(view=view, split=split, model=model, n=3,
                      mAP50_mean=statistics.mean(values), mAP50_std=statistics.stdev(values))
        if all("mAP50_95" in r for r in group):
            result.update(mAP50_95_mean=statistics.mean(r["mAP50_95"] for r in group),
                          mAP50_95_std=statistics.stdev(r["mAP50_95"] for r in group))
        summary.append(result)
    return summary


def prepare(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing overwrite: {out}")
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    frozen_path = plan_path.parent / "test_checkpoint_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    author = checked_author_source()
    dataset = audit_dataset(args.data_root.resolve())
    entries = checkpoint_entries(plan, frozen)
    out.mkdir(parents=True)
    manifest = {"status": "prepared", "schema_version": 1, "checkpoints": entries,
                "dataset": dataset, "profiles": PROFILES, "plan": str(plan_path), "plan_sha256": sha(plan_path),
                "checkpoint_manifest_sha256": sha(frozen_path), "author_source": author,
                "tool_sha256": sha(Path(__file__)), "common_scorer_sha256": sha(ROOT / "tools/evaluate_champion_audit.py"),
                "reporting": "Primary: common Ultralytics protocol on test, all seeds. Public metric views are diagnostics on both splits. No threshold/seed/model selection on these new outputs. Published-score equivalence remains unverified."}
    write(out / "manifest.json", manifest)
    write(out / "status.json", {"state": "prepared", "checkpoint_count": len(entries)})
    print(f"Prepared: {out} (18 checkpoints, no training)", flush=True)


def run(args):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    from tools.benchmark_speed import register_modules
    from tools.evaluate_champion_audit import score
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if (out / "summary.json").exists():
        raise FileExistsError("Completed evaluation already exists")
    if sha(Path(__file__)) != manifest["tool_sha256"] or sha(ROOT / "tools/evaluate_champion_audit.py") != manifest["common_scorer_sha256"]:
        raise ValueError("Evaluation code changed after preparation")
    checked_author_source()
    env = load_metric_functions()
    register_modules()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_hashes = {x["weights"]: x["sha256"] for x in manifest["checkpoints"]}
    for path, h in checkpoint_hashes.items():
        if sha(path) != h:
            raise ValueError(f"Frozen checkpoint changed: {path}")
    for split in manifest["dataset"]["splits"].values():
        for record in split["records"]:
            if sha(record["image"]) != record["image_sha256"] or sha(record["label"]) != record["label_sha256"]:
                raise ValueError("Dataset changed after preparation")
    write(out / "environment.json", {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
          "numpy": np.__version__, "python": sys.version, "metric_compatibility_aliases": env["compatibility_aliases"],
          "device": args.device, "speed_comparison": "not measured; no comparison to Apple M4 speed"})
    rows = []
    for entry in manifest["checkpoints"]:
        model = YOLO(entry["weights"])
        if len(model.names) != 4 or [model.names[i] for i in range(4)] != NAMES:
            raise ValueError("Unexpected checkpoint class mapping")
        for split in ("valid", "test"):
            records = manifest["dataset"]["splits"][split]["records"]
            images = [cv2.imread(r["image"]) for r in records]
            if any(im is None or im.shape[:2] != (640, 640) for im in images):
                raise ValueError("Image decode/shape error")
            for profile, settings in manifest["profiles"].items():
                key = f"{entry['job']}_{split}_{profile}"
                cache = out / "predictions" / (key + ".json")
                # Refuse partial-run reuse; interrupted evaluations get a fresh output namespace.
                if cache.exists():
                    raise FileExistsError(cache)
                predictions = []
                for image in images:
                    result = model.predict(image, imgsz=640, rect=False, batch=1, half=False,
                                           device=args.device, verbose=False, augment=False, **settings)[0]
                    predictions.append(result.boxes.data.detach().cpu().tolist())
                write(cache, {"checkpoint_sha256": entry["sha256"], "profile": settings,
                              "image_names": [r["name"] for r in records], "predictions": predictions})
                base = dict(job=entry["job"], model=entry["model"], seed=entry["seed"], split=split)
                if profile == "common":
                    metrics = score(predictions, records, [[640, 640]] * len(records), {"recall_conf": .25})
                    rows.append({**base, "view": "common_ultralytics101", **metrics})
                else:
                    p, g, pn, gn = author_inputs(predictions, records)
                    metric = env["evaluate_predictions"](p, g)
                    rows.append({**base, "view": "public_nms101_code", **metric})
                    metric = env["compute_map"](pn, gn, AUTHOR_NAMES)
                    rows.append({**base, "view": "public_wbf11_code_raw001", **metric})
                    filtered = {name: [p for p in pred if p["confidence"] >= .25] for name, pred in pn.items()}
                    metric = env["compute_map"](filtered, gn, AUTHOR_NAMES)
                    rows.append({**base, "view": "public_wbf11_code_conf025_sensitivity", **metric})
                print(f"Evaluated {key}", flush=True)
                write(out / "metrics.json", rows)
                write(out / "status.json", {"state": "running", "last_completed": key, "metric_rows": len(rows)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = aggregate(rows)
    write(out / "summary.json", summary)
    keys = ["view", "split", "model", "n", "mAP50_mean", "mAP50_std", "mAP50_95_mean", "mAP50_95_std"]
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    for path, h in checkpoint_hashes.items():
        if sha(path) != h:
            raise ValueError("Checkpoint changed during evaluation")
    write(out / "status.json", {"state": "completed", "checkpoint_count": 18, "metric_rows": len(rows),
          "summary_sha256": sha(out / "summary.json"), "published_protocol_equivalence": False})
    print(f"Completed {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--plan", type=Path, default=ROOT / "controlled_experiments/751837c92ab4/plan.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets/info17-original-fourclass-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
