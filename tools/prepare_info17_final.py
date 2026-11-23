#!/usr/bin/env python3
"""Prepare the original-label Information 17 comparison; never edit source data."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import NAMES, prepare, verify_plan, write_json

MAPPING = {2: 0, 4: 1, 7: 2, 9: 3}
EXPECTED_NAMES = {2: "Capacitor", 4: "Connector", 7: "Electrolytic Capacitor", 9: "IC"}


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_label(text):
    """Retain only four original classes; convert polygons to their tight YOLO box."""
    converted, polygons = [], []
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        cls = int(fields[0])
        if cls not in MAPPING:
            continue
        values = [float(x) for x in fields[1:]]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError(f"Invalid normalized label on line {number}")
        if len(values) == 4:
            cx, cy, w, h = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[::2], values[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            polygons.append(number)
        else:
            raise ValueError(f"Malformed detection/polygon on line {number}")
        if w <= 0 or h <= 0:
            raise ValueError(f"Degenerate box on line {number}")
        # Float round-trip precision; no clipping, merging, cleaning, or inferred boxes.
        converted.append(f"{MAPPING[cls]} " + " ".join(format(x, '.17g') for x in (cx, cy, w, h)))
    return converted, polygons


def build_original(source, output):
    from PIL import Image
    source, output = Path(source).resolve(), Path(output).resolve()
    cfg = yaml.safe_load((source / "data.yaml").read_text())
    names = cfg["names"]
    for idx, expected in EXPECTED_NAMES.items():
        actual = names[idx] if isinstance(names, list) else names.get(idx, names.get(str(idx)))
        if actual != expected:
            raise ValueError(f"Unexpected source class {idx}: {actual!r}")
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    # Validate everything before creating the new dataset.
    records, stats = [], {}
    for split in ("train", "valid", "test"):
        counts, kept, excluded = Counter(), 0, 0
        images = sorted(p for p in (source / split / "images").iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError(f"Empty source split: {split}")
        for image in images:
            label = source / split / "labels" / (image.stem + ".txt")
            rows, polygons = convert_label(label.read_text())
            with Image.open(image) as im:
                if im.size != (640, 640):
                    raise ValueError(f"Expected original 640x640 export: {image}")
            if not rows:
                excluded += 1
                continue
            kept += 1
            counts.update(int(row.split()[0]) for row in rows)
            records.append({"split": split, "image": str(image), "label": str(label),
                            "image_sha256": hash_file(image), "source_label_sha256": hash_file(label),
                            "polygon_lines_converted": polygons, "rows": rows})
        stats[split] = {"source_images": len(images), "retained_images": kept,
                        "excluded_without_target_classes": excluded,
                        "boxes": {NAMES[i]: counts[i] for i in range(4)}}
    output.mkdir(parents=True)
    for r in records:
        image = Path(r["image"])
        image_dir, label_dir = output / r["split"] / "images", output / r["split"] / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_dir / image.name)
        target = label_dir / (image.stem + ".txt")
        target.write_text("\n".join(r.pop("rows")) + "\n")
        r["derived_label_sha256"] = hash_file(target)
    (output / "data.yaml").write_text(yaml.safe_dump({"path": str(output), "nc": 4, "names": NAMES,
        "train": "train/images", "val": "valid/images", "test": "test/images"}, sort_keys=False))
    manifest = {"source": str(source), "original_metadata": cfg.get("roboflow"),
                "class_mapping": MAPPING, "stats": stats, "records": records,
                "policy": "Original four classes only; remove empty images; preserve boxes; polygons -> tight boxes."
                          " No Capacitor Jumper merge, cleaned labels, copy-paste, or external PCB checkpoints."}
    write_json(output / "dataset_manifest.json", manifest)
    print(json.dumps(stats, indent=2), flush=True)
    return manifest


def freeze_sources(root, out):
    paths = list(root.glob("*.py")) + list((root / "tools").glob("*.py"))
    paths += [root / "configs/info17_final_experiments.json", root / "sbatch/train_controlled_experiment.sh"]
    paths += [root / name for name in ("yolo26s.pt", "yolo12s.pt", "yolov8s.pt", "data/pcb_spectral_priors.json")
              if (root / name).is_file()]
    hashes = {str(p.relative_to(root)): hash_file(p) for p in paths}
    backup = out / "source_snapshot"
    for rel in hashes:
        dest = backup / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dest)
    write_json(out / "source_lock.json", hashes)


def verify_sources(root, out):
    expected = json.loads((out / "source_lock.json").read_text())
    changed = [rel for rel, digest in expected.items() if hash_file(root / rel) != digest]
    if changed:
        raise ValueError(f"Experiment code changed after preparation: {changed}")


def finalize(plan_path):
    """Freeze all validation-selected best.pt files before any test evaluation."""
    plan = json.loads(Path(plan_path).read_text())
    root, out = Path(plan["root"]), Path(plan["output"])
    verify_sources(root, out)
    verify_plan(plan)
    frozen = []
    for job in plan["jobs"]:
        run = root / "runs" / job["key"] / "pcb-filtered"
        with (run / "results.csv").open() as stream:
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(stream)]
        if len(rows) != job["epochs"] or not (out / "validation" / (job["key"] + ".json")).is_file():
            raise ValueError(f"Incomplete training/validation: {job['key']}")
        cfg = yaml.safe_load((run / "args.yaml").read_text())
        if cfg.get("seed") != job["seed"] or cfg.get("imgsz") != 640:
            raise ValueError(f"Run recipe mismatch: {job['key']}")
        weights = run / "weights/best.pt"
        frozen.append({"job": job["key"], "weights": str(weights), "sha256": hash_file(weights)})
    manifest = out / "test_checkpoint_manifest.json"
    if manifest.exists():
        raise FileExistsError("Test checkpoint manifest already frozen; refusing a repeated test selection")
    write_json(manifest, {"selection": "Ultralytics best.pt selected using validation fitness only",
                          "plan_identity": plan["identity"], "checkpoints": frozen})
    for job, entry in zip(plan["jobs"], frozen):
        verify_sources(root, out)
        if hash_file(entry["weights"]) != entry["sha256"]:
            raise ValueError("Frozen checkpoint changed")
        subprocess.run([sys.executable, str(root / "tools/controlled_experiments.py"), "evaluate",
                        "--plan", str(Path(plan_path).resolve()), "--job-key", job["key"],
                        "--split", "test", "--device", "0"], cwd=root, check=True)
    summary = []
    for split, folder in (("val", "validation"), ("test", "test")):
        for model in plan["config"]["models"]:
            jobs = [j for j in plan["jobs"] if j["model"] == model]
            results = [json.loads((out / folder / (j["key"] + ".json")).read_text()) for j in jobs]
            row = {"model": model, "split": split, "n": len(results), "seeds": [j["seed"] for j in jobs]}
            for metric in ("mAP50", "mAP50_95", "precision", "recall"):
                values = [r[metric] for r in results]
                row[metric + "_mean"] = statistics.mean(values)
                row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            row["per_class_ap50"] = {name: {"mean": statistics.mean(r["per_class_ap50"][name] for r in results),
                "std": statistics.stdev(r["per_class_ap50"][name] for r in results)} for name in NAMES}
            summary.append(row)
    write_json(out / "summary.json", summary)
    keys = [k for k in summary[0] if k not in {"seeds", "per_class_ap50"}]
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in summary)
    print(f"Final results: {out / 'summary.json'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "finalize", "verify-sources"))
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--source", default="datasets/pcb-filtered-yolov8")
    p.add_argument("--plan", type=Path)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    if args.action != "prepare":
        if args.plan is None:
            p.error("--plan required")
        if args.action == "finalize":
            finalize(args.plan)
        else:
            plan = json.loads(args.plan.read_text())
            verify_sources(Path(plan["root"]), Path(plan["output"]))
        return
    root = args.project_root.resolve()
    cfg = json.loads((root / "configs/info17_final_experiments.json").read_text())
    original = root / Path(cfg["datasets"]["paper"]).parent
    mixed = root / Path(cfg["datasets"]["mixed"]).parent
    source = Path(args.source)
    build_original(source if source.is_absolute() else root / source, original)
    subprocess.run([sys.executable, str(root / "tools/build_mixed_crop_dataset.py"),
                    "--source-dataset", str(original), "--output-dataset", str(mixed),
                    "--crop-sizes", "320", "480", "--sampling-ratio", "0.5", "--seed", "42"], check=True)
    options = argparse.Namespace(study="all", models=None, seeds=None,
                                 output="controlled_experiments", device="0")
    plan_path = prepare(options, cfg, root)
    out = plan_path.parent
    freeze_sources(root, out)
    write_json(root / "controlled_experiments/info17_final_v1_bundle.json", {"plan": str(plan_path),
               "job_count": 18, "paper_comparison": "Reproduced common protocol; published split unresolved",
               "test_is_exploratory": "These RF100 boards were used in earlier development; not an unseen external dataset."})
    if args.submit:
        plan = json.loads(plan_path.read_text())
        ids = {}
        # Each job gets an immutable sbatch script and a code-hash guard; generic submit remains unchanged.
        for job in plan["jobs"]:
            if (root / "runs" / job["key"]).exists():
                raise FileExistsError(job["key"])
        for job in plan["jobs"]:
            import shlex
            script = out / (job["key"] + ".sh")
            commands = ["#!/bin/bash", "#SBATCH --partition=research", "#SBATCH --mem=48G",
                        "#SBATCH --cpus-per-task=8", "#SBATCH --gres=gpu:1", "#SBATCH --time=12:00:00",
                        f"#SBATCH --job-name={job['key']}", f"#SBATCH --output={out}/slurm_%j.out",
                        "set -euo pipefail", f"cd {shlex.quote(str(root))}",
                        "source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh",
                        "conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo",
                        "export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics",
                        shlex.join([sys.executable, str(root / "tools/prepare_info17_final.py"), "verify-sources", "--plan", str(plan_path)]),
                        shlex.join([sys.executable, str(root / "tools/controlled_experiments.py"), "run", "--plan", str(plan_path), "--job-key", job["key"], "--device", "0"])]
            script.write_text("\n".join(commands) + "\n")
            result = subprocess.run(["sbatch", "--parsable", str(script)], check=True, capture_output=True, text=True)
            ids[job["key"]] = result.stdout.strip().split(";")[0]
            write_json(out / "submission.json", ids)
            print("SUBMITTED", job["key"], ids[job["key"]], flush=True)
        script = out / "finalize.sh"
        commands = commands[:commands.index("set -euo pipefail")] + ["set -euo pipefail"] + commands[commands.index("set -euo pipefail") + 1:-2]
        commands = [l if not l.startswith("#SBATCH --job-name=") else "#SBATCH --job-name=info17_final_test" for l in commands]
        commands.append(shlex.join([sys.executable, str(root / "tools/prepare_info17_final.py"), "finalize", "--plan", str(plan_path)]))
        script.write_text("\n".join(commands) + "\n")
        result = subprocess.run(["sbatch", "--parsable", "--dependency=afterok:" + ":".join(ids.values()),
                                 str(script)], check=True, capture_output=True, text=True)
        write_json(out / "finalization_job.json", {"job_id": result.stdout.strip().split(";")[0], "depends_on": ids})
        print("FINALIZATION_JOB", result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
