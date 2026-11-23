#!/usr/bin/env python3
"""Plan, audit, train and evaluate controlled PCB experiments. No GPU imports in planning."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=None)
def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root, path):
    p = Path(path)
    return p if p.is_absolute() else root / p


def read_dataset(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if names != NAMES or cfg.get("nc", 4) != 4:
        raise ValueError(f"Expected canonical four-class mapping: {path}")
    base = resolve(path.parent, cfg.get("path", ".")).resolve()
    splits = {}
    for split in ("train", "val", "test"):
        value = cfg.get(split)
        if not isinstance(value, str):
            raise ValueError(f"Expected directory or image-list path for {split}: {path}")
        location = resolve(base, value)
        if location.is_dir():
            images = sorted(p for p in location.rglob("*") if p.suffix.lower() in EXTENSIONS)
        elif location.is_file() and location.suffix == ".txt":
            images = []
            for line in location.read_text().splitlines():
                if line.strip():
                    # Ultralytics resolves ./ relative to the image-list file.
                    p = location.parent / line[2:] if line.startswith("./") else Path(line)
                    if not p.is_absolute():
                        raise ValueError(f"Use absolute image paths or ./ paths in {location}")
                    images.append(p)
        else:
            raise FileNotFoundError(f"Missing {split} images: {location}")
        if not images:
            raise ValueError(f"Empty {split} split: {location}")
        records = []
        for image in images:
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError(f"Cannot resolve YOLO labels for {image}")
            index = len(parts) - 1 - parts[::-1].index("images")
            parts[index] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label (use an empty file for background): {label}")
            for line in label.read_text().splitlines():
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5 or fields[0] not in {"0", "1", "2", "3"}:
                    raise ValueError(f"Expected five-column detection labels: {label}")
                cx, cy, w, h = map(float, fields[1:])
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                    raise ValueError(f"Invalid normalized box: {label}: {line}")
            records.append({"name": image.name, "image": str(image.absolute()),
                            "label": str(label.absolute()), "image_sha256": file_hash(str(image.resolve())),
                            "label_sha256": file_hash(str(label.resolve()))})
        splits[split] = records
    return {"yaml": str(path), "splits": splits}


def split_signature(records):
    return digest(sorted((r["name"], r["image_sha256"], r["label_sha256"]) for r in records))


def board_id(name):
    stem = Path(name).stem
    if stem.startswith("crop_"):
        stem = stem.split("_", 3)[-1]
    return stem.split(".rf.")[0].removesuffix("_cappaste")


def audit_datasets(datasets, evaluation):
    reference = datasets[evaluation]["splits"]
    heldout = reference["val"] + reference["test"]
    if {r["image_sha256"] for r in reference["val"]} & {r["image_sha256"] for r in reference["test"]}:
        raise ValueError("Validation and test contain duplicate image content")
    if {board_id(r["name"]) for r in reference["val"]} & {board_id(r["name"]) for r in reference["test"]}:
        raise ValueError("Validation and test share board identifiers")
    for name, ds in datasets.items():
        for split in ("val", "test"):
            if split_signature(ds["splits"][split]) != split_signature(reference[split]):
                raise ValueError(f"{name}: {split} images/labels differ from fixed evaluation dataset")
        train = ds["splits"]["train"]
        if {r["image_sha256"] for r in train} & {r["image_sha256"] for r in heldout}:
            raise ValueError(f"{name}: training image content overlaps held-out data")
        if {board_id(r["name"]) for r in train} & {board_id(r["name"]) for r in heldout}:
            raise ValueError(f"{name}: training board identifiers overlap held-out data")


def matched_budget(reference_count, sample_count, batch, reference_epochs):
    target = math.ceil(reference_count / batch) * reference_epochs
    per_epoch = math.ceil(sample_count / batch)
    epochs = max(1, math.floor(target / per_epoch + 0.5))
    actual = epochs * per_epoch
    return {"epochs": epochs, "target_batches": target, "planned_batches": actual,
            "relative_difference": (actual - target) / target,
            "sample_presentations": epochs * sample_count}


def make_jobs(config, studies, models=None, seeds=None, counts=None):
    jobs = {}
    for study in studies:
        spec = config["studies"][study]
        for model in spec["models"]:
            if models and model not in models:
                continue
            for dataset in spec.get("datasets", [spec.get("dataset")]):
                for seed in config["seeds"] if seeds is None else seeds:
                    mode = "budget" if spec.get("match_budget") else study
                    key = f"{config['version']}_{digest(config)[:8]}_{mode}_{model}_{dataset}_s{seed}"
                    budget = None
                    if spec.get("match_budget") and counts is not None:
                        budget = matched_budget(counts[config["reference_dataset"]], counts[dataset],
                                                config["common"]["batch"], config["reference_epochs"])
                        if abs(budget["relative_difference"]) > 0.02:
                            raise ValueError(f"{key}: epoch-rounded budget differs by >2%; revise reference budget")
                    epochs = budget["epochs"] if budget else spec.get("epochs", config["reference_epochs"])
                    if key in jobs:
                        jobs[key]["studies"].append(study)
                    else:
                        jobs[key] = {"key": key, "model": model, "dataset": dataset, "seed": seed,
                                     "epochs": epochs, "studies": [study], "budget": budget}
    if not jobs:
        raise ValueError("Selection matches no experiments")
    return list(jobs.values())


def flags(options):
    argv = []
    for key, value in options.items():
        if value is False or value is None:
            continue
        argv.append("--" + key.replace("_", "-"))
        if value is not True:
            argv.extend(map(str, value if isinstance(value, list) else [value]))
    return argv


def command_for(config, job, root, out, data_yaml, eval_yaml, device="0"):
    options = dict(config["common"])
    options.update(config["models"][job["model"]])
    if job["budget"]:
        # Keep warmup batch count approximately equal across different dataset sizes.
        options["warmup_epochs"] *= job["epochs"] / config["reference_epochs"]
    options.update(data=str(data_yaml), eval_data=str(eval_yaml), eval_split="val", epochs=job["epochs"],
                   seed=job["seed"], run_key=job["key"], project_root=str(root),
                   results_dir=str(out / "validation"), device=device)
    return [sys.executable, str(root / "train.py")] + flags(options)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(args, config, root):
    import yaml
    studies = list(config["studies"]) if args.study == "all" else [args.study]
    if set(studies) - set(config["studies"]):
        raise ValueError(f"Config supports studies: {', '.join(config['studies'])}")
    preliminary = make_jobs(config, studies, args.models, args.seeds)
    needed = {j["dataset"] for j in preliminary} | {config["evaluation_dataset"], config["reference_dataset"]}
    datasets = {name: read_dataset(resolve(root, config["datasets"][name])) for name in sorted(needed)}
    audit_datasets(datasets, config["evaluation_dataset"])
    counts = {name: len(ds["splits"]["train"]) for name, ds in datasets.items()}
    jobs = make_jobs(config, studies, args.models, args.seeds, counts)
    fingerprints = {name: {split: split_signature(rs) for split, rs in ds["splits"].items()}
                    for name, ds in datasets.items()}
    identity = digest({"config": config, "jobs": jobs, "datasets": fingerprints})
    out = resolve(root, args.output) / identity[:12]
    out.mkdir(parents=True, exist_ok=True)
    canonical = datasets[config["evaluation_dataset"]]["splits"]
    # Image lists freeze membership and use absolute paths; labels remain read-only in source datasets.
    for name, ds in datasets.items():
        directory = out / "datasets" / name
        directory.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            records = ds["splits"][split] if split == "train" else canonical[split]
            (directory / f"{split}.txt").write_text("".join(r["image"] + "\n" for r in records))
        cfg = {"path": str(directory), "names": NAMES, "nc": 4,
               **{s: str(directory / f"{s}.txt") for s in ("train", "val", "test")}}
        (directory / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    eval_yaml = out / "datasets" / config["evaluation_dataset"] / "data.yaml"
    for job in jobs:
        job["command"] = command_for(config, job, root, out,
                                     out / "datasets" / job["dataset"] / "data.yaml", eval_yaml, args.device)
    plan = {"identity": identity, "root": str(root), "output": str(out), "config": config,
            "dataset_fingerprints": fingerprints, "datasets": datasets, "jobs": jobs}
    write_json(out / "plan.json", plan)
    print(f"Prepared {len(jobs)} jobs: {out / 'plan.json'}")
    for job in jobs:
        print(f"  {job['key']}: {job['epochs']} epochs; budget={job['budget']}")
    return out / "plan.json"


def verify_plan(plan):
    file_hash.cache_clear()
    for name, expected in plan["dataset_fingerprints"].items():
        current = read_dataset(plan["datasets"][name]["yaml"])
        actual = {s: split_signature(rs) for s, rs in current["splits"].items()}
        if actual != expected:
            raise ValueError(f"Dataset changed since preparation: {name}; prepare a new plan")
    # Also verify the generated snapshots have not been edited.
    for name in plan["datasets"]:
        current = read_dataset(Path(plan["output"]) / "datasets" / name / "data.yaml")
        for split, records in current["splits"].items():
            source = name if split == "train" else plan["config"]["evaluation_dataset"]
            if split_signature(records) != plan["dataset_fingerprints"][source][split]:
                raise ValueError(f"Prepared image list changed: {name}/{split}")


def run_job(plan, key, device):
    verify_plan(plan)
    required = plan["config"]["ultralytics_version"]
    installed = importlib.metadata.version("ultralytics")
    if installed != required:
        raise ValueError(f"Protocol requires ultralytics=={required}; installed {installed}. Use the pinned environment.")
    job = next(j for j in plan["jobs"] if j["key"] == key)
    root, out = Path(plan["root"]), Path(plan["output"])
    run_dir = root / "runs" / key
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    evidence = out / "provenance" / key
    evidence.mkdir(parents=True, exist_ok=False)
    argv = list(job["command"])
    argv[0] = sys.executable
    argv[argv.index("--device") + 1] = device
    write_json(evidence / "job.json", {**job, "command": argv, "plan_identity": plan["identity"]})
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (evidence / "environment.txt").write_text(freeze.stdout)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    tracked_diff = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, check=True)
    sources = {str(p.relative_to(root)): file_hash(str(p)) for p in root.rglob("*.py")
               if not any(part.startswith(".") or part in {"runs", "datasets"} for part in p.relative_to(root).parts)}
    write_json(evidence / "source.json", {"commit": revision.stdout.strip(),
               "tracked_diff_sha256": hashlib.sha256(tracked_diff.stdout).hexdigest(), "python_sources": sources})
    print(shlex.join(argv), flush=True)
    subprocess.run(argv, cwd=root, check=True)
    inputs = {}
    for option in ("--weights", "--pretrained-weights", "--spectral-priors"):
        if option in argv:
            path = resolve(root, argv[argv.index(option) + 1])
            if path.is_file():
                inputs[str(path)] = file_hash(str(path.resolve()))
    priors = root / "data/pcb_spectral_priors.json"
    if priors.is_file():
        inputs[str(priors)] = file_hash(str(priors.resolve()))
    write_json(evidence / "initialization_files.json", inputs)


def evaluate_job(plan, key, device, split):
    verify_plan(plan)
    if importlib.metadata.version("ultralytics") != plan["config"]["ultralytics_version"]:
        raise ValueError("Evaluation must use the protocol's pinned Ultralytics version")
    job = next(j for j in plan["jobs"] if j["key"] == key)
    checkpoint = Path(plan["root"]) / "runs" / key / "pcb-filtered/weights/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    argv = list(job["command"])
    argv[0] = sys.executable
    argv[argv.index("--weights") + 1] = str(checkpoint)
    argv[argv.index("--eval-split") + 1] = split
    argv[argv.index("--results-dir") + 1] = str(Path(plan["output"]) / ("test" if split == "test" else "validation"))
    argv[argv.index("--device") + 1] = device
    # Loading the saved checkpoint directly needs no model reconstruction or weight transfer.
    if "--pretrained-weights" in argv:
        i = argv.index("--pretrained-weights")
        del argv[i:i + 2]
    target = Path(argv[argv.index("--results-dir") + 1]) / f"{key}.json"
    if split == "test" and target.exists():
        raise FileExistsError(f"Test result already exists: {target}")
    subprocess.run(argv + ["--skip-train"], cwd=plan["root"], check=True)


def select_crop_pair(plan, seed, models=None):
    """Never silently mix architectures when a plan contains several crop pairs."""
    jobs = [j for j in plan["jobs"] if "crop" in j["studies"] and j["seed"] == seed]
    available = {j["model"] for j in jobs}
    if models is not None:
        if len(models) != 1 or models[0] not in available:
            raise ValueError(f"Select exactly one crop model using --models: {', '.join(sorted(available))}")
        jobs = [j for j in jobs if j["model"] == models[0]]
    elif len(available) > 1:
        raise ValueError(f"Multiple crop models: specify --models {' or '.join(sorted(available))}")
    pair = {j["dataset"]: j for j in jobs}
    if len(jobs) != 2 or set(pair) != {"clean", "mixed"}:
        raise ValueError("Crop evaluation requires exactly one clean and one mixed run for the selected model/seed")
    return pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["list", "prepare", "run", "submit", "cancel", "evaluate", "crop-eval", "summary"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/controlled_experiments.json")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--output", default="controlled_experiments")
    p.add_argument("--study", choices=["all", "main", "augmentation", "crop"], default="all")
    p.add_argument("--models", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--plan", type=Path)
    p.add_argument("--job-key")
    p.add_argument("--device", default="0")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--max-parallel", type=int, default=None,
                   help="Optional concurrent-job limit. Omit to let Slurm schedule all jobs independently.")
    p.add_argument("--run-tag", help="Fresh run namespace for a restart, e.g. restart1; applies during prepare/list")
    p.add_argument("--seed", type=int, default=0, help="Paired crop evaluation seed")
    args = p.parse_args()
    root = args.project_root.resolve()
    if args.action in {"list", "prepare"}:
        config = json.loads(args.config.read_text())
        if args.run_tag:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", args.run_tag):
                p.error("--run-tag must contain 1-40 letters, digits, underscores or hyphens")
            config["version"] += "_" + args.run_tag
        if args.models and set(args.models) - set(config["models"]):
            p.error("Unknown model selection")
        if config["common"]["nbs"] != config["common"]["batch"]:
            p.error("Controlled update-budget protocol requires nbs == batch")
        if args.action == "prepare":
            prepare(args, config, root)
        else:
            studies = list(config["studies"]) if args.study == "all" else [args.study]
            if set(studies) - set(config["studies"]):
                p.error(f"Config supports studies: {', '.join(config['studies'])}")
            jobs = make_jobs(config, studies, args.models, args.seeds)
            print(json.dumps(jobs, indent=2))
            print(f"{len(jobs)} jobs; ablation epochs calculated during prepare from actual dataset counts.")
        return
    if args.plan is None:
        p.error("--plan is required")
    plan = json.loads(args.plan.read_text())
    if args.job_key and args.job_key not in {j["key"] for j in plan["jobs"]}:
        p.error("--job-key does not exist in this plan")
    if args.action == "run":
        if not args.job_key:
            p.error("--job-key is required")
        run_job(plan, args.job_key, args.device)
    elif args.action == "evaluate":
        if not args.job_key:
            p.error("--job-key is required; test evaluation must select a frozen checkpoint explicitly")
        evaluate_job(plan, args.job_key, args.device, args.split)
    elif args.action == "submit":
        verify_plan(plan)
        if args.max_parallel is not None and args.max_parallel < 1:
            p.error("--max-parallel must be positive")
        if (Path(plan["output"]) / "submission.json").exists():
            raise FileExistsError("This plan has already been submitted. Prepare with a new --run-tag for a restart.")
        ids = []
        for job in plan["jobs"]:
            if (Path(plan["root"]) / "runs" / job["key"]).exists():
                raise FileExistsError(f"Run already exists: {job['key']}; submit a fresh selection")
        for i, job in enumerate(plan["jobs"]):
            argv = ["sbatch", "--parsable"]
            if args.max_parallel is not None and i >= args.max_parallel:
                argv += [f"--dependency=afterany:{ids[i - args.max_parallel]}"]
            argv += [str(Path(plan["root"]) / "sbatch/train_controlled_experiment.sh"),
                     str(args.plan.resolve()), job["key"]]
            result = subprocess.run(argv, cwd=plan["root"], check=True, capture_output=True, text=True)
            ids.append(result.stdout.strip().split(";")[0])
            write_json(Path(plan["output"]) / "submission.json", dict(zip([j["key"] for j in plan["jobs"]], ids)))
            print(job["key"], ids[-1], flush=True)
    elif args.action == "cancel":
        submitted = json.loads((Path(plan["output"]) / "submission.json").read_text())
        allowed = {j["key"] for j in plan["jobs"]}
        if set(submitted) - allowed or not submitted or not all(str(v).isdigit() for v in submitted.values()):
            raise ValueError("Invalid submission records; refusing to cancel unrelated jobs")
        subprocess.run(["scancel", *map(str, submitted.values())], cwd=plan["root"], check=True)
        print(f"Cancellation requested for {len(submitted)} jobs belonging to this plan only.")
    elif args.action == "crop-eval":
        verify_plan(plan)
        if importlib.metadata.version("ultralytics") != plan["config"]["ultralytics_version"]:
            raise ValueError("Crop evaluation must use the protocol's pinned Ultralytics version")
        pair = select_crop_pair(plan, args.seed, args.models)
        if set(pair) != {"clean", "mixed"}:
            p.error("Plan must include both crop study arms for the selected seed")
        weights = {name: Path(plan["root"]) / "runs" / job["key"] / "pcb-filtered/weights/best.pt" for name, job in pair.items()}
        for path in weights.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        records = plan["datasets"][plan["config"]["evaluation_dataset"]]["splits"]["val"]
        images = {str(Path(r["image"]).parent) for r in records}
        labels = {str(Path(r["label"]).parent) for r in records}
        if len(images) != 1 or len(labels) != 1:
            raise ValueError("Existing dual-branch evaluator requires one image directory and one label directory")
        subprocess.run([sys.executable, str(Path(plan["root"]) / "tools/run_val_comparison_matrix.py"),
                        "--global-weights", str(weights["clean"]), "--crop-weights", str(weights["mixed"]),
                        "--images-dir", next(iter(images)), "--labels-dir", next(iter(labels)),
                        "--out-dir", str(Path(plan["output"]) / f"crop_validation_{pair['clean']['model']}_s{args.seed}"),
                        "--device", args.device,
                        *flags(plan["config"]["crop_inference"]),
                        "--conf-thresh", str(plan["config"]["common"]["eval_conf"]),
                        "--max-det", str(plan["config"]["common"]["max_det"])], cwd=plan["root"], check=True)
    else:
        groups = {}
        for job in plan["jobs"]:
            path = Path(plan["output"]) / "validation" / f"{job['key']}.json"
            if path.exists():
                result = json.loads(path.read_text())
                groups.setdefault((tuple(job["studies"]), job["model"], job["dataset"]), []).append(result)
        for key, values in groups.items():
            print(key, "n=", len(values), {metric: {"mean": statistics.mean(v[metric] for v in values),
                  "std": statistics.stdev(v[metric] for v in values) if len(values) > 1 else None}
                  for metric in ("mAP50", "mAP50_95")})
        print(f"Completed {sum(map(len, groups.values()))}/{len(plan['jobs'])} validation results")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit(f"Controlled experiment error: {exc}") from None
