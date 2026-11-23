#!/usr/bin/env python3
"""Prepare 19 clean-only main jobs, reusing two matching seed-0 clean controls."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import prepare, read_dataset, split_signature, write_json

REUSED_MODELS = ("physot_yolo26s", "yolov12s")


def validate_controls(config, plan, root):
    """Require matching recipes, initializers and canonical split fingerprints."""
    previous = plan["config"]
    for key in ("ultralytics_version", "common", "evaluation_dataset"):
        if previous[key] != config[key]:
            raise ValueError(f"Existing controls do not match {key}; do not reuse them")
    if Path(plan["root"]).resolve() != root.resolve():
        raise ValueError("Existing controls belong to another project root")
    dataset = read_dataset(root / config["datasets"]["clean"])
    actual = {s: split_signature(rs) for s, rs in dataset["splits"].items()}
    if actual != plan["dataset_fingerprints"]["clean"]:
        raise ValueError("Clean dataset changed since the existing controls were prepared")
    reused = []
    for model in REUSED_MODELS:
        if previous["models"][model] != config["models"][model]:
            raise ValueError(f"Existing initializer/loss/modules differ: {model}")
        matches = [j for j in plan["jobs"] if j["model"] == model
                   and j["dataset"] == "clean" and j["seed"] == 0]
        if len(matches) != 1 or matches[0]["epochs"] != config["studies"]["main"]["epochs"]:
            raise ValueError(f"Missing unique matching 100-epoch seed-0 control: {model}")
        job = matches[0]
        reused.append({"model": model, "dataset": "clean", "seed": 0,
                       "job_key": job["key"],
                       "validation_result": str(Path(plan["output"]) / "validation" / (job["key"] + ".json")),
                       "weights": str(root / "runs" / job["key"] / "pcb-filtered/weights/best.pt"),
                       "status": "reuse only after successful completion; an existing best.pt alone is not completion"})
    return reused


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--control-plan", type=Path, default=Path("controlled_experiments/5c2e711e5229/plan.json"))
    p.add_argument("--submit", action="store_true", help="Also submit the 19 new jobs to Slurm")
    args = p.parse_args()
    root = args.project_root.resolve()
    config = json.loads((root / "configs/clean_main_experiments.json").read_text())
    source = args.control_plan if args.control_plan.is_absolute() else root / args.control_plan
    control = json.loads(source.read_text())
    reused = validate_controls(config, control, root)
    remaining = [m for m in config["studies"]["main"]["models"] if m not in REUSED_MODELS]
    plans = []
    for models, seeds in ((remaining, [0, 1, 2]), (list(REUSED_MODELS), [1, 2])):
        options = argparse.Namespace(study="main", models=models, seeds=seeds,
                                     output="controlled_experiments", device="0")
        plans.append(prepare(options, config, root))
    # A manifest joins the plans without relabeling or copying existing results.
    bundle = root / "controlled_experiments" / "clean_main_noextraaug_v1_bundle.json"
    write_json(bundle, {"config": str(root / "configs/clean_main_experiments.json"),
                        "new_plans": [str(x) for x in plans], "new_job_count": 19,
                        "control_plan": str(source.resolve()), "reused_controls": reused,
                        "total_model_seed_results_when_complete": 21,
                        "evaluation_split": "val", "test_evaluation_performed": False})
    print(f"Bundle: {bundle}\n19 new jobs + 2 existing seed-0 controls = 21 results when complete.")
    commands = [[sys.executable, str(root / "tools/controlled_experiments.py"),
                 "submit", "--plan", str(path)] for path in plans]
    if args.submit:
        # Check both plans before submitting either, avoiding an accidental duplicate launch.
        for path in plans:
            plan = json.loads(path.read_text())
            if (Path(plan["output"]) / "submission.json").exists():
                raise FileExistsError(f"Plan already submitted: {path}")
            for job in plan["jobs"]:
                if (root / "runs" / job["key"]).exists():
                    raise FileExistsError(f"Run already exists: {job['key']}")
        for command in commands:
            subprocess.run(command, cwd=root, check=True)
    else:
        print("Prepared only. Submit commands:")
        for command in commands:
            print(shlex.join(command))


if __name__ == "__main__":
    main()
