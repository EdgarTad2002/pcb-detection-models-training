#!/usr/bin/env python3
"""Inspect the pinned DCD-Net source without executing its training script."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

PAPER_COMMIT = "91a531bef413d597a1fe2fcb3470dba8b018b2ab"


def audit_source(source):
    source = Path(source).resolve()
    required = ["train.py", "ultralytics/__init__.py", "ultralytics/cfg/models/DCD.yaml",
                "ultralytics/cfg/models/yolo11.yaml", "ultralytics/nn/modules/block.py",
                "ultralytics/tricks/dscconv.py", "ultralytics/utils/loss.py", "ultralytics/utils/metrics.py"]
    for filename in required:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"Missing {filename}; the code is on the master branch, not the placeholder main branch")
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    changed = subprocess.run(["git", "-C", str(source), "diff", "--name-only", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    # The author repository tracks bytecode caches, which Python rewrites on import.
    # Audit source edits separately so merely importing the package cannot invalidate a run.
    dirty = [p for p in changed if "__pycache__" not in Path(p).parts and not p.endswith(".pyc")]
    tree = ast.parse((source / "ultralytics/utils/loss.py").read_text())
    bbox = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BboxLoss")
    forward = next(n for n in bbox.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    calls = []
    for node in ast.walk(forward):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "bbox_iou":
            calls.append({k.arg: ast.literal_eval(k.value) for k in node.keywords if isinstance(k.value, ast.Constant)})
    active_fc = any(c.get("FCIoU") is True for c in calls)
    active_ciou = any(c.get("CIoU") is True for c in calls)
    package = ast.parse((source / "ultralytics/__init__.py").read_text())
    version = next(ast.literal_eval(n.value) for n in package.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets))
    return {"repository": "https://github.com/lqin0818/DCD-Net", "branch": "master", "source": str(source),
            "commit": commit, "matches_inspected_commit": commit == PAPER_COMMIT, "tracked_changes": dirty,
            "ignored_bytecode_changes": [p for p in changed if p not in dirty],
            "ultralytics_version": version,
            "source_sha256": {f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in required},
            "bbox_iou_calls_in_BboxLoss_forward": calls,
            "author_code_uses_ciou": active_ciou, "author_code_activates_fciou": active_fc,
            "notes": ["The model YAML calls the published blocks GSConv and C3k2DS.",
                      "The repository helper's FCIoU score uses CIoU + IoU - focaler(IoU).",
                      "Under a 1-score loss that helper has the opposite correction sign from paper equation 5.",
                      "The helper is not enabled in the inspected BboxLoss.forward; do not silently label CIoU runs as full paper FC-IoU.",
                      "A separate root license file and dependency manifest were not present in the inspected master checkout."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_source(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("commit", "ultralytics_version", "author_code_uses_ciou", "author_code_activates_fciou")}, indent=2))


if __name__ == "__main__":
    main()
