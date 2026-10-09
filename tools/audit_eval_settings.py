#!/usr/bin/env python3
"""
Cheap sanity checks on the evaluation / training setup (no GPU needed).

1. max_det audit: Ultralytics keeps at most `max_det` (default 300) boxes per
   image at val/predict time. Dense PCB boards can carry more ground-truth
   objects than that, which hard-caps recall no matter how good the model is.
2. DFL audit: YOLO26 may not use Distribution Focal Loss. If the champion's
   results.csv has no dfl_loss column (or it's always 0), then `--dfl 2.0` was
   a no-op and should be dropped from future runs.

Usage:
    python tools/audit_eval_settings.py \
        --data datasets/pcb-retinex-cappaste-1280/data.yaml \
        --split test \
        --run-dir runs/yolov26s_ultimate_retinex_copypaste_1280/pcb-filtered
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    p.add_argument("--run-dir", type=Path, default=None, help="Ultralytics run dir containing results.csv")
    p.add_argument("--max-det", type=int, default=300)
    return p.parse_args()


def resolve_label_dir(data_yaml: Path, split: str) -> Path:
    cfg = yaml.safe_load(open(data_yaml))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute() and not root.exists():
        root = data_yaml.parent / root
    key = {"valid": "val"}.get(split, split)
    rel = cfg.get(key, f"{split}/images")
    img_dir = Path(rel) if Path(rel).is_absolute() else root / rel
    if not img_dir.exists():
        img_dir = data_yaml.parent / split / "images"
    return img_dir.parent / "labels", cfg


def audit_max_det(args):
    lbl_dir, cfg = resolve_label_dir(args.data, args.split)
    names = cfg.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    files = sorted(lbl_dir.glob("*.txt"))
    print("=" * 72)
    print(f"[1] max_det audit  ({lbl_dir}, {len(files)} label files)")
    print("=" * 72)
    if not files:
        print("  !! no label files found")
        return

    per_img = []
    for f in files:
        cls = Counter(int(l.split()[0]) for l in f.read_text().splitlines() if l.strip())
        per_img.append((sum(cls.values()), f.stem, cls))
    per_img.sort(reverse=True)
    totals = [n for n, _, _ in per_img]
    over = [x for x in per_img if x[0] > args.max_det]
    lost = sum(n - args.max_det for n, _, _ in over)

    print(f"  GT objects / image: mean={sum(totals)/len(totals):.1f}  max={totals[0]}  min={totals[-1]}")
    print(f"  Images with > {args.max_det} GT objects: {len(over)} / {len(totals)}")
    print(f"  GT objects that can NEVER be recalled at max_det={args.max_det}: "
          f"{lost} / {sum(totals)} ({100*lost/max(sum(totals),1):.2f}%)")
    print("  Densest boards:")
    for n, stem, cls in per_img[:5]:
        breakdown = ", ".join(f"{names.get(c, c)}={k}" for c, k in sorted(cls.items()))
        print(f"    {n:5d}  {stem[:50]:50s}  {breakdown}")
    if over:
        print(f"  -> VERDICT: max_det={args.max_det} caps recall. Re-evaluate with --max-det 1000.")
    else:
        print(f"  -> VERDICT: max_det={args.max_det} is not limiting on this split.")
    print("  (Note: even below the cap, at conf=0.001 low-score false positives compete for the")
    print("   300 slots, so --max-det 1000 can still help on boards with ~150-300 objects.)")


def audit_dfl(args):
    print("\n" + "=" * 72)
    print("[2] DFL loss audit")
    print("=" * 72)
    if args.run_dir is None:
        print("  (skipped: no --run-dir)")
        return
    csv_path = args.run_dir / "results.csv"
    if not csv_path.exists():
        print(f"  !! {csv_path} not found")
        return
    rows = list(csv.DictReader(open(csv_path)))
    cols = [c.strip() for c in rows[0].keys()] if rows else []
    loss_cols = [c for c in cols if "loss" in c]
    print(f"  loss columns: {loss_cols}")
    dfl_cols = [c for c in cols if "dfl" in c]
    if not dfl_cols:
        print("  -> VERDICT: no dfl_loss column. `--dfl` is a NO-OP for this model; drop it.")
        return
    for c in dfl_cols:
        vals = []
        for r in rows:
            for k, v in r.items():
                if k.strip() == c:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
        nz = sum(1 for v in vals if abs(v) > 1e-9)
        print(f"  {c}: first={vals[0] if vals else 'n/a'}  last={vals[-1] if vals else 'n/a'}  nonzero epochs={nz}/{len(vals)}")
        if nz == 0:
            print(f"  -> VERDICT: {c} is always 0. `--dfl` is a NO-OP; drop it.")
        else:
            print(f"  -> VERDICT: {c} is active; `--dfl` has an effect.")


def main():
    args = parse_args()
    audit_max_det(args)
    audit_dfl(args)


if __name__ == "__main__":
    main()
