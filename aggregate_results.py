#!/usr/bin/env python3
"""
Combines every <run-key>.json written by train.py into one comparison table.

Run this any time -- after one job finishes or after all of them have --
since it just reads whatever JSON files currently exist in RESULTS_DIR.
This avoids the read-modify-write race that a single shared CSV would have
if multiple Slurm jobs finished around the same time and both tried to
append to it concurrently.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir /mnt/weka/etadevosyan/pcb-yolo/results
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            __import__("os").environ.get(
                "PCB_RESULTS_DIR", "/mnt/weka/etadevosyan/pcb-yolo/results"
            )
        ),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/comparison_table.csv",
    )
    args = p.parse_args()

    json_files = sorted(args.results_dir.glob("*.json"))
    if not json_files:
        print(f"No result JSON files found in {args.results_dir}")
        return

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        row = {k: v for k, v in data.items() if k != "per_class_ap50"}
        for cls, ap in data.get("per_class_ap50", {}).items():
            row[f"AP50_{cls.replace(' ', '_')}"] = ap
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

    out_csv = args.out_csv or (args.results_dir / "comparison_table.csv")
    df.to_csv(out_csv, index=False)

    print(f"Aggregated {len(df)} runs from {args.results_dir}")
    print(f"Saved to: {out_csv}\n")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(df)


if __name__ == "__main__":
    main()
