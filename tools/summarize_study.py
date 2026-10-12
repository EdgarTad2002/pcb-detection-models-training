#!/usr/bin/env python3
"""
tools/summarize_study.py - aggregate results/sweep_*.json into a seed-aware table.

Groups runs that differ only by seed, reports mean +/- std (n), and flags whether each
config's gain over its matching baseline exceeds run-to-run noise.

    python tools/summarize_study.py [--results-dir results] [--out results/study_summary.md]
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

METRICS = [("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"), ("recall", "Recall")]
SEED_RE = re.compile(r"_s(\d+)_")


def load(results_dir: Path):
    groups = defaultdict(list)
    for p in sorted(results_dir.glob("sweep_*.json")):
        d = json.loads(p.read_text())
        key = SEED_RE.sub("_s*_", d["model"])
        d["_cap"] = d.get("per_class_ap50", {}).get("Capacitor")
        groups[key].append(d)
    return groups


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else None)


def fmt(m, s):
    if m is None:
        return "-"
    return f"{m:.4f}" + (f" ± {s:.4f}" if s is not None else "")


def baseline_for(key: str) -> str:
    """Matching control = same hsv_v / architecture with mode 'base'."""
    return re.sub(r"^sweep_(bilat_a[\d.]+_b[\d.]+|cappaste_bilat_a[\d.]+_b[\d.]+|stem|base)_",
                  "sweep_base_", key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/study_summary.md"))
    args = ap.parse_args()

    groups = load(args.results_dir)
    if not groups:
        raise SystemExit(f"no sweep_*.json found in {args.results_dir}")

    stats = {}
    for key, runs in groups.items():
        stats[key] = {
            "n": len(runs),
            **{name: ms([r.get(name) for r in runs]) for name, _ in METRICS},
            "cap": ms([r["_cap"] for r in runs]),
        }

    lines = ["| config | n | mAP50 | mAP50-95 | Recall | Capacitor AP50 | verdict vs matching base |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for key in sorted(stats):
        s = stats[key]
        base = stats.get(baseline_for(key))
        verdict = "(control)" if key.startswith("sweep_base_") else "no matching base"
        if base is not None and not key.startswith("sweep_base_"):
            diff = s["mAP50"][0] - base["mAP50"][0]
            noise = max(x for x in [s["mAP50"][1], base["mAP50"][1], 0.0] if x is not None)
            if s["n"] < 2 or base["n"] < 2:
                verdict = f"Δ={diff:+.4f} (single seed: cannot separate from noise)"
            elif abs(diff) > 2 * noise:
                verdict = f"Δ={diff:+.4f} > 2σ ({noise:.4f}) -> likely real"
            else:
                verdict = f"Δ={diff:+.4f} within 2σ ({noise:.4f}) -> NOT distinguishable"
        lines.append(f"| `{key}` | {s['n']} | " + " | ".join(fmt(*s[n]) for n, _ in METRICS)
                     + f" | {fmt(*s['cap'])} | {verdict} |")

    text = "\n".join(lines)
    args.out.write_text(text + "\n")
    print(text)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
