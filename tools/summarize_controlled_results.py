#!/usr/bin/env python3
"""Audit and combine downloaded controlled-result bundles without changing the cluster."""
import argparse
import json
import math
from pathlib import Path
import statistics

LABELS = {
    "yolov12s": "YOLOv12s", "yolov10s": "YOLOv10s", "yolo26s": "YOLO26s",
    "yolo26s_sdice": "YOLO26s + Dice",
    "yolo26s_agwd_sdice": "AGWD + Dice (lambda 0.50)",
    "yolo26s_agwd_sdice_l035": "AGWD + Dice (lambda 0.35)",
    "physot_yolo26s": "PhysOT-YOLO26s",
}


def estimate(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Require at least two finite seed measurements")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def summarize(bundle, training_audit):
    plans, records = bundle["plans"], bundle["records"]
    if len(records) != 21:
        raise ValueError(f"Expected 21 successful results; got {len(records)}")
    common = plans[0]["config"]["common"]
    fingerprints = plans[0]["dataset_fingerprints"]
    for plan in plans[1:]:
        if plan["config"]["common"] != common or plan["dataset_fingerprints"] != fingerprints:
            raise ValueError("Repair and original plans have different common settings or data fingerprints")
        for key in ("ultralytics_version", "evaluation_dataset"):
            if plan["config"][key] != plans[0]["config"][key]:
                raise ValueError(f"Different {key} across plans")
    audit = {a["key"]: a for a in training_audit}
    groups, seen = {}, set()
    for record in records:
        job, result = record["job"], record["result"]
        model, seed = job["model"], job["seed"]
        if (model, seed) in seen:
            raise ValueError(f"Duplicate model/seed: {model}/{seed}")
        seen.add((model, seed))
        if model not in LABELS:
            raise ValueError(f"Unexpected model: {model}")
        expected = {"seed": seed, "epochs": 100, "imgsz": 1280, "batch": 8,
                    "eval_split": "val", "max_det": 1000, "eval_conf": 0.001, "eval_iou": 0.5}
        if any(result.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Result metadata drift: {job['key']}")
        row = audit[job["key"]]
        if not row["checkpoint_exists"] or int(row["last_epoch"]) != 100 or row["curve_rows"] != 100:
            raise ValueError(f"Missing checkpoint or incomplete training: {job['key']}")
        protocol = result["training_protocol"]
        for k, v in common.items():
            mapped = {"no_cache": "cache", "eval_conf": "conf", "eval_iou": "iou"}.get(k, k)
            expected_value = not v if k == "no_cache" else v
            if protocol.get(mapped) != expected_value:
                raise ValueError(f"Training protocol drift for {job['key']}: {mapped}")
        if protocol.get("amp") is not True or protocol.get("split") != "val":
            raise ValueError(f"Unexpected AMP or validation split: {job['key']}")
        plan = next(p for p in plans if record["source"].startswith(str(Path(p["path"]).parent) + "/"))
        recipe = plan["config"]["models"][model]
        for key in ("agwd_ota", "squared_dice", "use_igr_stem", "use_fg_csfa", "use_aca_supcon"):
            if bool(result.get(key)) != bool(recipe.get(key)):
                raise ValueError(f"Model recipe mismatch: {job['key']}/{key}")
        for key in ("lambda_dice", "smooth_dice"):
            if key in recipe and result.get(key) != recipe[key]:
                raise ValueError(f"Dice parameter mismatch: {job['key']}/{key}")
        groups.setdefault(model, []).append(record)
    if set(groups) != set(LABELS):
        raise ValueError("Missing model configurations")
    summary = []
    for model, rows in groups.items():
        rows.sort(key=lambda r: r["job"]["seed"])
        if [r["job"]["seed"] for r in rows] != [0, 1, 2]:
            raise ValueError(f"Incomplete seed coverage: {model}")
        fields = {
            "ap50": lambda r: r["mAP50"], "ap50_95": lambda r: r["mAP50_95"],
            **{f"ap50_{k}": (lambda r, name=k: r["per_class_ap50"][name])
               for k in ("Capacitor", "Connector", "Electrolytic Capacitor", "IC")},
        }
        summary.append({"model": model, "label": LABELS[model], "seeds": [0, 1, 2],
                        "metrics": {k: estimate([100 * getter(r["result"]) for r in rows]) for k, getter in fields.items()},
                        "sources": [r["source"] for r in rows],
                        "checkpoint_paths": [r["result"]["weights"] for r in rows]})
    summary.sort(key=lambda r: r["metrics"]["ap50"]["mean"], reverse=True)
    indexed = {r["model"]: r for r in summary}
    deltas = {}
    for model, entry in indexed.items():
        deltas[model] = {baseline: {metric: entry["metrics"][metric]["mean"] - indexed[baseline]["metrics"][metric]["mean"]
                         for metric in ("ap50", "ap50_95", "ap50_Capacitor")}
                         for baseline in ("yolo26s", "yolov12s")}
    return {"date": "2026-10-24", "split": "val", "units": "percent; differences in percentage points",
            "uncertainty": "sample standard deviation across seeds, not confidence intervals",
            "audit": {"successful_runs": 21, "models": 7, "seeds_per_model": 3,
                      "epochs_each": 100, "all_checkpoints_verified": True,
                      "common_training_settings_match": True, "dataset_fingerprints_match": True,
                      "model_recipe_metadata_match": True},
            "plans": [p["path"] for p in plans], "common_settings": common,
            "dataset_fingerprints": fingerprints, "summary": summary, "deltas": deltas}


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = summary["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.7), sharey=True)
    for ax, key, title in zip(axes, ("ap50", "ap50_95", "ap50_Capacitor"),
                              ("Overall mAP50", "Overall mAP50:95", "Capacitor AP50")):
        values = []
        for y, row in enumerate(rows):
            m = row["metrics"][key]
            values += m["values"] + [m["mean"] - m["sample_sd"], m["mean"] + m["sample_sd"]]
            ax.scatter(m["values"], [y - .13, y, y + .13], color="#8a939b", s=20, zorder=2)
            ax.errorbar(m["mean"], y, xerr=m["sample_sd"], fmt="D", color="#176184", capsize=4,
                        markersize=5, elinewidth=1.5, zorder=3)
            ax.text(1.03, y, f"{m['mean']:.2f} ± {m['sample_sd']:.2f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9, clip_on=False)
        extent = max(values) - min(values)
        ax.set_xlim(min(values) - extent * .12 - .1, max(values) + extent * .12 + .1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=13)
        ax.set_xlabel("Validation score (%)", fontsize=10)
        ax.set_yticks(range(len(rows)))
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e3e6e8", linewidth=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_yticklabels([r["label"] for r in rows], fontsize=10)
    fig.suptitle("Controlled PCB comparison: seven models, three seeds each", x=.02, ha="left", fontsize=16, fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], marker="D", color="#176184", label="Mean ± sample SD", markersize=5),
                        Line2D([0], [0], marker="o", linestyle="", color="#8a939b", label="Individual seeds 0, 1, 2", markersize=4)],
               loc="upper left", bbox_to_anchor=(.015, .935), ncol=2, frameon=False, fontsize=10)
    fig.text(.02, .04, "Source: 21 saved cluster validation results, 24 Oct 2026. All runs completed 100 epochs; matching training data and shared settings verified.", fontsize=9)
    fig.text(.02, .012, "Whiskers show seed variability, not confidence intervals. Axis ranges differ between panels. Historical test-set scores are excluded.", fontsize=9)
    fig.subplots_adjust(left=.23, right=.91, top=.79, bottom=.16, wspace=.9)
    fig.savefig(output / "validation-comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "validation-comparison.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--training-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(json.loads(args.input.read_text()), json.loads(args.training_audit.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.output)
    for row in result["summary"]:
        print(row["label"], " | ".join(f"{key}: {row['metrics'][key]['mean']:.2f} ± {row['metrics'][key]['sample_sd']:.2f}"
              for key in ("ap50", "ap50_95", "ap50_Capacitor")))
    print("Audit passed: all 21 checkpoints, 100 completed epochs per run, matching datasets/settings.")


if __name__ == "__main__":
    main()
