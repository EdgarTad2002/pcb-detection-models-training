#!/usr/bin/env python3
"""
Phase 6: Main Experiment Runner

Runs all voting ensemble experiments:
1. Prepare/load cached predictions
2. Method 2: Model-Weighted Voting (3 static weight configs)
3. Method 3: Dynamic Weighted Voting (grid search)
4. Optional: Per-class grid search
5. Compare all methods and save results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ensemble.voting_methods.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    GRID_SEARCH_FINE_STEP,
    GRID_SEARCH_METRIC,
    GRID_SEARCH_STEP,
    IOU_THRESHOLD,
    MODEL_ORDER,
    MODEL_PERFORMANCE,
    OUTPUT_DIR,
    TOP3_WEIGHTS,
)
from ensemble.voting_methods.dynamic_weighted_voting import (
    coarse_to_fine_search,
    grid_search,
    per_class_grid_search,
)
from ensemble.voting_methods.evaluate import (
    compare_methods,
    compute_fei,
    evaluate_voting_predictions,
)
from ensemble.voting_methods.model_weighted_voting import (
    model_weighted_fusion,
    run_weighted_voting_experiments,
)
from ensemble.voting_methods.prepare_predictions import (
    load_cached_data,
    prepare_all_predictions,
)


def run_individual_model_evaluation(
    predictions: Dict,
    ground_truth: Dict,
) -> Dict[str, Dict]:
    """Evaluate each individual model's predictions."""
    print("\n" + "=" * 60)
    print("Individual Model Evaluation")
    print("=" * 60)

    individual_results = {}
    for model_name in MODEL_ORDER:
        if model_name not in predictions:
            continue

        # Each model's predictions treated as fused (single-model).
        # Pass all raw detections (conf >= 0.001) — compute_precision_recall_f1
        # sweeps confidence thresholds internally to find the F1-optimal point,
        # matching YOLO val() reporting style.
        model_preds = dict(predictions[model_name])

        metrics = evaluate_voting_predictions(model_preds, ground_truth)
        individual_results[model_name] = metrics

        print(f"\n  {model_name}:")
        print(f"    mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        if "per_class_AP" in metrics:
            for cls_name, ap in metrics["per_class_AP"].items():
                print(f"    AP({cls_name}): {ap:.4f}")

    return individual_results


def run_method2(
    predictions: Dict,
    ground_truth: Dict,
) -> Dict[str, Dict]:
    """Run Method 2: Model-Weighted Voting with static weights."""
    print("\n" + "=" * 60)
    print("Method 2: Model-Weighted Voting (Static Weights)")
    print("=" * 60)

    method2_fused = run_weighted_voting_experiments(predictions)

    method2_results = {}
    for config_name, fused_preds in method2_fused.items():
        metrics = evaluate_voting_predictions(fused_preds, ground_truth)
        method2_results[config_name] = metrics

        weights = TOP3_WEIGHTS[config_name]
        w_str = ", ".join(f"{m}={w}" for m, w in weights.items())
        print(f"\n  {config_name} [{w_str}]:")
        print(f"    mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")

    return method2_results


def run_method3(
    predictions: Dict,
    ground_truth: Dict,
    step: float = GRID_SEARCH_STEP,
    metric: str = GRID_SEARCH_METRIC,
) -> Dict:
    """Run Method 3: Dynamic Weighted Voting with grid search."""
    print("\n" + "=" * 60)
    print("Method 3: Dynamic Weighted Voting (Grid Search)")
    print("=" * 60)

    result = grid_search(
        predictions,
        ground_truth,
        evaluate_voting_predictions,
        model_names=MODEL_ORDER,
        step=step,
        metric=metric,
    )

    # Print top 5
    print(f"\nTop 5 Configurations:")
    print(f"{'Rank':<6} {'Weights (v11, v5, v8)':<30} {'mAP@0.5':>8} {'F1':>8}")
    print("-" * 56)
    for i, (weights, metrics) in enumerate(result["all_results"][:5]):
        w_str = ", ".join(f"{weights[m]:.2f}" for m in MODEL_ORDER)
        print(
            f"{i+1:<6} ({w_str}){'':<10} "
            f"{metrics.get('mAP50', 0):.4f}   "
            f"{metrics.get('f1', 0):.4f}"
        )

    return result


def run_per_class_search(
    predictions: Dict,
    ground_truth: Dict,
    step: float = GRID_SEARCH_STEP,
) -> Dict:
    """Run per-class grid search."""
    print("\n" + "=" * 60)
    print("Per-Class Grid Search")
    print("=" * 60)

    result = per_class_grid_search(
        predictions,
        ground_truth,
        evaluate_voting_predictions,
        model_names=MODEL_ORDER,
        class_names=CLASS_NAMES,
        step=step,
    )

    print(f"\n{'Class':<25} {'Best Weights (v11, v5, v8)':<35} {'AP@0.5':>8}")
    print("-" * 70)
    for cls_id, r in sorted(result.items()):
        w = r["best_weights"]
        w_str = ", ".join(f"{w[m]:.2f}" for m in MODEL_ORDER)
        print(f"{r['class_name']:<25} ({w_str}){'':<13} {r['best_score']:.4f}")

    return result


def build_comparison_table(
    individual_results: Dict,
    method2_results: Dict,
    method3_result: Dict,
) -> Dict[str, Dict]:
    """Build the final comparison table."""
    comparison = {}

    # Individual models
    for model_name, metrics in individual_results.items():
        label = f"{model_name} (Individual)"
        comparison[label] = metrics

    # Method 2
    for config_name, metrics in method2_results.items():
        label = f"Method 2: {config_name}"
        comparison[label] = metrics

    # Method 3
    if method3_result and method3_result.get("best_metrics"):
        comparison["Method 3: Grid Search Optimized"] = method3_result["best_metrics"]

    return comparison


def save_results(
    individual_results: Dict,
    method2_results: Dict,
    method3_result: Dict,
    per_class_result: Dict = None,
    output_dir: Path = None,
):
    """Save all results to JSON files."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual model results
    with open(output_dir / "individual_results.json", "w") as f:
        json.dump(_make_serializable(individual_results), f, indent=2)

    # Save Method 2 results
    with open(output_dir / "method2_results.json", "w") as f:
        json.dump(_make_serializable(method2_results), f, indent=2)

    # Save Method 3 results
    method3_save = {
        "best_weights": method3_result.get("best_weights"),
        "best_score": method3_result.get("best_score"),
        "best_metrics": method3_result.get("best_metrics"),
        "search_time_s": method3_result.get("search_time_s"),
        "num_combinations": method3_result.get("num_combinations"),
        "top_10": [
            {"weights": w, "metrics": m}
            for w, m in method3_result.get("all_results", [])[:10]
        ],
    }
    with open(output_dir / "method3_results.json", "w") as f:
        json.dump(_make_serializable(method3_save), f, indent=2)

    # Save per-class results
    if per_class_result:
        per_class_save = {}
        for cls_id, r in per_class_result.items():
            per_class_save[str(cls_id)] = {
                "class_name": r["class_name"],
                "best_weights": r["best_weights"],
                "best_score": r["best_score"],
            }
        with open(output_dir / "per_class_results.json", "w") as f:
            json.dump(_make_serializable(per_class_save), f, indent=2)

    # Save full comparison
    comparison = build_comparison_table(
        individual_results, method2_results, method3_result
    )
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(_make_serializable(comparison), f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def _make_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_make_serializable(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def generate_report(
    individual_results: Dict,
    method2_results: Dict,
    method3_result: Dict,
    per_class_result: Dict = None,
    output_dir: Path = None,
):
    """Generate a text report summarizing all experiments."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("VOTING ENSEMBLE EXPERIMENT REPORT")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Models: {', '.join(MODEL_ORDER)}")
    lines.append("=" * 70)

    # Individual models
    lines.append("\n--- Individual Model Results ---")
    lines.append(f"{'Model':<20} {'mAP@0.5':>8} {'P':>8} {'R':>8} {'F1':>8}")
    lines.append("-" * 56)
    for name, m in individual_results.items():
        lines.append(
            f"{name:<20} {m['mAP50']:>8.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f}"
        )

    # Method 2
    lines.append("\n--- Method 2: Model-Weighted Voting ---")
    lines.append(f"{'Config':<25} {'Weights':<30} {'mAP@0.5':>8} {'F1':>8}")
    lines.append("-" * 75)
    for config_name, m in method2_results.items():
        w = TOP3_WEIGHTS[config_name]
        w_str = ", ".join(f"{w[mn]:.2f}" for mn in MODEL_ORDER)
        lines.append(
            f"{config_name:<25} ({w_str}){'':<8} {m['mAP50']:>8.4f} {m['f1']:>8.4f}"
        )

    # Method 3
    lines.append("\n--- Method 3: Grid Search Optimized ---")
    lines.append(f"Search space: {method3_result['num_combinations']} combinations")
    lines.append(f"Search time: {method3_result['search_time_s']:.1f}s")
    bw = method3_result["best_weights"]
    w_str = ", ".join(f"{bw[mn]:.2f}" for mn in MODEL_ORDER)
    lines.append(f"Best weights: ({w_str})")
    lines.append(f"Best mAP@0.5: {method3_result['best_score']:.4f}")

    if method3_result.get("best_metrics"):
        bm = method3_result["best_metrics"]
        lines.append(
            f"Best P/R/F1: {bm.get('precision',0):.4f} / "
            f"{bm.get('recall',0):.4f} / {bm.get('f1',0):.4f}"
        )

    lines.append("\nTop 5:")
    for i, (w, m) in enumerate(method3_result.get("all_results", [])[:5]):
        w_str = ", ".join(f"{w[mn]:.2f}" for mn in MODEL_ORDER)
        lines.append(
            f"  {i+1}. ({w_str}) -> mAP50={m.get('mAP50',0):.4f}, F1={m.get('f1',0):.4f}"
        )

    # Per-class
    if per_class_result:
        lines.append("\n--- Per-Class Optimal Weights ---")
        lines.append(f"{'Class':<25} {'Best Weights':<35} {'AP@0.5':>8}")
        lines.append("-" * 70)
        for cls_id, r in sorted(per_class_result.items()):
            w = r["best_weights"]
            w_str = ", ".join(f"{w[mn]:.2f}" for mn in MODEL_ORDER)
            lines.append(f"{r['class_name']:<25} ({w_str}){'':<13} {r['best_score']:.4f}")

    # Final comparison
    lines.append("\n--- Final Comparison ---")
    comparison = build_comparison_table(
        individual_results, method2_results, method3_result
    )
    lines.append(compare_methods(comparison))

    report = "\n".join(lines)
    print(f"\n{report}")

    report_path = output_dir / "experiment_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Voting Ensemble Experiments")
    parser.add_argument(
        "--force-inference", action="store_true",
        help="Force re-run model inference (ignore cache)",
    )
    parser.add_argument(
        "--skip-per-class", action="store_true",
        help="Skip per-class grid search",
    )
    parser.add_argument(
        "--grid-step", type=float, default=GRID_SEARCH_STEP,
        help=f"Grid search step size (default: {GRID_SEARCH_STEP})",
    )
    parser.add_argument(
        "--metric", type=str, default=GRID_SEARCH_METRIC,
        help=f"Metric to optimize (default: {GRID_SEARCH_METRIC})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    print("=" * 60)
    print("VOTING ENSEMBLE EXPERIMENTS")
    print("=" * 60)
    print(f"Models: {MODEL_ORDER}")
    print(f"Grid search step: {args.grid_step}")
    print(f"Optimization metric: {args.metric}")
    print(f"Output: {output_dir}")
    print()

    # 1. Prepare predictions (or load cached)
    print("Step 1: Preparing predictions...")
    predictions, ground_truth, model_perf = prepare_all_predictions(
        force_rerun=args.force_inference
    )

    # 2. Evaluate individual models
    individual_results = run_individual_model_evaluation(predictions, ground_truth)

    # 3. Method 2: Model-Weighted Voting
    method2_results = run_method2(predictions, ground_truth)

    # 4. Method 3: Grid Search
    method3_result = run_method3(
        predictions, ground_truth,
        step=args.grid_step,
        metric=args.metric,
    )

    # 5. Per-class grid search (optional)
    per_class_result = None
    if not args.skip_per_class:
        per_class_result = run_per_class_search(
            predictions, ground_truth, step=args.grid_step
        )

    # 6. Save results
    save_results(
        individual_results, method2_results, method3_result,
        per_class_result, output_dir,
    )

    # 7. Generate report
    generate_report(
        individual_results, method2_results, method3_result,
        per_class_result, output_dir,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
