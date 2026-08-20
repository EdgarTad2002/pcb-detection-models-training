"""
Phase 4: Method 3 - Dynamic Weighted Voting with Grid Search

Use grid search to find optimal model weights for top-3 models
that maximize detection performance (mAP@0.5 or F1-score).

Reference: Şimşek et al. - "Automatic Meniscus Segmentation Using YOLO-Based Deep
Learning Models with Ensemble Methods in Knee MRI"
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from ensemble.voting_methods.box_utils import fuse_image_detections
from ensemble.voting_methods.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    GRID_SEARCH_FINE_STEP,
    GRID_SEARCH_METRIC,
    GRID_SEARCH_STEP,
    IOU_THRESHOLD,
    MODEL_ORDER,
)


def generate_weight_combinations(
    n_models: int = 3,
    step: float = 0.1,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> List[Tuple[float, ...]]:
    """
    Generate all valid weight combinations where sum(weights) = 1.0.

    Args:
        n_models: number of models (default 3)
        step: weight increment step
        min_weight: minimum weight per model
        max_weight: maximum weight per model

    Returns:
        List of weight tuples, e.g. [(0.5, 0.3, 0.2), ...]
    """
    # Number of discrete steps
    n_steps = round((max_weight - min_weight) / step)
    target = round(1.0 / step)

    combinations = []

    if n_models == 3:
        for i in range(n_steps + 1):
            w1 = min_weight + i * step
            if w1 > max_weight:
                break
            for j in range(n_steps + 1):
                w2 = min_weight + j * step
                if w2 > max_weight:
                    break
                w3_raw = 1.0 - w1 - w2
                # Check w3 is on grid and in range
                w3 = round(w3_raw, 10)
                if w3 < min_weight - 1e-9 or w3 > max_weight + 1e-9:
                    continue
                # Verify sum == 1.0
                if abs(w1 + w2 + w3 - 1.0) < 1e-9:
                    combinations.append((round(w1, 4), round(w2, 4), round(w3, 4)))
    else:
        # General recursive approach for arbitrary n_models
        _generate_recursive(
            n_models, step, min_weight, max_weight, target, [], combinations
        )

    return combinations


def _generate_recursive(
    n_remaining: int,
    step: float,
    min_weight: float,
    max_weight: float,
    target_sum: int,
    current: list,
    results: list,
):
    """Recursive helper for generating weight combinations."""
    if n_remaining == 1:
        w = round(target_sum * step, 4)
        if min_weight - 1e-9 <= w <= max_weight + 1e-9:
            results.append(tuple(current + [w]))
        return

    n_steps = round((max_weight - min_weight) / step)
    for i in range(n_steps + 1):
        w = round(min_weight + i * step, 4)
        remaining = target_sum - round(w / step)
        if remaining < 0:
            break
        _generate_recursive(
            n_remaining - 1, step, min_weight, max_weight, remaining,
            current + [w], results
        )


def evaluate_weight_combination(
    predictions: Dict[str, Dict[str, List[Dict]]],
    ground_truth: Dict[str, List[Dict]],
    model_names: List[str],
    weights: Tuple[float, ...],
    eval_func: Callable,
    iou_threshold: float = IOU_THRESHOLD,
    conf_threshold: float = CONF_THRESHOLD,
) -> Dict:
    """
    Apply fusion with given weights and evaluate.

    Args:
        predictions: {model_name: {image_name: [detections]}}
        ground_truth: {image_name: [gt_dicts]}
        model_names: ordered list of model names matching weights order
        weights: tuple of weights matching model_names order
        eval_func: evaluation function(fused_preds, ground_truth) -> metrics dict
        iou_threshold: IoU threshold for matching and NMS
        conf_threshold: confidence threshold

    Returns:
        Metrics dict from eval_func
    """
    model_weights = dict(zip(model_names, weights))

    # Fuse predictions for all images (no progress bar for speed)
    image_names = sorted(set().union(*(p.keys() for p in predictions.values())))
    fused_preds = {}

    for img_name in image_names:
        model_dets = {}
        for model_name in model_names:
            dets = predictions.get(model_name, {}).get(img_name, [])
            if dets:
                model_dets[model_name] = dets

        if model_dets:
            fused_preds[img_name] = fuse_image_detections(
                model_dets, model_weights, iou_threshold, conf_threshold
            )
        else:
            fused_preds[img_name] = []

    return eval_func(fused_preds, ground_truth)


def grid_search(
    predictions: Dict[str, Dict[str, List[Dict]]],
    ground_truth: Dict[str, List[Dict]],
    eval_func: Callable,
    model_names: List[str] = None,
    step: float = GRID_SEARCH_STEP,
    metric: str = GRID_SEARCH_METRIC,
    iou_threshold: float = IOU_THRESHOLD,
    conf_threshold: float = CONF_THRESHOLD,
    verbose: bool = True,
) -> Dict:
    """
    Exhaustive grid search for optimal model weights.

    Args:
        predictions: {model_name: {image_name: [detections]}}
        ground_truth: {image_name: [gt_dicts]}
        eval_func: evaluation function(fused_preds, ground_truth) -> metrics dict
                   Must return a dict containing the metric key.
        model_names: ordered list of model names (default: MODEL_ORDER)
        step: weight step size
        metric: metric key to optimize (from eval_func output)
        iou_threshold: IoU threshold
        conf_threshold: confidence threshold
        verbose: print progress

    Returns:
        {
            "best_weights": dict,
            "best_score": float,
            "best_metrics": dict,
            "all_results": list of (weights_dict, metrics),
            "search_time_s": float,
            "num_combinations": int,
        }
    """
    if model_names is None:
        model_names = MODEL_ORDER

    combinations = generate_weight_combinations(
        n_models=len(model_names), step=step
    )

    if verbose:
        print(f"Grid search: {len(combinations)} combinations, step={step}")
        print(f"Optimizing: {metric}")

    all_results = []
    best_score = -1.0
    best_weights = None
    best_metrics = None

    start_time = time.time()

    iterator = tqdm(combinations, desc="Grid search") if verbose else combinations
    for weights in iterator:
        metrics = evaluate_weight_combination(
            predictions, ground_truth, model_names, weights,
            eval_func, iou_threshold, conf_threshold,
        )

        weights_dict = dict(zip(model_names, weights))
        score = metrics.get(metric, 0.0)
        all_results.append((weights_dict, metrics))

        if score > best_score:
            best_score = score
            best_weights = weights_dict
            best_metrics = metrics

        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(best=f"{best_score:.4f}")

    elapsed = time.time() - start_time

    # Sort all results by metric descending
    all_results.sort(key=lambda x: x[1].get(metric, 0.0), reverse=True)

    if verbose:
        print(f"\nSearch completed in {elapsed:.1f}s")
        print(f"Best weights: {best_weights}")
        print(f"Best {metric}: {best_score:.4f}")

    return {
        "best_weights": best_weights,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "all_results": all_results,
        "search_time_s": elapsed,
        "num_combinations": len(combinations),
    }


def coarse_to_fine_search(
    predictions: Dict[str, Dict[str, List[Dict]]],
    ground_truth: Dict[str, List[Dict]],
    eval_func: Callable,
    model_names: List[str] = None,
    coarse_step: float = 0.2,
    fine_step: float = GRID_SEARCH_FINE_STEP,
    metric: str = GRID_SEARCH_METRIC,
    iou_threshold: float = IOU_THRESHOLD,
    conf_threshold: float = CONF_THRESHOLD,
) -> Dict:
    """
    Two-stage grid search: coarse pass then fine-grained refinement.

    Args:
        predictions, ground_truth, eval_func: same as grid_search
        model_names: ordered model names
        coarse_step: step size for first pass
        fine_step: step size for refinement around best
        metric: metric to optimize

    Returns:
        Same format as grid_search
    """
    if model_names is None:
        model_names = MODEL_ORDER

    print("=== Stage 1: Coarse search ===")
    coarse_result = grid_search(
        predictions, ground_truth, eval_func, model_names,
        step=coarse_step, metric=metric,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )

    best_coarse = coarse_result["best_weights"]
    print(f"Coarse best: {best_coarse} -> {coarse_result['best_score']:.4f}")

    # Generate fine-grained combinations around best coarse weights
    print(f"\n=== Stage 2: Fine search (step={fine_step}) ===")
    fine_combos = []
    best_vals = [best_coarse[m] for m in model_names]
    search_range = coarse_step  # search +/- coarse_step around best

    n_fine_steps = round(2 * search_range / fine_step) + 1
    for i in range(n_fine_steps):
        w1 = best_vals[0] - search_range + i * fine_step
        for j in range(n_fine_steps):
            w2 = best_vals[1] - search_range + j * fine_step
            w3 = round(1.0 - w1 - w2, 10)
            if w1 < 0 or w2 < 0 or w3 < 0:
                continue
            if w1 > 1 or w2 > 1 or w3 > 1:
                continue
            if abs(w1 + w2 + w3 - 1.0) < 1e-9:
                fine_combos.append((round(w1, 4), round(w2, 4), round(w3, 4)))

    # Deduplicate
    fine_combos = list(set(fine_combos))
    print(f"Fine search: {len(fine_combos)} combinations")

    all_results = []
    best_score = coarse_result["best_score"]
    best_weights = best_coarse
    best_metrics = coarse_result["best_metrics"]

    start_time = time.time()
    for weights in tqdm(fine_combos, desc="Fine search"):
        metrics = evaluate_weight_combination(
            predictions, ground_truth, model_names, weights,
            eval_func, iou_threshold, conf_threshold,
        )
        weights_dict = dict(zip(model_names, weights))
        score = metrics.get(metric, 0.0)
        all_results.append((weights_dict, metrics))

        if score > best_score:
            best_score = score
            best_weights = weights_dict
            best_metrics = metrics

    elapsed = time.time() - start_time
    total_time = coarse_result["search_time_s"] + elapsed

    all_results.sort(key=lambda x: x[1].get(metric, 0.0), reverse=True)

    print(f"\nFine search completed in {elapsed:.1f}s (total: {total_time:.1f}s)")
    print(f"Best weights: {best_weights}")
    print(f"Best {metric}: {best_score:.4f}")

    return {
        "best_weights": best_weights,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "all_results": all_results,
        "search_time_s": total_time,
        "num_combinations": coarse_result["num_combinations"] + len(fine_combos),
    }


def per_class_grid_search(
    predictions: Dict[str, Dict[str, List[Dict]]],
    ground_truth: Dict[str, List[Dict]],
    eval_func: Callable,
    model_names: List[str] = None,
    class_names: List[str] = None,
    step: float = GRID_SEARCH_STEP,
    metric: str = GRID_SEARCH_METRIC,
    iou_threshold: float = IOU_THRESHOLD,
    conf_threshold: float = CONF_THRESHOLD,
) -> Dict:
    """
    Find optimal weights for each class separately.

    Filters predictions and ground truth to single class before searching.

    Args:
        predictions, ground_truth, eval_func: same as grid_search
        model_names: ordered model names
        class_names: list of class names (for display)
        step: weight step size
        metric: metric to optimize

    Returns:
        {class_id: {"class_name": str, "best_weights": dict, "best_score": float, ...}}
    """
    if model_names is None:
        model_names = MODEL_ORDER
    if class_names is None:
        class_names = CLASS_NAMES

    # Find all class IDs present in ground truth
    class_ids = set()
    for gts in ground_truth.values():
        for gt in gts:
            class_ids.add(gt["class"])
    class_ids = sorted(class_ids)

    results = {}
    for cls_id in class_ids:
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"\n{'='*50}")
        print(f"Grid search for class {cls_id}: {cls_name}")
        print(f"{'='*50}")

        # Filter predictions to this class only
        cls_preds = {}
        for model_name, model_preds in predictions.items():
            cls_preds[model_name] = {}
            for img_name, dets in model_preds.items():
                cls_preds[model_name][img_name] = [
                    d for d in dets if d["class"] == cls_id
                ]

        # Filter ground truth to this class only
        cls_gt = {}
        for img_name, gts in ground_truth.items():
            cls_gt[img_name] = [g for g in gts if g["class"] == cls_id]

        result = grid_search(
            cls_preds, cls_gt, eval_func, model_names,
            step=step, metric=metric,
            iou_threshold=iou_threshold, conf_threshold=conf_threshold,
            verbose=True,
        )

        result["class_name"] = cls_name
        results[cls_id] = result

    return results


if __name__ == "__main__":
    print("Dynamic weighted voting module loaded.")
    combos = generate_weight_combinations(n_models=3, step=0.1)
    print(f"3 models, step=0.1: {len(combos)} combinations")
    combos_fine = generate_weight_combinations(n_models=3, step=0.05)
    print(f"3 models, step=0.05: {len(combos_fine)} combinations")
