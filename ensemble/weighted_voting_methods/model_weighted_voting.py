"""
Phase 3: Method 2 - Model-Weighted Voting

Assign static weights to top-3 models based on their individual performance ranking.
Top-performing models get higher weights in box fusion.

Reference: Şimşek et al. - "Automatic Meniscus Segmentation Using YOLO-Based Deep
Learning Models with Ensemble Methods in Knee MRI"
"""

from typing import Dict, List, Tuple

from tqdm import tqdm

from ensemble.voting_methods.box_utils import fuse_image_detections
from ensemble.voting_methods.config import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MODEL_ORDER,
    TOP3_WEIGHTS,
)


def model_weighted_fusion(
    predictions: Dict[str, Dict[str, List[Dict]]],
    model_weights: Dict[str, float],
    iou_threshold: float = IOU_THRESHOLD,
    conf_threshold: float = CONF_THRESHOLD,
) -> Dict[str, List[Dict]]:
    """
    Apply model-weighted box fusion across all images.

    Args:
        predictions: {model_name: {image_name: [detections]}}
        model_weights: {model_name: weight} where weights sum to ~1.0
        iou_threshold: IoU threshold for matching and NMS
        conf_threshold: confidence threshold for final filtering

    Returns:
        {image_name: [fused_detections]}
    """
    # Collect all image names across models
    image_names = set()
    for model_preds in predictions.values():
        image_names.update(model_preds.keys())
    image_names = sorted(image_names)

    results = {}
    for img_name in tqdm(image_names, desc="Model-weighted fusion"):
        # Gather detections from each model for this image
        model_dets = {}
        for model_name in predictions:
            dets = predictions[model_name].get(img_name, [])
            if dets:
                model_dets[model_name] = dets

        if not model_dets:
            results[img_name] = []
            continue

        fused = fuse_image_detections(
            model_dets, model_weights, iou_threshold, conf_threshold
        )
        results[img_name] = fused

    return results


def run_weighted_voting_experiments(
    predictions: Dict[str, Dict[str, List[Dict]]],
    weight_configs: Dict[str, Dict[str, float]] = None,
    iou_threshold: float = IOU_THRESHOLD,
    conf_threshold: float = CONF_THRESHOLD,
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Run model-weighted fusion with all weight configurations.

    Args:
        predictions: {model_name: {image_name: [detections]}}
        weight_configs: {config_name: {model_name: weight}}
                        Defaults to TOP3_WEIGHTS from config.
        iou_threshold: IoU threshold for matching and NMS
        conf_threshold: confidence threshold for filtering

    Returns:
        {config_name: {image_name: [fused_detections]}}
    """
    if weight_configs is None:
        weight_configs = TOP3_WEIGHTS

    all_results = {}
    for config_name, weights in weight_configs.items():
        print(f"\n--- Config: {config_name} ---")
        print(f"    Weights: {weights}")

        fused_preds = model_weighted_fusion(
            predictions, weights, iou_threshold, conf_threshold
        )

        total_dets = sum(len(d) for d in fused_preds.values())
        print(f"    Total fused detections: {total_dets}")

        all_results[config_name] = fused_preds

    return all_results


if __name__ == "__main__":
    from ensemble.voting_methods.prepare_predictions import load_cached_data

    predictions, ground_truth, model_perf = load_cached_data()

    print("Running Method 2: Model-Weighted Voting")
    print(f"Models: {MODEL_ORDER}")
    print(f"Images: {len(next(iter(predictions.values())))}")

    results = run_weighted_voting_experiments(predictions)

    print("\n=== Summary ===")
    for config_name, preds in results.items():
        total = sum(len(d) for d in preds.values())
        print(f"  {config_name}: {total} total detections")
