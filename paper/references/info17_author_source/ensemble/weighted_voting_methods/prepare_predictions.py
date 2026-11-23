"""
Phase 1: Data Preparation

Generate and cache predictions from top-3 YOLO models for the test dataset.
Saves predictions.json, ground_truth.json, and model_performance.json.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO

from ensemble.voting_methods.config import (
    CACHE_DIR,
    CLASS_NAMES,
    CONF_THRESHOLD,
    MODEL_ORDER,
    MODEL_PATHS,
    MODEL_PERFORMANCE,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
)


def load_models(model_paths: Dict[str, Path]) -> Dict[str, YOLO]:
    """Load top-3 YOLO models."""
    models = {}
    for name, path in model_paths.items():
        print(f"Loading {name} from {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        models[name] = YOLO(str(path))
    print(f"Loaded {len(models)} models.")
    return models


def run_inference(
    model: YOLO,
    image_path: str,
    img_width: int,
    img_height: int,
    conf_thresh: float = 0.001,
) -> List[Dict]:
    """
    Run inference on a single image.

    Returns list of detections with normalized [x1, y1, x2, y2] coordinates.
    Uses a very low conf_thresh during inference to capture all potential detections;
    filtering happens later during fusion.
    """
    results = model(image_path, conf=conf_thresh, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            if cls_id >= len(CLASS_NAMES):
                continue

            # Normalize coordinates to [0, 1]
            box_norm = [
                float(xyxy[0]) / img_width,
                float(xyxy[1]) / img_height,
                float(xyxy[2]) / img_width,
                float(xyxy[3]) / img_height,
            ]

            detections.append({
                "class": cls_id,
                "confidence": conf,
                "box_norm": box_norm,
            })

    return detections


def load_ground_truth(
    label_dir: Path, image_dir: Path
) -> Tuple[Dict, Dict[str, Tuple[int, int]]]:
    """
    Parse YOLO format labels and convert to normalized [x1, y1, x2, y2].

    Returns:
        ground_truth: {image_filename: [{"class": int, "box_norm": [x1,y1,x2,y2]}]}
        image_sizes: {image_filename: (width, height)}
    """
    ground_truth = {}
    image_sizes = {}

    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    for img_path in tqdm(image_files, desc="Loading ground truth"):
        label_path = label_dir / (img_path.stem + ".txt")
        img = Image.open(img_path)
        w, h = img.size
        image_sizes[img_path.name] = (w, h)

        gts = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Convert from YOLO center format to normalized xyxy
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    gts.append({
                        "class": cls_id,
                        "box_norm": [x1, y1, x2, y2],
                    })

        ground_truth[img_path.name] = gts

    return ground_truth, image_sizes


def prepare_all_predictions(
    model_paths: Dict[str, Path] = None,
    image_dir: Path = None,
    label_dir: Path = None,
    output_dir: Path = None,
    conf_thresh: float = 0.001,
    force_rerun: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    """
    Main function: generate predictions.json, ground_truth.json, model_performance.json.

    If cached files exist and force_rerun is False, loads from cache.

    Returns:
        predictions, ground_truth, model_performance
    """
    model_paths = model_paths or MODEL_PATHS
    image_dir = image_dir or TEST_IMAGES_DIR
    label_dir = label_dir or TEST_LABELS_DIR
    output_dir = output_dir or CACHE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    pred_file = output_dir / "predictions.json"
    gt_file = output_dir / "ground_truth.json"
    perf_file = output_dir / "model_performance.json"
    sizes_file = output_dir / "image_sizes.json"

    # Check cache
    if not force_rerun and pred_file.exists() and gt_file.exists():
        print("Loading cached predictions and ground truth...")
        with open(pred_file, "r") as f:
            predictions = json.load(f)
        with open(gt_file, "r") as f:
            ground_truth = json.load(f)
        with open(perf_file, "r") as f:
            model_performance = json.load(f)
        print(f"  Loaded predictions for {len(predictions)} models")
        print(f"  Loaded ground truth for {len(ground_truth)} images")
        return predictions, ground_truth, model_performance

    # Load ground truth and image sizes
    print("Loading ground truth labels...")
    ground_truth, image_sizes = load_ground_truth(label_dir, image_dir)
    print(f"  {len(ground_truth)} images, "
          f"{sum(len(gts) for gts in ground_truth.values())} total GT boxes")

    # Load models
    models = load_models(model_paths)

    # Run inference for each model
    predictions = {}
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    for model_name in MODEL_ORDER:
        if model_name not in models:
            continue
        print(f"\nRunning inference with {model_name}...")
        model = models[model_name]
        model_preds = {}

        for img_path in tqdm(image_files, desc=f"  {model_name}"):
            w, h = image_sizes[img_path.name]
            dets = run_inference(model, str(img_path), w, h, conf_thresh)
            model_preds[img_path.name] = dets

        predictions[model_name] = model_preds
        total_dets = sum(len(d) for d in model_preds.values())
        print(f"  {model_name}: {total_dets} total detections across {len(model_preds)} images")

    # Save to cache
    print("\nSaving to cache...")
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)
    with open(gt_file, "w") as f:
        json.dump(ground_truth, f, indent=2)
    with open(perf_file, "w") as f:
        json.dump(MODEL_PERFORMANCE, f, indent=2)
    with open(sizes_file, "w") as f:
        json.dump(image_sizes, f, indent=2)

    print(f"Saved to {output_dir}/")
    return predictions, ground_truth, MODEL_PERFORMANCE


def load_cached_data(cache_dir: Path = None) -> Tuple[Dict, Dict, Dict]:
    """Load previously cached predictions and ground truth."""
    cache_dir = cache_dir or CACHE_DIR
    pred_file = cache_dir / "predictions.json"
    gt_file = cache_dir / "ground_truth.json"
    perf_file = cache_dir / "model_performance.json"

    for f in [pred_file, gt_file, perf_file]:
        if not f.exists():
            raise FileNotFoundError(
                f"Cache file not found: {f}\n"
                "Run prepare_all_predictions() first."
            )

    with open(pred_file, "r") as f:
        predictions = json.load(f)
    with open(gt_file, "r") as f:
        ground_truth = json.load(f)
    with open(perf_file, "r") as f:
        model_performance = json.load(f)

    return predictions, ground_truth, model_performance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare predictions for voting ensemble")
    parser.add_argument("--force", action="store_true", help="Force re-run inference")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for inference")
    args = parser.parse_args()

    predictions, ground_truth, model_perf = prepare_all_predictions(
        conf_thresh=args.conf,
        force_rerun=args.force,
    )

    print("\n=== Summary ===")
    for model_name, preds in predictions.items():
        total = sum(len(d) for d in preds.values())
        print(f"  {model_name}: {total} detections across {len(preds)} images")
    total_gt = sum(len(gts) for gts in ground_truth.values())
    print(f"  Ground truth: {total_gt} boxes across {len(ground_truth)} images")
