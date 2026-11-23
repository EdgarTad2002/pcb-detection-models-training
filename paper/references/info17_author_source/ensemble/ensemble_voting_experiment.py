#!/usr/bin/env python3
"""
Ensemble Voting Strategies Experiment

Uses the three voting strategies from ensembleObjectDetection library:
1. Affirmative - Any model's detection is valid
2. Consensus - Majority of models must agree
3. Unanimous - All models must agree

Tests with Top 2, 3, 4, and 7 YOLO models.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Configuration
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
DATASET_DIR = Path(__file__).parent.parent / "datasets" / "pcb-filtered-yolov8"
VAL_IMAGES_DIR = DATASET_DIR / "valid" / "images"
VAL_LABELS_DIR = DATASET_DIR / "valid" / "labels"
OUTPUT_DIR = Path(__file__).parent / "results_voting_strategies"

# Class names
CLASS_NAMES = ['IC', 'Capacitor', 'Connector', 'Electrolytic_Capacitor']
CLASS_NAME_MAP = {0: 'IC', 1: 'Capacitor', 2: 'Connector', 3: 'Electrolytic_Capacitor'}

# Model configurations ranked by overall mAP@0.5
# Using the best.pt weights from training runs
RUNS_DIR = Path(__file__).parent.parent / "runs"
MODEL_CONFIGS = {
    'yolov11': {'weight': RUNS_DIR / 'yolov11/pcb-filtered/weights/best.pt', 'type': 'ultralytics'},
    'yolov9': {'weight': RUNS_DIR / 'yolov9s_ultralytics/yolov9s-ultralytics/weights/best.pt', 'type': 'ultralytics'},
    'yolov8': {'weight': RUNS_DIR / 'yolov8/pcb-filtered/weights/best.pt', 'type': 'ultralytics'},
    'yolov12': {'weight': RUNS_DIR / 'yolov12/pcb-filtered/weights/best.pt', 'type': 'ultralytics'},
    'yolov10': {'weight': RUNS_DIR / 'yolov10/pcb-filtered/weights/best.pt', 'type': 'ultralytics'},
    'yolov5': {'weight': RUNS_DIR / 'yolov5_ultralytics/yolov5s-ultralytics2/weights/best.pt', 'type': 'ultralytics'},
    'yolov3': {'weight': RUNS_DIR / 'yolov3_ultralytics/yolov3-ultralytics5/weights/best.pt', 'type': 'ultralytics'},
}

# Top model combinations to test (ranked by mAP@0.5)
# Available trained models: yolov11, yolov9, yolov8, yolov12, yolov10, yolov5, yolov3
MODEL_COMBINATIONS = {
    'top2': ['yolov11', 'yolov9'],
    'top3': ['yolov11', 'yolov9', 'yolov8'],
    'top4': ['yolov11', 'yolov9', 'yolov8', 'yolov12'],
    'top7': ['yolov11', 'yolov9', 'yolov8', 'yolov12', 'yolov10', 'yolov5', 'yolov3'],
}


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image dimensions using PIL."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def prettify_xml(elem) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def predictions_to_pascal_voc_xml(
    image_name: str,
    predictions: List[Dict],
    image_width: int,
    image_height: int,
    output_path: str
):
    """Convert predictions to Pascal VOC XML format."""
    top = ET.Element('annotation')

    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'

    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = image_name

    childPath = ET.SubElement(top, 'path')
    childPath.text = str(output_path)

    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'

    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(image_width)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(image_height)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = '3'

    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = '0'

    for pred in predictions:
        childObject = ET.SubElement(top, 'object')

        childName = ET.SubElement(childObject, 'name')
        childName.text = pred['class_name']

        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'

        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'

        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'

        childConfidence = ET.SubElement(childObject, 'confidence')
        childConfidence.text = f"{pred['confidence']:.4f}"

        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(int(pred['bbox'][0]))
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(int(pred['bbox'][1]))
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(int(pred['bbox'][2]))
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(int(pred['bbox'][3]))

    xml_str = prettify_xml(top)

    xml_filename = Path(image_name).stem + '.xml'
    xml_path = Path(output_path) / xml_filename
    with open(xml_path, 'w') as f:
        f.write(xml_str)


def run_model_inference(model_name: str, config: Dict, images_dir: Path, output_dir: Path) -> Dict:
    """Run inference for a single model and save predictions as Pascal VOC XML."""
    weight_path = config['weight']

    if not weight_path.exists():
        print(f"  Warning: Weight file not found: {weight_path}")
        return {}

    print(f"  Loading {model_name}...")
    model = YOLO(str(weight_path))

    # Create output directory for this model
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all validation images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    predictions_by_image = {}

    print(f"  Running inference on {len(image_files)} images...")
    for img_path in image_files:
        # Get image dimensions
        width, height = get_image_dimensions(str(img_path))

        # Run inference
        results = model.predict(str(img_path), conf=0.001, verbose=False)

        predictions = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)

                for box, score, cls in zip(boxes, scores, classes):
                    predictions.append({
                        'bbox': box.tolist(),
                        'confidence': float(score),
                        'class_id': int(cls),
                        'class_name': CLASS_NAME_MAP[int(cls)]
                    })

        # Save as Pascal VOC XML
        predictions_to_pascal_voc_xml(
            img_path.name,
            predictions,
            width,
            height,
            model_output_dir
        )

        predictions_by_image[img_path.name] = predictions

    return predictions_by_image


def bb_intersection_over_union(boxA: List[float], boxB: List[float]) -> float:
    """Calculate IoU between two boxes [xmin, ymin, xmax, ymax]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def group_overlapping_boxes(all_predictions: List[Dict], iou_threshold: float = 0.5) -> List[List[Dict]]:
    """Group overlapping predictions from different models."""
    if not all_predictions:
        return []

    groups = []
    remaining = all_predictions.copy()

    while remaining:
        current = remaining.pop(0)
        current_group = [current]

        i = 0
        while i < len(remaining):
            candidate = remaining[i]
            # Check if same class and overlapping
            if (current['class_name'] == candidate['class_name'] and
                bb_intersection_over_union(current['bbox'], candidate['bbox']) > iou_threshold):
                current_group.append(candidate)
                remaining.pop(i)
            else:
                i += 1

        groups.append(current_group)

    return groups


def apply_nms_to_group(group: List[Dict], nms_threshold: float = 0.3) -> Dict:
    """Apply NMS to a group and return the best detection."""
    if len(group) == 1:
        return group[0]

    # Sort by confidence
    sorted_group = sorted(group, key=lambda x: x['confidence'], reverse=True)

    # Average the bounding boxes weighted by confidence
    total_conf = sum(p['confidence'] for p in group)
    avg_bbox = [0, 0, 0, 0]
    for p in group:
        weight = p['confidence'] / total_conf
        for i in range(4):
            avg_bbox[i] += p['bbox'][i] * weight

    # Average confidence
    avg_conf = total_conf / len(group)

    return {
        'bbox': avg_bbox,
        'confidence': avg_conf,
        'class_id': group[0]['class_id'],
        'class_name': group[0]['class_name'],
        'num_votes': len(group)
    }


def ensemble_predictions(
    model_predictions: Dict[str, Dict[str, List[Dict]]],
    strategy: str,
    num_models: int
) -> Dict[str, List[Dict]]:
    """
    Ensemble predictions using the specified voting strategy.

    Args:
        model_predictions: Dict[model_name][image_name] -> List[predictions]
        strategy: 'affirmative', 'consensus', or 'unanimous'
        num_models: Total number of models

    Returns:
        Dict[image_name] -> List[ensembled_predictions]
    """
    # Get all image names
    all_images = set()
    for model_preds in model_predictions.values():
        all_images.update(model_preds.keys())

    ensembled = {}

    for image_name in all_images:
        # Collect all predictions for this image from all models
        all_preds = []
        for model_name, model_preds in model_predictions.items():
            if image_name in model_preds:
                for pred in model_preds[image_name]:
                    pred_copy = pred.copy()
                    pred_copy['model'] = model_name
                    all_preds.append(pred_copy)

        # Group overlapping predictions
        groups = group_overlapping_boxes(all_preds)

        # Apply voting strategy
        final_predictions = []
        for group in groups:
            num_votes = len(group)

            if strategy == 'affirmative':
                # Any detection is valid
                final_pred = apply_nms_to_group(group)
                final_predictions.append(final_pred)

            elif strategy == 'consensus':
                # Majority must agree (>= ceil(num_models/2))
                threshold = math.ceil(num_models / 2)
                if num_votes >= threshold:
                    final_pred = apply_nms_to_group(group)
                    final_predictions.append(final_pred)

            elif strategy == 'unanimous':
                # All models must agree
                if num_votes == num_models:
                    final_pred = apply_nms_to_group(group)
                    final_predictions.append(final_pred)

        ensembled[image_name] = final_predictions

    return ensembled


def load_ground_truth() -> Tuple[Dict, Dict]:
    """Load ground truth annotations in COCO format."""
    images = []
    annotations = []
    ann_id = 1

    label_files = list(VAL_LABELS_DIR.glob("*.txt"))

    for img_id, label_file in enumerate(label_files, 1):
        image_name = label_file.stem + ".jpg"
        image_path = VAL_IMAGES_DIR / image_name

        if not image_path.exists():
            image_name = label_file.stem + ".png"
            image_path = VAL_IMAGES_DIR / image_name

        if not image_path.exists():
            continue

        width, height = get_image_dimensions(str(image_path))

        images.append({
            'id': img_id,
            'file_name': image_name,
            'width': width,
            'height': height
        })

        # Parse YOLO format labels
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height

                    x1 = x_center - w / 2
                    y1 = y_center - h / 2

                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': cls_id + 1,  # COCO uses 1-indexed
                        'bbox': [x1, y1, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    ann_id += 1

    # Create image name to ID mapping
    image_name_to_id = {img['file_name']: img['id'] for img in images}

    coco_gt = {
        'info': {'description': 'PCB Filtered Dataset'},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i+1, 'name': name, 'supercategory': 'component'} for i, name in enumerate(CLASS_NAMES)]
    }

    return coco_gt, image_name_to_id


def convert_to_coco_results(
    predictions: Dict[str, List[Dict]],
    image_name_to_id: Dict[str, int]
) -> List[Dict]:
    """Convert predictions to COCO results format."""
    results = []

    for image_name, preds in predictions.items():
        if image_name not in image_name_to_id:
            continue

        image_id = image_name_to_id[image_name]

        for pred in preds:
            x1, y1, x2, y2 = pred['bbox']
            w = x2 - x1
            h = y2 - y1

            results.append({
                'image_id': image_id,
                'category_id': pred['class_id'] + 1,  # COCO uses 1-indexed
                'bbox': [x1, y1, w, h],
                'score': pred['confidence']
            })

    return results


def evaluate_coco(coco_gt_dict: Dict, results: List[Dict], verbose: bool = False) -> Dict:
    """Evaluate using COCO metrics."""
    import tempfile
    import io
    import sys

    if not results:
        return {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0}

    # Save ground truth to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_gt_dict, f)
        gt_path = f.name

    # Save results to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(results, f)
        results_path = f.name

    try:
        # Suppress COCO output
        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(results_path)

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if not verbose:
            sys.stdout = old_stdout

        metrics = {
            'mAP@0.5': float(coco_eval.stats[1]),  # AP @ IoU=0.50
            'mAP@0.5:0.95': float(coco_eval.stats[0]),  # AP @ IoU=0.50:0.95
        }
    except Exception as e:
        if not verbose:
            sys.stdout = old_stdout
        print(f"  Evaluation error: {e}")
        metrics = {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0}
    finally:
        os.unlink(gt_path)
        os.unlink(results_path)

    return metrics


def calculate_precision_recall_f1(
    predictions: Dict[str, List[Dict]],
    coco_gt_dict: Dict,
    image_name_to_id: Dict[str, int],
    conf_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 at a given confidence threshold."""
    # Build ground truth lookup
    gt_by_image = defaultdict(list)
    for ann in coco_gt_dict['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    tp = 0
    fp = 0
    total_gt = len(coco_gt_dict['annotations'])

    for image_name, preds in predictions.items():
        if image_name not in image_name_to_id:
            continue

        image_id = image_name_to_id[image_name]
        gts = gt_by_image[image_id].copy()

        # Filter by confidence
        filtered_preds = [p for p in preds if p['confidence'] >= conf_threshold]
        sorted_preds = sorted(filtered_preds, key=lambda x: x['confidence'], reverse=True)

        for pred in sorted_preds:
            best_iou = 0
            best_gt_idx = -1

            pred_bbox = pred['bbox']
            pred_class = pred['class_id'] + 1  # COCO 1-indexed

            for gt_idx, gt in enumerate(gts):
                if gt['category_id'] != pred_class:
                    continue

                gt_bbox = gt['bbox']
                # Convert COCO format [x, y, w, h] to [x1, y1, x2, y2]
                gt_box = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]

                iou = bb_intersection_over_union(pred_bbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= 0.5:
                tp += 1
                gts.pop(best_gt_idx)
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def main():
    print("=" * 70)
    print("ENSEMBLE VOTING STRATEGIES EXPERIMENT")
    print("=" * 70)
    print(f"\nStrategies: Affirmative, Consensus, Unanimous")
    print(f"Model Combinations: Top 2, 3, 4, 7")
    print(f"Validation Images: {VAL_IMAGES_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Run inference for all models and cache results
    print("\n" + "=" * 70)
    print("STEP 1: Running Inference for All Models")
    print("=" * 70)

    inference_cache_dir = OUTPUT_DIR / "inference_cache"
    inference_cache_dir.mkdir(exist_ok=True)

    all_model_predictions = {}

    for model_name, config in MODEL_CONFIGS.items():
        cache_file = inference_cache_dir / f"{model_name}_predictions.json"

        if cache_file.exists():
            print(f"\n{model_name}: Loading from cache...")
            with open(cache_file, 'r') as f:
                all_model_predictions[model_name] = json.load(f)
        else:
            print(f"\n{model_name}: Running inference...")
            preds = run_model_inference(model_name, config, VAL_IMAGES_DIR, inference_cache_dir)
            all_model_predictions[model_name] = preds

            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(preds, f)

    # Step 2: Load ground truth
    print("\n" + "=" * 70)
    print("STEP 2: Loading Ground Truth")
    print("=" * 70)

    coco_gt_dict, image_name_to_id = load_ground_truth()
    print(f"  Loaded {len(coco_gt_dict['images'])} images, {len(coco_gt_dict['annotations'])} annotations")

    # Step 3: Run experiments
    print("\n" + "=" * 70)
    print("STEP 3: Running Ensemble Experiments")
    print("=" * 70)

    results = []

    # First, evaluate individual models
    print("\n--- Individual Model Performance ---")
    for model_name in MODEL_CONFIGS.keys():
        preds = {k: v for k, v in all_model_predictions[model_name].items()}
        coco_results = convert_to_coco_results(preds, image_name_to_id)

        if coco_results:
            metrics = evaluate_coco(coco_gt_dict, coco_results)
            precision, recall, f1 = calculate_precision_recall_f1(
                preds, coco_gt_dict, image_name_to_id
            )

            result = {
                'config': f'individual_{model_name}',
                'models': [model_name],
                'num_models': 1,
                'strategy': 'none',
                'mAP@0.5': metrics['mAP@0.5'],
                'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            results.append(result)

            print(f"\n{model_name}:")
            print(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
            print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
            print(f"  P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")

    # Run ensemble experiments
    strategies = ['affirmative', 'consensus', 'unanimous']

    for combo_name, model_list in MODEL_COMBINATIONS.items():
        print(f"\n\n--- {combo_name.upper()} ({', '.join(model_list)}) ---")

        # Get predictions for this combination
        combo_predictions = {m: all_model_predictions[m] for m in model_list if m in all_model_predictions}
        num_models = len(combo_predictions)

        if num_models < 2:
            print(f"  Skipping: Only {num_models} models available")
            continue

        for strategy in strategies:
            print(f"\n  Strategy: {strategy}")

            # Apply ensemble
            ensembled = ensemble_predictions(combo_predictions, strategy, num_models)

            # Convert to COCO format
            coco_results = convert_to_coco_results(ensembled, image_name_to_id)

            if not coco_results:
                print(f"    No predictions (strategy too strict)")
                result = {
                    'config': f'{combo_name}_{strategy}',
                    'models': model_list,
                    'num_models': num_models,
                    'strategy': strategy,
                    'mAP@0.5': 0.0,
                    'mAP@0.5:0.95': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
            else:
                metrics = evaluate_coco(coco_gt_dict, coco_results)
                precision, recall, f1 = calculate_precision_recall_f1(
                    ensembled, coco_gt_dict, image_name_to_id
                )

                result = {
                    'config': f'{combo_name}_{strategy}',
                    'models': model_list,
                    'num_models': num_models,
                    'strategy': strategy,
                    'mAP@0.5': metrics['mAP@0.5'],
                    'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }

                print(f"    mAP@0.5: {metrics['mAP@0.5']:.4f}")
                print(f"    mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
                print(f"    P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")

            results.append(result)

    # Save results
    results_file = OUTPUT_DIR / "voting_strategies_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {results_file}")




if __name__ == "__main__":
    main()
