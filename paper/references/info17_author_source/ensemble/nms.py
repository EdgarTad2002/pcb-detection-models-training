#!/usr/bin/env python3
"""
YOLO Ensemble Experiments
"""

import sys
import time
import warnings
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RUNS_DIR = BASE_DIR / "runs"
DATASET_DIR = BASE_DIR / "datasets" / "pcb-filtered-yolov8"
VAL_IMAGES_DIR = DATASET_DIR / "valid" / "images"
VAL_LABELS_DIR = DATASET_DIR / "valid" / "labels"
RESULTS_DIR = BASE_DIR / "ensemble" / "results"

MODEL_CONFIGS = {
    'yolov5': RUNS_DIR / 'yolov5_ultralytics/yolov5s-ultralytics2/weights/best.pt',
    'yolov8': RUNS_DIR / 'yolov8/pcb-filtered/weights/best.pt',
    'yolov9': RUNS_DIR / 'yolov9s_ultralytics/yolov9s-ultralytics/weights/best.pt',
    'yolov10': RUNS_DIR / 'yolov10/pcb-filtered/weights/best.pt',
    'yolov11': RUNS_DIR / 'yolov11/pcb-filtered/weights/best.pt',
    'yolov12': RUNS_DIR / 'yolov12/pcb-filtered/weights/best.pt',
}

# Explicit Class Names
CLASS_NAMES = ['IC', 'Capacitor', 'Connector', 'Electrolytic_Capacitor']

try:
    from ultralytics import YOLO
    from ensemble_boxes import nms as ensemble_nms
except ImportError:
    print("Error: Missing libraries. Run: pip install ultralytics ensemble-boxes")
    sys.exit(1)


# ----------------------------------------------------------------------------
# ROBUST METRICS
# ----------------------------------------------------------------------------

def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    x = np.linspace(0, 1, 101)  # 101-point interpolation (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves. """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)

    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            recall = tpc / (n_gt + 1e-16)
            r.append(recall[-1])

            precision = tpc / (tpc + fpc)
            p.append(precision[-1])

            ap.append(compute_ap(recall, precision))

    return ap, unique_classes.astype(int), r, p


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)

    if not isinstance(detections, torch.Tensor): detections = torch.tensor(detections)
    if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels)

    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))

    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        # --- FIX IS HERE: Convert iouv Tensor to numpy ---
        correct[matches[:, 1].astype(int)] = matches[:, 2:3] >= iouv.cpu().numpy()

    return torch.tensor(correct, dtype=torch.bool)


def evaluate_predictions(all_preds, all_targets):
    """ Evaluate metrics globally using accumulated predictions and targets. """
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()

    stats = []

    preds_by_img = defaultdict(list)
    targets_by_img = defaultdict(list)

    for p in all_preds: preds_by_img[p['image_id']].append(p)
    for t in all_targets: targets_by_img[t['image_id']].append(t)

    all_ids = set(preds_by_img.keys()) | set(targets_by_img.keys())

    for img_id in all_ids:
        preds = preds_by_img[img_id]
        targets = targets_by_img[img_id]

        tcls = torch.tensor([t['class_id'] for t in targets]) if targets else torch.tensor([])
        pcls = torch.tensor([p['class_id'] for p in preds]) if preds else torch.tensor([])

        if targets:
            tbox = torch.tensor([t['box_xyxy'] for t in targets])
            targets_tensor = torch.cat((tcls.view(-1, 1), tbox), 1)
        else:
            targets_tensor = torch.zeros((0, 5))

        if preds:
            pbox = torch.tensor([p['box_xyxy'] for p in preds])
            pconf = torch.tensor([p['score'] for p in preds])
            preds_tensor = torch.cat((pbox, pconf.view(-1, 1), pcls.view(-1, 1)), 1)
        else:
            preds_tensor = torch.zeros((0, 6))

        if preds_tensor.shape[0] == 0:
            if targets_tensor.shape[0] > 0:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        if targets_tensor.shape[0] > 0:
            correct = process_batch(preds_tensor, targets_tensor, iouv)
        else:
            correct = torch.zeros(preds_tensor.shape[0], niou, dtype=torch.bool)

        stats.append((correct, preds_tensor[:, 4], preds_tensor[:, 5], tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, conf, pred_cls, target_cls = stats
        ap, unique_classes, r, p = ap_per_class(tp[:, 0], conf, pred_cls, target_cls)

        mAP50 = np.mean(ap)
        per_class_res = {CLASS_NAMES[int(c)]: a for c, a in zip(unique_classes, ap)}

        return {'mAP50': mAP50, 'per_class': per_class_res}
    else:
        return {'mAP50': 0.0, 'per_class': {}}


# ----------------------------------------------------------------------------
# EXPERIMENTS
# ----------------------------------------------------------------------------

def load_data():
    print(f"Scanning {VAL_IMAGES_DIR}...")
    image_paths = sorted(list(VAL_IMAGES_DIR.glob("*.jpg")) +
                         list(VAL_IMAGES_DIR.glob("*.png")) +
                         list(VAL_IMAGES_DIR.glob("*.jpeg")))

    if not image_paths:
        print("❌ Error: No images found!")
        sys.exit(1)

    all_targets = []
    print("Loading Ground Truth...")
    for img_path in tqdm(image_paths):
        img_id = img_path.stem
        txt_path = VAL_LABELS_DIR / (img_id + ".txt")

        if txt_path.exists():
            with Image.open(img_path) as img:
                w, h = img.size
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h

                        all_targets.append({
                            'image_id': img_id,
                            'class_id': cls,
                            'box_xyxy': [x1, y1, x2, y2]
                        })
    return image_paths, all_targets


def run_experiment_1(models, image_paths, all_targets):
    print("\n=== Experiment 1: Individual Models ===")
    results = {}
    cached_preds = {m: {} for m in models}

    for name, model in models.items():
        print(f"\nTesting {name}...")
        all_preds = []
        model.predict(str(image_paths[0]), conf=0.001, verbose=False)  # Warmup

        for img_path in tqdm(image_paths):
            img_id = img_path.stem
            with Image.open(img_path) as img:
                w, h = img.size

            preds = model.predict(str(img_path), conf=0.001, max_det=300, verbose=False)

            img_preds_list = []
            for r in preds:
                if r.boxes:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()

                    for box, score, cls in zip(boxes, scores, classes):
                        p = {
                            'image_id': img_id,
                            'class_id': int(cls),
                            'box_xyxy': list(box),
                            'score': float(score),
                            'model': name
                        }
                        img_preds_list.append(p)
                        all_preds.append(p)

            cached_preds[name][img_id] = {'preds': img_preds_list, 'size': (w, h)}

        metrics = evaluate_predictions(all_preds, all_targets)
        print(f"   >>> {name} mAP@0.5: {metrics['mAP50']:.4f}")
        for cls, ap in metrics['per_class'].items():
            print(f"       {cls}: {ap:.4f}")

        results[name] = metrics

    return results, cached_preds


def run_experiment_2_nms(cached_preds, rankings, all_targets, image_paths):
    print("\n=== Experiment 2: NMS Ensemble ===")
    configs = ['top_2', 'top_3', 'top_4']
    results = {}

    for config in configs:
        n = int(config.split('_')[1])
        models = rankings[:n]
        print(f"\nEnsemble: {models}")

        all_ens_preds = []
        for img_path in tqdm(image_paths):
            img_id = img_path.stem
            boxes_list, scores_list, labels_list = [], [], []
            w, h = 0, 0

            for m in models:
                if img_id in cached_preds[m]:
                    data = cached_preds[m][img_id]
                    w, h = data['size']
                    if not data['preds']: continue

                    b = [[p['box_xyxy'][0] / w, p['box_xyxy'][1] / h, p['box_xyxy'][2] / w, p['box_xyxy'][3] / h] for p
                         in data['preds']]
                    s = [p['score'] for p in data['preds']]
                    l = [p['class_id'] for p in data['preds']]

                    boxes_list.append(b)
                    scores_list.append(s)
                    labels_list.append(l)

            if not boxes_list: continue

            boxes, scores, labels = ensemble_nms(boxes_list, scores_list, labels_list, iou_thr=0.5)

            for b, s, l in zip(boxes, scores, labels):
                all_ens_preds.append({
                    'image_id': img_id,
                    'class_id': int(l),
                    'score': float(s),
                    'box_xyxy': [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
                })

        metrics = evaluate_predictions(all_ens_preds, all_targets)
        print(f"   >>> {config} mAP@0.5: {metrics['mAP50']:.4f}")
        results[config] = metrics

    return results


def main():
    print("Loading Data...")
    image_paths, all_targets = load_data()

    models = {n: YOLO(str(p)) for n, p in MODEL_CONFIGS.items() if p.exists()}
    if not models: return

    # Exp 1
    exp1_res, cached_preds = run_experiment_1(models, image_paths, all_targets)
    rankings = sorted(exp1_res.keys(), key=lambda x: exp1_res[x]['mAP50'], reverse=True)

    # Exp 2
    exp2_res = run_experiment_2_nms(cached_preds, rankings, all_targets, image_paths)

    # Save Report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ENSEMBLE_REPORT_REFINED.md", "w") as f:
        f.write("# Ensemble Report (Refined)\n\n")
        f.write("## Individual Models\n")
        for k, v in exp1_res.items():
            f.write(f"- {k}: {v['mAP50']:.4f}\n")
            for cls, ap in v['per_class'].items():
                f.write(f"    - {cls}: {ap:.4f}\n")
        f.write("\n## Ensembles\n")
        for k, v in exp2_res.items():
            f.write(f"- {k}: {v['mAP50']:.4f}\n")

    print(f"\nDone. Report saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()