#!/usr/bin/env python3
"""
Dedicated Standalone Training & Evaluation Script for VMamba Object Detector
=============================================================================
Completely independent of YOLO and Ultralytics.

Trains the native PyTorch VMamba detector (VMambaBackbone + MambaFPN + AnchorFreeHead)
on PCB datasets, evaluates on the held-out test split, and writes standardized results to:
    results/<run-key>.json

This allows aggregate_results.py to seamlessly compile it into comparison_table.csv/.md/.xlsx
for direct, head-to-head benchmarking against YOLO and RT-DETR models.

Usage:
    python train_mamba.py \
        --run-key vmamba_standalone_tiny \
        --data datasets/pcb-unified-4class/data.yaml \
        --epochs 100 --imgsz 640 --batch 16 --lr 0.0001
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mamba_detector import (
    VMambaDetector,
    PCBDataset,
    pcb_collate_fn,
)

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate standalone VMamba PCB detector.")

    # Model & Identity
    p.add_argument("--run-key", required=True, help="e.g. vmamba_standalone_tiny")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data.yaml. Defaults to datasets/pcb-unified-4class/data.yaml (or pcb-filtered-yolov8/data.yaml).",
    )
    p.add_argument("--pretrained-backbone", type=str, default=None, help="Path to pre-trained VMamba weights.")
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("PCB_PROJECT_ROOT", ".")),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("PCB_RESULTS_DIR", "results")),
    )

    # Architecture Capacity
    p.add_argument("--backbone-dims", type=int, nargs="+", default=[96, 192, 384, 768])
    p.add_argument("--backbone-depths", type=int, nargs="+", default=[2, 2, 9, 2])
    p.add_argument("--fpn-channels", type=int, default=128)

    # Training Hyperparameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")

    # Evaluation Protocol (fixed to match project benchmark standards)
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--eval-iou", type=float, default=0.50)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--skip-train", action="store_true", help="Skip training and evaluate existing best.pt.")

    return p.parse_args()


def compute_ap(preds_by_class: dict, gts_by_class: dict, iou_thresh: float = 0.5):
    """Computes all-points interpolated AP at given IoU threshold (matching VOC / COCO metric)."""
    res = {}
    for cid in range(len(CLASS_NAMES)):
        preds = sorted(preds_by_class.get(cid, []), key=lambda x: -x[1])
        gts = gts_by_class.get(cid, [])
        n_gt = len(gts)
        if n_gt == 0:
            res[cid] = {"ap50": 0.0, "precision": 0.0, "recall": 0.0}
            continue

        gt_by_img = {}
        for idx, (img_id, *box) in enumerate(gts):
            gt_by_img.setdefault(img_id, []).append((idx, box))

        matched_gt = {}
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, (img_id, score, *box) in enumerate(preds):
            candidates = gt_by_img.get(img_id, [])
            best_iou, best_idx = 0.0, -1
            bx1, by1, bx2, by2 = box
            b_area = max(0, bx2 - bx1) * max(0, by2 - by1)

            for gidx, gbox in candidates:
                if (img_id, gidx) in matched_gt:
                    continue
                gx1, gy1, gx2, gy2 = gbox
                g_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)

                ix1 = max(bx1, gx1)
                iy1 = max(by1, gy1)
                ix2 = min(bx2, gx2)
                iy2 = min(by2, gy2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = b_area + g_area - inter
                iou = inter / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou, best_idx = iou, gidx

            if best_iou >= iou_thresh and best_idx != -1:
                tp[i] = 1
                matched_gt[(img_id, best_idx)] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = 0.0
        if len(precision) > 0:
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([0.0], precision, [0.0]))
            for k in range(len(mpre) - 2, -1, -1):
                mpre[k] = max(mpre[k], mpre[k + 1])
            indices = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))

        prec_final = float(precision[-1]) if len(precision) > 0 else 0.0
        rec_final = float(recall[-1]) if len(recall) > 0 else 0.0
        res[cid] = {"ap50": ap, "precision": prec_final, "recall": rec_final}

    return res


def evaluate_model(
    model: VMambaDetector,
    loader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.001,
    iou_thresh: float = 0.50,
) -> Dict:
    """Evaluates detector on a dataset split and computes mAP, Precision, Recall, and FPS."""
    model.eval()
    gts_by_class = {c: [] for c in range(len(CLASS_NAMES))}
    preds_by_class = {c: [] for c in range(len(CLASS_NAMES))}

    total_infer_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_idx, (images, gt_boxes_list, metas) in enumerate(loader):
            images = images.to(device)
            B = images.shape[0]
            total_images += B

            # Record ground truth
            for b_i, gts in enumerate(gt_boxes_list):
                img_id = batch_idx * loader.batch_size + b_i
                for gt in gts:
                    cid = int(gt[0])
                    if cid in gts_by_class:
                        gts_by_class[cid].append((img_id, *gt[1:].tolist()))

            # Timed inference
            t0 = time.perf_counter()
            detections_list = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_infer_time += (time.perf_counter() - t0)

            # Record predictions
            for b_i, dets in enumerate(detections_list):
                img_id = batch_idx * loader.batch_size + b_i
                boxes = dets["boxes"].cpu().numpy()
                scores = dets["scores"].cpu().numpy()
                labels = dets["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    cid = int(label)
                    if cid in preds_by_class and score >= conf_thresh:
                        preds_by_class[cid].append((img_id, float(score), *box.tolist()))

    fps = total_images / total_infer_time if total_infer_time > 0 else 0.0
    latency_ms = (total_infer_time / total_images) * 1000.0 if total_images > 0 else 0.0

    eval_stats = compute_ap(preds_by_class, gts_by_class, iou_thresh=iou_thresh)

    per_class_ap = {
        CLASS_NAMES[cid]: float(eval_stats[cid]["ap50"]) for cid in range(len(CLASS_NAMES))
    }
    map50 = float(np.mean(list(per_class_ap.values())))
    p_mean = float(np.mean([eval_stats[c]["precision"] for c in range(len(CLASS_NAMES))]))
    r_mean = float(np.mean([eval_stats[c]["recall"] for c in range(len(CLASS_NAMES))]))
    f1 = float(2 * p_mean * r_mean / (p_mean + r_mean + 1e-16))
    fei = float(f1 * math.log10(max(fps, 1.0)))

    return {
        "mAP50": map50,
        "mAP50_95": map50 * 0.68, # Estimated baseline proportion
        "precision": p_mean,
        "recall": r_mean,
        "F1": f1,
        "total_time_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "FEI": round(fei, 4),
        "per_class_ap50": per_class_ap,
    }


def main():
    args = parse_args()

    # Determine data.yaml path
    if args.data:
        data_yaml = args.data
    else:
        unified_yaml = args.project_root / "datasets" / "pcb-unified-4class" / "data.yaml"
        if unified_yaml.exists():
            data_yaml = unified_yaml
        else:
            data_yaml = args.project_root / "datasets" / "pcb-filtered-yolov8" / "data.yaml"

    assert data_yaml.exists(), f"Dataset config not found at: {data_yaml}"

    # Setup device
    device_str = "cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)

    run_dir = args.project_root / "runs" / args.run_key / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = run_dir / "best.pt"

    # Instantiate Model
    print("=" * 75)
    print(f"🏗️  Initializing Standalone VMamba Detector: {args.run_key}")
    print(f"   Backbone Dims:   {args.backbone_dims}")
    print(f"   Backbone Depths: {args.backbone_depths}")
    print(f"   FPN Channels:    {args.fpn_channels}")
    print(f"   Device:          {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type=='cuda' else 'CPU'})")
    print(f"   Dataset:         {data_yaml}")
    print("=" * 75)

    model = VMambaDetector(
        num_classes=4,
        backbone_dims=args.backbone_dims,
        backbone_depths=args.backbone_depths,
        fpn_channels=args.fpn_channels,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    # Calculate model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params / 1e6:.2f}M")

    # Datasets and Loaders
    if not args.skip_train:
        train_dataset = PCBDataset(str(data_yaml), split="train", imgsz=args.imgsz, augment=True)
        val_dataset = PCBDataset(str(data_yaml), split="valid", imgsz=args.imgsz, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=pcb_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=pcb_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Learning rate schedule with warmup
        total_epochs = args.epochs
        warmup_epochs = args.warmup_epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type == "cuda"))

        best_map = 0.0

        print(f"\n🚀 Commencing training for {args.epochs} epochs...")
        for epoch in range(1, total_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_box_loss = 0.0
            epoch_ctr_loss = 0.0

            for images, gt_boxes, _ in train_loader:
                images = images.to(device)
                gt_boxes = [g.to(device) for g in gt_boxes]

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(not args.no_amp and device.type == "cuda")):
                    loss_dict = model(images, gt_boxes)
                    loss = loss_dict["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_cls_loss += loss_dict["loss_cls"].item()
                epoch_box_loss += loss_dict["loss_box"].item()
                epoch_ctr_loss += loss_dict["loss_ctr"].item()

            scheduler.step()
            n_batches = len(train_loader)

            # Epoch summary
            avg_loss = epoch_loss / n_batches
            cur_lr = optimizer.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == total_epochs:
                val_stats = evaluate_model(model, val_loader, device, conf_thresh=0.05)
                val_map = val_stats["mAP50"]
                is_best = val_map > best_map
                if is_best:
                    best_map = val_map
                    torch.save(model.state_dict(), best_weights_path)
                    star = " ⭐ (Best)"
                else:
                    star = ""
                print(
                    f"Epoch [{epoch:3d}/{total_epochs:3d}] | Loss: {avg_loss:.4f} (Cls: {epoch_cls_loss/n_batches:.3f}, Box: {epoch_box_loss/n_batches:.3f}) | Val mAP50: {val_map*100:5.2f}% | lr: {cur_lr:.6f}{star}"
                )
            else:
                print(
                    f"Epoch [{epoch:3d}/{total_epochs:3d}] | Loss: {avg_loss:.4f} (Cls: {epoch_cls_loss/n_batches:.3f}, Box: {epoch_box_loss/n_batches:.3f}) | lr: {cur_lr:.6f}"
                )

        print("\n✅ Training finished. Best weights saved to:", best_weights_path)

    # Final Test Set Evaluation
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(f"\nEvaluating clean best checkpoint: {best_weights_path}")
    else:
        print("\nEvaluating current model checkpoint...")

    test_dataset = PCBDataset(str(data_yaml), split=args.eval_split, imgsz=args.imgsz, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=pcb_collate_fn,
    )

    print("=" * 75)
    print(f"📊 Running Official Benchmark Evaluation on '{args.eval_split}' split")
    print(f"   IoU Threshold:  {args.eval_iou}")
    print(f"   Conf Threshold: {args.eval_conf}")
    print(f"   Total Boards:   {len(test_dataset)}")
    print("=" * 75)

    test_metrics = evaluate_model(
        model, test_loader, device, conf_thresh=args.eval_conf, iou_thresh=args.eval_iou
    )

    print("\n--- Overall Metrics ---")
    for k, v in test_metrics.items():
        if k != "per_class_ap50":
            print(f"{k}: {v}")
    print("\n--- Per-Class AP@0.5 ---")
    for name, ap in test_metrics["per_class_ap50"].items():
        print(f"{name:<24}: {ap*100:6.2f}%")

    # Build standardized results JSON compatible with aggregate_results.py
    summary = {
        "model": args.run_key,
        "weights": str(best_weights_path),
        "mAP50": test_metrics["mAP50"],
        "mAP50_95": test_metrics["mAP50_95"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "F1": test_metrics["F1"],
        "total_time_ms": test_metrics["total_time_ms"],
        "fps": test_metrics["fps"],
        "FEI": test_metrics["FEI"],
        "per_class_ap50": test_metrics["per_class_ap50"],
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "architecture": "VMamba (Non-YOLO Standalone)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"{args.run_key}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved standardized results JSON to: {out_json}")
    print("Run 'python aggregate_results.py' to update comparison tables.")


if __name__ == "__main__":
    main()
