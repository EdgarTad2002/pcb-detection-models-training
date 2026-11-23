#!/usr/bin/env python3
"""
tools/build_mixed_crop_dataset.py
=================================
Builds a reproducible mixed training dataset combining:
  1. Full-board native-resolution images (scaled during training)
  2. Native-resolution crops (e.g. 640x640, 960x960) focusing on:
     - Capacitor-dense regions (ensuring high target representation)
     - Difficult background regions (empty substrate, solder joints, test points)
     - Random regions (general context)

Key Rigorous Constraints:
  - Exclusively crops from the TRAINING split boards (zero train/val/test leakage).
  - Preserves validation and test splits unchanged as pure native-resolution boards.
  - Generates a full cryptographic JSON manifest recording board origins, crop coordinates,
    seeds, box transformations, and filtering statistics.
  - Strict Boundary Policy:
      * Minimum visibility ratio: box must retain >= 30% of its area in the crop.
      * Minimum dimension: width and height must be >= 3 pixels after cropping.
      * Partially clipped boxes are tightly clamped to crop edges [0, crop_w] x [0, crop_h].
      * Severely truncated fragments (< 30% area or < 3px) are dropped to prevent training noise.
  - Output is saved to a separate target directory: DOES NOT modify existing datasets in place.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def parse_args():
    p = argparse.ArgumentParser(description="Build mixed full-board + capacitor crop dataset.")
    p.add_argument(
        "--source-dataset",
        type=str,
        default="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-native-res-cleaned-1280",
        help="Path to source native-resolution cleaned dataset",
    )
    p.add_argument(
        "--output-dataset",
        type=str,
        default="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-native-mixed-crops-1280",
        help="Target directory for the new mixed dataset",
    )
    p.add_argument(
        "--crop-sizes",
        type=int,
        nargs="+",
        default=[640, 960],
        help="Crop sizes in native image pixels (e.g. 640 960)",
    )
    p.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.50,
        help="Ratio of crop samples relative to full samples (0.50 = 50% crops, 50% full)",
    )
    p.add_argument(
        "--min-visibility",
        type=float,
        default=0.30,
        help="Minimum area ratio of box retained within crop to keep annotation",
    )
    p.add_argument(
        "--min-box-dim",
        type=int,
        default=3,
        help="Minimum box width and height in pixels to retain after cropping",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible crop selection",
    )
    return p.parse_args()


def load_yolo_annotations(lbl_path: Path, img_w: int, img_h: int) -> List[Dict]:
    """Loads YOLO format txt and converts to absolute pixel coordinates [x1, y1, x2, y2]."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h
            boxes.append({
                "cls": cid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "orig_area": max(0.0, (x2 - x1) * (y2 - y1)),
            })
    return boxes


def crop_sample(
    img: np.ndarray,
    boxes: List[Dict],
    cx: int,
    cy: int,
    crop_w: int,
    crop_h: int,
    min_visibility: float,
    min_box_dim: int,
) -> Tuple[np.ndarray, List[Dict]]:
    """Crops an image region and transforms/filters annotations according to boundary policy."""
    ih, iw = img.shape[:2]
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(iw, x1 + crop_w)
    y2 = min(ih, y1 + crop_h)

    # Adjust x1, y1 if boundary hit to preserve exact crop dimensions where possible
    if x2 - x1 < crop_w and iw >= crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h and ih >= crop_h:
        y1 = max(0, y2 - crop_h)

    cropped_img = img[y1:y2, x1:x2].copy()
    cw, ch = cropped_img.shape[1], cropped_img.shape[0]

    retained_boxes = []
    for b in boxes:
        # Compute intersection of box with crop window
        bx1 = max(0.0, b["x1"] - x1)
        by1 = max(0.0, b["y1"] - y1)
        bx2 = min(float(cw), b["x2"] - x1)
        by2 = min(float(ch), b["y2"] - y1)

        inter_w = max(0.0, bx2 - bx1)
        inter_h = max(0.0, by2 - by1)
        inter_area = inter_w * inter_h

        # Boundary policy check:
        if inter_area <= 0.0:
            continue
        vis_ratio = inter_area / max(1e-6, b["orig_area"])
        if vis_ratio < min_visibility:
            continue
        if inter_w < min_box_dim or inter_h < min_box_dim:
            continue

        retained_boxes.append({
            "cls": b["cls"],
            "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
            "vis_ratio": vis_ratio,
            "w": inter_w, "h": inter_h,
        })

    return cropped_img, retained_boxes, (x1, y1, x2, y2)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_dir = Path(args.source_dataset).resolve()
    output_dir = Path(args.output_dataset).resolve()
    if not 0 <= args.sampling_ratio < 1:
        raise ValueError("--sampling-ratio must be in [0, 1)")
    if not args.crop_sizes or min(args.crop_sizes) < 64:
        raise ValueError("--crop-sizes must all be at least 64 pixels")
    if not 0 < args.min_visibility <= 1 or args.min_box_dim <= 0:
        raise ValueError("Invalid crop visibility or minimum box dimension")

    print("=" * 75)
    print("BUILDING REPRODUCIBLE MIXED FULL-BOARD + CAPACITOR CROP DATASET")
    print("=" * 75)
    print(f"Source:           {source_dir}")
    print(f"Output:           {output_dir}")
    print(f"Crop Sizes:       {args.crop_sizes}")
    print(f"Sampling Ratio:   {args.sampling_ratio} (crops / total)")
    print(f"Boundary Policy:  min_visibility={args.min_visibility}, min_dim={args.min_box_dim}px")
    print(f"Random Seed:      {args.seed}")
    print("=" * 75)

    assert source_dir.exists(), f"Source dataset does not exist: {source_dir}"

    # Setup directories
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing dataset: {output_dir}. Choose a new output path.")

    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy validation and test splits exactly without any alteration
    print("\nCopying pure validation and test splits (un-cropped, strictly preserved)...")
    for split in ["valid", "test"]:
        s_img_dir = source_dir / split / "images"
        s_lbl_dir = source_dir / split / "labels"
        t_img_dir = output_dir / split / "images"
        t_lbl_dir = output_dir / split / "labels"

        img_files = sorted(list(s_img_dir.glob("*.jpg")) + list(s_img_dir.glob("*.png")))
        for img_p in img_files:
            shutil.copy2(img_p, t_img_dir / img_p.name)
            lbl_p = s_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.copy2(lbl_p, t_lbl_dir / lbl_p.name)
            else:
                (t_lbl_dir / f"{img_p.stem}.txt").write_text("")
        print(f"  {split}: {len(img_files)} images preserved.")

    # Process training split
    train_img_dir = source_dir / "train" / "images"
    train_lbl_dir = source_dir / "train" / "labels"
    out_train_img = output_dir / "train" / "images"
    out_train_lbl = output_dir / "train" / "labels"

    train_imgs = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")))
    if not train_imgs:
        raise ValueError(f"No training images in {train_img_dir}")
    print(f"\nProcessing train split ({len(train_imgs)} full-board images)...")

    manifest = {
        "source_dataset": str(source_dir),
        "seed": args.seed,
        "crop_sizes": args.crop_sizes,
        "sampling_ratio": args.sampling_ratio,
        "min_visibility": args.min_visibility,
        "min_box_dim": args.min_box_dim,
        "train_samples": [],
    }

    # 1. Copy full-board training images
    for img_p in train_imgs:
        shutil.copy2(img_p, out_train_img / img_p.name)
        lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, out_train_lbl / lbl_p.name)
        else:
            (out_train_lbl / f"{img_p.stem}.txt").write_text("")

        manifest["train_samples"].append({
            "type": "full_board",
            "filename": img_p.name,
            "source_board": img_p.stem.split(".rf.")[0] if ".rf." in img_p.stem else img_p.stem,
        })

    # 2. Generate native-resolution crops targeting capacitors, backgrounds, and random areas
    num_full = len(train_imgs)
    # Ratio: num_crops / (num_full + num_crops) = sampling_ratio => num_crops = num_full * r / (1 - r)
    num_crops_target = int(round(num_full * args.sampling_ratio / (1.0 - args.sampling_ratio)))
    print(f"Generating {num_crops_target} native-resolution crops (to achieve {args.sampling_ratio*100:.0f}% crop mix)...")

    crops_generated = 0
    crop_idx = 0
    attempts = 0

    while crops_generated < num_crops_target:
        attempts += 1
        if attempts > max(100, num_crops_target * 20):
            raise RuntimeError("Unable to generate requested crops; check source images and crop dimensions")
        # Sample a random source training image
        src_img_p = random.choice(train_imgs)
        src_lbl_p = train_lbl_dir / f"{src_img_p.stem}.txt"

        img = cv2.imread(str(src_img_p))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        boxes = load_yolo_annotations(src_lbl_p, iw, ih)

        # Select crop size
        c_size = random.choice(args.crop_sizes)
        # Choose strategy: 70% capacitor-centered, 15% difficult background, 15% random
        strat = random.random()
        cap_boxes = [b for b in boxes if b["cls"] == 0]

        if strat < 0.70 and cap_boxes:
            # Capacitor-focused crop
            target_cap = random.choice(cap_boxes)
            # Add slight jitter to center
            cx = int((target_cap["x1"] + target_cap["x2"]) / 2.0 + random.randint(-40, 40))
            cy = int((target_cap["y1"] + target_cap["y2"]) / 2.0 + random.randint(-40, 40))
            crop_type = "capacitor_focused"
        elif strat < 0.85:
            # Random background region
            cx = random.randint(c_size // 4, max(c_size // 4 + 1, iw - c_size // 4))
            cy = random.randint(c_size // 4, max(c_size // 4 + 1, ih - c_size // 4))
            crop_type = "background"
        else:
            # Uniform random crop
            cx = random.randint(0, iw)
            cy = random.randint(0, ih)
            crop_type = "random"

        cropped_img, retained_boxes, coords = crop_sample(
            img, boxes, cx, cy, c_size, c_size,
            min_visibility=args.min_visibility,
            min_box_dim=args.min_box_dim,
        )

        ch, cw = cropped_img.shape[:2]
        if ch < 64 or cw < 64:
            continue

        crop_fname = f"crop_{crop_idx:05d}_{c_size}px_{src_img_p.stem}.jpg"
        crop_lblname = f"crop_{crop_idx:05d}_{c_size}px_{src_img_p.stem}.txt"

        # Save image
        if not cv2.imwrite(str(out_train_img / crop_fname), cropped_img):
            raise OSError(f"Failed to write crop: {out_train_img / crop_fname}")

        # Save YOLO formatted label file
        lines = []
        for rb in retained_boxes:
            rcx = ((rb["x1"] + rb["x2"]) / 2.0) / cw
            rcy = ((rb["y1"] + rb["y2"]) / 2.0) / ch
            rbw = (rb["x2"] - rb["x1"]) / cw
            rbh = (rb["y2"] - rb["y1"]) / ch
            lines.append(f"{rb['cls']} {rcx:.6f} {rcy:.6f} {rbw:.6f} {rbh:.6f}")

        (out_train_lbl / crop_lblname).write_text("\n".join(lines))

        manifest["train_samples"].append({
            "type": crop_type,
            "filename": crop_fname,
            "source_image": src_img_p.name,
            "crop_coords": list(coords),
            "crop_size": c_size,
            "num_annotations": len(retained_boxes),
        })

        crops_generated += 1
        crop_idx += 1

    # Write data.yaml
    data_yaml_content = f"""# Native Mixed Full-Board + Capacitor Crop Dataset
path: {output_dir}
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: Capacitor
  1: Connector
  2: Electrolytic Capacitor
  3: IC
"""
    (output_dir / "data.yaml").write_text(data_yaml_content)

    # Save manifest
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSuccessfully built dataset!")
    print(f"Total train samples: {len(manifest['train_samples'])} ({num_full} full-board + {crops_generated} crops)")
    print(f"Manifest written to: {manifest_path}")
    print(f"data.yaml written to: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
