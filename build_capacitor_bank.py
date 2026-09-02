#!/usr/bin/env python3
"""
Builds a bank of cropped Capacitor images from the training set, for use by
bbox_copy_paste.py's augmentation. Run once before training.

Usage:
    python build_capacitor_bank.py \
        --source datasets/pcb-native-res \
        --dest capacitor_bank \
        --margin 0.15
"""

import argparse
import json
from pathlib import Path

import cv2

CAPACITOR_CLASS_ID = 2  # raw dataset class id


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Dataset root (containing train/images, train/labels)")
    p.add_argument("--dest", type=Path, required=True, help="Output folder for cropped capacitor images")
    p.add_argument("--margin", type=float, default=0.4, help="Extra margin around each box, as a fraction of box size")
    p.add_argument("--min-crop-size", type=int, default=32, help="Force crops to be at least this many pixels on each side")
    p.add_argument("--min-size", type=int, default=8, help="Skip crops smaller than this many pixels on either side")
    return p.parse_args()


def main():
    args = parse_args()
    img_dir = args.source / "train" / "images"
    lbl_dir = args.source / "train" / "labels"
    args.dest.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    n_crops = 0
    metadata = {}

    for img_path in img_files:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = None  # lazy-load only if this image has a capacitor box
        h = w = None

        for i, line in enumerate(label_path.read_text().strip().splitlines()):
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id != CAPACITOR_CLASS_ID:
                continue

            if img is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    break
                h, w = img.shape[:2]

            cx, cy, bw, bh = map(float, parts[1:5])
            orig_box_x1 = (cx - bw / 2.0) * w
            orig_box_y1 = (cy - bh / 2.0) * h
            orig_box_x2 = (cx + bw / 2.0) * w
            orig_box_y2 = (cy + bh / 2.0) * h

            box_w, box_h = bw * w, bh * h
            margin_w, margin_h = box_w * args.margin, box_h * args.margin

            x1 = int(max(0, orig_box_x1 - margin_w))
            y1 = int(max(0, orig_box_y1 - margin_h))
            x2 = int(min(w, orig_box_x2 + margin_w))
            y2 = int(min(h, orig_box_y2 + margin_h))

            # Force a minimum crop size so tiny boxes still get real board
            # context, not just a near-exact crop of the component itself.
            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w < args.min_crop_size:
                pad = (args.min_crop_size - crop_w) / 2
                x1 = int(max(0, x1 - pad))
                x2 = int(min(w, x2 + pad))
            if crop_h < args.min_crop_size:
                pad = (args.min_crop_size - crop_h) / 2
                y1 = int(max(0, y1 - pad))
                y2 = int(min(h, y2 + pad))

            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w < args.min_size or crop_h < args.min_size:
                continue

            # Compute exact relative bounding box within the crop [0, 1]
            rel_x1 = max(0.0, min(1.0, (orig_box_x1 - x1) / crop_w))
            rel_y1 = max(0.0, min(1.0, (orig_box_y1 - y1) / crop_h))
            rel_x2 = max(0.0, min(1.0, (orig_box_x2 - x1) / crop_w))
            rel_y2 = max(0.0, min(1.0, (orig_box_y2 - y1) / crop_h))

            crop = img[y1:y2, x1:x2]
            crop_filename = f"{img_path.stem}_cap{i}.png"
            out_path = args.dest / crop_filename
            cv2.imwrite(str(out_path), crop)

            metadata[crop_filename] = {
                "rel_box": [rel_x1, rel_y1, rel_x2, rel_y2]
            }
            n_crops += 1

    metadata_path = args.dest / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extracted {n_crops} capacitor crops to {args.dest}")
    print(f"Saved crop relative box metadata to {metadata_path}")


if __name__ == "__main__":
    main()

