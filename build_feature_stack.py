#!/usr/bin/env python3
"""
Replaces RGB with a 3-channel engineered feature stack:
  Channel 0: CLAHE-enhanced grayscale (local contrast boost)
  Channel 1: HSV Saturation
  Channel 2: Sobel edge magnitude

Stays inside Ultralytics' standard 3-channel image pipeline -- no
architecture change, no custom dataset loader, train.py runs unmodified.
This is NOT synthesized infrared/hyperspectral data (that's not physically
possible from RGB alone) -- it's a deterministic recombination of the same
captured pixels, testing whether a different representation surfaces
existing contrast more usefully than raw RGB.

Labels are copied through unchanged -- object locations don't move.

Usage:
    python build_feature_stack.py \
        --source datasets/pcb-native-res \
        --dest datasets/pcb-native-res-featstack
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    return p.parse_args()


def build_feature_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_ch = clahe.apply(gray)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_ch = hsv[:, :, 1]

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_ch = np.clip(edge_mag / edge_mag.max() * 255 if edge_mag.max() > 0 else edge_mag, 0, 255).astype(np.uint8)

    return cv2.merge([contrast_ch, sat_ch, edge_ch])


def main():
    args = parse_args()

    for split in args.splits:
        src_img_dir = args.source / split / "images"
        src_lbl_dir = args.source / split / "labels"
        if not src_img_dir.exists():
            print(f"Skipping {split} -- {src_img_dir} not found")
            continue

        dest_img_dir = args.dest / split / "images"
        dest_lbl_dir = args.dest / split / "labels"
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
        print(f"Transforming {split}: {len(img_files)} images")
        for img_path in tqdm(img_files, desc=split):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feat_img = build_feature_image(img)
            cv2.imwrite(str(dest_img_dir / f"{img_path.stem}.jpg"), feat_img)

            label_path = src_lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, dest_lbl_dir / label_path.name)

    src_yaml = args.source / "data.yaml"
    with open(src_yaml) as f:
        src_cfg = yaml.safe_load(f)

    new_cfg = {
        "path": str(args.dest.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": src_cfg["nc"],
        "names": src_cfg["names"],
    }
    with open(args.dest / "data.yaml", "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    print(f"\nDone. New data.yaml: {args.dest / 'data.yaml'}")
    print("NOTE: this replaces RGB with [CLAHE-contrast, Saturation, Edge-magnitude] "
          "channels -- still a valid 3-channel image, no architecture change needed.")


if __name__ == "__main__":
    main()
