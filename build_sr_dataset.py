#!/usr/bin/env python3
"""
Builds the paired LR/HR dataset for SuperYOLO-style training, using ONLY
the standard 640x640 dataset -- no native-res source needed.

Since there's no genuinely higher-resolution image to pull from within this
dataset, we simulate the SR training signal within the same image:
  images/     -- the SAME 640x640 image, deliberately degraded (downsampled
                 small, then upsampled back to 640x640) -- real information
                 loss, but identical pixel dimensions to your baseline's
                 imgsz=640, so results stay directly comparable.
  images_hr/  -- the ORIGINAL, undegraded 640x640 image -- the SR
                 reconstruction target.
Labels are copied through unchanged -- box positions don't move (both LR
and HR are the same 640x640 frame, no cropping/resizing distortion).

Usage:
    python build_sr_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest datasets/pcb-sr-640 \
        --degradation-factor 4
"""

import argparse
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--degradation-factor", type=int, default=4,
                    help="Downsample by this factor before upsampling back -- higher = more information loss for the SR branch to recover")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    for split in args.splits:
        src_img_dir = args.source / split / "images"
        src_lbl_dir = args.source / split / "labels"
        if not src_img_dir.exists():
            print(f"Skipping {split} -- not found")
            continue

        dest_lr_dir = args.dest / split / "images"
        dest_hr_dir = args.dest / split / "images_hr"
        dest_lbl_dir = args.dest / split / "labels"
        dest_lr_dir.mkdir(parents=True, exist_ok=True)
        dest_hr_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
        print(f"{split}: {len(img_files)} images")
        for img_path in tqdm(img_files, desc=split):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # HR target: the original image, untouched
            cv2.imwrite(str(dest_hr_dir / img_path.name), img)

            # LR input: genuinely degraded, but resized back to the same
            # HxW so the detector's actual input tensor size never changes
            small_w = max(1, w // args.degradation_factor)
            small_h = max(1, h // args.degradation_factor)
            degraded = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dest_lr_dir / img_path.name), degraded)

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
    print("NOTE: 'images' contains DEGRADED versions (detector input), "
          "'images_hr' contains the original sharp images (SR target only).")


if __name__ == "__main__":
    main()
