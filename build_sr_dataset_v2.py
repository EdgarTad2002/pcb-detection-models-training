#!/usr/bin/env python3
"""
Builds the paired LR/HR dataset for SuperYOLO using the GENUINE Native
Resolution dataset as the high-resolution target, and the 640x640 dataset
as the low-resolution input.

This provides the true "SuperYOLO" training signal, as the SR branch will be
forced to reconstruct actual high-frequency details from the native images
that are genuinely missing in the 640px downsampled versions.

Usage:
    python build_sr_dataset_v2.py \
        --source-lr datasets/pcb-filtered-yolov8 \
        --source-hr datasets/pcb-native-res \
        --dest datasets/pcb-sr-native
"""

import argparse
import shutil
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm


def get_base_name(filename):
    """
    Strips Roboflow's random hashes to match images across different dataset versions.
    Example: 'IMG_123_jpg.rf.a1b2c3d4e5f6.jpg' -> 'IMG_123'
    """
    name = filename.stem
    if "_jpg.rf." in name:
        name = name.split("_jpg.rf.")[0]
    elif ".rf." in name:
        name = name.split(".rf.")[0]
    return name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-lr", type=Path, required=True, help="Path to 640px dataset")
    p.add_argument("--source-hr", type=Path, required=True, help="Path to Native Res dataset")
    p.add_argument("--dest", type=Path, required=True, help="Output SR dataset path")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    return p.parse_args()


def main():
    args = parse_args()

    for split in args.splits:
        lr_img_dir = args.source_lr / split / "images"
        lr_lbl_dir = args.source_lr / split / "labels"
        
        hr_img_dir = args.source_hr / split / "images"

        if not lr_img_dir.exists():
            print(f"Skipping {split} -- LR images not found")
            continue
        if not hr_img_dir.exists():
            print(f"Skipping {split} -- HR images not found")
            continue

        dest_lr_dir = args.dest / split / "images"
        dest_hr_dir = args.dest / split / "images_hr"
        dest_lbl_dir = args.dest / split / "labels"
        
        dest_lr_dir.mkdir(parents=True, exist_ok=True)
        dest_hr_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Build lookup table for HR images based on base name
        hr_files = sorted(list(hr_img_dir.glob("*.jpg")) + list(hr_img_dir.glob("*.png")))
        hr_lookup = {get_base_name(f): f for f in hr_files}

        lr_files = sorted(list(lr_img_dir.glob("*.jpg")) + list(lr_img_dir.glob("*.png")))
        print(f"{split}: Matching {len(lr_files)} LR images to HR targets...")
        
        matched_count = 0
        for lr_path in tqdm(lr_files, desc=split):
            base_name = get_base_name(lr_path)
            
            if base_name not in hr_lookup:
                print(f"\n[WARN] No HR match found for: {lr_path.name}")
                continue
                
            hr_path = hr_lookup[base_name]

            # We don't need to resize or degrade the LR image this time.
            # We just copy it as-is, because it is already 640x640,
            # and the HR target is the much larger native resolution image.
            
            # LR input (detector input)
            shutil.copy(lr_path, dest_lr_dir / lr_path.name)
            
            # HR target (SR reconstruction target)
            # We name it exactly the same as the LR file so SRYOLODataset finds it
            shutil.copy(hr_path, dest_hr_dir / lr_path.name)

            # Label (stays identical to the LR dataset)
            label_path = lr_lbl_dir / f"{lr_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, dest_lbl_dir / label_path.name)
                
            matched_count += 1
            
        print(f"  Successfully paired {matched_count}/{len(lr_files)} images for {split}.")

    # Write data.yaml
    src_yaml = args.source_lr / "data.yaml"
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
    print("NOTE: 'images' contains the 640px versions, 'images_hr' contains the genuine native resolution images.")


if __name__ == "__main__":
    main()
