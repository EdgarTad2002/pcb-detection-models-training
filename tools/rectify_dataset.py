#!/usr/bin/env python3
"""
tools/rectify_dataset.py - Unified 4-Class PCB Dataset Generator

Converts the raw Roboflow PCB dataset (with 23 classes and split capacitor taxonomy)
into a clean, native 4-class dataset (nc: 4):
  - Class 1 (Capacitor Jumper) + Class 2 (Capacitor) -> Class 0 (Capacitor)
  - Class 4 (Connector)                              -> Class 1 (Connector)
  - Class 7 (Electrolytic Capacitor)                 -> Class 2 (Electrolytic Capacitor)
  - Class 9 (IC) & Class 22 (iC)                     -> Class 3 (IC)
  - Discards all other 19 non-target classes (Resistors, Pins, Pads, etc.)

Features:
  - Non-destructive: Leaves original dataset 100% intact as a backup.
  - Storage-efficient: Uses relative or absolute symlinks for images.
  - Purges old .cache files so YOLO rebuilds clean, fast label caches.
  - Generates verified data.yaml with nc: 4.

Usage:
  python tools/rectify_dataset.py \
      --source datasets/pcb-filtered-yolov8 \
      --dest datasets/pcb-unified-4class
"""

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# Mapping from raw Roboflow class IDs to native 4-class IDs
CLASS_MAPPING = {
    1: 0,   # Capacitor Jumper -> Capacitor
    2: 0,   # Capacitor        -> Capacitor
    4: 1,   # Connector        -> Connector
    7: 2,   # Electrolytic Capacitor -> Electrolytic Capacitor
    9: 3,   # IC               -> IC
    22: 3,  # iC (typo)        -> IC
}

TARGET_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]


def process_split(src_dir: Path, dst_dir: Path, split_name: str = "", use_symlinks: bool = True):
    """Processes images and labels for one split (train, valid, test, or flat folder)."""
    if split_name:
        src_img_dir = src_dir / split_name / "images"
        src_lbl_dir = src_dir / split_name / "labels"
        dst_img_dir = dst_dir / split_name / "images"
        dst_lbl_dir = dst_dir / split_name / "labels"
        cache_parent = dst_dir / split_name
    else:
        src_img_dir = src_dir / "images"
        src_lbl_dir = src_dir / "labels"
        dst_img_dir = dst_dir / "images"
        dst_lbl_dir = dst_dir / "labels"
        cache_parent = dst_dir
    
    if not src_img_dir.exists() or not src_lbl_dir.exists():
        split_display = split_name if split_name else str(src_dir)
        print(f"⚠️  Split '{split_display}' not found, skipping.")
        return None

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Symlink or copy images
    image_files = sorted(list(src_img_dir.glob("*.[jJ][pP][gG]")) + 
                         list(src_img_dir.glob("*.[pP][nN][gG]")) + 
                         list(src_img_dir.glob("*.[jJ][pP][eE][gG]")))
    
    for img_p in image_files:
        dst_p = dst_img_dir / img_p.name
        if dst_p.is_symlink() or dst_p.exists():
            continue
        if use_symlinks:
            os.symlink(img_p.resolve(), dst_p)
        else:
            shutil.copy2(img_p, dst_p)

    # 2. Remap label files
    raw_counts = Counter()
    rectified_counts = Counter()
    label_files = sorted(list(src_lbl_dir.glob("*.txt")))

    for lbl_p in label_files:
        new_lines = []
        with open(lbl_p, "r") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 5:
                    raw_cid = int(parts[0])
                    raw_counts[raw_cid] += 1
                    
                    if raw_cid in CLASS_MAPPING:
                        new_cid = CLASS_MAPPING[raw_cid]
                        rectified_counts[new_cid] += 1
                        new_line = f"{new_cid} " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)

        dst_lbl_p = dst_lbl_dir / lbl_p.name
        with open(dst_lbl_p, "w") as fp:
            fp.writelines(new_lines)

    # Remove any stale cache in destination (both inside labels/ and sibling labels.cache)
    for cache_file in dst_lbl_dir.glob("*.cache"):
        cache_file.unlink(missing_ok=True)
    if (cache_parent / "labels.cache").exists():
        (cache_parent / "labels.cache").unlink()

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "raw_counts": raw_counts,
        "rectified_counts": rectified_counts,
    }


def write_yaml(dest_dir: Path, names: list):
    """Writes clean data.yaml for 4-class YOLO training."""
    yaml_path = dest_dir / "data.yaml"
    abs_dest = dest_dir.resolve()
    content = f"""# PCB Unified 4-Class Dataset Configuration
path: {abs_dest}
train: train/images
val: valid/images
test: test/images

nc: {len(names)}
names:
"""
    for idx, name in enumerate(names):
        content += f"  {idx}: {name}\n"

    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"📝 Created clean YAML configuration: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Rectify Roboflow PCB dataset into clean 4-class dataset.")
    parser.add_argument("--source", type=Path, default=Path("datasets/pcb-filtered-yolov8"),
                        help="Path to raw Roboflow dataset (or flat sample dir)")
    parser.add_argument("--dest", type=Path, default=Path("datasets/pcb-unified-4class"),
                        help="Destination path for clean 4-class dataset")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of creating symlinks (symlinks are default)")
    args = parser.parse_args()

    src = args.source.resolve()
    dst = args.dest.resolve()

    if not src.exists():
        print(f"❌ Error: Source directory does not exist: {src}")
        sys.exit(1)

    print("=" * 80)
    print("🚀 Starting PCB Dataset Rectification (Native 4-Class Conversion)")
    print(f"📁 Source:      {src}")
    print(f"📁 Destination: {dst}")
    print(f"🔗 Mode:        {'Copy Images' if args.copy_images else 'Symlink Images (Zero Duplication)'}")
    print("=" * 80)

    dst.mkdir(parents=True, exist_ok=True)

    # Detect if source has train/valid/test or is a flat images/labels directory
    has_splits = (src / "train").exists() or (src / "valid").exists() or (src / "test").exists()

    stats = {}
    if has_splits:
        splits = ["train", "valid", "test"]
        for split in splits:
            print(f"\n⚙️  Processing split: '{split}'...")
            res = process_split(src, dst, split, use_symlinks=not args.copy_images)
            if res:
                stats[split] = res
                print(f"   ✓ Processed {res['images']} images and {res['labels']} label files.")
                print("   📊 Class Breakdown:")
                for cid, cname in enumerate(TARGET_NAMES):
                    cnt = res["rectified_counts"][cid]
                    print(f"      • Class {cid} ({cname:24s}): {cnt:5d} boxes")

        # Write data.yaml
        write_yaml(dst, TARGET_NAMES)

        print("\n" + "=" * 80)
        print("✅ DATASET RECTIFICATION COMPLETE!")
        print("=" * 80)
        print(f"{'Split':<10} | {'Images':<8} | {'Capacitors':<12} | {'Connectors':<12} | {'Elec. Caps':<12} | {'ICs':<8}")
        print("-" * 75)
        for split in splits:
            if split in stats:
                s = stats[split]["rectified_counts"]
                imgs = stats[split]["images"]
                print(f"{split:<10} | {imgs:<8} | {s[0]:<12} | {s[1]:<12} | {s[2]:<12} | {s[3]:<8}")
        print("-" * 75)
        print(f"\nReady for training with '{dst}/data.yaml'!")

    else:
        # Flat directory (e.g. data_samples)
        print(f"\n⚙️  Processing flat directory: '{src}'...")
        res = process_split(src, dst, "", use_symlinks=not args.copy_images)
        if res:
            print(f"   ✓ Processed {res['images']} images and {res['labels']} label files.")
            print("   📊 Class Breakdown:")
            for cid, cname in enumerate(TARGET_NAMES):
                cnt = res["rectified_counts"][cid]
                print(f"      • Class {cid} ({cname:24s}): {cnt:5d} boxes")
        print("\n✅ Directory rectification complete!")


if __name__ == "__main__":
    main()
