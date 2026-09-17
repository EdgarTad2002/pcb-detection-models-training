#!/usr/bin/env python3
"""
Applies Bilateral Edge-Preserving Filtering to the PCB dataset.

Preserves sharp component edges (solder joints, capacitor boundaries, IC leads)
while smoothing out high-frequency fiberglass substrate noise.

Technique adopted from SanderGi/PCB-Detection (correct4d.py).
Formula parameters: d=5, sigmaColor=75, sigmaSpace=75.

Usage:
    python tools/build_bilateral_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest datasets/pcb-bilateral-640
"""

import argparse
import concurrent.futures
import shutil
import time
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate Bilateral-Filtered PCB Dataset")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/pcb-filtered-yolov8"),
        help="Path to source YOLO dataset",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("datasets/pcb-bilateral-640"),
        help="Path to destination dataset",
    )
    p.add_argument("--d", type=int, default=5, help="Diameter of each pixel neighborhood")
    p.add_argument("--sigma-color", type=float, default=75.0, help="Filter sigma in the color space")
    p.add_argument("--sigma-space", type=float, default=75.0, help="Filter sigma in the coordinate space")
    p.add_argument("--workers", type=int, default=8, help="Number of multiprocessing workers")
    return p.parse_args()


def process_image(args_tuple):
    src_path, dst_path, d, sigma_color, sigma_space = args_tuple
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    cv2.imwrite(str(dst_path), filtered)
    return True


def main():
    args = parse_args()
    assert args.source.exists(), f"Source dataset not found at {args.source}"

    print("=" * 70)
    print("Building Bilateral-Filtered PCB Dataset")
    print(f"Source: {args.source}")
    print(f"Dest:   {args.dest}")
    print(f"Params: d={args.d}, sigmaColor={args.sigma_color}, sigmaSpace={args.sigma_space}")
    print("=" * 70)

    t0 = time.time()
    splits = ["train", "valid", "test"]

    for split in splits:
        src_img_dir = args.source / split / "images"
        src_lbl_dir = args.source / split / "labels"
        if not src_img_dir.exists():
            continue

        dst_img_dir = args.dest / split / "images"
        dst_lbl_dir = args.dest / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(
            [p for p in src_img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        print(f"\nProcessing '{split}' split ({len(img_files)} images)...")

        tasks = [
            (f, dst_img_dir / f.name, args.d, args.sigma_color, args.sigma_space)
            for f in img_files
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc=split))

        # Copy labels unchanged (exact coordinate preservation)
        print(f"Copying labels for '{split}' split...")
        lbl_files = list(src_lbl_dir.glob("*.txt"))
        for lbl_path in lbl_files:
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

    # Generate data.yaml pointing to new dataset
    src_yaml = args.source / "data.yaml"
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    new_cfg = {
        "path": str(args.dest.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": cfg.get("nc", 23),
        "names": cfg.get("names", []),
    }

    dst_yaml = args.dest / "data.yaml"
    with open(dst_yaml, "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    print(f"\n✅ Bilateral dataset built successfully in {time.time() - t0:.2f}s!")
    print(f"Data config saved to: {dst_yaml}")


if __name__ == "__main__":
    main()
