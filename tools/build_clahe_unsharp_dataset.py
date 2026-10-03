#!/usr/bin/env python3
"""
Applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) and
Gaussian Unsharp Masking with configurable (sigma, gamma, clip_limit) to a PCB dataset.

Parameters:
- sigma: Gaussian blur kernel standard deviation (spatial scale / frequency cutoff)
- gamma: Sharpening boost strength (I_sharp = (1 + gamma)*I - gamma*Blurred)
- clip_limit: CLAHE contrast threshold on LAB L-channel (prevents noise over-amplification)
- tile_size: CLAHE grid size (default 8x8)

Usage:
    python tools/build_clahe_unsharp_dataset.py \
        --source datasets/pcb-unified-4class \
        --dest datasets/pcb-clahe-s0.5-g0.3-640 \
        --sigma 0.5 --gamma 0.3 --clip-limit 1.5 --workers 8
"""

import argparse
import concurrent.futures
import shutil
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate CLAHE + Unsharp-Masked PCB Dataset")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/pcb-unified-4class"),
        help="Path to source YOLO dataset (e.g. datasets/pcb-unified-4class)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Path to destination dataset directory",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian blur standard deviation for unsharp mask (spatial scale)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="Unsharp mask scaling / strength parameter (edge boost factor)",
    )
    p.add_argument(
        "--clip-limit",
        type=float,
        default=1.5,
        help="CLAHE contrast clip limit on LAB L-channel (0 to disable CLAHE)",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=8,
        help="CLAHE grid tile size (default: 8 for 8x8 grid)",
    )
    p.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE and apply Gaussian unsharp masking only",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel multiprocessing workers",
    )
    return p.parse_args()


def process_image(args_tuple: Tuple) -> bool:
    src_path, dst_path, sigma, gamma, clip_limit, tile_size, use_clahe = args_tuple
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        return False

    # 1. CLAHE in LAB color space
    if use_clahe and clip_limit > 0:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        base_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        base_bgr = img_bgr

    # 2. Gaussian Unsharp Masking
    if gamma > 0 and sigma > 0:
        blurred = cv2.GaussianBlur(base_bgr, (0, 0), sigma)
        sharpened = cv2.addWeighted(base_bgr, 1.0 + gamma, blurred, -gamma, 0)
        out_bgr = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        out_bgr = base_bgr

    cv2.imwrite(str(dst_path), out_bgr)
    return True


def main():
    args = parse_args()
    assert args.source.exists(), f"Source dataset not found at {args.source}"

    use_clahe = (not args.no_clahe) and (args.clip_limit > 0)
    print("=" * 75)
    print("🛠️  Building CLAHE + Unsharp-Masked PCB Dataset")
    print(f"Source:       {args.source}")
    print(f"Destination:  {args.dest}")
    print(f"Parameters:   sigma={args.sigma}, gamma={args.gamma}")
    print(f"CLAHE:        {'Enabled (clipLimit=' + str(args.clip_limit) + ', tile=' + str(args.tile_size) + ')' if use_clahe else 'Disabled'}")
    print("=" * 75)

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
            (
                f,
                dst_img_dir / f.name,
                args.sigma,
                args.gamma,
                args.clip_limit,
                args.tile_size,
                use_clahe,
            )
            for f in img_files
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc=split))

        # Copy label annotations unchanged (exact bounding box preservation)
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
        "nc": cfg.get("nc", 4),
        "names": cfg.get("names", ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]),
    }

    dst_yaml = args.dest / "data.yaml"
    with open(dst_yaml, "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    print(f"\n✅ CLAHE/Unsharp dataset built successfully in {time.time() - t0:.2f}s!")
    print(f"Data config saved to: {dst_yaml}")


if __name__ == "__main__":
    main()
