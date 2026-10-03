#!/usr/bin/env python3
"""
build_omni_scale_dataset.py
===========================
Generates the Omni-Scale Compound Training Dataset for PCB component detection.

Unifies all winning techniques:
1. Data Rectification & 4-Class Unification (no legacy label drift).
2. Photometric Conditioning (CLAHE + Unsharp Masking) to normalize specular reflections
   and metallic sheen on electrolytic capacitor cans and solder joints.
3. Multi-Context Mixed-Scale Training:
   - Native-resolution 640px tiles: preserves 0402 chip capacitors at true 30px scale.
   - Full PCB boards: preserves global context, board outline, and large IC/Connector bounds.
4. Held-out validation and clean native-res test split for Dual-Stream SAHI Hyper-Inference.

Usage:
    python tools/build_omni_scale_dataset.py \
        --source datasets/pcb-native-res-unified-4class \
        --dest datasets/pcb-omni-scale-clahe \
        --tile-size 640 \
        --stride 320 \
        --sigma 1.0 \
        --gamma 0.6 \
        --clip-limit 1.5 \
        --workers 8
"""

import argparse
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Build Omni-Scale CLAHE PCB Dataset")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/pcb-native-res-unified-4class"),
        help="Path to source native-resolution dataset",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("datasets/pcb-omni-scale-clahe"),
        help="Path to output destination dataset",
    )
    p.add_argument("--tile-size", type=int, default=640, help="Tile dimension in pixels")
    p.add_argument("--stride", type=int, default=320, help="Stride step between tile windows")
    p.add_argument("--min-visibility", type=float, default=0.25, help="Minimum visible box fraction inside tile")
    p.add_argument("--sigma", type=float, default=1.0, help="Unsharp mask blur sigma")
    p.add_argument("--gamma", type=float, default=0.6, help="Unsharp mask boost strength")
    p.add_argument("--clip-limit", type=float, default=1.5, help="CLAHE contrast clip limit")
    p.add_argument("--tile-clahe-grid", type=int, default=8, help="CLAHE tile grid size (8x8)")
    p.add_argument("--val-ratio", type=float, default=0.10, help="Held-out validation ratio")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG write quality")
    p.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Photometric enhancement (CLAHE + Unsharp Mask)
# ─────────────────────────────────────────────────────────────────────────────
def enhance_image(
    img_bgr: np.ndarray,
    sigma: float = 1.0,
    gamma: float = 0.6,
    clip_limit: float = 1.5,
    tile_size: int = 8,
) -> np.ndarray:
    if clip_limit > 0:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        base_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        base_bgr = img_bgr

    if gamma > 0 and sigma > 0:
        blurred = cv2.GaussianBlur(base_bgr, (0, 0), sigma)
        sharpened = cv2.addWeighted(base_bgr, 1.0 + gamma, blurred, -gamma, 0)
        out_bgr = np.clip(sharpened, 0, 255).astype(np.uint8)
    else:
        out_bgr = base_bgr
    return out_bgr


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Tiling Helpers
# ─────────────────────────────────────────────────────────────────────────────
def tile_positions(W: int, H: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    def positions_1d(dim: int) -> List[int]:
        if dim <= tile_size:
            return [0]
        ps = list(range(0, dim - tile_size, stride))
        if not ps or ps[-1] + tile_size < dim:
            ps.append(dim - tile_size)
        return sorted(set(ps))

    xs = positions_1d(W)
    ys = positions_1d(H)
    return [(x, y) for y in ys for x in xs]


def clip_boxes_to_tile(
    label_lines: List[str],
    tx: int,
    ty: int,
    tile_size: int,
    img_W: int,
    img_H: int,
    min_visibility: float,
) -> List[Tuple]:
    kept = []
    tx2, ty2 = tx + tile_size, ty + tile_size

    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue

        x1 = (cx - bw / 2) * img_W
        y1 = (cy - bh / 2) * img_H
        x2 = (cx + bw / 2) * img_W
        y2 = (cy + bh / 2) * img_H

        orig_area = max((x2 - x1) * (y2 - y1), 1e-6)

        ix1, iy1 = max(x1, tx), max(y1, ty)
        ix2, iy2 = min(x2, tx2), min(y2, ty2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        if inter_area / orig_area < min_visibility:
            continue

        ncx = (ix1 + ix2) / 2 - tx
        ncy = (iy1 + iy2) / 2 - ty
        nw = ix2 - ix1
        nh = iy2 - iy1

        kept.append((
            cls,
            ncx / tile_size,
            ncy / tile_size,
            nw / tile_size,
            nh / tile_size,
        ))
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Per-Image Processing Worker
# ─────────────────────────────────────────────────────────────────────────────
def process_omni_image(args_bundle: Tuple) -> Tuple[int, int]:
    (
        img_path,
        lbl_dir,
        dst_img_dir,
        dst_lbl_dir,
        tile_size,
        stride,
        min_visibility,
        sigma,
        gamma,
        clip_limit,
        tile_clahe_grid,
        jpeg_quality,
        include_full,
    ) = args_bundle

    raw = cv2.imread(str(img_path))
    if raw is None:
        return 0, 0

    enhanced = enhance_image(
        raw, sigma=sigma, gamma=gamma, clip_limit=clip_limit, tile_size=tile_clahe_grid
    )
    H, W = enhanced.shape[:2]
    stem = img_path.stem

    lbl_path = lbl_dir / (stem + ".txt")
    label_lines = open(lbl_path).readlines() if lbl_path.exists() else []

    tiles_written = 0

    # 1. Write full image (Macro Context)
    if include_full:
        full_out_img = dst_img_dir / f"{stem}_full.jpg"
        full_out_lbl = dst_lbl_dir / f"{stem}_full.txt"
        cv2.imwrite(str(full_out_img), enhanced, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        with open(full_out_lbl, "w") as f:
            f.writelines(label_lines)
        tiles_written += 1

    # 2. Slice and write native-scale tiles (Micro Detail)
    positions = tile_positions(W, H, tile_size, stride)
    for tx, ty in positions:
        boxes = clip_boxes_to_tile(label_lines, tx, ty, tile_size, W, H, min_visibility)
        if not boxes:
            continue  # drop empty background tiles

        tile_crop = enhanced[ty : ty + tile_size, tx : tx + tile_size]
        tile_name = f"{stem}_tile_{tx}_{ty}"
        out_img = dst_img_dir / f"{tile_name}.jpg"
        out_lbl = dst_lbl_dir / f"{tile_name}.txt"

        cv2.imwrite(str(out_img), tile_crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        with open(out_lbl, "w") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        tiles_written += 1

    return tiles_written, len(positions)


def build_split(
    image_paths: List[Path],
    lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    args: argparse.Namespace,
    split_name: str,
    include_full: bool = True,
):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    bundles = [
        (
            p,
            lbl_dir,
            dst_img_dir,
            dst_lbl_dir,
            args.tile_size,
            args.stride,
            args.min_visibility,
            args.sigma,
            args.gamma,
            args.clip_limit,
            args.tile_clahe_grid,
            args.jpeg_quality,
            include_full,
        )
        for p in image_paths
    ]

    total_written = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_omni_image, b): b[0] for b in bundles}
        for f in as_completed(futures):
            written, _ = f.result()
            total_written += written

    print(f"  [{split_name.upper():<5}] Produced {total_written} training samples from {len(image_paths)} source boards")


def main():
    args = parse_args()
    assert args.source.exists(), f"Source directory not found: {args.source}"

    print("=" * 70)
    print("Omni-Scale Compound PCB Dataset Generator")
    print(f"Source:       {args.source}")
    print(f"Destination:  {args.dest}")
    print(f"Tile Params:  size={args.tile_size}px, stride={args.stride}px")
    print(f"CLAHE Params: sigma={args.sigma}, gamma={args.gamma}, clip={args.clip_limit}")
    print("=" * 70)

    # Resolve train image and label paths
    src_train_img = args.source / "train" / "images"
    src_train_lbl = args.source / "train" / "labels"
    all_train_imgs = sorted(
        p for p in src_train_img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not all_train_imgs:
        raise FileNotFoundError(f"No train images in {src_train_img}")

    random.seed(42)
    random.shuffle(all_train_imgs)
    n_val = max(1, int(len(all_train_imgs) * args.val_ratio))
    val_imgs = all_train_imgs[:n_val]
    train_imgs = all_train_imgs[n_val:]

    # 1. Build Train split (Macro Full + Micro Tiles)
    build_split(
        train_imgs,
        src_train_lbl,
        args.dest / "train" / "images",
        args.dest / "train" / "labels",
        args,
        "train",
        include_full=True,
    )

    # 2. Build Validation split (Macro Full + Micro Tiles)
    build_split(
        val_imgs,
        src_train_lbl,
        args.dest / "valid" / "images",
        args.dest / "valid" / "labels",
        args,
        "valid",
        include_full=True,
    )

    # 3. Process Test split (Native resolution with CLAHE applied for clean SAHI inference)
    test_src_img = None
    for name in ["test", "valid", "val"]:
        p = args.source / name / "images"
        if p.exists() and any(p.iterdir()):
            test_src_img = p
            test_src_lbl = args.source / name / "labels"
            break

    if test_src_img:
        dst_test_img = args.dest / "test" / "images"
        dst_test_lbl = args.dest / "test" / "labels"
        dst_test_img.mkdir(parents=True, exist_ok=True)
        dst_test_lbl.mkdir(parents=True, exist_ok=True)

        test_files = sorted(
            p for p in test_src_img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"  [TEST ] Processing {len(test_files)} test boards with CLAHE...")
        for p in test_files:
            img = cv2.imread(str(p))
            if img is not None:
                enh = enhance_image(
                    img,
                    sigma=args.sigma,
                    gamma=args.gamma,
                    clip_limit=args.clip_limit,
                    tile_size=args.tile_clahe_grid,
                )
                cv2.imwrite(str(dst_test_img / f"{p.stem}.jpg"), enh, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            lbl_file = test_src_lbl / (p.stem + ".txt")
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_test_lbl / lbl_file.name)

    # 4. Generate data.yaml
    names = {0: "Capacitor", 1: "Connector", 2: "Electrolytic Capacitor", 3: "IC"}
    src_yaml = args.source / "data.yaml"
    if src_yaml.exists():
        cfg = yaml.safe_load(src_yaml.read_text())
        if "names" in cfg:
            raw_names = cfg["names"]
            names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else raw_names

    data_cfg = {
        "path": str(args.dest),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
        "_meta": {
            "type": "Omni-Scale CLAHE Mixed Dataset",
            "tile_size": args.tile_size,
            "stride": args.stride,
            "sigma": args.sigma,
            "gamma": args.gamma,
            "clip_limit": args.clip_limit,
        },
    }

    out_yaml = args.dest / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Omni-Scale dataset written to: {args.dest}")
    print(f"   data.yaml: {out_yaml}\n")


if __name__ == "__main__":
    main()
