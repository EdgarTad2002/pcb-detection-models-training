#!/usr/bin/env python3
"""
build_sahi_tile_dataset.py
==========================
Generates a SAHI-style tiled training dataset from high-resolution PCB images.

Problem this solves
-------------------
At 640px, a 2500×1500px PCB image is downsampled 4×. Chip capacitors
that are 30×20px in the original image become 7×5px — at or below the
Nyquist limit for reliable detection.  SAHI tile training fixes this:

  640px standard:   2500px image → resize to 640px → cap = 7px  ← aliased
  SAHI tile train:  2500px image → extract 640px tiles → cap = 30px ← native

Strategy
--------
  • Train split  : source images sliced into tile_size×tile_size tiles
                   with stride overlap.  Annotations clipped per tile.
                   Empty tiles (no annotations) dropped by default.
  • Val split    : random 10% of train images, also tiled (no data leak).
  • Test split   : UNCHANGED — full images copied from source (for SAHI
                   eval and fair comparison vs. other models).

Usage
-----
  python tools/build_sahi_tile_dataset.py \\
      --source  datasets/pcb-native-res-unified-4class \\
      --dest    datasets/pcb-sahi-tile-640 \\
      --tile-size 640 --stride 320 \\
      --min-visibility 0.25 \\
      --val-ratio 0.10 \\
      --workers 8

Output
------
  datasets/pcb-sahi-tile-640/
    data.yaml
    train/images/  train/labels/   (tiles)
    valid/images/  valid/labels/   (tiles from held-out 10%)
    test/images/   test/labels/    (full images, for SAHI inference eval)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def tile_positions(W: int, H: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    """
    Generate (x, y) top-left corners for all tiles covering a W×H image.
    The last tile in each dimension is clamped so the tile stays in-bounds.
    """
    def positions_1d(dim: int) -> List[int]:
        if dim <= tile_size:
            return [0]
        ps = list(range(0, dim - tile_size, stride))
        # Ensure last tile covers the far edge
        if not ps or ps[-1] + tile_size < dim:
            ps.append(dim - tile_size)
        return sorted(set(ps))

    xs = positions_1d(W)
    ys = positions_1d(H)
    return [(x, y) for y in ys for x in xs]


def clip_boxes(
    label_lines: List[str],
    tx: int, ty: int,
    tile_size: int,
    img_W: int, img_H: int,
    min_visibility: float,
) -> List[Tuple]:
    """
    Clip YOLO-format labels to a tile window.

    Parameters
    ----------
    label_lines    : raw label file lines (one per box: cls cx cy w h)
    tx, ty         : tile top-left in image pixels
    tile_size      : tile width and height in pixels
    img_W, img_H   : full image dimensions
    min_visibility : minimum fraction of a box's area that must lie inside
                     the tile to keep the annotation (avoids tiny slivers)

    Returns
    -------
    List of (cls, cx, cy, w, h) in tile-normalised coords [0, 1].
    """
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

        # Absolute pixel coords in full image
        x1 = (cx - bw / 2) * img_W
        y1 = (cy - bh / 2) * img_H
        x2 = (cx + bw / 2) * img_W
        y2 = (cy + bh / 2) * img_H

        orig_area = max((x2 - x1) * (y2 - y1), 1e-6)

        # Intersection with tile
        ix1, iy1 = max(x1, tx), max(y1, ty)
        ix2, iy2 = min(x2, tx2), min(y2, ty2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue  # box completely outside tile

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        if inter_area / orig_area < min_visibility:
            continue  # too small a fraction visible

        # Re-normalise to tile coords
        ncx = (ix1 + ix2) / 2 - tx
        ncy = (iy1 + iy2) / 2 - ty
        nw = ix2 - ix1
        nh = iy2 - iy1

        kept.append((
            cls,
            ncx / tile_size, ncy / tile_size,
            nw  / tile_size, nh  / tile_size,
        ))
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Per-image tile worker
# ─────────────────────────────────────────────────────────────────────────────
def process_image(
    img_path: Path,
    lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    tile_size: int,
    stride: int,
    min_visibility: float,
    keep_empty: bool,
    jpeg_quality: int,
) -> Tuple[int, int]:
    """
    Slice one image into tiles and write tile image + label files.

    Returns (n_tiles_written, n_empty_tiles_dropped).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0

    H, W = img.shape[:2]
    stem = img_path.stem

    lbl_path = lbl_dir / (stem + ".txt")
    label_lines = open(lbl_path).readlines() if lbl_path.exists() else []

    positions = tile_positions(W, H, tile_size, stride)
    written = 0
    skipped_empty = 0

    for tx, ty in positions:
        tile_img = img[ty : ty + tile_size, tx : tx + tile_size]

        # Pad edge tiles to exact tile_size × tile_size
        th, tw = tile_img.shape[:2]
        if th < tile_size or tw < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:th, :tw] = tile_img
            tile_img = padded

        boxes = clip_boxes(label_lines, tx, ty, tile_size, W, H, min_visibility)

        if not boxes:
            if not keep_empty:
                skipped_empty += 1
                continue

        tile_stem = f"{stem}_t{ty:04d}_{tx:04d}"
        cv2.imwrite(
            str(dst_img_dir / f"{tile_stem}.jpg"),
            tile_img,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

        with open(dst_lbl_dir / f"{tile_stem}.txt", "w") as f:
            for cls, cx, cy, bw, bh in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        written += 1

    return written, skipped_empty


# ─────────────────────────────────────────────────────────────────────────────
# Split builder
# ─────────────────────────────────────────────────────────────────────────────
def build_tiled_split(
    img_paths: List[Path],
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    tile_size: int,
    stride: int,
    min_visibility: float,
    keep_empty: bool,
    jpeg_quality: int,
    workers: int,
    split_name: str,
) -> Tuple[int, int]:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0

    print(f"\n  [{split_name}] Processing {len(img_paths)} images "
          f"(tile={tile_size}px, stride={stride}px, vis≥{min_visibility:.0%}) ...")

    if workers <= 1:
        for img_path in img_paths:
            w, s = process_image(
                img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir,
                tile_size, stride, min_visibility, keep_empty, jpeg_quality,
            )
            total_written += w
            total_skipped += s
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    process_image,
                    img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir,
                    tile_size, stride, min_visibility, keep_empty, jpeg_quality,
                ): img_path
                for img_path in img_paths
            }
            for fut in as_completed(futures):
                w, s = fut.result()
                total_written += w
                total_skipped += s

    print(f"  [{split_name}] Done: {total_written} tiles written, "
          f"{total_skipped} empty tiles dropped.")
    return total_written, total_skipped


# ─────────────────────────────────────────────────────────────────────────────
# Test split: copy full images unchanged (for SAHI inference eval)
# ─────────────────────────────────────────────────────────────────────────────
def copy_split(src_img_dir: Path, src_lbl_dir: Path,
               dst_img_dir: Path, dst_lbl_dir: Path, split_name: str):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_img in src_img_dir.iterdir():
        if src_img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(src_img, dst_img_dir / src_img.name)
        lbl = src_lbl_dir / (src_img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl_dir / lbl.name)
        n += 1
    print(f"  [{split_name}] Copied {n} full images (for SAHI inference).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Build SAHI tile training dataset from native-resolution PCB images."
    )
    p.add_argument("--source", type=Path,
                   default=Path("datasets/pcb-native-res-unified-4class"),
                   help="Source dataset root (must contain train/ and test/ splits)")
    p.add_argument("--dest", type=Path,
                   default=Path("datasets/pcb-sahi-tile-640"),
                   help="Output dataset root")
    p.add_argument("--tile-size", type=int, default=640,
                   help="Tile width and height in pixels (default 640)")
    p.add_argument("--stride", type=int, default=320,
                   help="Tile stride in pixels; overlap = tile_size - stride "
                        "(default 320 → 50%% overlap)")
    p.add_argument("--min-visibility", type=float, default=0.25,
                   help="Minimum fraction of a bounding box that must be visible "
                        "inside a tile to keep the annotation (default 0.25)")
    p.add_argument("--val-ratio", type=float, default=0.10,
                   help="Fraction of train images reserved for validation "
                        "(split at image level, not tile level; default 0.10)")
    p.add_argument("--keep-empty", action="store_true",
                   help="Keep tiles with no annotations (background tiles). "
                        "Default: drop empty tiles.")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality for saved tiles (default 95)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val split")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker processes (default 8)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)

    src = args.source.resolve()
    dst = args.dest.resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    print(f"\n{'='*60}")
    print(f"  SAHI Tile Dataset Builder")
    print(f"  Source  : {src}")
    print(f"  Dest    : {dst}")
    print(f"  Tile    : {args.tile_size}×{args.tile_size}px, stride={args.stride}px "
          f"(overlap={args.tile_size - args.stride}px, "
          f"{100*(args.tile_size - args.stride)//args.tile_size}%)")
    print(f"  Min vis : {args.min_visibility:.0%}")
    print(f"  Val     : {args.val_ratio:.0%} of train images (held out at image level)")
    print(f"{'='*60}\n")

    # ── Collect train images ──────────────────────────────────────────────
    src_train_img = src / "train" / "images"
    src_train_lbl = src / "train" / "labels"

    all_train_imgs = sorted(
        p for p in src_train_img.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not all_train_imgs:
        raise FileNotFoundError(f"No images found in {src_train_img}")

    random.shuffle(all_train_imgs)
    n_val = max(1, int(len(all_train_imgs) * args.val_ratio))
    val_imgs   = all_train_imgs[:n_val]
    train_imgs = all_train_imgs[n_val:]

    print(f"  Source train images : {len(all_train_imgs)}")
    print(f"  → Train subset      : {len(train_imgs)}")
    print(f"  → Val   subset      : {len(val_imgs)}")

    # ── Build train tiles ─────────────────────────────────────────────────
    build_tiled_split(
        train_imgs, src_train_lbl,
        dst / "train" / "images", dst / "train" / "labels",
        args.tile_size, args.stride, args.min_visibility,
        args.keep_empty, args.jpeg_quality, args.workers, "train",
    )

    # ── Build val tiles ───────────────────────────────────────────────────
    build_tiled_split(
        val_imgs, src_train_lbl,
        dst / "valid" / "images", dst / "valid" / "labels",
        args.tile_size, args.stride, args.min_visibility,
        args.keep_empty, args.jpeg_quality, args.workers, "valid",
    )

    # ── Copy test split (full images for SAHI eval) ───────────────────────
    # Use whichever test dir exists
    for test_dir_name in ["test", "valid", "val"]:
        src_test_img = src / test_dir_name / "images"
        src_test_lbl = src / test_dir_name / "labels"
        if src_test_img.exists() and any(src_test_img.iterdir()):
            copy_split(
                src_test_img, src_test_lbl,
                dst / "test" / "images", dst / "test" / "labels",
                "test",
            )
            break

    # ── Write data.yaml ───────────────────────────────────────────────────
    # Read class names from source
    src_yaml = src / "data.yaml"
    names = {0: "Capacitor", 1: "Connector", 2: "Electrolytic Capacitor", 3: "IC"}
    if src_yaml.exists():
        src_cfg = yaml.safe_load(src_yaml.read_text())
        if "names" in src_cfg:
            raw = src_cfg["names"]
            names = dict(enumerate(raw)) if isinstance(raw, list) else raw

    data_cfg = {
        "path": str(dst),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(names),
        "names": names,
        # Metadata for reference
        "_tile_size": args.tile_size,
        "_stride":    args.stride,
        "_source":    str(src),
    }
    out_yaml = dst / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    # ── Summary ───────────────────────────────────────────────────────────
    n_train = sum(1 for _ in (dst / "train" / "images").iterdir())
    n_valid = sum(1 for _ in (dst / "valid" / "images").iterdir())
    n_test  = sum(1 for _ in (dst / "test"  / "images").iterdir())

    print(f"\n{'='*60}")
    print(f"  Dataset written to: {dst}")
    print(f"  train tiles : {n_train}")
    print(f"  valid tiles : {n_valid}")
    print(f"  test images : {n_test}  (full, for SAHI inference)")
    print(f"  data.yaml   : {out_yaml}")
    print(f"\n  Next steps:")
    print(f"    sbatch sbatch/train_sahi_tile_640.sh")
    print(f"    # then SAHI evaluation:")
    print(f"    python tools/eval_sahi_benchmark.py \\")
    print(f"        --weights runs/sahi_tile_640/pcb-filtered/weights/best.pt \\")
    print(f"        --data {out_yaml}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
