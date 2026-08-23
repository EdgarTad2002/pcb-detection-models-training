#!/usr/bin/env python3
"""
Slices the train/valid splits of a YOLO-format dataset into overlapping
tiles, so small/dense objects (Capacitors) occupy proportionally more
pixels per training example. The test split is left untouched at full
resolution -- evaluation happens via SAHI sliced inference (see
eval_sahi.py), which better reflects real deployment than evaluating on
pre-sliced test tiles.

Usage:
    python slice_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest datasets/pcb-tiled-640 \
        --tile-size 640 \
        --overlap 0.2

Output layout (mirrors the source dataset's structure):
    <dest>/
        train/images/*.jpg   train/labels/*.txt
        valid/images/*.jpg   valid/labels/*.txt
        data.yaml            # train/valid point here, test points at the
                              # ORIGINAL untiled test images
"""

import argparse
import random
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Path to source YOLO dataset root (containing train/valid/test)")
    p.add_argument("--dest", type=Path, required=True, help="Output path for the tiled dataset")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2, help="Fraction of tile size to overlap between adjacent tiles")
    p.add_argument("--min-visibility", type=float, default=0.3, help="Minimum fraction of a box's area that must survive clipping to keep it")
    p.add_argument("--keep-empty-prob", type=float, default=0.1, help="Probability of keeping a tile with zero boxes (background/hard-negative mining)")
    p.add_argument("--splits", nargs="+", default=["train", "valid"], help="Which splits to tile (test is normally left alone)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_yolo_labels(label_path):
    """Returns list of (class_id, cx, cy, w, h), all normalized [0,1]."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    cls_id, cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return cls_id, x1, y1, x2, y2


def xyxy_to_yolo(cls_id, x1, y1, x2, y2, tile_w, tile_h):
    cx = ((x1 + x2) / 2) / tile_w
    cy = ((y1 + y2) / 2) / tile_h
    w = (x2 - x1) / tile_w
    h = (y2 - y1) / tile_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def compute_tile_origins(img_size, tile_size, overlap):
    """Returns list of tile top-left coordinates along one axis, covering the full image."""
    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        stride = tile_size
    origins = list(range(0, max(img_size - tile_size, 0) + 1, stride))
    last_possible = max(img_size - tile_size, 0)
    if not origins or origins[-1] != last_possible:
        origins.append(last_possible)
    return sorted(set(origins))


def slice_image(img_path, label_path, out_img_dir, out_lbl_dir, tile_size, overlap, min_visibility, keep_empty_prob, rng):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"WARNING: could not read {img_path}, skipping")
        return 0
    img_h, img_w = img.shape[:2]
    boxes = read_yolo_labels(label_path)
    boxes_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in boxes]

    xs = compute_tile_origins(img_w, tile_size, overlap)
    ys = compute_tile_origins(img_h, tile_size, overlap)

    stem = img_path.stem
    n_written = 0

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            tx1, ty1 = x, y
            tx2, ty2 = min(x + tile_size, img_w), min(y + tile_size, img_h)
            tile_w, tile_h = tx2 - tx1, ty2 - ty1

            tile_boxes = []
            for cls_id, bx1, by1, bx2, by2 in boxes_xyxy:
                orig_area = max(bx2 - bx1, 0) * max(by2 - by1, 0)
                if orig_area <= 0:
                    continue
                ix1, iy1 = max(bx1, tx1), max(by1, ty1)
                ix2, iy2 = min(bx2, tx2), min(by2, ty2)
                inter_w, inter_h = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
                inter_area = inter_w * inter_h
                if inter_area / orig_area < min_visibility:
                    continue
                # box coords relative to tile origin
                nx1, ny1 = ix1 - tx1, iy1 - ty1
                nx2, ny2 = ix2 - tx1, iy2 - ty1
                tile_boxes.append(xyxy_to_yolo(cls_id, nx1, ny1, nx2, ny2, tile_w, tile_h))

            if not tile_boxes and rng.random() > keep_empty_prob:
                continue  # drop empty tile (mostly) to avoid overwhelming with background

            tile_img = img[ty1:ty2, tx1:tx2]
            # pad up to tile_size if this tile is at the image edge and smaller
            if tile_h < tile_size or tile_w < tile_size:
                padded = 255 * (0 * tile_img.copy())  # placeholder, replaced below
                import numpy as np
                padded = np.full((tile_size, tile_size, 3), 114, dtype=tile_img.dtype)  # YOLO-standard grey pad
                padded[:tile_h, :tile_w] = tile_img
                tile_img = padded

            out_name = f"{stem}_r{row}c{col}"
            cv2.imwrite(str(out_img_dir / f"{out_name}.jpg"), tile_img)
            (out_lbl_dir / f"{out_name}.txt").write_text("\n".join(tile_boxes))
            n_written += 1

    return n_written


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    src_yaml = args.source / "data.yaml"
    assert src_yaml.exists(), f"Source data.yaml not found at {src_yaml}"
    with open(src_yaml) as f:
        src_cfg = yaml.safe_load(f)

    total_tiles = 0
    for split in args.splits:
        img_dir = args.source / split / "images"
        lbl_dir = args.source / split / "labels"
        assert img_dir.exists(), f"Missing {img_dir}"

        out_img_dir = args.dest / split / "images"
        out_lbl_dir = args.dest / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"Tiling {split}: {len(img_files)} source images")
        split_tiles = 0
        for img_path in tqdm(img_files, desc=split):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            n = slice_image(
                img_path, label_path, out_img_dir, out_lbl_dir,
                args.tile_size, args.overlap, args.min_visibility,
                args.keep_empty_prob, rng,
            )
            split_tiles += n
        print(f"  -> wrote {split_tiles} tiles for {split}")
        total_tiles += split_tiles

    # Build the new data.yaml: train/valid point at tiled data, test stays
    # at the ORIGINAL full-resolution images (for SAHI sliced inference).
    new_cfg = {
        "path": str(args.dest.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": str((args.source / "test" / "images").resolve()),
        "nc": src_cfg["nc"],
        "names": src_cfg["names"],
    }
    with open(args.dest / "data.yaml", "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    print(f"\nDone. {total_tiles} total tiles written.")
    print(f"New data.yaml: {args.dest / 'data.yaml'}")
    print("NOTE: 'test' in this yaml points at the ORIGINAL untiled images -- "
          "evaluate this model with eval_sahi.py, not a plain model.val() call, "
          "since the model was trained on tiles but should be evaluated with "
          "sliced inference on full boards to reflect real deployment.")


if __name__ == "__main__":
    main()
