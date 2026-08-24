#!/usr/bin/env python3
"""
Bbox-based copy-paste augmentation (no segmentation masks needed).

Builds an expanded training set: every original image is copied through
unchanged, PLUS an augmented sibling version with 1-4 random capacitor
crops (from build_capacitor_bank.py's output) pasted onto it at random
non-overlapping positions. Valid/test splits are left untouched.

This is a static, one-time expansion (not re-randomized every epoch like
Ultralytics' native copy_paste) -- simpler and more robust than hooking
into the live augmentation pipeline, at the cost of somewhat less variety
per epoch. Run build_capacitor_bank.py first.

Usage:
    python bbox_copy_paste.py \
        --source datasets/pcb-native-res \
        --bank capacitor_bank \
        --dest datasets/pcb-native-res-cappaste \
        --paste-prob 0.7 --min-pastes 1 --max-pastes 4
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

CAPACITOR_CLASS_ID = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--paste-prob", type=float, default=0.7, help="Probability of creating an augmented sibling per source image")
    p.add_argument("--min-pastes", type=int, default=1)
    p.add_argument("--max-pastes", type=int, default=4)
    p.add_argument("--scale-jitter", type=float, nargs=2, default=[0.7, 1.3])
    p.add_argument("--max-iou-overlap", type=float, default=0.05, help="Reject a paste position if it overlaps an existing box more than this")
    p.add_argument("--max-attempts", type=int, default=15, help="Position-sampling attempts per paste before giving up")
    p.add_argument("--feather-px", type=int, default=6, help="Alpha-blend edge width, in pixels, to avoid a hard paste seam")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_labels(label_path):
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


def to_xyxy_px(box, img_w, img_h):
    _, cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def feathered_paste(base_img, crop, x1, y1, feather_px=6):
    """
    Pastes `crop` onto `base_img` at (x1, y1) with a soft alpha-feathered
    edge, so the paste blends into the new background instead of leaving a
    hard rectangular seam (which otherwise gives the model an easy,
    unrealistic "pasted-looking" shortcut to key on rather than learning
    real capacitor appearance).
    """
    ch, cw = crop.shape[:2]
    # build a mask: 1.0 in the interior, fading to 0.0 over `feather_px`
    # pixels at each edge
    mask = np.ones((ch, cw), dtype=np.float32)
    f = min(feather_px, ch // 2, cw // 2)
    if f > 0:
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        mask[:f, :] *= ramp[:, None]
        mask[-f:, :] *= ramp[::-1][:, None]
        mask[:, :f] *= ramp[None, :]
        mask[:, -f:] *= ramp[None, ::-1]
    mask3 = mask[:, :, None]

    region = base_img[y1:y1 + ch, x1:x1 + cw].astype(np.float32)
    blended = region * (1 - mask3) + crop.astype(np.float32) * mask3
    base_img[y1:y1 + ch, x1:x1 + cw] = blended.astype(np.uint8)
    return base_img


def match_brightness(crop, base_img, x1, y1):
    """
    Cheap photometric matching: shifts the crop's average brightness toward
    the target region's average brightness, so a bright crop pasted onto a
    dark board (or vice versa) doesn't stand out as an obvious mismatch.
    """
    ch, cw = crop.shape[:2]
    target_region = base_img[y1:y1 + ch, x1:x1 + cw]
    crop_mean = crop.reshape(-1, 3).mean(axis=0)
    target_mean = target_region.reshape(-1, 3).mean(axis=0)
    shift = target_mean - crop_mean
    matched = crop.astype(np.float32) + shift * 0.6  # partial correction, not full match
    return np.clip(matched, 0, 255).astype(np.uint8)


def paste_crops(img, existing_boxes_px, bank_paths, args, rng):
    h, w = img.shape[:2]
    n_pastes = rng.randint(args.min_pastes, args.max_pastes)
    new_labels = []

    for _ in range(n_pastes):
        crop_path = rng.choice(bank_paths)
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        scale = rng.uniform(*args.scale_jitter)
        ch, cw = crop.shape[:2]
        ch, cw = max(1, int(ch * scale)), max(1, int(cw * scale))
        if ch >= h or cw >= w:
            continue
        crop = cv2.resize(crop, (cw, ch))

        if rng.random() < 0.5:
            crop = cv2.flip(crop, 1)  # cheap horizontal-flip variety

        placed = False
        for _ in range(args.max_attempts):
            x1 = rng.randint(0, w - cw)
            y1 = rng.randint(0, h - ch)
            candidate_box = (x1, y1, x1 + cw, y1 + ch)
            max_iou = max(
                [iou_xyxy(candidate_box, b) for b in existing_boxes_px] or [0.0]
            )
            if max_iou <= args.max_iou_overlap:
                crop = match_brightness(crop, img, x1, y1)
                feathered_paste(img, crop, x1, y1, feather_px=args.feather_px)
                existing_boxes_px.append(candidate_box)
                cx = (x1 + cw / 2) / w
                cy = (y1 + ch / 2) / h
                nw, nh = cw / w, ch / h
                new_labels.append(f"{CAPACITOR_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                placed = True
                break
        # if not placed after max_attempts, just skip this paste -- no space found

    return img, new_labels


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    bank_paths = sorted(list(args.bank.glob("*.png")) + list(args.bank.glob("*.jpg")))
    assert bank_paths, f"No crops found in {args.bank} -- run build_capacitor_bank.py first"
    print(f"Loaded {len(bank_paths)} capacitor crops from bank")

    src_img_dir = args.source / "train" / "images"
    src_lbl_dir = args.source / "train" / "labels"
    dest_img_dir = args.dest / "train" / "images"
    dest_lbl_dir = args.dest / "train" / "labels"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
    n_originals, n_augmented, n_pasted_boxes = 0, 0, 0

    for img_path in img_files:
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        # 1. Copy the original through unchanged -- never lose real data
        shutil.copy(img_path, dest_img_dir / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, dest_lbl_dir / label_path.name)
        n_originals += 1

        # 2. Probabilistically create an augmented sibling
        if rng.random() >= args.paste_prob:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        existing_boxes = load_labels(label_path)
        existing_boxes_px = [list(to_xyxy_px(b, w, h)) for b in existing_boxes]

        aug_img, new_labels = paste_crops(img.copy(), existing_boxes_px, bank_paths, args, rng)
        if not new_labels:
            continue  # no room found, skip this sibling

        out_stem = f"{img_path.stem}_cappaste"
        cv2.imwrite(str(dest_img_dir / f"{out_stem}.jpg"), aug_img)

        all_lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in existing_boxes]
        all_lines.extend(new_labels)
        (dest_lbl_dir / f"{out_stem}.txt").write_text("\n".join(all_lines))

        n_augmented += 1
        n_pasted_boxes += len(new_labels)

    # valid/test stay untouched, pointed at the ORIGINAL source
    src_yaml = args.source / "data.yaml"
    with open(src_yaml) as f:
        src_cfg = yaml.safe_load(f)

    new_cfg = {
        "path": str(args.dest.resolve()),
        "train": "train/images",
        "val": str((args.source / "valid" / "images").resolve()),
        "test": str((args.source / "test" / "images").resolve()),
        "nc": src_cfg["nc"],
        "names": src_cfg["names"],
    }
    with open(args.dest / "data.yaml", "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    print(f"\nDone.")
    print(f"  Original images copied through: {n_originals}")
    print(f"  Augmented siblings created:      {n_augmented}")
    print(f"  Total pasted capacitor boxes:    {n_pasted_boxes}")
    print(f"  Total training images:           {n_originals + n_augmented}")
    print(f"New data.yaml: {args.dest / 'data.yaml'}")


if __name__ == "__main__":
    main()
