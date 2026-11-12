#!/usr/bin/env python3
"""
Multi-Class Photometric-Matched Feathered BBox Copy-Paste Builder for R4-Final
===============================================================================
Why this is critical for `pcb-native-reviewed-r4-final`:
1. Ultralytics' built-in `--copy-paste` augmentation ONLY operates on polygon
   instance segmentation masks (`segments`). On bounding-box datasets like
   `r4-final`, `--copy-paste 0.30` is a 100% NO-OP!
2. `r4-final/train` (492 images) has extreme class imbalance:
     - Class 0 (Capacitor):              36,442 instances (tiny scale bottleneck)
     - Class 1 (Connector):               2,609 instances (14x fewer -> 33.3% val AP!)
     - Class 2 (Electrolytic Capacitor):  1,981 instances (18x fewer)
     - Class 3 (IC):                      5,816 instances (6.3x fewer)
3. This script extracts a high-resolution crop bank of all 4 classes from
   `r4-final/train` (with context margin for seamless cosine alpha feathering)
   and creates an expanded training split where every original training board
   is preserved untouched PLUS augmented sibling boards are generated with:
     - 6 to 12 Micro-Capacitors (Class 0)
     - 3 to 6 Connectors (Class 1, with 90/180/270 deg orthogonal PCB rotations)
     - 2 to 5 Electrolytic Capacitors (Class 2)
     - 1 to 3 ICs (Class 3)
   pasted onto unoccupied substrate regions (IoU <= 0.03) with local photometric
   brightness matching and 6px cosine alpha border feathering.
4. `valid` and `test` splits point directly to the untouched canonical
   `pcb-native-reviewed-r4-final` splits.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Build Multi-Class Feathered Copy-Paste Dataset for R4-Final")
    p.add_argument(
        "--source",
        "--src-dir",
        dest="source",
        type=Path,
        default=Path("datasets/pcb-native-reviewed-r4-final"),
        help="Canonical R4-Final dataset directory",
    )
    p.add_argument(
        "--bank",
        type=Path,
        default=Path("datasets/r4_final_component_bank"),
        help="Output directory for extracted multi-class component crops",
    )
    p.add_argument(
        "--dest",
        "--dst-dir",
        dest="dest",
        type=Path,
        default=Path("datasets/pcb-r4-final-apex-copypaste"),
        help="Destination dataset directory with expanded train split",
    )
    p.add_argument("--margin", type=float, default=0.30, help="Context margin fraction around each box")
    p.add_argument("--feather-px", type=int, default=6, help="Cosine alpha feathering width in pixels")
    p.add_argument("--max-iou", type=float, default=0.03, help="Max allowed IoU overlap with existing boxes")
    p.add_argument("--aug-prob", type=float, default=1.0, help="Probability of generating augmented sibling per board")
    p.add_argument("--siblings-per-img", type=int, default=1, help="Number of augmented sibling boards per train image")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cid, cx, cy, w, h))
    return boxes


def to_xyxy_px(box: Tuple[int, float, float, float, float], W: int, H: int) -> Tuple[float, float, float, float]:
    _, cx, cy, bw, bh = box
    x1 = (cx - bw * 0.5) * W
    y1 = (cy - bh * 0.5) * H
    x2 = (cx + bw * 0.5) * W
    y2 = (cy + bh * 0.5) * H
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-7)


def build_multiclass_bank(source_dir: Path, bank_dir: Path, margin: float = 0.30) -> Dict[int, List[dict]]:
    meta_path = bank_dir / "bank_metadata.json"
    if meta_path.exists():
        print(f"[Bank] Loading existing multi-class component bank from {meta_path}")
        raw = json.loads(meta_path.read_text())
        return {int(k): v for k, v in raw.items()}

    train_img_dir = source_dir / "train" / "images"
    train_lbl_dir = source_dir / "train" / "labels"
    bank_dir.mkdir(parents=True, exist_ok=True)

    bank_by_class: Dict[int, List[dict]] = {0: [], 1: [], 2: [], 3: []}
    max_per_class = {0: 3500, 1: 2500, 2: 2000, 3: 2000}

    img_files = sorted(
        [p for p in train_img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    )
    print(f"[Bank] Extracting multi-class component crops from {len(img_files)} train images...")

    for img_path in img_files:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(lbl_path)
        if not boxes:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        all_xyxy = [to_xyxy_px(b, W, H) for b in boxes]

        for idx, b in enumerate(boxes):
            cid, cx, cy, bw, bh = b
            if cid not in bank_by_class or len(bank_by_class[cid]) >= max_per_class[cid]:
                continue

            bx1, by1, bx2, by2 = all_xyxy[idx]
            pw, ph = bx2 - bx1, by2 - by1
            if pw < 8 or ph < 8:
                continue
            # Avoid overcrowded overlapping boxes when building clean bank templates
            other_ious = [iou_xyxy((bx1, by1, bx2, by2), all_xyxy[j]) for j in range(len(boxes)) if j != idx]
            if other_ious and max(other_ious) > 0.05:
                continue

            mx, my = pw * margin, ph * margin
            x1 = int(max(0, round(bx1 - mx)))
            y1 = int(max(0, round(by1 - my)))
            x2 = int(min(W, round(bx2 + mx)))
            y2 = int(min(H, round(by2 + my)))
            cw, ch = x2 - x1, y2 - y1
            if cw < 12 or ch < 12 or cw > W // 2 or ch > H // 2:
                continue

            rel_x1 = float(np.clip((bx1 - x1) / cw, 0.0, 1.0))
            rel_y1 = float(np.clip((by1 - y1) / ch, 0.0, 1.0))
            rel_x2 = float(np.clip((bx2 - x1) / cw, 0.0, 1.0))
            rel_y2 = float(np.clip((by2 - y1) / ch, 0.0, 1.0))

            cls_sub = bank_dir / f"class_{cid}"
            cls_sub.mkdir(parents=True, exist_ok=True)
            crop_name = f"{img_path.stem}_c{cid}_{idx}.png"
            crop_path = cls_sub / crop_name
            cv2.imwrite(str(crop_path), img[y1:y2, x1:x2])

            bank_by_class[cid].append({
                "path": str(crop_path),
                "rel_box": [rel_x1, rel_y1, rel_x2, rel_y2],
            })

    meta_path.write_text(json.dumps({str(k): v for k, v in bank_by_class.items()}, indent=2))
    for cid, items in bank_by_class.items():
        print(f"  Class {cid}: {len(items)} clean high-res templates banked.")
    return bank_by_class


def rotate_crop_and_relbox(
    crop: np.ndarray, rel_box: List[float], k_rot: int
) -> Tuple[np.ndarray, List[float]]:
    """Rotates crop by k_rot * 90 degrees counter-clockwise and updates relative box."""
    k = k_rot % 4
    if k == 0:
        return crop, list(rel_box)
    rx1, ry1, rx2, ry2 = rel_box
    if k == 1:  # 90 deg CCW: (x, y) -> (y, 1 - x)
        new_box = [ry1, 1.0 - rx2, ry2, 1.0 - rx1]
        return np.rot90(crop, 1).copy(), new_box
    if k == 2:  # 180 deg: (x, y) -> (1 - x, 1 - y)
        new_box = [1.0 - rx2, 1.0 - ry2, 1.0 - rx1, 1.0 - ry1]
        return np.rot90(crop, 2).copy(), new_box
    # k == 3: 270 deg CCW: (x, y) -> (1 - y, x)
    new_box = [1.0 - ry2, rx1, 1.0 - ry1, rx2]
    return np.rot90(crop, 3).copy(), new_box


def match_local_photometry(crop: np.ndarray, base_img: np.ndarray, x1: int, y1: int) -> np.ndarray:
    ch, cw = crop.shape[:2]
    region = base_img[y1 : y1 + ch, x1 : x1 + cw].astype(np.float32)
    crop_f = crop.astype(np.float32)
    shift = region.reshape(-1, 3).mean(axis=0) - crop_f.reshape(-1, 3).mean(axis=0)
    return np.clip(crop_f + 0.55 * shift, 0.0, 255.0).astype(np.uint8)


def cosine_feathered_paste(base_img: np.ndarray, crop: np.ndarray, x1: int, y1: int, feather_px: int = 6):
    ch, cw = crop.shape[:2]
    mask = np.ones((ch, cw), dtype=np.float32)
    f = min(feather_px, ch // 3, cw // 3)
    if f > 1:
        t = np.linspace(0.0, np.pi, f, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(t))  # smooth C1 cosine ramp from 0 to 1
        mask[:f, :] *= ramp[:, None]
        mask[-f:, :] *= ramp[::-1, None]
        mask[:, :f] *= ramp[None, :]
        mask[:, -f:] *= ramp[None, ::-1]
    mask3 = mask[:, :, None]
    reg = base_img[y1 : y1 + ch, x1 : x1 + cw].astype(np.float32)
    base_img[y1 : y1 + ch, x1 : x1 + cw] = (reg * (1.0 - mask3) + crop.astype(np.float32) * mask3).astype(np.uint8)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    out_yaml = args.dest / "data.yaml"
    if out_yaml.exists() and (args.dest / "train" / "images").exists():
        n_existing = len(list((args.dest / "train" / "images").glob("*")))
        if n_existing > 700:
            print(f"[CopyPaste] Dataset already prepared at {args.dest} ({n_existing} train images). Skipping.")
            return

    bank_by_class = build_multiclass_bank(args.source, args.bank, margin=args.margin)

    train_src_img = args.source / "train" / "images"
    train_src_lbl = args.source / "train" / "labels"
    train_dst_img = args.dest / "train" / "images"
    train_dst_lbl = args.dest / "train" / "labels"
    train_dst_img.mkdir(parents=True, exist_ok=True)
    train_dst_lbl.mkdir(parents=True, exist_ok=True)

    # Link or copy valid and test splits 100% untouched
    for split in ("valid", "test"):
        src_sp = (args.source / split).resolve()
        dst_sp = args.dest / split
        if dst_sp.exists() or dst_sp.is_symlink():
            if dst_sp.is_symlink() or dst_sp.is_file():
                dst_sp.unlink()
            else:
                shutil.rmtree(dst_sp)
        try:
            os.symlink(str(src_sp), str(dst_sp), target_is_directory=True)
        except Exception:
            shutil.copytree(str(src_sp), str(dst_sp))

    # Paste schedule per sibling board: (class_id, min_count, max_count, scale_range)
    paste_plan = [
        (1, 3, 6, (0.85, 1.15)),  # Connector (minority class boost!)
        (2, 2, 5, (0.85, 1.15)),  # Electrolytic Capacitor (minority class boost!)
        (3, 1, 3, (0.85, 1.15)),  # IC
        (0, 6, 12, (0.80, 1.25)), # Micro-Capacitor
    ]

    img_files = sorted(
        [p for p in train_src_img.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    )
    total_written = 0

    for img_path in img_files:
        lbl_path = train_src_lbl / f"{img_path.stem}.txt"
        # 1. Copy original board & label untouched
        dst_orig_img = train_dst_img / img_path.name
        if not dst_orig_img.exists():
            try:
                os.link(str(img_path), str(dst_orig_img))
            except Exception:
                shutil.copy2(str(img_path), str(dst_orig_img))
        if lbl_path.exists():
            shutil.copy2(str(lbl_path), str(train_dst_lbl / lbl_path.name))
        total_written += 1

        # 2. Generate Photometric-Matched Feathered Copy-Paste sibling(s)
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            continue
        H, W = orig_img.shape[:2]
        orig_boxes = load_yolo_labels(lbl_path)
        orig_lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in orig_boxes]

        for s_idx in range(args.siblings_per_img):
            aug_img = orig_img.copy()
            occ_xyxy = [to_xyxy_px(b, W, H) for b in orig_boxes]
            new_lines = list(orig_lines)

            for cid, min_n, max_n, (s_min, s_max) in paste_plan:
                templates = bank_by_class.get(cid, [])
                if not templates:
                    continue
                n_to_paste = rng.randint(min_n, max_n)
                for _ in range(n_to_paste):
                    item = rng.choice(templates)
                    crop = cv2.imread(item["path"])
                    if crop is None:
                        continue
                    k_rot = rng.choice([0, 1, 2, 3])
                    crop, rel_box = rotate_crop_and_relbox(crop, item["rel_box"], k_rot)
                    sc = rng.uniform(s_min, s_max)
                    ch, cw = max(8, int(round(crop.shape[0] * sc))), max(8, int(round(crop.shape[1] * sc)))
                    if ch >= H // 2 or cw >= W // 2:
                        continue
                    crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)

                    for _attempt in range(18):
                        x1 = rng.randint(4, max(5, W - cw - 4))
                        y1 = rng.randint(4, max(5, H - ch - 4))
                        cand_xyxy = (float(x1), float(y1), float(x1 + cw), float(y1 + ch))
                        if all(iou_xyxy(cand_xyxy, existing) <= args.max_iou for existing in occ_xyxy):
                            crop_matched = match_local_photometry(crop, aug_img, x1, y1)
                            cosine_feathered_paste(aug_img, crop_matched, x1, y1, feather_px=args.feather_px)
                            occ_xyxy.append(cand_xyxy)

                            rx1, ry1, rx2, ry2 = rel_box
                            bx1 = x1 + rx1 * cw
                            by1 = y1 + ry1 * ch
                            bx2 = x1 + rx2 * cw
                            by2 = y1 + ry2 * ch
                            bw_px = max(2.0, bx2 - bx1)
                            bh_px = max(2.0, by2 - by1)
                            bcx = (bx1 + bx2) * 0.5 / W
                            bcy = (by1 + by2) * 0.5 / H
                            bnw = bw_px / W
                            bnh = bh_px / H
                            new_lines.append(f"{cid} {bcx:.6f} {bcy:.6f} {bnw:.6f} {bnh:.6f}")
                            break

            sib_stem = f"{img_path.stem}_apexcp{s_idx}"
            cv2.imwrite(str(train_dst_img / f"{sib_stem}{img_path.suffix}"), aug_img)
            (train_dst_lbl / f"{sib_stem}.txt").write_text("\n".join(new_lines) + "\n")
            total_written += 1

    # Write data.yaml
    src_cfg = yaml.safe_load((args.source / "data.yaml").read_text())
    dst_cfg = {
        "path": str(args.dest.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": src_cfg.get("nc", 4),
        "names": src_cfg.get("names", ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]),
    }
    out_yaml.write_text(yaml.safe_dump(dst_cfg, sort_keys=False))
    print(f"[CopyPaste] Built {total_written} training boards at {args.dest} (val/test untouched).")


if __name__ == "__main__":
    main()
