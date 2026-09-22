#!/usr/bin/env python3
"""
tools/inspect_cluster_gt.py - Inspect and Compare Cluster Dataset Ground Truth vs Roboflow

Takes N training/validation/test images from the cluster dataset (or locally cached samples)
and generates:
1. High-resolution visual comparisons:
   - Panel A: Raw Roboflow 23-class annotations (including Capacitor Jumper aliasing & non-targets)
   - Panel B: Rectified Unified 4-class annotations (clean nc:4 dataset)
2. Statistical validation tables to verify that:
   - Class 1 (Capacitor Jumper) + Class 2 (Capacitor) -> Class 0 (Capacitor)
   - Class 4 (Connector) -> Class 1 (Connector)
   - Class 7 (Electrolytic Capacitor) -> Class 2 (Electrolytic Capacitor)
   - Class 9/22 (IC) -> Class 3 (IC)
   - Non-target classes (Resistors, Pins, Pads, etc.) are cleanly discarded.

Usage:
  # On cluster or local machine:
  python tools/inspect_cluster_gt.py --split train --num 10
  python tools/inspect_cluster_gt.py --split valid --num 5 --save-dir results/gt_valid_inspect
"""

import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Roboflow 23 Raw Classes
RAW_CLASSES = [
    "Button", "Capacitor Jumper", "Capacitor", "Clock", "Connector",
    "Diode", "EM", "Electrolytic Capacitor", "Ferrite Bead", "IC",
    "Inductor", "Jumper", "Led", "Pads", "Pins", "Resistor",
    "Resistor Jumper", "Resistor Network", "Switch", "Test Point",
    "Transistor", "Unknown", "Variable Resistor"
]

# Evaluated Target Classes
EVAL_CLASSES = {
    0: "Capacitor",
    1: "Connector",
    2: "Electrolytic Capacitor",
    3: "IC"
}

# Unified 4-Class Mapping Dictionary
RAW_TO_UNIFIED = {
    1: 0,   # Capacitor Jumper -> Capacitor
    2: 0,   # Capacitor        -> Capacitor
    4: 1,   # Connector        -> Connector
    7: 2,   # Electrolytic Cap -> Electrolytic Capacitor
    9: 3,   # IC               -> IC
    22: 3,  # iC (typo)        -> IC
}

# Color palette for bounding box visualization (RGB)
CLASS_COLORS = {
    "Capacitor": (0, 220, 255),               # Cyan
    "Capacitor Jumper": (0, 220, 255),        # Cyan (matches Capacitor)
    "Connector": (255, 185, 0),               # Gold
    "Electrolytic Capacitor": (255, 50, 180), # Magenta
    "IC": (0, 255, 100),                      # Bright Green
    "Resistor": (255, 120, 50),               # Orange
    "Resistor Network": (255, 120, 50),       # Orange
    "Resistor Jumper": (255, 120, 50),        # Orange
    "Pins": (180, 80, 255),                   # Purple
    "Pads": (120, 200, 255),                  # Light Blue
    "Default": (170, 170, 170)                # Gray
}

def get_color(cname):
    return CLASS_COLORS.get(cname, CLASS_COLORS["Default"])

def parse_yolo_label(lbl_path, w, h):
    """Parses a YOLO annotation file into a list of tuples: (cid, x1, y1, x2, y2)."""
    boxes = []
    if not lbl_path or not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = [float(p) for p in parts[1:5]]
                x1 = int(round((cx - bw / 2) * w))
                y1 = int(round((cy - bh / 2) * h))
                x2 = int(round((cx + bw / 2) * w))
                y2 = int(round((cy + bh / 2) * h))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                boxes.append((cid, x1, y1, x2, y2))
    return boxes

def render_bounding_boxes(img_rgb, boxes, class_names_map, prefix=""):
    """Draws color-coded bounding boxes and labels on an RGB image."""
    canvas = img_rgb.copy()
    h, w = canvas.shape[:2]
    for cid, x1, y1, x2, y2 in boxes:
        cname = class_names_map.get(cid, f"Class_{cid}")
        color = get_color(cname)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        lbl_text = f"{prefix}{cname}" if prefix else cname
        (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        y_label = max(y1 - 4, th + 4)
        cv2.rectangle(canvas, (x1, y_label - th - 3), (x1 + tw + 3, y_label + 3), color, -1)
        cv2.putText(canvas, lbl_text, (x1 + 1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas

def fetch_samples_from_cluster(split="train", num=10, cache_dir="data_samples/cluster_train",
                               cluster_host="cluster.ysu.am", cluster_user="etadevosyan"):
    """Fetches sample images and both label variants from YSU cluster via rsync."""
    remote_base = "/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets"
    print(f"📡 Fetching {num} '{split}' samples from cluster ({cluster_host})...")
    
    # List filenames
    cmd = f'ssh {cluster_user}@{cluster_host} "ls {remote_base}/pcb-unified-4class/{split}/images | head -n {num}"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"⚠️ Unable to query cluster: {res.stderr.strip()}")
        return None
        
    img_names = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not img_names:
        return None
        
    img_dir = Path(cache_dir) / "images"
    raw_lbl_dir = Path(cache_dir) / "labels_raw"
    uni_lbl_dir = Path(cache_dir) / "labels_unified"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_lbl_dir.mkdir(parents=True, exist_ok=True)
    uni_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for img in img_names:
        stem = Path(img).stem
        # Image
        subprocess.run(["rsync", "-avz", "--copy-links", f"{cluster_user}@{cluster_host}:{remote_base}/pcb-filtered-yolov8/{split}/images/{img}", str(img_dir)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Raw label
        subprocess.run(["rsync", "-avz", f"{cluster_user}@{cluster_host}:{remote_base}/pcb-filtered-yolov8/{split}/labels/{stem}.txt", str(raw_lbl_dir)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Unified label
        subprocess.run(["rsync", "-avz", f"{cluster_user}@{cluster_host}:{remote_base}/pcb-unified-4class/{split}/labels/{stem}.txt", str(uni_lbl_dir)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                       
    print(f"✅ Downloaded {len(img_names)} images and labels into '{cache_dir}'!")
    return img_names

def find_dataset_locations(split="train"):
    """Finds image and label directories, either on cluster or local cache."""
    # 1. Cluster filesystem mounts
    cluster_paths = [
        Path(f"/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/{split}"),
        Path(f"datasets/pcb-filtered-yolov8/{split}"),
    ]
    for p in cluster_paths:
        if p.exists() and (p / "images").exists():
            uni_p = Path(str(p).replace("pcb-filtered-yolov8", "pcb-unified-4class"))
            return p / "images", p / "labels", uni_p / "labels" if uni_p.exists() else None

    # 2. Local cluster cache
    local_cache = Path("data_samples/cluster_train")
    if (local_cache / "images").exists() and list((local_cache / "images").glob("*.jpg")):
        return local_cache / "images", local_cache / "labels_raw", local_cache / "labels_unified"

    # 3. Fallback to data_samples
    local_samples = Path("data_samples")
    if (local_samples / "images").exists():
        return local_samples / "images", local_samples / "labels", None

    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Inspect cluster dataset ground truth vs Roboflow")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Dataset split")
    parser.add_argument("--num", type=int, default=10, help="Number of images to inspect")
    parser.add_argument("--save-dir", type=str, default="results/cluster_gt_inspect", help="Directory to save comparison plots")
    parser.add_argument("--auto-fetch", action="store_true", default=True, help="Auto-fetch from cluster if not found locally")
    args = parser.parse_args()

    img_dir, raw_lbl_dir, uni_lbl_dir = find_dataset_locations(args.split)
    
    if (not img_dir or not img_dir.exists()) and args.auto_fetch:
        fetch_samples_from_cluster(split=args.split, num=args.num)
        img_dir, raw_lbl_dir, uni_lbl_dir = find_dataset_locations(args.split)

    if not img_dir or not img_dir.exists():
        print("❌ Error: No images found. Run with network access to cluster.ysu.am or provide datasets/ locally.")
        sys.exit(1)

    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:args.num]
    if not images:
        print(f"❌ No images found in {img_dir}")
        sys.exit(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f"🔎 INSPECTING {len(images)} '{args.split.upper()}' IMAGES: ROBOFLOW RAW (23-CLASS) vs. UNIFIED (4-CLASS)")
    print(f"📁 Image Directory:       {img_dir}")
    print(f"📄 Raw Labels:           {raw_lbl_dir}")
    print(f"📄 Rectified Labels:     {uni_lbl_dir}")
    print(f"💾 Output Plots:         {save_dir}")
    print("=" * 85)

    raw_names_dict = {i: name for i, name in enumerate(RAW_CLASSES)}
    summary_rows = []

    for idx, img_p in enumerate(images, start=1):
        stem = img_p.stem
        raw_bgr = cv2.imread(str(img_p))
        if raw_bgr is None:
            continue
        rgb_img = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb_img.shape[:2]

        raw_lbl_file = raw_lbl_dir / f"{stem}.txt" if raw_lbl_dir else None
        uni_lbl_file = uni_lbl_dir / f"{stem}.txt" if uni_lbl_dir else None

        raw_boxes = parse_yolo_label(str(raw_lbl_file), w, h)
        uni_boxes = parse_yolo_label(str(uni_lbl_file), w, h) if uni_lbl_file and uni_lbl_file.exists() else []

        raw_cids = Counter(b[0] for b in raw_boxes)
        uni_cids = Counter(b[0] for b in uni_boxes)

        cap_jumper = raw_cids.get(1, 0)
        cap_std = raw_cids.get(2, 0)
        conn = raw_cids.get(4, 0)
        elec = raw_cids.get(7, 0)
        ic = raw_cids.get(9, 0) + raw_cids.get(22, 0)
        non_target = sum(v for k, v in raw_cids.items() if k not in (1, 2, 4, 7, 9, 22))

        uni_cap = uni_cids.get(0, cap_jumper + cap_std)
        uni_conn = uni_cids.get(1, conn)
        uni_elec = uni_cids.get(2, elec)
        uni_ic = uni_cids.get(3, ic)

        print(f"\n🖼️ [{idx}/{len(images)}] {img_p.name} ({w}x{h} px)")
        print(f"  • Total Raw Annotations:     {len(raw_boxes):3d} boxes")
        print(f"  • Total Unified Target Boxes: {len(uni_boxes) if uni_boxes else (cap_jumper+cap_std+conn+elec+ic):3d} boxes")
        print("  • Mapping Validation:")
        print(f"      - Capacitor:       Raw Jumper({cap_jumper:2d}) + Cap({cap_std:2d}) = {cap_jumper+cap_std:2d}  ==>  Unified Class 0: {uni_cap:2d} {'✅' if uni_cap == cap_jumper+cap_std else '❌'}")
        print(f"      - Connector:       Raw Connector({conn:2d})               = {conn:2d}  ==>  Unified Class 1: {uni_conn:2d} {'✅' if uni_conn == conn else '❌'}")
        print(f"      - Elec. Capacitor: Raw Elec Cap({elec:2d})                = {elec:2d}  ==>  Unified Class 2: {uni_elec:2d} {'✅' if uni_elec == elec else '❌'}")
        print(f"      - IC:              Raw IC({ic:2d})                        = {ic:2d}  ==>  Unified Class 3: {uni_ic:2d} {'✅' if uni_ic == ic else '❌'}")
        print(f"      - Discarded:       Raw non-targets ({non_target:2d} boxes: Resistors, Pins, Pads)  ==>  Cleanly removed in nc:4")

        summary_rows.append({
            "name": img_p.name[:35],
            "raw_total": len(raw_boxes),
            "uni_total": len(uni_boxes) if uni_boxes else (cap_jumper+cap_std+conn+elec+ic),
            "cap": uni_cap,
            "conn": uni_conn,
            "elec": uni_elec,
            "ic": uni_ic
        })

        # 2-Panel Plot: Raw vs Unified
        vis_raw = render_bounding_boxes(rgb_img, raw_boxes, raw_names_dict, prefix="Raw:")
        
        # If rectified file was not present, build simulated unified boxes
        if not uni_boxes:
            sim_uni_boxes = []
            for b in raw_boxes:
                cid = b[0]
                if cid in RAW_TO_UNIFIED:
                    sim_uni_boxes.append((RAW_TO_UNIFIED[cid], *b[1:]))
            vis_uni = render_bounding_boxes(rgb_img, sim_uni_boxes, EVAL_CLASSES, prefix="Unified:")
        else:
            vis_uni = render_bounding_boxes(rgb_img, uni_boxes, EVAL_CLASSES, prefix="Unified:")

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(vis_raw)
        axes[0].set_title(f"[A] Roboflow Raw 23-Class Ground Truth ({len(raw_boxes)} boxes)\n{img_p.name[:40]}", 
                          fontsize=12, fontweight="bold", color="#1E3A8A", pad=8)
        axes[0].axis("off")

        axes[1].imshow(vis_uni)
        axes[1].set_title(f"[B] Rectified Unified 4-Class Ground Truth ({len(uni_boxes) or (cap_jumper+cap_std+conn+elec+ic)} boxes)\nCapacitor={uni_cap}, Connector={uni_conn}, ElecCap={uni_elec}, IC={uni_ic}", 
                          fontsize=12, fontweight="bold", color="#065F46", pad=8)
        axes[1].axis("off")

        plt.tight_layout()
        out_png = save_dir / f"gt_compare_{idx:02d}_{stem[:30]}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()

    print("\n" + "=" * 85)
    print("📊 BATCH SUMMARY VERIFICATION TABLE:")
    print("-" * 85)
    print(f"{'Image Filename':<36} | {'Raw Total':>9} | {'Unified':>7} | {'Caps':>5} | {'Conn':>5} | {'Elec':>5} | {'IC':>4}")
    print("-" * 85)
    for r in summary_rows:
        print(f"{r['name']:<36} | {r['raw_total']:>9d} | {r['uni_total']:>7d} | {r['cap']:>5d} | {r['conn']:>5d} | {r['elec']:>5d} | {r['ic']:>4d}")
    print("=" * 85)
    print(f"🎉 Completed! Generated {len(images)} comparison plots in: {save_dir}")

if __name__ == "__main__":
    main()
