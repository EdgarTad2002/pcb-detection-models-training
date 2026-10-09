#!/usr/bin/env python3
"""
Generate publication-grade visual diffs between the Original Dataset
and the Cleaned Dataset for key PCB boards.
Produces side-by-side comparisons and zoomed-in crops of missing components.
"""

import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Class definitions & colors
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
CLASS_COLORS = {
    "Capacitor": "#2ca02c",            # Green
    "Connector": "#e377c2",            # Pink
    "Electrolytic Capacitor": "#ff7f0e", # Orange
    "IC": "#1f77b4",                   # Blue
}
INJECTED_COLOR = "#d62728"             # Vibrant Red / Crimson
INJECTED_IC_COLOR = "#9467bd"          # Purple

def load_yolo_boxes(txt_path, img_w, img_h):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"Class_{cid}"
                boxes.append({
                    "cid": cid,
                    "class": cname,
                    "coords": (x1, y1, x2, y2),
                    "norm": (xc, yc, w, h)
                })
    return boxes

def plot_board_diff(img_name, out_path, zoom_region=None):
    root = Path(__file__).resolve().parent.parent
    img_dir = root / "data_samples" / "native_images"
    orig_lbl_dir = root / "data_samples" / "native_labels"
    clean_lbl_dir = root / "data_samples" / "native_cleaned" / "labels"
    changelog_path = root / "data_samples" / "native_cleaned" / "rectification_changelog.json"
    
    img_path = img_dir / img_name
    orig_lbl = orig_lbl_dir / (Path(img_name).stem + ".txt")
    clean_lbl = clean_lbl_dir / (Path(img_name).stem + ".txt")
    
    if not img_path.exists():
        print(f"Error: {img_path} not found")
        return
        
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    orig_boxes = load_yolo_boxes(orig_lbl, w, h)
    clean_boxes = load_yolo_boxes(clean_lbl, w, h)
    
    # Load changelog additions for this specific image
    with open(changelog_path) as f:
        changelog = json.load(f)["changelog"]
    injected_items = [c for c in changelog if c["image"] == img_name]
    
    print(f"[{img_name}] Original: {len(orig_boxes)}, Cleaned: {len(clean_boxes)}, Injected diff: {len(injected_items)}")
    
    # Setup Figure (Side-by-Side Dual View + Zoom Crop)
    fig = plt.figure(figsize=(24, 12), dpi=250)
    
    # Main Panel 1: Original Dataset
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title(f"ORIGINAL UNCLEANED DATASET\nTotal Ground Truth: {len(orig_boxes)} boxes", 
                  fontsize=14, fontweight="bold", color="#1a1a1a", pad=12)
    ax1.axis("off")
    
    for b in orig_boxes:
        x1, y1, x2, y2 = b["coords"]
        col = CLASS_COLORS.get(b["class"], "#333333")
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=1.2, edgecolor=col, facecolor="none")
        ax1.add_patch(rect)
        
    # Main Panel 2: Cleaned Dataset (with Injected Highlighted)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img_rgb)
    ax2.set_title(f"CLEANED RECTIFIED DATASET\nTotal Ground Truth: {len(clean_boxes)} boxes  (★ +{len(injected_items)} Missing Recovered)", 
                  fontsize=14, fontweight="bold", color="#d62728", pad=12)
    ax2.axis("off")
    
    # Draw original boxes with dashed/subtle green
    for b in orig_boxes:
        x1, y1, x2, y2 = b["coords"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=1.0, edgecolor="#2ca02c", facecolor="none", alpha=0.6)
        ax2.add_patch(rect)
        
    # Draw injected boxes with thick bright crimson / cyan and a star indicator
    for item in injected_items:
        xc, yc, bw, bh = item["box"]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        col = INJECTED_COLOR if item["class"] == "Capacitor" else INJECTED_IC_COLOR
        rect = patches.Rectangle((x1, y1), bw_px, bh_px, 
                                 linewidth=2.2, edgecolor=col, facecolor=col, alpha=0.25)
        ax2.add_patch(rect)
        rect_border = patches.Rectangle((x1, y1), bw_px, bh_px, 
                                       linewidth=2.2, edgecolor=col, facecolor="none")
        ax2.add_patch(rect_border)
        
    # Add legend to ax2
    legend_elements = [
        patches.Patch(facecolor="none", edgecolor="#2ca02c", linewidth=1.5, label=f"Original GT ({len(orig_boxes)})"),
        patches.Patch(facecolor="#d62728", edgecolor="#d62728", alpha=0.5, linewidth=2, label=f"Injected Missing Capacitors (+{sum(1 for x in injected_items if x['class']=='Capacitor')})"),
    ]
    if any(x["class"] == "IC" for x in injected_items):
        legend_elements.append(
            patches.Patch(facecolor="#9467bd", edgecolor="#9467bd", alpha=0.5, linewidth=2, label=f"Injected Missing ICs (+{sum(1 for x in injected_items if x['class']=='IC')})")
        )
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=11, frameon=True, shadow=True)
    
    plt.suptitle(f"Dataset Comparison: {img_name}\nHuman Labeling Omissions Rectified via Matched-Filter Verification", 
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    root = Path(__file__).resolve().parent.parent
    res_dir = root / "results" / "dataset_comparisons"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. pcb123rec1 (+54 additions)
    plot_board_diff(
        "pcb123rec1_jpg.rf.2cc75b8b236f73d9bdfe05e02d5b0340.jpg",
        res_dir / "diff_pcb123rec1_54added.png"
    )
    
    # 2. pcb12rec1 (+43 additions)
    plot_board_diff(
        "pcb12rec1_jpg.rf.ded545184f6c4b53af85d7fd1f965922.jpg",
        res_dir / "diff_pcb12rec1_43added.png"
    )
    
    # 3. pcb135rec1 (+49 additions)
    plot_board_diff(
        "pcb135rec1_jpg.rf.330a68639859330504dff5db96faa59d.jpg",
        res_dir / "diff_pcb135rec1_49added.png"
    )

if __name__ == "__main__":
    main()
