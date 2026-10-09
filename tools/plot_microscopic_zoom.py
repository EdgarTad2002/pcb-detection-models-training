#!/usr/bin/env python3
"""
Generate microscopic zoom-in figure showing the physical surface of missing components
"""

import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def make_zoom_figure():
    root = Path(__file__).resolve().parent.parent
    img_dir = root / "data_samples" / "native_images"
    changelog_path = root / "data_samples" / "native_cleaned" / "rectification_changelog.json"
    out_path = root / "results" / "dataset_comparisons" / "microscopic_zoom_crops.png"
    
    with open(changelog_path) as f:
        changelog = json.load(f)["changelog"]
        
    # Select 6 diverse high-conviction missing components across different boards
    picks = [
        # (img_name, candidate_idx_in_board, label_text)
        ("pcb123rec1_jpg.rf.2cc75b8b236f73d9bdfe05e02d5b0340.jpg", 0, "Missing Decoupling Capacitor (pcb123)"),
        ("pcb123rec1_jpg.rf.2cc75b8b236f73d9bdfe05e02d5b0340.jpg", 1, "Missing SMD Capacitor (pcb123)"),
        ("pcb12rec1_jpg.rf.ded545184f6c4b53af85d7fd1f965922.jpg", 0, "Missing Bus Filter Capacitor (pcb12)"),
        ("pcb12rec1_jpg.rf.ded545184f6c4b53af85d7fd1f965922.jpg", 5, "Missing Rail Capacitor (pcb12)"),
        ("pcb135rec1_jpg.rf.330a68639859330504dff5db96faa59d.jpg", 0, "Missing Small SOIC/QFN IC (pcb135)"),
        ("pcb135rec1_jpg.rf.330a68639859330504dff5db96faa59d.jpg", 2, "Missing Bypass Capacitor (pcb135)"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=250)
    axes = axes.flatten()
    
    for ax_idx, (img_name, cand_idx, title) in enumerate(picks):
        img_p = img_dir / img_name
        b_items = [c for c in changelog if c["image"] == img_name]
        item = b_items[cand_idx]
        
        img_bgr = cv2.imread(str(img_p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        xc, yc, bw, bh = item["box"]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        
        pad_x = max(25, int(bw * w * 1.5))
        pad_y = max(25, int(bh * h * 1.5))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        
        crop = img_rgb[cy1:cy2, cx1:cx2]
        
        ax = axes[ax_idx]
        ax.imshow(crop)
        
        rx1 = x1 - cx1
        ry1 = y1 - cy1
        rw = x2 - x1
        rh = y2 - y1
        
        col = "#d62728" if item["class"] == "Capacitor" else "#9467bd"
        rect = patches.Rectangle((rx1, ry1), rw, rh, 
                                 linewidth=3.0, edgecolor=col, facecolor=col, alpha=0.2)
        ax.add_patch(rect)
        rect_border = patches.Rectangle((rx1, ry1), rw, rh, 
                                       linewidth=3.0, edgecolor=col, facecolor="none")
        ax.add_patch(rect_border)
        
        ax.set_title(
            f"{title}\nClass: {item['class']} | Detector Conf: {item.get('confidence', 0):.2f} | Filter Corr: {item.get('correlation', 0):.2f}",
            fontsize=11, fontweight="bold", pad=8
        )
        ax.axis("off")
        
    plt.suptitle("Microscopic Evidence: Physical Components Omitted in Original Dataset, Recovered in Cleaned Dataset", 
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    make_zoom_figure()
