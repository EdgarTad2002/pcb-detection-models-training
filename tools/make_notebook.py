#!/usr/bin/env python3
"""
tools/make_notebook.py
Generates enhanced compare_original_vs_cleaned_dataset.ipynb
with interactive confidence threshold filtering.
"""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Cell 0: Title and Introduction
    add_md("""# 🔎 PCB Dataset Inspector: Original vs. Cleaned Dataset
### Visualizing Human Annotation Omissions & High-Conviction Component Recovery

This interactive notebook allows you to visually compare images between:
1. **The Original Dataset (`datasets/pcb-unified-4class`)**: The raw dataset containing severe missing annotations (omissions by human labelers).
2. **The Cleaned Dataset (`datasets/pcb-unified-4class-cleaned`)**: The rectified dataset where missing physical components were verified via Normalized Cross-Correlation Matched-Filtering.

---

### Key Capabilities:
- 🖼️ **Side-by-Side Dual View**: Compares Original Ground Truth vs. Cleaned Ground Truth.
- 🟥 **Visual Highlighting**: Original annotations are outlined in subtle green lines; **newly injected annotations** are highlighted in bold crimson/purple boxes with tags.
- 🎚️ **Interactive Confidence Filtering Slider**: Filter out low-confidence candidate detections (e.g. `conf < 0.50`) in real time to prevent pseudo-label noise!
- 🔬 **Microscopic Zoom Inspector**: Zooms in directly on the physical PCB surface to inspect individual component packages, solder fillets, and IC pins.
- 🎛️ **Interactive Dropdown**: Select any modified PCB from the dataset to inspect.""")

    # Cell 1: Setup & Imports
    add_code("""import os
import json
from pathlib import Path
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Try importing ipywidgets for interactive selector
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False
    print("ℹ️ ipywidgets not found; static plotting functions remain fully functional.")

# High-contrast color palette
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
CLASS_COLORS = {
    "Capacitor": "#2ca02c",             # Green
    "Connector": "#e377c2",             # Pink
    "Electrolytic Capacitor": "#ff7f0e", # Orange
    "IC": "#1f77b4",                    # Blue
}
INJECTED_CAP_COLOR = "#d62728"          # Bold Crimson
INJECTED_IC_COLOR = "#9467bd"           # Purple
INJECTED_CONN_COLOR = "#ff7f0e"         # Amber/Orange

# Workspace directories
ROOT_DIR = Path.cwd()
IMG_DIR = ROOT_DIR / "data_samples" / "native_images"
ORIG_LBL_DIR = ROOT_DIR / "data_samples" / "native_labels"
CLEAN_LBL_DIR = ROOT_DIR / "data_samples" / "native_cleaned" / "labels"
CHANGELOG_PATH = ROOT_DIR / "data_samples" / "native_cleaned" / "rectification_changelog.json"

print("✅ Environment initialized.")
print(f"📁 Image Directory: {IMG_DIR} (Exists: {IMG_DIR.exists()})")
print(f"📁 Original Labels: {ORIG_LBL_DIR} (Exists: {ORIG_LBL_DIR.exists()})")
print(f"📁 Cleaned Labels:  {CLEAN_LBL_DIR} (Exists: {CLEAN_LBL_DIR.exists()})")""")

    # Cell 2: Index Modified Boards
    add_code("""# Load changelog and index all boards with missing annotations
with open(CHANGELOG_PATH) as f:
    changelog_data = json.load(f)

changelog = changelog_data.get("changelog", [])
print(f"📊 Total verified missing annotations in local samples: {len(changelog)}")

# Group by image
board_additions = Counter(item["image"] for item in changelog)
sorted_boards = board_additions.most_common()

print(f"🎯 Total boards with modified annotations: {len(sorted_boards)}")
print("\\nTop 10 Boards with Most Missing Components Recovered:")
print("-" * 75)
for idx, (b_name, count) in enumerate(sorted_boards[:10], 1):
    items = [x for x in changelog if x["image"] == b_name]
    caps = sum(1 for x in items if x["class"] == "Capacitor")
    ics = sum(1 for x in items if x["class"] == "IC")
    conns = sum(1 for x in items if x["class"] == "Connector")
    print(f"{idx:2d}. {b_name} -> +{count:2d} Total (+{caps:2d} Caps, +{ics:2d} ICs, +{conns:2d} Connectors)")""")

    # Cell 3: Helper Functions
    add_code("""def load_yolo_boxes(txt_path, img_w, img_h):
    \"\"\"Loads normalized YOLO boxes and converts to pixel (x1, y1, x2, y2).\"\"\"
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

def compare_original_vs_cleaned(img_name, min_conf=0.50, min_corr=0.0, allowed_classes=None, figsize=(20, 10), save_path=None):
    \"\"\"
    Renders a side-by-side comparison of the Original Dataset vs Cleaned Dataset.
    Left: Original Ground Truth
    Right: Cleaned Ground Truth with Injected Annotations filtered by min_conf!
    \"\"\"
    img_path = IMG_DIR / img_name
    orig_lbl = ORIG_LBL_DIR / (Path(img_name).stem + ".txt")
    
    if not img_path.exists():
        print(f"❌ Error: Image not found at {img_path}")
        return
        
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    orig_boxes = load_yolo_boxes(orig_lbl, w, h)
    
    # Filter injected items by user-defined confidence and correlation threshold
    all_injected = [c for c in changelog if c["image"] == img_name]
    filtered_injected = [
        c for c in all_injected
        if c.get("confidence", 0.0) >= min_conf
        and c.get("correlation", 0.0) >= min_corr
        and (allowed_classes is None or c.get("class") in allowed_classes)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=200)
    
    # --- PANEL 1: ORIGINAL UNCLEANED ---
    ax1 = axes[0]
    ax1.imshow(img_rgb)
    ax1.set_title(
        f"ORIGINAL UNCLEANED DATASET\\nTotal Ground Truth: {len(orig_boxes)} boxes",
        fontsize=13, fontweight="bold", pad=10
    )
    ax1.axis("off")
    for b in orig_boxes:
        x1, y1, x2, y2 = b["coords"]
        col = CLASS_COLORS.get(b["class"], "#2ca02c")
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=1.2, edgecolor=col, facecolor="none")
        ax1.add_patch(rect)
        
    # --- PANEL 2: CLEANED WITH INJECTIONS ---
    ax2 = axes[1]
    ax2.imshow(img_rgb)
    total_effective = len(orig_boxes) + len(filtered_injected)
    ax2.set_title(
        f"CLEANED DATASET (Confidence Threshold >= {min_conf:.2f})\\nTotal: {total_effective} boxes  (★ +{len(filtered_injected)} Filtered Additions)",
        fontsize=13, fontweight="bold", color="#d62728", pad=10
    )
    ax2.axis("off")
    
    # Original boxes in subtle green
    for b in orig_boxes:
        x1, y1, x2, y2 = b["coords"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=1.0, edgecolor="#2ca02c", facecolor="none", alpha=0.5)
        ax2.add_patch(rect)
        
    # Injected boxes in thick crimson/purple/amber
    for item in filtered_injected:
        xc, yc, bw, bh = item["box"]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        
        cname = item["class"]
        if cname == "Capacitor":
            col = INJECTED_CAP_COLOR
        elif cname == "IC":
            col = INJECTED_IC_COLOR
        else:
            col = INJECTED_CONN_COLOR
        
        rect_fill = patches.Rectangle((x1, y1), bw_px, bh_px, 
                                      linewidth=0, facecolor=col, alpha=0.3)
        rect_border = patches.Rectangle((x1, y1), bw_px, bh_px, 
                                        linewidth=2.2, edgecolor=col, facecolor="none")
        ax2.add_patch(rect_fill)
        ax2.add_patch(rect_border)
        
    # Legend
    n_caps = sum(1 for x in filtered_injected if x['class'] == 'Capacitor')
    n_ics = sum(1 for x in filtered_injected if x['class'] == 'IC')
    n_conns = sum(1 for x in filtered_injected if x['class'] == 'Connector')
    
    legend_handles = [
        patches.Patch(facecolor="none", edgecolor="#2ca02c", linewidth=1.5, label=f"Original GT ({len(orig_boxes)})"),
    ]
    if n_caps > 0:
        legend_handles.append(patches.Patch(facecolor=INJECTED_CAP_COLOR, edgecolor=INJECTED_CAP_COLOR, alpha=0.5, linewidth=1.5, label=f"Injected Capacitors (+{n_caps})"))
    if n_ics > 0:
        legend_handles.append(patches.Patch(facecolor=INJECTED_IC_COLOR, edgecolor=INJECTED_IC_COLOR, alpha=0.5, linewidth=1.5, label=f"Injected ICs (+{n_ics})"))
    if n_conns > 0:
        legend_handles.append(patches.Patch(facecolor=INJECTED_CONN_COLOR, edgecolor=INJECTED_CONN_COLOR, alpha=0.5, linewidth=1.5, label=f"Injected Connectors (+{n_conns})"))
        
    ax2.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True, shadow=True)
    
    plt.suptitle(f"Board: {img_name}\\nFilter: Min Confidence = {min_conf:.2f} (Excluded {len(all_injected) - len(filtered_injected)} low-conf candidates)", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.show()""")

    # Cell 4: Zoom Inspector Function with Confidence Threshold
    add_code("""def inspect_injected_crops(img_name, min_conf=0.50, max_crops=6, crop_padding=30):
    \"\"\"
    Displays close-up zoom crops of newly injected components matching min_conf.
    \"\"\"
    img_path = IMG_DIR / img_name
    all_items = [c for c in changelog if c["image"] == img_name]
    injected_items = [c for c in all_items if c.get("confidence", 0.0) >= min_conf]
    
    if not injected_items:
        print(f"No injected annotations for this image with confidence >= {min_conf:.2f} (originally {len(all_items)} lower-confidence candidates).")
        return
        
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    n_crops = min(max_crops, len(injected_items))
    fig, axes = plt.subplots(1, n_crops, figsize=(4 * n_crops, 4), dpi=150)
    if n_crops == 1:
        axes = [axes]
        
    for i, item in enumerate(injected_items[:n_crops]):
        xc, yc, bw, bh = item["box"]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        
        # Padded crop coordinates
        cx1 = max(0, x1 - crop_padding)
        cy1 = max(0, y1 - crop_padding)
        cx2 = min(w, x2 + crop_padding)
        cy2 = min(h, y2 + crop_padding)
        
        crop = img_rgb[cy1:cy2, cx1:cx2].copy()
        
        rx1 = x1 - cx1
        ry1 = y1 - cy1
        rx2 = x2 - cx1
        ry2 = y2 - cy1
        
        ax = axes[i]
        ax.imshow(crop)
        
        cname = item["class"]
        col = INJECTED_CAP_COLOR if cname == "Capacitor" else (INJECTED_IC_COLOR if cname == "IC" else INJECTED_CONN_COLOR)
        rect = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, 
                                 linewidth=2.5, edgecolor=col, facecolor="none")
        ax.add_patch(rect)
        
        conf = item.get("confidence", 0.0)
        corr = item.get("correlation", 0.0)
        ax.set_title(f"#{i+1} {cname}\\nConf: {conf:.2f} | Corr: {corr:.2f}", fontsize=10, fontweight="bold")
        ax.axis("off")
        
    plt.suptitle(f"Microscopic Close-Ups of High-Conviction (Conf >= {min_conf:.2f}) Injected Components on {img_name}", fontsize=12, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.show()""")

    # Cell 5: Showcase Arty_Top (User's Example!)
    add_md("""### Showcase: `Arty_Top` (Inspection of Low-Conf vs High-Conf Additions)
Notice how setting `min_conf = 0.50` cleanly **eliminates the spurious connector false alarm (`conf=0.27`)** on the test pads, while preserving genuine ICs and header pins!""")
    add_code("""arty_img = "Arty_Top_jpg.rf.7bc260a89099530b771500c983e1669e.jpg"
print("1. With low threshold (min_conf = 0.25) -> Spurious connector (0.27) is included:")
inspect_injected_crops(arty_img, min_conf=0.25, max_crops=4)

print("\\n2. With High-Conviction threshold (min_conf = 0.50) -> Spurious connector is filtered out!")
inspect_injected_crops(arty_img, min_conf=0.50, max_crops=4)
compare_original_vs_cleaned(arty_img, min_conf=0.50)""")

    # Cell 6: Showcase 1 (pcb123rec1: High-Confidence Missing Capacitors)
    add_md("""### Example 1: `pcb123rec1` (Dense Capacitor Decoupling Bank)
Even at a strict `min_conf = 0.40`, **dozens of unannotated capacitors remain verified**!""")
    add_code("""example_1 = "pcb123rec1_jpg.rf.2cc75b8b236f73d9bdfe05e02d5b0340.jpg"
compare_original_vs_cleaned(example_1, min_conf=0.40)
inspect_injected_crops(example_1, min_conf=0.40, max_crops=5)""")

    # Cell 7: Showcase 2 (pcb12rec1: High-Confidence Voltage Rail Capacitors)
    add_md("""### Example 2: `pcb12rec1` (Voltage Rail Capacitors)""")
    add_code("""example_2 = "pcb12rec1_jpg.rf.ded545184f6c4b53af85d7fd1f965922.jpg"
compare_original_vs_cleaned(example_2, min_conf=0.40)
inspect_injected_crops(example_2, min_conf=0.40, max_crops=5)""")

    # Cell 8: Interactive Board Selector with Confidence Slider
    add_md("""### 🎛️ Interactive PCB Selector with Dynamic Confidence Slider
Use the slider below to dynamically adjust the confidence threshold (`min_conf`) and observe how false positives drop out in real time:""")
    add_code("""if HAS_WIDGETS:
    board_dropdown = widgets.Dropdown(
        options=[(f"{name} (+{cnt} added)", name) for name, cnt in sorted_boards],
        description="Select PCB:",
        layout=widgets.Layout(width="60%")
    )
    conf_slider = widgets.FloatSlider(
        value=0.50,
        min=0.20,
        max=0.90,
        step=0.05,
        description="Min Conf:",
        readout_format=".2f",
        layout=widgets.Layout(width="50%")
    )
    ui_controls = widgets.VBox([board_dropdown, conf_slider])
    output_area = widgets.Output()

    def update_view(*args):
        with output_area:
            clear_output(wait=True)
            bname = board_dropdown.value
            cval = conf_slider.value
            compare_original_vs_cleaned(bname, min_conf=cval)
            inspect_injected_crops(bname, min_conf=cval, max_crops=5)

    board_dropdown.observe(update_view, names='value')
    conf_slider.observe(update_view, names='value')
    
    display(ui_controls, output_area)
    update_view()
else:
    print("Run `compare_original_vs_cleaned(board_name, min_conf=0.50)`")""")

    out_nb = Path("compare_original_vs_cleaned_dataset.ipynb")
    with open(out_nb, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated: {out_nb}")

if __name__ == "__main__":
    create_notebook()
