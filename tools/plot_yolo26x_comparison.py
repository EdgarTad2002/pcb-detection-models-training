#!/usr/bin/env python3
"""
Generate comprehensive comparison figures and analysis for:
YOLO26x (Original Dataset) vs YOLO26x (Cleaned Dataset)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    runs_dir = root / "runs"
    
    # 1. Load results JSONs
    with open(results_dir / "rectified_yolov26x_1280.json") as f:
        res_orig = json.load(f)
    with open(results_dir / "rectified_yolov26x_cleaned_1280.json") as f:
        res_clean = json.load(f)
        
    # Baseline comparison (YOLO26s 1280)
    baseline_path = results_dir / "yolov26s_native_res_rectified_1280.json"
    res_s_baseline = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            res_s_baseline = json.load(f)
            
    # 2. Load training logs (results.csv)
    csv_orig = runs_dir / "rectified_yolov26x_1280" / "pcb-filtered" / "results.csv"
    csv_clean = runs_dir / "rectified_yolov26x_cleaned_1280" / "pcb-filtered" / "results.csv"
    
    df_orig = pd.read_csv(csv_orig)
    df_clean = pd.read_csv(csv_clean)
    
    # Strip whitespace from column names
    df_orig.columns = [c.strip() for c in df_orig.columns]
    df_clean.columns = [c.strip() for c in df_clean.columns]
    
    # 3. Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # Color palette
    c_orig = "#2b5c8f"    # Deep Blue (Original)
    c_clean = "#2ca02c"   # Vibrant Green (Cleaned)
    c_base = "#ff7f0e"    # Orange (YOLO26s Baseline)
    
    # Subplot 1: Validation mAP50 across Epochs
    ax1 = axes[0, 0]
    ax1.plot(df_orig["epoch"], df_orig["metrics/mAP50(B)"], label="YOLO26x Original (1280px)", color=c_orig, linewidth=2.2)
    ax1.plot(df_clean["epoch"], df_clean["metrics/mAP50(B)"], label="YOLO26x Cleaned (1280px)", color=c_clean, linewidth=2.2)
    ax1.set_title("Validation mAP@0.5 Trajectory (100 Epochs)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("mAP@0.5", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, shadow=True)
    ax1.set_ylim(0, 0.65)
    
    # Subplot 2: Validation mAP50-95 across Epochs
    ax2 = axes[0, 1]
    ax2.plot(df_orig["epoch"], df_orig["metrics/mAP50-95(B)"], label="YOLO26x Original (1280px)", color=c_orig, linewidth=2.2)
    ax2.plot(df_clean["epoch"], df_clean["metrics/mAP50-95(B)"], label="YOLO26x Cleaned (1280px)", color=c_clean, linewidth=2.2)
    ax2.set_title("Validation mAP@0.5:0.95 Trajectory (Strict Localization)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("mAP@0.5:0.95", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, shadow=True)
    ax2.set_ylim(0, 0.50)
    
    # Subplot 3: Per-Class AP@0.5 Comparison Bar Chart
    ax3 = axes[1, 0]
    classes = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
    x = np.arange(len(classes))
    width = 0.35
    
    vals_orig = [res_orig["per_class_ap50"][c] for c in classes]
    vals_clean = [res_clean["per_class_ap50"][c] for c in classes]
    
    bars1 = ax3.bar(x - width/2, vals_orig, width, label="YOLO26x Original", color=c_orig, alpha=0.9, edgecolor="black")
    bars2 = ax3.bar(x + width/2, vals_clean, width, label="YOLO26x Cleaned (+1,175 labels)", color=c_clean, alpha=0.9, edgecolor="black")
    
    # Add value annotations on bars
    for bar in bars1:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.012, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.012, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax3.set_title("Per-Class AP@0.5 Breakdown (Test Split)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=15, ha="right", fontsize=10)
    ax3.set_ylabel("Average Precision (AP@0.5)", fontsize=11)
    ax3.set_ylim(0, 0.85)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", frameon=True, shadow=True)
    
    # Highlight the Capacitor gain!
    cap_gain = ((vals_clean[0] - vals_orig[0]) / vals_orig[0]) * 100
    ax3.annotate(f"+{cap_gain:.1f}% Boost!\n(Cleaned Labels)", 
                 xy=(0 + width/2, vals_clean[0]), 
                 xytext=(0.4, 0.45),
                 arrowprops=dict(facecolor="#d62728", shrink=0.08, width=2, headwidth=8),
                 fontsize=10, fontweight="bold", color="#d62728",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebeb", edgecolor="#d62728", lw=1.5))
    
    # Subplot 4: Precision, Recall, and Overall Tradeoff
    ax4 = axes[1, 1]
    metrics_names = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    x_m = np.arange(len(metrics_names))
    m_orig = [res_orig["precision"], res_orig["recall"], res_orig["mAP50"], res_orig["mAP50_95"]]
    m_clean = [res_clean["precision"], res_clean["recall"], res_clean["mAP50"], res_clean["mAP50_95"]]
    
    bars_m1 = ax4.bar(x_m - width/2, m_orig, width, label="YOLO26x Original", color=c_orig, alpha=0.9, edgecolor="black")
    bars_m2 = ax4.bar(x_m + width/2, m_clean, width, label="YOLO26x Cleaned", color=c_clean, alpha=0.9, edgecolor="black")
    
    for bar in bars_m1:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.012, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_m2:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.012, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax4.set_title("Overall Benchmark Metrics (1280px Test Set)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xticks(x_m)
    ax4.set_xticklabels(metrics_names, fontsize=10)
    ax4.set_ylabel("Score", fontsize=11)
    ax4.set_ylim(0, 0.85)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper right", frameon=True, shadow=True)
    
    plt.suptitle("A/B Ablation: YOLO26x Flagship (59M Params, 1280px) on Original vs Cleaned Dataset", 
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_path = results_dir / "yolo26x_original_vs_cleaned_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved comparison plot to: {out_path}")

if __name__ == "__main__":
    main()
