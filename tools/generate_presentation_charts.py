#!/usr/bin/env python3
"""
tools/generate_presentation_charts.py - Generate publication-quality figures
for Edgar's PCB Detection research presentation.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.2,
    "grid.color": "#e5e7eb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

OUT_DIR = Path("results/presentation_charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Chart 1: Seed Variance & Confidence Intervals (Multi-Seed Scientific Validation)
# -----------------------------------------------------------------------------
def make_seed_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: Multi-seed comparison with error bars and individual points
    configs = ["Raw Baseline\n(Cleaned 1280px)", "Bilateral Retinex\n(α=0.3, β=0.1)", "3-Seed Bilateral\nWBF Ensemble"]
    means = [0.5862, 0.5927, 0.6191]
    stds = [0.0170, 0.0119, 0.0]
    colors = ["#4B5563", "#2563EB", "#059669"]

    # Seeds data
    base_seeds = [0.5840, 0.6042, 0.5705]
    bilat_seeds = [0.5997, 0.5994, 0.5789]

    x = np.arange(len(configs))
    bars = ax1.bar(x, means, yerr=[stds[0], stds[1], 0], capsize=6, color=colors, alpha=0.85, width=0.55, edgecolor="#111827", linewidth=1.2)
    
    # Overlay individual seed points
    ax1.scatter([0]*3, base_seeds, color="#1F2937", s=65, zorder=5, label="Individual Seeds (s0, s1, s2)")
    ax1.scatter([1]*3, bilat_seeds, color="#1D4ED8", s=65, zorder=5)
    ax1.scatter([2], [0.6191], color="#047857", marker="*", s=140, zorder=5, label="Ensemble Output")

    # Annotate values
    ax1.text(0, means[0] + 0.022, f"0.5862 ± 0.0170\n(Range: 3.37%)", ha="center", fontsize=9.5, fontweight="bold", color="#374151")
    ax1.text(1, means[1] + 0.018, f"0.5927 ± 0.0119\n(Δ = +0.0065, not significant)", ha="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax1.text(2, means[2] + 0.007, f"0.6191\n(ensemble of 3 seeds)", ha="center", fontsize=9.5, fontweight="bold", color="#065F46")

    ax1.set_ylabel("mAP50 (4-Class Cleaned Test Split)")
    ax1.set_title("Seed-to-seed spread is as large as the method gap")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylim(0.54, 0.65)
    ax1.grid(True, axis="y")
    ax1.legend(loc="lower left", framealpha=0.9)

    # Right: mAP50 vs Recall Trade-off Frontier
    models_scatter = [
        ("Base s0", 0.5822, 0.5840, "#9CA3AF"),
        ("Base s1", 0.5548, 0.6042, "#9CA3AF"),
        ("Base s2", 0.5791, 0.5705, "#9CA3AF"),
        ("Bilat s0", 0.6008, 0.5997, "#3B82F6"),
        ("Bilat s1", 0.5317, 0.5994, "#3B82F6"),
        ("Bilat s2", 0.5350, 0.5789, "#3B82F6"),
        ("YOLO26x", 0.5616, 0.5975, "#F59E0B"),
        ("Variational Ret.", 0.5716, 0.5591, "#EF4444"),
        ("3-Seed Ensemble", 0.6336, 0.6191, "#10B981"),
    ]

    for label, rec, map50, col in models_scatter:
        if "Ensemble" in label:
            ax2.scatter(rec, map50, color=col, s=160, marker="*", edgecolor="#064E3B", linewidth=1.5, zorder=6, label=f"★ {label} (0.6336, 0.6191)")
            ax2.annotate(label, (rec, map50), textcoords="offset points", xytext=(8, -6), fontweight="bold", color="#065F46", fontsize=10)
        else:
            ax2.scatter(rec, map50, color=col, s=70, alpha=0.85, zorder=4)
            ax2.annotate(label, (rec, map50), textcoords="offset points", xytext=(5, 4), fontsize=8.5, color="#374151")

    ax2.set_xlabel("Detection Recall")
    ax2.set_ylabel("mAP50")
    ax2.set_title("Recall vs. mAP50 (single runs)")
    ax2.set_xlim(0.52, 0.65)
    ax2.set_ylim(0.55, 0.63)
    ax2.grid(True)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    p = OUT_DIR / "seed_variance_and_pareto_frontier.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {p}")

# -----------------------------------------------------------------------------
# Chart 2: Per-Class Breakdown Across Models
# -----------------------------------------------------------------------------
def make_per_class_chart():
    classes = ["Capacitor", "Connector", "Electrolytic Cap", "IC"]
    
    # Data for the 4 core representations
    baseline_mean = [0.3000, 0.7078, 0.7082, 0.6289]
    bilat_mean    = [0.2832, 0.7163, 0.7416, 0.6295]
    yolo26x       = [0.2973, 0.7375, 0.7317, 0.6235]
    ensemble      = [0.3120, 0.7366, 0.7604, 0.6672]

    x = np.arange(len(classes))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    r1 = ax.bar(x - 1.5*width, baseline_mean, width, label="Raw Baseline (Mean of 3 Seeds)", color="#6B7280", edgecolor="#374151")
    r2 = ax.bar(x - 0.5*width, bilat_mean,    width, label="Bilateral Retinex (Mean of 3 Seeds)", color="#3B82F6", edgecolor="#1D4ED8")
    r3 = ax.bar(x + 0.5*width, yolo26x,       width, label="YOLO26x (Bilateral, 1 seed)", color="#F59E0B", edgecolor="#D97706")
    r4 = ax.bar(x + 1.5*width, ensemble,      width, label="3-Seed Bilateral WBF Ensemble", color="#10B981", edgecolor="#047857")

    # Annotate ensemble bars with values
    for i, bar in enumerate(r4):
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color="#065F46")

    ax.set_ylabel("AP50 (Average Precision @ IoU=0.50)")
    ax.set_title("Per-class AP50 (test split, 42 images)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y")

    # Highlight capacitor bottleneck
    ax.axvspan(-0.4, 0.4, color='#FEE2E2', alpha=0.35, zorder=0)
    ax.text(0, 0.38, "Lowest AP50\nin every model", ha="center", fontsize=8.5, color="#991B1B", fontweight="bold")

    plt.tight_layout()
    p = OUT_DIR / "per_class_breakdown_comparison.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {p}")

# -----------------------------------------------------------------------------
# Chart 3: Comprehensive Study Ablation Leaderboard
# -----------------------------------------------------------------------------
def make_ablation_leaderboard():
    models = [
        ("3-Seed Bilateral WBF Ensemble (1280px)", 0.6191, "#059669", "ensemble of 3 seeds"),
        ("Raw Baseline (Seed 1)", 0.6042, "#6B7280", "High-seed variance"),
        ("Bilateral Retinex (Seed 0)", 0.5997, "#2563EB", "Single seed peak"),
        ("Bilateral Retinex (Seed 1)", 0.5994, "#2563EB", "Single seed peak"),
        ("YOLO26x Scale (Bilateral, Seed 0)", 0.5975, "#D97706", "1 seed, larger model"),
        ("Bilateral Retinex (3-Seed Mean)", 0.5927, "#1D4ED8", "Rigorous statistical mean"),
        ("Raw Baseline (Seed 0, hsv_v=0.1)", 0.5895, "#4B5563", "Augmentation tweak"),
        ("Bilateral Retinex (Cleaned 1280px Orig)", 0.5864, "#3B82F6", "First benchmark run"),
        ("Raw Baseline (3-Seed Mean)", 0.5862, "#374151", "Rigorous statistical control"),
        ("Raw Baseline (Cleaned 1280px Orig)", 0.5860, "#4B5563", "Initial control"),
        ("Raw Baseline (Seed 0)", 0.5840, "#4B5563", "Seed 0 control"),
        ("Capacitor Copy-Paste + Bilateral", 0.5790, "#9333EA", "1 seed"),
        ("Bilateral Retinex (Seed 2)", 0.5789, "#3B82F6", "Low-seed variance"),
        ("Bilateral Retinex (hsv_v=0.1)", 0.5765, "#2563EB", "1 seed, within noise"),
        ("Raw Baseline (Seed 2)", 0.5705, "#6B7280", "Low-seed variance"),
        ("P2 High-Resolution Head (1280px)", 0.5670, "#DC2626", "1 seed"),
        ("Variational Retinex (3-Channel Fusion)", 0.5591, "#E11D48", "1 seed"),
    ]

    names = [m[0] for m in reversed(models)]
    scores = [m[1] for m in reversed(models)]
    colors = [m[2] for m in reversed(models)]
    tags   = [m[3] for m in reversed(models)]

    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    bars = ax.barh(range(len(names)), scores, color=colors, height=0.68, edgecolor="#111827", linewidth=0.8, alpha=0.9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("mAP50 on Rectified 4-Class Test Set (42 Images)")
    ax.set_title("All runs ranked by mAP50 (single seeds unless marked Mean)", pad=15)
    ax.set_xlim(0.53, 0.635)
    ax.grid(True, axis="x")

    # Add reference dashed line at Baseline Mean
    ax.axvline(0.5862, color="#374151", linestyle="--", linewidth=1.5, alpha=0.75, label="Raw Baseline Mean (0.5862)")
    ax.axvline(0.6191, color="#059669", linestyle="-.", linewidth=1.5, alpha=0.75, label="3-seed ensemble (0.6191)")

    # Score and tag annotations
    for i, (bar, score, tag) in enumerate(zip(bars, scores, tags)):
        offset = 0.0015
        text_col = "#047857" if score >= 0.61 else ("#1E40AF" if score >= 0.592 else "#374151")
        weight = "bold" if (score >= 0.61 or "Mean" in names[i]) else "normal"
        ax.text(score + offset, bar.get_y() + bar.get_height()/2, f"{score:.4f}  ({tag})",
                va="center", fontsize=8.5, color=text_col, fontweight=weight)

    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    p = OUT_DIR / "comprehensive_ablation_benchmarks.png"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {p}")

if __name__ == "__main__":
    make_seed_chart()
    make_per_class_chart()
    make_ablation_leaderboard()
    print("🎉 All publication charts successfully generated!")
