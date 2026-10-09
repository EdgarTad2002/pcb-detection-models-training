#!/usr/bin/env python3
"""
tools/plot_retinex_comparison.py
Plots side-by-side comparison of the Fu et al. CVPR 2016 weighted variational Retinex decomposition.
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison():
    demo_dir = Path("results/variational_retinex_demo")
    sample_path = Path("data_samples/native_images/Arty_Top_jpg.rf.7bc260a89099530b771500c983e1669e.jpg")
    
    img_orig = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)
    img_refl = cv2.cvtColor(cv2.imread(str(demo_dir / "reflectance.jpg")), cv2.COLOR_BGR2RGB)
    img_illum = cv2.imread(str(demo_dir / "illumination.jpg"), cv2.IMREAD_GRAYSCALE)
    img_enh = cv2.cvtColor(cv2.imread(str(demo_dir / "enhanced.jpg")), cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=200)
    
    # 1. Original
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title("1. Observed Image S = R · L (Original Camera Capture)", fontsize=13, fontweight="bold", pad=8)
    axes[0, 0].axis("off")
    
    # 2. Illumination L
    axes[0, 1].imshow(img_illum, cmap="inferno")
    axes[0, 1].set_title("2. Estimated Illumination L (Spatial Light Intensity / Glare)", fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].axis("off")
    
    # 3. Reflectance R
    axes[1, 0].imshow(img_refl)
    axes[1, 0].set_title("3. Intrinsic Reflectance R (Shadow-Invariant Physical Albedo)", fontsize=13, fontweight="bold", pad=8)
    axes[1, 0].axis("off")
    
    # 4. Enhanced S_enhanced
    axes[1, 1].imshow(img_enh)
    axes[1, 1].set_title("4. Illumination-Enhanced S_enhanced = R · L^(1/γ) (Fu et al., CVPR 2016)", fontsize=13, fontweight="bold", pad=8)
    axes[1, 1].axis("off")
    
    plt.suptitle("Weighted Variational Retinex Decomposition on PCB (Fu et al., CVPR 2016)", 
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_file = demo_dir / "retinex_decomposition_quad.jpg"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved figure to: {out_file}")

if __name__ == "__main__":
    plot_comparison()
