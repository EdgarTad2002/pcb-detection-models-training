#!/usr/bin/env python3
"""
tools/demo_retinex_fusions.py

Generates side-by-side diagnostic figures and high-resolution microscopic zoom crops
comparing two channel-wise Retinex fusion methodologies:
1. Method 1: Edge-Preserving Bilateral Retinex Residual Fusion (Land's Model)
2. Method 2: Fu et al. (CVPR 2016) 3-Channel Variational Retinex Residual Fusion

Saves presentation-ready visual panels for academic reporting.
"""

import time
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.enhance_pcb_retinex import enhance_pcb_image, decompose_retinex, extract_reflectance_canny
from tools.weighted_variational_retinex import decompose_weighted_variational


def run_variational_rgb_fusion(img_bgr: np.ndarray, alpha: float = 0.30, max_iter: int = 10, device: str = "cpu"):
    """
    Decomposes each RGB channel independently using Fu et al. (CVPR 2016) ADMM,
    then combines original channel + reflectance channel:
        I_fused = (1 - alpha) * I_orig + alpha * R_refl
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype(np.float32) / 255.0  # (H, W, 3)
    
    # Shape (3, H, W) for batched per-channel decomposition
    tensor = torch.from_numpy(img_f).permute(2, 0, 1).to(device)
    
    with torch.no_grad():
        R_tensor, L_tensor, S_enh_tensor = decompose_weighted_variational(
            tensor,
            c1=0.01,
            c2=0.1,
            lambd=1.0,
            gamma=2.2,
            max_iter=max_iter,
        )
    
    R_np = R_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    L_np = L_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    S_enh_np = S_enh_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Residual combination: (1 - alpha) * orig + alpha * reflectance
    fused_rgb = (1.0 - alpha) * img_f + alpha * R_np
    fused_rgb = np.clip(fused_rgb * 255.0, 0, 255).astype(np.uint8)
    fused_bgr = cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2BGR)
    
    return fused_bgr, L_np, R_np, S_enh_np


def generate_presentation_visuals(sample_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(sample_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    
    h, w, _ = img_bgr.shape
    print(f"📷 Processing sample board ({w}x{h}): {sample_path.name}")
    
    # For fast presentation demo on 4K board, downsample to 1280px or take full crop
    scale = 1280.0 / max(h, w)
    w_eval, h_eval = int(w * scale), int(h * scale)
    img_bgr_1280 = cv2.resize(img_bgr, (w_eval, h_eval), interpolation=cv2.INTER_AREA)
    
    orig_rgb = cv2.cvtColor(img_bgr_1280, cv2.COLOR_BGR2RGB)
    
    # =========================================================================
    # 1. Method 1: Bilateral Retinex Residual Fusion
    # =========================================================================
    print("⏳ Running Method 1: Edge-Preserving Bilateral Retinex...")
    t0 = time.time()
    enh_bilateral_bgr, l_bilateral, r_bilateral, edges_bilateral = enhance_pcb_image(
        img_bgr_1280, alpha=0.30, beta=0.10
    )
    t_bilateral = time.time() - t0
    print(f"   Done in {t_bilateral:.3f}s")
    
    enh_bilateral_rgb = cv2.cvtColor(enh_bilateral_bgr, cv2.COLOR_BGR2RGB)
    r_bilateral_rgb = cv2.cvtColor((r_bilateral * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    # Save Method 1 5-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=180)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("1. Original PCB Capture", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(l_bilateral, cmap="inferno")
    axes[1].set_title("2. Illumination Map L\n[Bilateral Edge-Preserving]", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(r_bilateral_rgb)
    axes[2].set_title("3. Reflectance Map R\n[Per-Channel Physical Albedo]", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(edges_bilateral, cmap="gray")
    axes[3].set_title("4. Canny Structural Edges\n[Suppressed Fiberglass Weave]", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    
    axes[4].imshow(enh_bilateral_rgb)
    axes[4].set_title("5. Fused Output\n[0.70·Orig + 0.30·R + 0.10·Edge]", fontsize=11, fontweight="bold", color="darkgreen")
    axes[4].axis("off")
    
    plt.suptitle("Method 1: Edge-Preserving Bilateral Retinex Residual Fusion (Land's Model)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    m1_path = out_dir / "method1_bilateral_retinex_panel.png"
    plt.savefig(str(m1_path), bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Method 1 figure to: {m1_path}")
    
    # =========================================================================
    # 2. Method 2: Fu et al. (CVPR 2016) 3-Channel Variational Retinex Fusion
    # =========================================================================
    print("⏳ Running Method 2: 3-Channel Variational Retinex (Fu et al.)...")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enh_var_bgr, l_var, r_var, s_enh_var = run_variational_rgb_fusion(
        img_bgr_1280, alpha=0.30, max_iter=8, device=device
    )
    t_var = time.time() - t0
    print(f"   Done in {t_var:.3f}s")
    
    enh_var_rgb = cv2.cvtColor(enh_var_bgr, cv2.COLOR_BGR2RGB)
    r_var_rgb = np.clip(r_var * 255.0, 0, 255).astype(np.uint8)
    l_var_mean = np.mean(l_var, axis=2)
    
    # Save Method 2 5-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=180)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("1. Original PCB Capture", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(l_var_mean, cmap="inferno")
    axes[1].set_title("2. 3-Ch Illumination L\n[Variational Smoothness Prior]", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(r_var_rgb)
    axes[2].set_title("3. 3-Ch Reflectance R\n[L1-Sparsity / Soft-Thresholding]", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(np.clip(s_enh_var * 255.0, 0, 255).astype(np.uint8))
    axes[3].set_title("4. Variational Enhanced\n[R · L^(1/γ) Recombination]", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    
    axes[4].imshow(enh_var_rgb)
    axes[4].set_title("5. Fused Output\n[0.70·Orig + 0.30·R_refl]", fontsize=11, fontweight="bold", color="darkblue")
    axes[4].axis("off")
    
    plt.suptitle("Method 2: Fu et al. (CVPR 2016) 3-Channel Variational Retinex Residual Fusion", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    m2_path = out_dir / "method2_variational_retinex_panel.png"
    plt.savefig(str(m2_path), bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Method 2 figure to: {m2_path}")
    
    # =========================================================================
    # 3. Microscopic Zoom Crops Comparison (Micro-Capacitors & Solder Fillets)
    # =========================================================================
    print("🔬 Generating microscopic zoom crops comparison...")
    # Two informative regions: (A) Shadowed capacitor cluster, (B) SMT IC & traces
    crops = [
        ("Crop A: Large dark BGA package + logo (contrast check)", int(h_eval * 0.40), int(h_eval * 0.65), int(w_eval * 0.45), int(w_eval * 0.70)),
        ("Crop B: IC pins, small R/C passives & silkscreen", int(h_eval * 0.15), int(h_eval * 0.40), int(w_eval * 0.15), int(w_eval * 0.40)),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=180)
    for row_idx, (title, y1, y2, x1, x2) in enumerate(crops):
        crop_orig = orig_rgb[y1:y2, x1:x2]
        crop_m1 = enh_bilateral_rgb[y1:y2, x1:x2]
        crop_m2 = enh_var_rgb[y1:y2, x1:x2]
        
        axes[row_idx, 0].imshow(crop_orig)
        axes[row_idx, 0].set_title(f"{title}\n[Original Raw Capture]", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")
        
        axes[row_idx, 1].imshow(crop_m1)
        axes[row_idx, 1].set_title(f"{title}\n[Method 1: Bilateral + Canny]", fontsize=11, fontweight="bold", color="darkgreen")
        axes[row_idx, 1].axis("off")
        
        axes[row_idx, 2].imshow(crop_m2)
        axes[row_idx, 2].set_title(f"{title}\n[Method 2: Variational ADMM]", fontsize=11, fontweight="bold", color="darkblue")
        axes[row_idx, 2].axis("off")
        
    plt.suptitle("Microscopic Component-Level Comparison: Original vs Method 1 vs Method 2", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    zoom_path = out_dir / "microscopic_zoom_comparison.png"
    plt.savefig(str(zoom_path), bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Zoom figure to: {zoom_path}")
    
    # Save standalone demo image files for training evaluation
    cv2.imwrite(str(out_dir / "arty_top_enhanced_bilateral.jpg"), enh_bilateral_bgr)
    cv2.imwrite(str(out_dir / "arty_top_enhanced_variational.jpg"), enh_var_bgr)
    
    print("\n🎉 All presentation visual assets generated successfully!")


if __name__ == "__main__":
    sample = Path("data_samples/native_images/Arty_Top_jpg.rf.7bc260a89099530b771500c983e1669e.jpg")
    out = Path("results/retinex_fusion_presentation")
    generate_presentation_visuals(sample, out)
