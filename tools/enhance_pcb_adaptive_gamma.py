#!/usr/bin/env python3
"""
tools/enhance_pcb_adaptive_gamma.py

Adaptive Gamma Correction & Retinex Material Enhancement for PCB Inspection.
=============================================================================
Replaces heuristic Canny binary edge detection with continuous Adaptive Gamma
Correction with Weighting Distribution (AGCWD) and illumination tone mapping.

Key Improvements over Canny:
1. Zero binary edge discontinuities (no hard 0/255 lines that corrupt conv kernels).
2. Suppresses microscopic FR-4 fiberglass weave noise instead of amplifying it.
3. Normalizes severe solder specular glare while illuminating dark component shadows.
4. Preserves continuous physical albedo and color fidelity.

Mathematical Formulation:
1. Illumination-Reflectance Decomposition: I = R ⊙ L (via edge-preserving bilateral filter).
2. Material Reflectance extraction: R = I / (L + eps).
3. Adaptive Gamma Illumination Mapping:
       P_w(l) = P_max * ((P(l) - P_min) / (P_max - P_min))^alpha
       gamma(l) = 1.0 - C_w(l)
       L_adapted = L^(gamma(L))
4. Photometric Fusion (Zero Canny):
       I_enhanced = (1 - alpha) * I_orig + alpha * (R ⊙ L_adapted)

Usage:
  1. Preview on a single image:
     python tools/enhance_pcb_adaptive_gamma.py --preview --input <path_to_image.jpg> --output preview.png

  2. Process full dataset:
     python tools/enhance_pcb_adaptive_gamma.py \
         --source datasets/pcb-unified-4class-cleaned \
         --dest datasets/pcb-adaptive-gamma-1280 \
         --alpha 0.35 --workers 8
"""

import argparse
import concurrent.futures
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Core Image Enhancement Functions
# ---------------------------------------------------------------------------

def compute_agcwd_lut(v_channel: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Computes a 256-element Look-Up Table (LUT) for Adaptive Gamma Correction
    with Weighting Distribution (AGCWD, Huang et al. 2013).
    """
    hist, _ = np.histogram(v_channel, bins=256, range=(0, 256))
    pdf = hist / float(v_channel.size)
    p_min, p_max = pdf.min(), pdf.max()

    if p_max > p_min:
        pdf_w = p_max * (((pdf - p_min) / (p_max - p_min)) ** alpha)
    else:
        pdf_w = pdf

    cdf_w = np.cumsum(pdf_w) / np.maximum(np.sum(pdf_w), 1e-9)
    l_norm = np.arange(256) / 255.0
    gamma = 1.0 - cdf_w
    lut = np.clip(255.0 * np.power(l_norm, gamma), 0, 255).astype(np.uint8)
    return lut


def decompose_bilateral_retinex(img_bgr: np.ndarray, d: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0):
    """
    Decomposes an image into smooth Illumination (L) and Reflectance (R) maps
    using edge-preserving bilateral filtering without spatial edge injection.
    """
    img_f = img_bgr.astype(np.float32) / 255.0
    l_init = np.max(img_f, axis=2)
    l_init_u8 = (l_init * 255.0).astype(np.uint8)

    # Edge-preserving bilateral filter on illumination
    l_smooth_u8 = cv2.bilateralFilter(l_init_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    l_smooth = (l_smooth_u8.astype(np.float32) / 255.0) + 1e-4

    # Reflectance map
    r_map = np.zeros_like(img_f)
    for c in range(3):
        r_map[:, :, c] = img_f[:, :, c] / l_smooth

    r_norm = np.clip(r_map, 0.0, 1.0)
    return l_smooth, r_norm, img_f, l_smooth_u8


def enhance_pcb_adaptive_gamma(img_bgr: np.ndarray, alpha: float = 0.35, agc_alpha: float = 0.5):
    """
    Applies Continuous Adaptive Gamma Retinex Fusion.
    REPLACES Canny binary edges with smooth illumination tone mapping:
        I_enhanced = (1 - alpha) * I_orig + alpha * (R * L_adapted)
    """
    l_smooth, r_norm, img_f, l_smooth_u8 = decompose_bilateral_retinex(img_bgr)

    # Compute Adaptive Gamma LUT on the illumination field
    lut = compute_agcwd_lut(l_smooth_u8, alpha=agc_alpha)
    l_adapted_u8 = cv2.LUT(l_smooth_u8, lut)
    l_adapted = (l_adapted_u8.astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Reconstructed material field with adaptive gamma illumination
    adapted_refl = r_norm * l_adapted

    # Smooth physical blending (zero binary edge injection!)
    enhanced_f = (1.0 - alpha) * img_f + alpha * adapted_refl
    enhanced = np.clip(enhanced_f * 255.0, 0, 255).astype(np.uint8)

    return enhanced, l_smooth, r_norm, l_adapted_u8


# ---------------------------------------------------------------------------
# Dataset Processing Worker
# ---------------------------------------------------------------------------

def process_single_image(args_tuple):
    src_img_path, dst_img_path, alpha, agc_alpha = args_tuple
    try:
        img_bgr = cv2.imread(str(src_img_path))
        if img_bgr is None:
            return False, f"Failed to read {src_img_path}"

        enhanced, _, _, _ = enhance_pcb_adaptive_gamma(img_bgr, alpha=alpha, agc_alpha=agc_alpha)

        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_img_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True, str(src_img_path.name)
    except Exception as e:
        return False, f"Error processing {src_img_path}: {e}"


# ---------------------------------------------------------------------------
# CLI & Full Dataset Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Adaptive Gamma PCB dataset (Zero Canny).")
    parser.add_argument("--source", type=Path, default=Path("datasets/pcb-unified-4class-cleaned"),
                        help="Source dataset directory with train/val/test splits.")
    parser.add_argument("--dest", type=Path, default=Path("datasets/pcb-adaptive-gamma-1280"),
                        help="Destination dataset directory.")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="Reflectance blend weight (default: 0.35).")
    parser.add_argument("--agc-alpha", type=float, default=0.50,
                        help="AGCWD weighting parameter alpha (default: 0.50).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes.")
    parser.add_argument("--preview", action="store_true",
                        help="Generate a visual diagnostic panel on a single image and exit.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input image for --preview.")
    parser.add_argument("--output", type=Path, default=Path("adaptive_gamma_preview.png"),
                        help="Output path for --preview panel.")
    args = parser.parse_args()

    # Preview Mode
    if args.preview:
        if args.input is None:
            # Fallback to sample image if available
            sample = Path("data_samples/native_images/Arty_Top_jpg.rf.7bc260a89099530b771500c983e1669e.jpg")
            if sample.exists():
                args.input = sample
            else:
                raise ValueError("Must provide --input <image.jpg> for --preview mode.")

        img = cv2.imread(str(args.input))
        assert img is not None, f"Could not read {args.input}"

        enhanced, l_smooth, r_norm, l_adapted_u8 = enhance_pcb_adaptive_gamma(img, alpha=args.alpha, agc_alpha=args.agc_alpha)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        r_rgb = cv2.cvtColor((r_norm * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=180)
        axes[0].imshow(orig_rgb)
        axes[0].set_title("1. Original PCB Capture", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(l_smooth, cmap="inferno")
        axes[1].set_title("2. Smooth Illumination L\n[Bilateral Filter - No Edges]", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(l_adapted_u8, cmap="inferno")
        axes[2].set_title("3. Adaptive Gamma L_adapted\n[AGCWD - Solder Glare Protected]", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(enh_rgb)
        axes[3].set_title("4. Adaptive Gamma Fused Output\n[Zero Canny Discontinuities]", fontsize=11, fontweight="bold", color="darkgreen")
        axes[3].axis("off")

        plt.suptitle("Adaptive Gamma Retinex (Replacing Canny with Tone Mapping)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(str(args.output), bbox_inches="tight")
        plt.close()
        print(f"✅ Saved preview diagnostic panel to: {args.output}")
        return

    # Dataset Processing Mode
    src_path = args.source.resolve()
    dst_path = args.dest.resolve()

    assert src_path.exists(), f"Source dataset not found: {src_path}"
    print("=" * 75)
    print("🚀 BUILDING ADAPTIVE GAMMA DATASET (NO CANNY EDGES)")
    print(f"Source:       {src_path}")
    print(f"Destination:  {dst_path}")
    print(f"Parameters:   alpha={args.alpha}, agc_alpha={args.agc_alpha}")
    print("=" * 75)

    dst_path.mkdir(parents=True, exist_ok=True)

    tasks = []
    # Identify splits: train, valid/val, test
    splits = ["train", "valid", "val", "test"]
    found_splits = []

    for sp in splits:
        sp_img_dir = src_path / sp / "images"
        if not sp_img_dir.exists():
            sp_img_dir = src_path / sp
            if not sp_img_dir.exists():
                continue

        target_sp = "val" if sp == "valid" else sp
        if target_sp not in found_splits:
            found_splits.append(target_sp)

        dst_img_dir = dst_path / target_sp / "images"
        dst_lbl_dir = dst_path / target_sp / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Collect images
        img_files = sorted([p for p in sp_img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        for img_file in img_files:
            dst_file = dst_img_dir / img_file.name
            tasks.append((img_file, dst_file, args.alpha, args.agc_alpha))

        # Copy corresponding label files directly
        sp_lbl_dir = src_path / sp / "labels" if (src_path / sp / "images").exists() else src_path / f"{sp}_labels"
        if sp_lbl_dir.exists():
            for lbl_file in sp_lbl_dir.glob("*.txt"):
                shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)

    print(f"📦 Total images to process across {len(found_splits)} splits: {len(tasks)}")

    # Execute processing in parallel
    t0 = time.time()
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for ok, msg in executor.map(process_single_image, tasks):
            if ok:
                success_count += 1
            else:
                print(f"⚠️ {msg}")

    dt = time.time() - t0
    rate = len(tasks) / max(dt, 0.001)
    print(f"✅ Processed {success_count}/{len(tasks)} images in {dt:.1f}s ({rate:.1f} img/s).")

    # Generate data.yaml
    src_yaml = src_path / "data.yaml"
    if src_yaml.exists():
        cfg = yaml.safe_load(src_yaml.read_text())
    else:
        cfg = {
            "nc": 4,
            "names": ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]
        }

    dst_yaml_content = {
        "path": str(dst_path),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": cfg.get("nc", 4),
        "names": cfg.get("names", ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"])
    }

    dst_yaml = dst_path / "data.yaml"
    dst_yaml.write_text(yaml.safe_dump(dst_yaml_content, sort_keys=False))
    print(f"📄 Clean data.yaml written to: {dst_yaml}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
