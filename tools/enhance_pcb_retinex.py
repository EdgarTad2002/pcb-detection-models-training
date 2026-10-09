#!/usr/bin/env python3
"""
Retinex Material Reflectance & Edge-Preserving Enhancement for PCB Inspection.
=============================================================================
Applies:
1. Illumination-Reflectance Decomposition (I = R ⊙ L) via Edge-Preserving Bilateral Filter.
2. Log-domain / Normalized Material Reflectance extraction per color channel (R, G, B).
3. Illumination-invariant Canny edge extraction on the Reflectance field (suppressing fiberglass weave).
4. Additive Material & Edge Injection:
       I_enhanced = (1 - alpha) * I_orig + alpha * R + beta * Edge

Usage:
  1. Preview on a single image:
     python tools/enhance_pcb_retinex.py --preview --input <path_to_image.jpg> --output preview.png

  2. Process full dataset:
     python tools/enhance_pcb_retinex.py \
         --source datasets/pcb-unified-4class-cleaned \
         --dest datasets/pcb-retinex-enhanced-640 \
         --alpha 0.30 --beta 0.10
"""

import argparse
import concurrent.futures
from pathlib import Path
import shutil
import time

import cv2
import numpy as np


def decompose_retinex(img_bgr: np.ndarray, d: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0):
    """
    Decomposes an image into Illumination (L) and Reflectance (R) maps
    using edge-preserving bilateral filtering to prevent halo artifacts.
    """
    img_f = img_bgr.astype(np.float32) / 255.0
    
    # 1. Initial Illumination Estimate: max projection across RGB channels
    l_init = np.max(img_f, axis=2)
    
    # 2. Edge-preserving smoothing on L (uint8 space for OpenCV bilateralFilter)
    l_init_u8 = (l_init * 255.0).astype(np.uint8)
    l_smooth_u8 = cv2.bilateralFilter(l_init_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    l_smooth = (l_smooth_u8.astype(np.float32) / 255.0) + 1e-4
    
    # 3. Extract Reflectance for each channel: R^c = I^c / L
    r_map = np.zeros_like(img_f)
    for c in range(3):
        r_map[:, :, c] = img_f[:, :, c] / l_smooth
    
    # Normalize reflectance to [0, 1]
    r_norm = np.clip(r_map, 0.0, 1.0)
    
    return l_smooth, r_norm, img_f


def extract_reflectance_canny(r_norm: np.ndarray, blur_ksize: int = 3):
    """
    Extracts sharp structural edges from the shadow-free Reflectance map.
    Pre-smooths slightly to eliminate microscopic fiberglass weave texture of FR-4 substrate.
    """
    # Convert reflectance to grayscale luminance
    r_u8 = (r_norm * 255.0).astype(np.uint8)
    r_gray = cv2.cvtColor(r_u8, cv2.COLOR_BGR2GRAY)
    
    # Light bilateral blur to smooth fiberglass weave while preserving component edges
    r_clean = cv2.bilateralFilter(r_gray, d=blur_ksize, sigmaColor=30, sigmaSpace=30)
    
    # Otsu-adaptive Canny thresholds
    otsu_thresh, _ = cv2.threshold(r_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower_th = max(10, int(0.5 * otsu_thresh))
    upper_th = min(250, int(1.2 * otsu_thresh))
    
    edges = cv2.Canny(r_clean, lower_th, upper_th)
    edge_f = (edges.astype(np.float32) / 255.0)[:, :, np.newaxis]
    return edge_f, edges


def enhance_pcb_image(img_bgr: np.ndarray, alpha: float = 0.30, beta: float = 0.10):
    """
    Fuses the original photo with its corresponding reflectance and edge features:
        I_enhanced = (1 - alpha) * I_orig + alpha * R + beta * Edge
    """
    l_map, r_map, img_f = decompose_retinex(img_bgr)
    edge_f, raw_edges = extract_reflectance_canny(r_map)
    
    # Additive Material & Edge Blending
    enhanced = (1.0 - alpha) * img_f + alpha * r_map + beta * edge_f
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    
    return enhanced, l_map, r_map, raw_edges


def create_preview_panel(input_path: Path, output_path: Path, alpha: float = 0.30, beta: float = 0.10):
    """Creates a 5-panel side-by-side diagnostic visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")
    
    enhanced_bgr, l_map, r_map, edges = enhance_pcb_image(img_bgr, alpha=alpha, beta=beta)
    
    # Convert BGR to RGB for matplotlib
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    enh_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    r_rgb = cv2.cvtColor((r_map * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=150)
    
    axes[0].imshow(orig_rgb)
    axes[0].set_title("1. Original PCB (RGB)", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(l_map, cmap="inferno")
    axes[1].set_title("2. Illumination (L)\n[Bilateral Edge-Preserving]", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(r_rgb)
    axes[2].set_title("3. Material Reflectance (R)\n[Shadow & Glare Free]", fontsize=13, fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(edges, cmap="gray")
    axes[3].set_title("4. Canny Edges on R\n[Material Transitions]", fontsize=13, fontweight="bold")
    axes[3].axis("off")
    
    axes[4].imshow(enh_rgb)
    axes[4].set_title(f"5. Enhanced Output\n[Orig + {alpha*100:.0f}% R + {beta*100:.0f}% Edge]", fontsize=13, fontweight="bold")
    axes[4].axis("off")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    print(f"✅ Preview panel saved to: {output_path}")


def process_single_task(args_tuple):
    src_file, dst_file, alpha, beta = args_tuple
    img = cv2.imread(str(src_file))
    if img is None:
        return False
    enh, _, _, _ = enhance_pcb_image(img, alpha=alpha, beta=beta)
    cv2.imwrite(str(dst_file), enh)
    return True


def main():
    p = argparse.ArgumentParser(description="Retinex & Edge-Preserving PCB Enhancement Tool")
    p.add_argument("--preview", action="store_true", help="Generate 5-panel preview comparison on a single image")
    p.add_argument("--input", type=Path, default=None, help="Input image path for preview")
    p.add_argument("--output", type=Path, default=Path("preview_retinex_canny.png"), help="Output path for preview image")
    p.add_argument("--source", type=Path, default=None, help="Source dataset path (e.g. datasets/pcb-unified-4class-cleaned)")
    p.add_argument("--dest", type=Path, default=None, help="Destination dataset path")
    p.add_argument("--alpha", type=float, default=0.30, help="Weight of added reflectance (default: 0.30 = 30%%)")
    p.add_argument("--beta", type=float, default=0.10, help="Weight of added Canny edges (default: 0.10 = 10%%)")
    p.add_argument("--workers", type=int, default=8, help="Number of worker processes for batch processing")
    args = p.parse_args()

    if args.preview:
        if args.input is None:
            # Look for any available sample image
            candidates = list(Path("datasets").glob("**/*.jpg")) + list(Path(".").glob("**/*.jpg"))
            if not candidates:
                raise ValueError("No input image specified and no .jpg found in workspace!")
            args.input = candidates[0]
        create_preview_panel(args.input, args.output, alpha=args.alpha, beta=args.beta)
        return

    if args.source and args.dest:
        assert args.source.exists(), f"Source dataset {args.source} not found!"
        print("=" * 75)
        print("🚀 Generating Retinex Material & Edge-Enhanced PCB Dataset")
        print(f"   Source: {args.source}")
        print(f"   Dest:   {args.dest}")
        print(f"   Reflectance Weight (alpha): {args.alpha}")
        print(f"   Edge Weight (beta):        {args.beta}")
        print("=" * 75)

        t0 = time.time()
        splits = ["train", "valid", "test"]
        for split in splits:
            src_img_dir = args.source / split / "images"
            src_lbl_dir = args.source / split / "labels"
            if not src_img_dir.exists():
                continue

            dst_img_dir = args.dest / split / "images"
            dst_lbl_dir = args.dest / split / "labels"
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            img_files = sorted([f for f in src_img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            print(f"\nProcessing '{split}' split ({len(img_files)} images)...")

            tasks = [(f, dst_img_dir / f.name, args.alpha, args.beta) for f in img_files]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(process_single_task, tasks))
            print(f"Processed {sum(results)} images in '{split}'.")

            # Copy label .txt files 100% unchanged
            if src_lbl_dir.exists():
                print(f"Copying label files for '{split}' split...")
                for lbl in src_lbl_dir.glob("*.txt"):
                    shutil.copy2(lbl, dst_lbl_dir / lbl.name)

        # Copy data.yaml if present
        data_yaml = args.source / "data.yaml"
        if data_yaml.exists():
            shutil.copy2(data_yaml, args.dest / "data.yaml")

        print(f"\n✅ Enhanced dataset created successfully at {args.dest} in {time.time() - t0:.1f}s!")
    else:
        print("Please provide either --preview with --input, or --source and --dest.")


if __name__ == "__main__":
    main()
