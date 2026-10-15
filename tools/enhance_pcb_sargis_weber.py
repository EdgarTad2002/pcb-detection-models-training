"""
Sargis's Local Contrast / Weber-Fechner Edge Enhancement Filter for PCB AOI
Based on Chapter 6 (Eq. 6.3.8 / Eq. 5) of Ph.D. Dissertation (Hovhannisyan, 2026):

    f(L) = 300 * ((L_max - L_min) / (L_min + c))^alpha * log( (L_max - L_min) / (L_min + c) )

Replaces binary non-differentiable Canny edges with smooth, continuous,
physically grounded Weber contrast edge responses.
"""

import argparse
import concurrent.futures
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def compute_sargis_weber_edges(
    img: np.ndarray,
    ksize: int = 3,
    alpha: float = 0.10,
    c: float = 1.0,
    use_log1p: bool = True,
    normalize_output: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Sargis's Weber-contrast edge map on an input image.

    Args:
        img: Input image (H, W) or (H, W, 3) in uint8 [0, 255] or float [0, 1].
        ksize: Neighborhood window size (3 for 3x3, 5 for 5x5).
        alpha: Tunable exponent adjusting enhancement aggressiveness (e.g. 0.02, 0.1, 0.2, 0.95).
        c: Stabilization constant to prevent division by zero (default: 1.0 for uint8).
        use_log1p: If True, uses log(1 + ratio) so that flat areas (ratio=0) yield exactly 0,
                   guaranteeing non-negative, well-behaved edge gradients.
        normalize_output: If True, rescales output to [0, 1].
    """
    if img.ndim == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.dtype == np.uint8 else cv2.COLOR_RGB2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        gray = img.copy()

    if gray.dtype != np.float32 and gray.dtype != np.float64:
        gray_f = gray.astype(np.float32)
    else:
        if gray.max() <= 1.0:
            gray_f = gray.astype(np.float32) * 255.0
        else:
            gray_f = gray.astype(np.float32)

    # Local Neighborhood Window: L_max (Dilation) and L_min (Erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    l_max = cv2.dilate(gray_f, kernel)
    l_min = cv2.erode(gray_f, kernel)

    # Weber Contrast Ratio: (L_max - L_min) / (L_min + c)
    diff = np.maximum(l_max - l_min, 0.0)
    denom = l_min + float(c)
    c_ratio = diff / denom

    # Formula: 300 * (C_ratio)^alpha * log(C_ratio)
    ratio_pow = np.power(np.maximum(c_ratio, 1e-7), alpha)
    log_term = np.log1p(c_ratio) if use_log1p else np.maximum(0.0, np.log(np.maximum(c_ratio, 1e-4)))
    raw_edge = 300.0 * ratio_pow * log_term

    if normalize_output:
        p99 = np.percentile(raw_edge, 99.8)
        if p99 > 1e-6:
            edge_norm = np.clip(raw_edge / p99, 0.0, 1.0)
        else:
            edge_norm = np.zeros_like(raw_edge)
        return edge_norm, c_ratio, l_max, l_min

    return raw_edge, c_ratio, l_max, l_min


def decompose_bilateral_retinex(img_bgr: np.ndarray, d: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0):
    """Decomposes an image into Illumination L and Material Reflectance R."""
    img_f = img_bgr.astype(np.float32) / 255.0
    l_init = np.max(img_f, axis=2)
    l_init_u8 = (l_init * 255.0).astype(np.uint8)
    l_smooth_u8 = cv2.bilateralFilter(l_init_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    l_smooth = (l_smooth_u8.astype(np.float32) / 255.0) + 1e-4
    
    r_map = np.zeros_like(img_f)
    for c in range(3):
        r_map[:, :, c] = img_f[:, :, c] / l_smooth
    r_norm = np.clip(r_map, 0.0, 1.0)
    return l_smooth, r_norm, img_f


def enhance_pcb_sargis_image(
    img_bgr: np.ndarray,
    alpha_exp: float = 0.10,
    ksize: int = 3,
    weight_orig: float = 0.70,
    weight_r: float = 0.30,
    weight_edge: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes the PCB image using:
        I_enhanced = weight_orig * I_orig + weight_r * R + weight_edge * Edge_Weber(R)
    """
    l_map, r_map, img_f = decompose_bilateral_retinex(img_bgr)
    edge_norm, _, _, _ = compute_sargis_weber_edges(
        (r_map * 255.0).astype(np.uint8),
        ksize=ksize,
        alpha=alpha_exp,
        c=1.0,
        use_log1p=True,
        normalize_output=True,
    )
    edge_3c = edge_norm[:, :, None]
    enhanced = weight_orig * img_f + weight_r * r_map + weight_edge * edge_3c
    enhanced_u8 = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return enhanced_u8, l_map, r_map, edge_norm


def process_single_task(args_tuple):
    src_file, dst_file, alpha_exp, ksize, w_orig, w_r, w_edge = args_tuple
    img = cv2.imread(str(src_file))
    if img is None:
        return False
    enh, _, _, _ = enhance_pcb_sargis_image(
        img,
        alpha_exp=alpha_exp,
        ksize=ksize,
        weight_orig=w_orig,
        weight_r=w_r,
        weight_edge=w_edge,
    )
    cv2.imwrite(str(dst_file), enh)
    return True


def create_preview_panel(
    input_path: Path,
    output_path: Path,
    alpha_exp: float = 0.10,
    ksize: int = 3,
):
    """Creates a 5-panel side-by-side diagnostic visualization comparing Canny vs Sargis Weber."""
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    enh_bgr, l_map, r_map, weber_edges = enhance_pcb_sargis_image(
        img_bgr, alpha_exp=alpha_exp, ksize=ksize
    )

    r_gray = cv2.cvtColor((r_map * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    r_clean = cv2.bilateralFilter(r_gray, d=3, sigmaColor=30, sigmaSpace=30)
    otsu_th, _ = cv2.threshold(r_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_edges = cv2.Canny(r_clean, max(10, int(0.5 * otsu_th)), min(250, int(1.2 * otsu_th)))

    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
    r_rgb = cv2.cvtColor((r_map * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=150)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("1. Original PCB (RGB)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(r_rgb)
    axes[1].set_title("2. Material Reflectance (R)\n[Shadow-Free]", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(canny_edges, cmap="gray")
    axes[2].set_title("3. Classic Canny on R\n[Binary 1-bit Step Spikes]", fontsize=12, fontweight="bold", color="darkred")
    axes[2].axis("off")

    axes[3].imshow(weber_edges, cmap="inferno")
    axes[3].set_title(f"4. Sargis Weber on R\n[Smooth Continuous (a={alpha_exp}, k={ksize})]", fontsize=12, fontweight="bold", color="darkblue")
    axes[3].axis("off")

    axes[4].imshow(enh_rgb)
    axes[4].set_title("5. Sargis Enhanced Output\n[0.7I + 0.3R + 0.1Weber]", fontsize=12, fontweight="bold", color="darkgreen")
    axes[4].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight")
    plt.close()
    print(f"✅ Preview panel saved to: {output_path}")


def run_alpha_sweep_visualization(
    input_img_path: Union[str, Path],
    output_png_path: Union[str, Path],
    alphas: List[float] = [0.01, 0.02, 0.1, 0.2, 0.95],
    ksizes: List[int] = [3, 5],
):
    img_bgr = cv2.imread(str(input_img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image at {input_img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

    n_cols = len(alphas) + 2
    n_rows = len(ksizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=150)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, k in enumerate(ksizes):
        axes[r_idx, 0].imshow(img_rgb)
        axes[r_idx, 0].set_title(f"Original PCB\n(Window {k}x{k})", fontsize=11, fontweight="bold")
        axes[r_idx, 0].axis("off")

        axes[r_idx, 1].imshow(canny, cmap="gray")
        axes[r_idx, 1].set_title("Classic Canny\n[Binary 1-bit]", fontsize=11, fontweight="bold", color="darkred")
        axes[r_idx, 1].axis("off")

        for c_idx, a in enumerate(alphas):
            col = c_idx + 2
            edge_map, _, _, _ = compute_sargis_weber_edges(
                gray, ksize=k, alpha=a, c=1.0, use_log1p=True, normalize_output=True
            )
            axes[r_idx, col].imshow(edge_map, cmap="inferno")
            axes[r_idx, col].set_title(f"Sargis Weber (k={k})\nalpha = {a}", fontsize=11, fontweight="bold", color="darkblue")
            axes[r_idx, col].axis("off")

    plt.tight_layout()
    Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_png_path), bbox_inches="tight")
    plt.close()
    print(f"✅ Saved alpha sweep diagnostic comparison to: {output_png_path}")


def main():
    p = argparse.ArgumentParser(description="Sargis Weber Contrast PCB Enhancement & Dataset Generator")
    p.add_argument("--preview", action="store_true", help="Generate 5-panel preview comparison on a single image")
    p.add_argument("--input", type=Path, default=None, help="Input image path for preview")
    p.add_argument("--output", type=Path, default=Path("preview_sargis_weber.png"), help="Output path for preview image")
    p.add_argument("--sweep", action="store_true", help="Run multi-alpha sweep on a single image")
    p.add_argument("--source", type=Path, default=None, help="Source dataset path")
    p.add_argument("--dest", type=Path, default=None, help="Destination dataset path")
    p.add_argument("--alpha-exp", type=float, default=0.10, help="Sargis Weber exponent alpha (default: 0.10)")
    p.add_argument("--ksize", type=int, default=3, choices=[3, 5], help="Window size (default: 3)")
    p.add_argument("--w-orig", type=float, default=0.70, help="Weight of original image (default: 0.70)")
    p.add_argument("--w-r", type=float, default=0.30, help="Weight of reflectance (default: 0.30)")
    p.add_argument("--w-edge", type=float, default=0.10, help="Weight of Weber edges (default: 0.10)")
    p.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = p.parse_args()

    if args.sweep:
        if args.input is None:
            candidates = list(Path("datasets").glob("**/*.jpg")) + list(Path(".").glob("**/*.jpg"))
            if not candidates:
                raise ValueError("No input image found!")
            args.input = candidates[0]
        run_alpha_sweep_visualization(args.input, args.output)
        return

    if args.preview:
        if args.input is None:
            candidates = list(Path("datasets").glob("**/*.jpg")) + list(Path(".").glob("**/*.jpg"))
            if not candidates:
                raise ValueError("No input image found!")
            args.input = candidates[0]
        create_preview_panel(args.input, args.output, alpha_exp=args.alpha_exp, ksize=args.ksize)
        return

    if args.source and args.dest:
        assert args.source.exists(), f"Source dataset {args.source} not found!"
        print("=" * 75)
        print("🚀 Generating Sargis Weber-Contrast Enhanced PCB Dataset")
        print(f"   Source:     {args.source}")
        print(f"   Dest:       {args.dest}")
        print(f"   Alpha Exp:  {args.alpha_exp} | Window: {args.ksize}x{args.ksize}")
        print(f"   Weights:    Orig={args.w_orig}, Reflectance={args.w_r}, WeberEdge={args.w_edge}")
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

            tasks = [
                (f, dst_img_dir / f.name, args.alpha_exp, args.ksize, args.w_orig, args.w_r, args.w_edge)
                for f in img_files
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(process_single_task, tasks))
            print(f"Processed {sum(results)} images in '{split}'.")

            if src_lbl_dir.exists():
                print(f"Copying label files for '{split}' split...")
                for lbl in src_lbl_dir.glob("*.txt"):
                    shutil.copy2(lbl, dst_lbl_dir / lbl.name)

        # Write clean data.yaml with path pointing to dest
        src_yaml = args.source / "data.yaml"
        if src_yaml.exists():
            with open(src_yaml, "r") as f:
                cfg = yaml.safe_load(f)
            cfg["path"] = str(args.dest.resolve())
            with open(args.dest / "data.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

        print(f"\n✅ Sargis Weber dataset created successfully at {args.dest} in {time.time() - t0:.1f}s!")
    else:
        print("Please provide either --preview/--sweep with --input, or --source and --dest.")


if __name__ == "__main__":
    main()
