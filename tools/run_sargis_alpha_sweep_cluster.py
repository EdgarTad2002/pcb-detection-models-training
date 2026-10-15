"""
Runs a multi-alpha visual and quantitative ablation study of Sargis's Weber filter
across 10 representative PCB images on the cluster.

Evaluates alpha in [0.01, 0.02, 0.10, 0.20, 0.95] with window sizes 3x3 and 5x5.
Computes:
  1. EME (Measure of Enhancement, Agaian et al.)
  2. Edge Energy (Sharpness on component pins/traces)
  3. Background Noise (Substrate texture suppression)
  4. Contrast-to-Noise Ratio (CNR = Edge Energy / Background Noise)
"""

from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.enhance_pcb_sargis_weber import compute_sargis_weber_edges, run_alpha_sweep_visualization


def compute_eme(img_gray: np.ndarray, block_size: int = 8, c: float = 1.0) -> float:
    """Measure of Enhancement (Agaian et al., 2007). Higher = better contrast."""
    h, w = img_gray.shape
    k1, k2 = h // block_size, w // block_size
    if k1 == 0 or k2 == 0:
        return 0.0
    eme = 0.0
    for i in range(k1):
        for j in range(k2):
            block = img_gray[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size].astype(np.float32)
            b_min, b_max = block.min(), block.max()
            if b_min + c > 0 and b_max > 0:
                val = (b_max + c) / (b_min + c)
                if val > 1e-4:
                    eme += 20.0 * np.log(val)
    return float(eme / (k1 * k2))


def evaluate_edge_metrics(raw_gray: np.ndarray, edge_map: np.ndarray) -> dict:
    """
    Computes objective quality metrics for an edge map:
      - Edge Energy: Mean intensity in top 5% strongest edge pixels.
      - Background Noise: Mean intensity in bottom 50% flat areas.
      - CNR: Contrast-to-Noise Ratio (higher is sharper with less background clutter).
    """
    p95 = np.percentile(edge_map, 95)
    p50 = np.percentile(edge_map, 50)

    edge_energy = float(np.mean(edge_map[edge_map >= p95]))
    bg_noise = float(np.mean(edge_map[edge_map <= p50]))
    cnr = edge_energy / max(bg_noise, 1e-6)

    return {
        "edge_energy": edge_energy,
        "bg_noise": bg_noise,
        "cnr": cnr,
    }


def main():
    print("=" * 80)
    print("🔬 SARGIS WEBER FILTER: MULTI-ALPHA VISUAL & QUANTITATIVE ABLATION (10 IMAGES)")
    print("=" * 80)

    # 1. Locate 10 representative test images
    candidate_dirs = [
        Path("datasets/pcb-unified-4class-cleaned/test/images"),
        Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-unified-4class-cleaned/test/images"),
        Path("datasets/pcb-unified-4class/test/images"),
    ]
    img_dir = None
    for d in candidate_dirs:
        if d.exists():
            img_dir = d
            break

    if img_dir is None:
        raise FileNotFoundError(f"Could not find test images in candidate paths: {candidate_dirs}")

    sample_images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:10]
    print(f"Found {len(sample_images)} test images in: {img_dir}\n")

    out_dir = Path("results/sargis_alpha_ablation_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = [0.01, 0.02, 0.10, 0.20, 0.95]
    ksizes = [3, 5]

    # Metric accumulators: {(alpha, ksize): [metrics across 10 images]}
    metric_results = { (a, k): [] for a in alphas for k in ksizes }

    # 2. Process each image
    for idx, img_path in enumerate(sample_images):
        print(f"[{idx+1}/10] Analyzing {img_path.name}...")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Generate the side-by-side diagnostic panel
        out_panel = out_dir / f"board_{idx+1:02d}_{img_path.stem[:25]}_sweep.png"
        run_alpha_sweep_visualization(
            input_img_path=img_path,
            output_png_path=out_panel,
            alphas=alphas,
            ksizes=ksizes
        )

        # Compute quantitative metrics for each configuration
        for a in alphas:
            for k in ksizes:
                edge_map, c_ratio, _, _ = compute_sargis_weber_edges(
                    gray, ksize=k, alpha=a, c=1.0, use_log1p=True, normalize_output=True
                )
                m = evaluate_edge_metrics(gray, edge_map)
                # Compute EME on enhanced image: 0.70*I + 0.30*R + 0.10*Edge
                enh_gray = np.clip(gray.astype(np.float32) + 0.10 * (edge_map * 255.0), 0, 255).astype(np.uint8)
                m["eme"] = compute_eme(enh_gray)
                metric_results[(a, k)].append(m)

    # 3. Print Master Quantitative Ablation Table
    print("\n" + "=" * 85)
    print("📊 MASTER MULTI-ALPHA ABLATION RESULTS (Averaged across 10 PCB test boards)")
    print("=" * 85)
    print(f"{'Alpha (a)':<12} | {'Window (k)':<10} | {'Edge Energy':<12} | {'BG Noise (Substrate)':<22} | {'CNR (Sharpness)':<16} | {'EME Contrast'}")
    print("-" * 85)

    best_cnr = -1.0
    best_config = None

    for (a, k), metrics_list in sorted(metric_results.items()):
        avg_energy = np.mean([m["edge_energy"] for m in metrics_list])
        avg_noise = np.mean([m["bg_noise"] for m in metrics_list])
        avg_cnr = np.mean([m["cnr"] for m in metrics_list])
        avg_eme = np.mean([m["eme"] for m in metrics_list])

        if avg_cnr > best_cnr:
            best_cnr = avg_cnr
            best_config = (a, k)

        print(f"alpha = {a:<6} | {k}x{k:<7} | {avg_energy:<12.4f} | {avg_noise:<22.4f} | {avg_cnr:<16.2f} | {avg_eme:.2f}")

    print("=" * 85)
    print(f" Optimal Alpha based on Contrast-to-Noise Ratio (CNR): alpha = {best_config[0]} (window {best_config[1]}x{best_config[1]})")
    print(f" All 10 visual comparison panels saved in: {out_dir.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
