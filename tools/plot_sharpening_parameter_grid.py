#!/usr/bin/env python3
"""
Generates a multi-panel visual parameter grid (Sigma vs Gamma) for CLAHE + Unsharp Masking
on dense PCB micro-components.
"""

from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def apply_clahe_unsharp(image_bgr: np.ndarray, sigma: float, gamma: float, clip_limit: float = 1.5) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    base_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    if gamma > 0 and sigma > 0:
        blurred = cv2.GaussianBlur(base_bgr, (0, 0), sigma)
        sharpened = cv2.addWeighted(base_bgr, 1.0 + gamma, blurred, -gamma, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    return base_bgr


def main():
    img_path = Path("data_samples/images/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.jpg")
    if not img_path.exists():
        # Fallback to first available image
        import glob
        all_imgs = sorted(glob.glob("data_samples/images/*.jpg"))
        if not all_imgs:
            print("No images found in data_samples/images/")
            return
        img_path = Path(all_imgs[0])

    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]

    # Crop a dense region containing micro-capacitors and traces
    ymin, ymax = int(h * 0.35), int(h * 0.58)
    xmin, xmax = int(w * 0.42), int(w * 0.65)
    crop = img_bgr[ymin:ymax, xmin:xmax]

    sigmas = [0.5, 1.0, 1.5]
    gammas = [0.3, 0.6, 1.0]

    fig, axes = plt.subplots(len(sigmas), len(gammas), figsize=(14, 12))
    plt.subplots_adjust(wspace=0.04, hspace=0.15)

    for i, sigma in enumerate(sigmas):
        for j, gamma in enumerate(gammas):
            proc = apply_clahe_unsharp(crop, sigma=sigma, gamma=gamma, clip_limit=1.5)
            proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            ax = axes[i, j]
            ax.imshow(proc_rgb)
            ax.set_title(f"$\\sigma = {sigma}$, $\\gamma = {gamma}$", fontsize=12, fontweight="bold", color="navy")
            ax.axis("off")

    fig.suptitle(
        "CLAHE + Unsharp Mask Parameter Sweep (σ: Spatial Scale vs γ: Edge Boost Strength)\n"
        "Observed on dense 0402 SMD capacitor solder pads & FR-4 substrate",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_dirs = [Path("presentation_assets"), Path("runs/report_assets")]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        out_file = d / "clahe_sharpening_grid_sweep.png"
        fig.savefig(str(out_file), bbox_inches="tight", dpi=150)
        print(f"✅ Saved visual grid to: {out_file}")

    plt.close()


if __name__ == "__main__":
    main()
