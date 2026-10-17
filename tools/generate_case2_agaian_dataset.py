#!/usr/bin/env python3
"""
Case 2: Agaian Generalized Fractional Retinex Dataset Generator
================================================================
Implements the Generalized Fractional Retinex Formulation:
    I(x, y) = tau * [R(x, y)]^lambda * [L(x, y)]^vartheta
In logarithmic domain:
    ln(I) = ln(tau) + lambda * ln(R) + vartheta * ln(L_AGC)

Where:
  - lambda = 1.20: Contrast expansion on high-frequency material reflectance (R)
  - vartheta = 0.50: Compresses dynamic range of oblique lighting falloff (L)
  - L_AGC: Adaptive Gamma Corrected illumination (Agaian et al., IEEE Access 10365161, Eq. 23)
  - R: Guided by Sargis Weber-Fechner Edge operator (window 3x3, alpha=0.20)
  - tau: Dynamic range exposure factor mapping enhanced signals to [0, 255]
"""

import argparse
import concurrent.futures
from pathlib import Path
import shutil
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from tools.agaian_adaptive_gamma import apply_agaian_agc_numpy
from tools.enhance_pcb_sargis_weber import compute_sargis_weber_edges, decompose_bilateral_retinex


def process_single_image(
    src_img_path: Path,
    dest_img_path: Path,
    lambda_exp: float = 1.20,
    vartheta_exp: float = 0.50,
    alpha_weber: float = 0.20,
    ksize: int = 3,
    weber_weight: float = 0.25,
) -> bool:
    """
    Applies the Agaian Generalized Fractional Retinex transformation to one PCB image.
    """
    try:
        img_bgr = cv2.imread(str(src_img_path))
        if img_bgr is None:
            return False

        # 1. Physical Retinex Decomposition -> (L, R)
        l_smooth, r_norm, _ = decompose_bilateral_retinex(img_bgr, d=5, sigma_color=75.0, sigma_space=75.0)

        # 2. Agaian Adaptive Gamma Correction on Illumination (IEEE Access 10365161 Eq. 23)
        l_agc, _ = apply_agaian_agc_numpy(l_smooth, gamma_min=0.30, gamma_max=2.0)
        l_agc_3ch = np.stack([l_agc, l_agc, l_agc], axis=-1)

        # 3. Sargis Weber Contrast applied to Reflectance R (Chapter 6, Eq. 5, alpha=0.20)
        r_gray = cv2.cvtColor((r_norm * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        weber_edge, _, _, _ = compute_sargis_weber_edges(
            r_gray, ksize=ksize, alpha=alpha_weber, c=1.0, use_log1p=True, normalize_output=True
        )
        weber_edge_3ch = np.stack([weber_edge, weber_edge, weber_edge], axis=-1)

        # Edge-guided reflectance: enhance sub-pixel component edges
        r_guided = r_norm * (1.0 + weber_weight * weber_edge_3ch)
        r_guided = np.clip(r_guided, 1e-4, 1.0)
        l_safe = np.clip(l_agc_3ch, 1e-4, 1.0)

        # 4. Generalized Fractional Retinex Recombination
        # I = tau * (R ^ lambda) * (L ^ vartheta)
        # In log domain: ln(I_unscaled) = lambda * ln(R) + vartheta * ln(L)
        log_i = lambda_exp * np.log(r_guided) + vartheta_exp * np.log(l_safe)
        i_rec = np.exp(log_i)

        # 5. Dynamic exposure factor (tau) with percentile-based robust normalization
        p_low = np.percentile(i_rec, 0.2)
        p_high = np.percentile(i_rec, 99.8)
        if p_high > p_low:
            i_norm = (i_rec - p_low) / (p_high - p_low)
        else:
            i_norm = i_rec
        i_norm = np.clip(i_norm, 0.0, 1.0)
        enhanced_bgr = (i_norm * 255.0).astype(np.uint8)

        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_img_path), enhanced_bgr)
        return True
    except Exception as e:
        print(f"Error processing {src_img_path}: {e}")
        return False


def build_agaian_retinex_dataset(
    src_root: Path,
    dest_root: Path,
    lambda_exp: float = 1.20,
    vartheta_exp: float = 0.50,
    alpha_weber: float = 0.20,
    ksize: int = 3,
    weber_weight: float = 0.25,
    workers: int = 8,
):
    print("=" * 75)
    print("🚀 GENERATING AGAIAN GENERALIZED FRACTIONAL RETINEX DATASET (CASE 2)")
    print(f"Source:       {src_root}")
    print(f"Destination:  {dest_root}")
    print(f"Formula:      I = tau * [R_guided]^{lambda_exp} * [L_AGC]^{vartheta_exp}")
    print(f"Sargis Weber: alpha={alpha_weber}, window={ksize}x{ksize}, weight={weber_weight}")
    print(f"Agaian AGC:   IEEE Access 10365161 Eq. 23 on L")
    print(f"Workers:      {workers}")
    print("=" * 75)

    dest_root.mkdir(parents=True, exist_ok=True)

    # Copy data.yaml if exists
    src_yaml = src_root / "data.yaml"
    dest_yaml = dest_root / "data.yaml"
    if src_yaml.exists():
        with open(src_yaml, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        for split in ["train", "val", "test", "valid"]:
            if split in data_cfg and isinstance(data_cfg[split], str):
                data_cfg[split] = data_cfg[split].replace(str(src_root), str(dest_root))
                data_cfg[split] = data_cfg[split].replace(src_root.name, dest_root.name)
        with open(dest_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)
        print(f"✅ Created {dest_yaml}")

    splits = ["train", "valid", "val", "test"]
    total_imgs = 0
    t0 = time.time()

    for split in splits:
        src_split_img_dir = src_root / split / "images"
        if not src_split_img_dir.exists():
            continue

        dest_split_img_dir = dest_root / split / "images"
        dest_split_lbl_dir = dest_root / split / "labels"
        src_split_lbl_dir = src_root / split / "labels"

        dest_split_img_dir.mkdir(parents=True, exist_ok=True)
        dest_split_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Copy labels
        if src_split_lbl_dir.exists():
            for lbl_file in src_split_lbl_dir.glob("*.txt"):
                shutil.copy2(lbl_file, dest_split_lbl_dir / lbl_file.name)

        # Find all images
        img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(list(src_split_img_dir.glob(ext)))
        img_paths = sorted(set(img_paths))

        print(f"\nProcessing '{split}' split ({len(img_paths)} images)...")
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for src_p in img_paths:
                dest_p = dest_split_img_dir / src_p.name
                tasks.append(
                    executor.submit(
                        process_single_image,
                        src_p,
                        dest_p,
                        lambda_exp,
                        vartheta_exp,
                        alpha_weber,
                        ksize,
                        weber_weight,
                    )
                )

            done = 0
            for fut in concurrent.futures.as_completed(tasks):
                if fut.result():
                    done += 1
                if done % 100 == 0 or done == len(tasks):
                    print(f"  [{split}] {done}/{len(tasks)} processed...", flush=True)

        total_imgs += len(img_paths)

    # Ensure valid / val alias
    if (dest_root / "valid").exists() and not (dest_root / "val").exists():
        try:
            (dest_root / "val").symlink_to(dest_root / "valid", target_is_directory=True)
        except Exception:
            pass

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"✅ Case 2 Dataset Complete! {total_imgs} images processed in {elapsed:.1f}s ({elapsed/max(1, total_imgs):.2f}s/img)")
    print(f"Destination: {dest_root}")
    print("=" * 75)


def main():
    p = argparse.ArgumentParser(description="Generate Case 2 Agaian Generalized Fractional Retinex Dataset")
    p.add_argument("--source", default="datasets/pcb-unified-4class-cleaned", help="Source dataset path")
    p.add_argument("--dest", default="datasets/pcb-agaian-retinex-1280", help="Destination path")
    p.add_argument("--lambda-exp", type=float, default=1.20, help="Reflectance power factor lambda")
    p.add_argument("--vartheta-exp", type=float, default=0.50, help="Illumination power factor vartheta")
    p.add_argument("--alpha-weber", type=float, default=0.20, help="Sargis Weber exponent")
    p.add_argument("--ksize", type=int, default=3, help="Sargis Weber filter window size")
    p.add_argument("--weber-weight", type=float, default=0.25, help="Weber edge injection weight into R")
    p.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    args = p.parse_args()

    build_agaian_retinex_dataset(
        src_root=Path(args.source),
        dest_root=Path(args.dest),
        lambda_exp=args.lambda_exp,
        vartheta_exp=args.vartheta_exp,
        alpha_weber=args.alpha_weber,
        ksize=args.ksize,
        weber_weight=args.weber_weight,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
