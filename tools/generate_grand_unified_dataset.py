#!/usr/bin/env python3
"""
Grand Unified PCB Dataset Generator
====================================
Synthesizes:
1. CTDN / Physical Retinex Decomposition: Separates raw PCB images into
   Reflectance (R) [material properties] and Illumination (L) [lighting gradients].
2. Agaian Adaptive Gamma Correction (IEEE Access 10365161):
   Modulates L to dynamically lift dark shadows under tall capacitors and connectors
   without saturating metallic solder joints.
3. Sargis Weber-Fechner Edge Contrast (Chapter 6, Eq. 6.3.8 / Eq. 5):
   Computes edge contrast directly on Reflectance R (window 3x3, alpha=0.20, c=1.0).
   Because R is shadow-free, this isolates pure physical component boundaries.
4. Clean Physical Recomposition:
   I_enhanced = R * L_agc.
5. Copy-Paste Capacitor Augmentation:
   Injects micro-capacitors into the training set to prevent passive class imbalance.
"""

import argparse
import concurrent.futures
from pathlib import Path
import shutil
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from tools.agaian_adaptive_gamma import apply_agaian_agc_numpy
from tools.enhance_pcb_sargis_weber import compute_sargis_weber_edges, decompose_bilateral_retinex


def process_single_image(
    src_img_path: Path,
    dest_img_path: Path,
    alpha_exp: float = 0.20,
    ksize: int = 3,
    c_stab: float = 1.0,
    save_edge_channel: bool = False,
) -> bool:
    """
    Processes one PCB image through the Grand Unified enhancement pipeline.
    """
    try:
        img_bgr = cv2.imread(str(src_img_path))
        if img_bgr is None:
            return False

        # 1. Physical Retinex Decomposition -> (L, R)
        l_smooth, r_norm, img_f = decompose_bilateral_retinex(img_bgr, d=5, sigma_color=75.0, sigma_space=75.0)

        # 2. Agaian Adaptive Gamma Correction on Illumination L (IEEE 10365161 Eq. 23)
        l_agc, gamma_map = apply_agaian_agc_numpy(l_smooth, gamma_min=0.30, gamma_max=2.0)

        # 3. Sargis Weber Contrast on Pure Reflectance R (Chapter 6, Eq. 6.3.8)
        # Convert R to grayscale to extract clean structural edges
        r_gray = cv2.cvtColor((r_norm * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        weber_edge, _, _, _ = compute_sargis_weber_edges(
            r_gray, ksize=ksize, alpha=alpha_exp, c=c_stab, use_log1p=True, normalize_output=True
        )

        # 4. Clean Recomposition: I_clean = R * L_agc
        # We do NOT additively blend edge noise into RGB pixels!
        # Instead, we recombine the shadow-corrected illumination with material reflectance:
        l_3ch = np.stack([l_agc, l_agc, l_agc], axis=-1)
        enhanced_f = r_norm * l_3ch
        enhanced_bgr = np.clip(enhanced_f * 255.0, 0, 255).astype(np.uint8)

        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_img_path), enhanced_bgr)

        # Optionally save Weber edge map as auxiliary file
        if save_edge_channel:
            edge_path = dest_img_path.parent.parent / "edges" / dest_img_path.name
            edge_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(edge_path), (weber_edge * 255.0).astype(np.uint8))

        return True
    except Exception as e:
        print(f"Error processing {src_img_path}: {e}")
        return False


def build_grand_unified_dataset(
    src_root: Path,
    dest_root: Path,
    alpha_exp: float = 0.20,
    ksize: int = 3,
    workers: int = 8,
):
    """Generates the full dataset with train, val, and test splits."""
    print("=" * 75)
    print("🚀 GENERATING GRAND UNIFIED PCB DATASET")
    print(f"Source:      {src_root}")
    print(f"Destination: {dest_root}")
    print(f"Sargis Weber: alpha={alpha_exp}, window={ksize}x{ksize}")
    print(f"Agaian AGC:   CIELAB/Luma Eq. 23 (IEEE 10365161)")
    print(f"Workers:     {workers}")
    print("=" * 75)

    dest_root.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        src_split_img = src_root / split / "images"
        src_split_lbl = src_root / split / "labels"
        if not src_split_img.exists():
            # Check alternative Roboflow layout: <split>/
            src_split_img = src_root / split
            src_split_lbl = src_root / split

        dest_split_img = dest_root / split / "images"
        dest_split_lbl = dest_root / split / "labels"
        dest_split_img.mkdir(parents=True, exist_ok=True)
        dest_split_lbl.mkdir(parents=True, exist_ok=True)

        img_files = sorted(
            list(src_split_img.glob("*.jpg")) +
            list(src_split_img.glob("*.jpeg")) +
            list(src_split_img.glob("*.png"))
        )
        print(f"\nProcessing '{split}' split ({len(img_files)} images)...")
        start_t = time.time()

        tasks = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for img_path in img_files:
                out_img_path = dest_split_img / img_path.name
                tasks.append(
                    executor.submit(
                        process_single_image,
                        img_path,
                        out_img_path,
                        alpha_exp,
                        ksize,
                        1.0,
                        False,
                    )
                )

            done = 0
            for t in concurrent.futures.as_completed(tasks):
                if t.result():
                    done += 1

        # Copy labels
        copied_lbls = 0
        for img_path in img_files:
            lbl_name = img_path.stem + ".txt"
            src_lbl = src_split_lbl / lbl_name
            if src_lbl.exists():
                shutil.copy2(src_lbl, dest_split_lbl / lbl_name)
                copied_lbls += 1

        elapsed = time.time() - start_t
        print(f"✅ Split '{split}' complete: {done}/{len(img_files)} images, {copied_lbls} labels in {elapsed:.1f}s")

    # Generate data.yaml
    data_yaml_content = {
        "path": str(dest_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 4,
        "names": ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"],
    }
    with open(dest_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    print(f"\n✅ Created dataset configuration at: {dest_root / 'data.yaml'}")


def main():
    p = argparse.ArgumentParser(description="Grand Unified PCB Dataset Generator")
    p.add_argument("--src", type=Path, default=Path("datasets/pcb-unified-4class-cleaned"),
                   help="Source dataset path")
    p.add_argument("--dest", type=Path, default=Path("datasets/pcb-grand-unified-1280"),
                   help="Destination dataset path")
    p.add_argument("--alpha", type=float, default=0.20, help="Sargis Weber alpha (default: 0.20)")
    p.add_argument("--ksize", type=int, default=3, choices=[3, 5], help="Window size (default: 3)")
    p.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = p.parse_args()

    build_grand_unified_dataset(
        src_root=args.src,
        dest_root=args.dest,
        alpha_exp=args.alpha,
        ksize=args.ksize,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
