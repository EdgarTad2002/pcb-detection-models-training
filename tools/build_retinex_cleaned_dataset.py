#!/usr/bin/env python3
"""
tools/build_retinex_cleaned_dataset.py

Generates the Retinex-rectified PCB dataset using Fu et al. (CVPR 2016)
weighted variational decomposition.
Normalizes illumination, eliminates spotlight glare on solder joints,
and reveals shadowed micro-capacitors across all splits.

Usage:
    python tools/build_retinex_cleaned_dataset.py \
        --src-dir datasets/pcb-unified-4class-cleaned \
        --dst-dir datasets/pcb-unified-4class-cleaned-retinex \
        --device cuda \
        --c1 0.01 --c2 0.1 --lambd 1.0 --gamma 2.2 --max-iter 15
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weighted_variational_retinex import (
    process_rgb_batch,
    process_rgb_image,
)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> Optional[str]:
    """Retrieves current git commit hash if in a git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def verify_or_write_params(
    dst_path: Path,
    current_params: dict,
    overwrite: bool = False,
) -> None:
    """
    Checks params.json in dst_path against current_params.
    If params.json exists and differs:
      - If overwrite is False: raises RuntimeError with diff details.
      - If overwrite is True: warns and updates params.json.
    """
    params_file = dst_path / "params.json"
    if params_file.exists():
        try:
            with open(params_file, "r") as f:
                existing_params = json.load(f)
        except Exception:
            existing_params = {}

        compare_keys = ["c1", "c2", "lambd", "gamma", "max_iter", "eps1", "eps2", "retinex_script_sha256"]
        mismatches = {}
        for k in compare_keys:
            if existing_params.get(k) != current_params.get(k):
                mismatches[k] = (existing_params.get(k), current_params.get(k))

        if mismatches:
            msg = (
                f"Existing dataset parameters in {params_file} do not match the current run:\n"
                + "\n".join([f"  - {k}: existing={v[0]} vs current={v[1]}" for k, v in mismatches.items()])
                + "\nUse --overwrite to recompute and overwrite existing outputs."
            )
            if not overwrite:
                raise RuntimeError(msg)
            else:
                print(f"⚠️  WARNING: Dataset parameters mismatch detected! Overwrite is enabled. Updating {params_file}...")

    # Write / update params.json
    with open(params_file, "w") as f:
        json.dump(current_params, f, indent=2)


def build_retinex_dataset(
    src_dir: str = "datasets/pcb-unified-4class-cleaned",
    dst_dir: str = "datasets/pcb-unified-4class-cleaned-retinex",
    device: str = "cuda",
    c1: float = 0.01,
    c2: float = 0.1,
    lambd: float = 1.0,
    gamma: float = 2.2,
    max_iter: int = 15,
    eps1: float = 1e-3,
    eps2: float = 1e-3,
    batch_size: int = 1,
    overwrite: bool = False,
):
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_path}")

    # Hash the core Retinex decomposition engine script
    script_path = Path(__file__).resolve().parent / "weighted_variational_retinex.py"
    script_hash = compute_file_sha256(script_path) if script_path.exists() else "unknown"
    git_commit = get_git_commit()

    current_params = {
        "c1": c1,
        "c2": c2,
        "lambd": lambd,
        "gamma": gamma,
        "max_iter": max_iter,
        "eps1": eps1,
        "eps2": eps2,
        "retinex_script_sha256": script_hash,
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    dst_path.mkdir(parents=True, exist_ok=True)
    verify_or_write_params(dst_path, current_params, overwrite=overwrite)

    compute_device = torch.device("cuda" if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    print("=" * 70)
    print("🚀 BUILDING RETINEX-ENHANCED CLEANED PCB DATASET (Fu et al., CVPR 2016)")
    print("=" * 70)
    print(f"📁 Source Dataset:      {src_path}")
    print(f"📁 Destination Dataset: {dst_path}")
    print(f"⚡ Compute Device:       {compute_device} ({torch.cuda.get_device_name(0) if compute_device.type == 'cuda' else 'CPU'})")
    print(f"🔄 ADMM Max Iterations: {max_iter}")
    print(f"🎯 Penalties & Priors:  c1={c1}, c2={c2}, lambda={lambd}")
    print(f"☀️  Illumination Gamma:  {gamma} (Eq. 9 & 11: S_enh = R * L^(1/gamma))")
    print(f"🛑 Stopping Tolerances: eps1={eps1}, eps2={eps2}")
    print(f"📦 Batch Size:          {batch_size}")
    print(f"🔁 Overwrite:           {overwrite}")
    print(f"🔑 Retinex Engine SHA:  {script_hash[:12]}...")
    print("=" * 70)

    splits = ["train", "valid", "test"]
    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for split in splits:
        src_img_dir = src_path / split / "images"
        src_lbl_dir = src_path / split / "labels"
        dst_img_dir = dst_path / split / "images"
        dst_lbl_dir = dst_path / split / "labels"

        if not src_img_dir.exists():
            continue

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy label files directly (labels remain unchanged)
        if src_lbl_dir.exists():
            for lbl_p in src_lbl_dir.glob("*.txt"):
                shutil.copy2(lbl_p, dst_lbl_dir / lbl_p.name)

        # Remove stale label caches
        for cfile in dst_lbl_dir.glob("*.cache"):
            cfile.unlink()
        if (dst_path / split / "labels.cache").exists():
            (dst_path / split / "labels.cache").unlink()

        # 2. Process images with Weighted Variational Retinex
        image_files = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        print(f"\nProcessing '{split}' split: {len(image_files)} images...")

        split_processed = 0
        split_skipped = 0

        # Filter out images that are already processed and up-to-date
        pending_files = []
        for img_p in image_files:
            dst_img_p = dst_img_dir / img_p.name
            if dst_img_p.exists() and dst_img_p.stat().st_size > 1000 and not overwrite:
                split_skipped += 1
                total_skipped += 1
            else:
                pending_files.append(img_p)

        if not pending_files:
            print(f"  [{split}] All {len(image_files)} images are up to date (skipped).")
            continue

        t0 = time.time()

        if batch_size > 1:
            # Batch processing for images
            idx = 0
            while idx < len(pending_files):
                chunk_files = pending_files[idx : idx + batch_size]
                loaded_imgs = []
                valid_files = []

                for p in chunk_files:
                    bgr = cv2.imread(str(p))
                    if bgr is not None:
                        loaded_imgs.append(bgr)
                        valid_files.append(p)
                    else:
                        total_errors += 1

                if loaded_imgs:
                    try:
                        enhanced_bgrs = process_rgb_batch(
                            loaded_imgs,
                            device=compute_device,
                            c1=c1,
                            c2=c2,
                            lambd=lambd,
                            gamma=gamma,
                            max_iter=max_iter,
                            eps1=eps1,
                            eps2=eps2,
                        )
                        for f_p, enh_bgr in zip(valid_files, enhanced_bgrs):
                            dst_p = dst_img_dir / f_p.name
                            cv2.imwrite(str(dst_p), enh_bgr)
                            split_processed += 1
                            total_processed += 1
                    except Exception as e:
                        print(f"⚠️ Batch processing error: {e}. Falling back to single-image processing.")
                        for f_p in valid_files:
                            dst_p = dst_img_dir / f_p.name
                            try:
                                bgr = cv2.imread(str(f_p))
                                _, _, enh = process_rgb_image(
                                    bgr,
                                    device=compute_device,
                                    c1=c1,
                                    c2=c2,
                                    lambd=lambd,
                                    gamma=gamma,
                                    max_iter=max_iter,
                                    eps1=eps1,
                                    eps2=eps2,
                                )
                                cv2.imwrite(str(dst_p), cv2.cvtColor(enh, cv2.COLOR_RGB2BGR))
                                split_processed += 1
                                total_processed += 1
                            except Exception as err:
                                print(f"⚠️ Error on {f_p.name}: {err}. Copying original.")
                                shutil.copy2(f_p, dst_p)
                                total_errors += 1

                idx += len(chunk_files)
                if split_processed % 25 == 0 or idx >= len(pending_files):
                    elapsed = time.time() - t0
                    fps = split_processed / max(elapsed, 0.001)
                    print(f"  [{split}] {split_processed}/{len(pending_files)} pending images processed ({fps:.1f} imgs/sec)")
        else:
            # Single-image processing
            for idx, img_p in enumerate(pending_files):
                dst_img_p = dst_img_dir / img_p.name
                img_bgr = cv2.imread(str(img_p))
                if img_bgr is None:
                    total_errors += 1
                    continue

                try:
                    _, _, rgb_enhanced = process_rgb_image(
                        img_bgr,
                        device=compute_device,
                        c1=c1,
                        c2=c2,
                        lambd=lambd,
                        gamma=gamma,
                        max_iter=max_iter,
                        eps1=eps1,
                        eps2=eps2,
                    )
                    bgr_enhanced = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(dst_img_p), bgr_enhanced)
                    split_processed += 1
                    total_processed += 1
                except Exception as e:
                    print(f"⚠️ Error processing {img_p.name}: {e}. Copying original as fallback.")
                    shutil.copy2(img_p, dst_img_p)
                    total_errors += 1

                if (idx + 1) % 25 == 0 or (idx + 1) == len(pending_files):
                    elapsed = time.time() - t0
                    fps = (idx + 1) / max(elapsed, 0.001)
                    print(f"  [{split}] {idx + 1}/{len(pending_files)} pending processed ({fps:.1f} imgs/sec)")

    # 3. Create clean data.yaml
    src_yaml = src_path / "data.yaml"
    dst_yaml = dst_path / "data.yaml"
    if src_yaml.exists():
        with open(src_yaml, "r") as f:
            yaml_content = f.read()
        yaml_content = yaml_content.replace(src_path.name, dst_path.name)
        with open(dst_yaml, "w") as f:
            f.write(yaml_content)

    print("\n" + "=" * 70)
    print("🎉 RETINEX DATASET PREPARATION COMPLETE!")
    print(f"📁 Dataset location:    {dst_path}")
    print(f"🖼️  Processed images:    {total_processed}")
    print(f"⏭️  Skipped images:      {total_skipped}")
    if total_errors > 0:
        print(f"⚠️  Errors (fallbacks):  {total_errors}")
    print(f"📄 Parameters recorded: {dst_path / 'params.json'}")
    print(f"📄 Configuration YAML:  {dst_yaml}")
    print("=" * 70 + "\n")


def main():
    p = argparse.ArgumentParser(description="Build Retinex-enhanced cleaned PCB dataset (Fu et al., CVPR 2016)")
    p.add_argument("--src-dir", type=str, default="datasets/pcb-unified-4class-cleaned")
    p.add_argument("--dst-dir", type=str, default="datasets/pcb-unified-4class-cleaned-retinex")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--c1", type=float, default=0.01, help="Reflection penalty c1 (default: 0.01)")
    p.add_argument("--c2", type=float, default=0.1, help="Illumination penalty c2 (default: 0.1)")
    p.add_argument("--lambd", type=float, default=1.0, help="ADMM regularizer lambda (default: 1.0)")
    p.add_argument("--gamma", type=float, default=2.2, help="Illumination gamma correction exponent (default: 2.2)")
    p.add_argument("--max-iter", type=int, default=15, help="Max ADMM iterations (default: 15)")
    p.add_argument("--eps1", type=float, default=1e-3, help="Reflectance convergence tolerance (default: 1e-3)")
    p.add_argument("--eps2", type=float, default=1e-3, help="Illumination convergence tolerance (default: 1e-3)")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for same-resolution images (default: 1)")
    p.add_argument("--overwrite", action="store_true", help="Force overwrite existing dataset and params.json")
    args = p.parse_args()

    build_retinex_dataset(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        device=args.device,
        c1=args.c1,
        c2=args.c2,
        lambd=args.lambd,
        gamma=args.gamma,
        max_iter=args.max_iter,
        eps1=args.eps1,
        eps2=args.eps2,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
