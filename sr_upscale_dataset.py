#!/usr/bin/env python3
"""
Strategy A: Super-Resolution Preprocessing for PCB Detection.

Upscales the training split of a PCB dataset using Real-ESRGAN (x2 model),
then writes a new data.yaml pointing to the SR-upscaled training images while
keeping val/test splits pointed at the original source (so evaluation is fair
and consistent with all other runs).

Labels are copied unchanged -- YOLO labels are normalized [0,1] coordinates
and are therefore resolution-independent. No label modification needed.

Only the train split is upscaled. val/test are left untouched to preserve
the standard evaluation protocol used across all runs in this project.

Run build once before training:
    python sr_upscale_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest   datasets/pcb-sr-2x

Then train with:
    python train.py \
        --run-key yolov26s_sr_preprocess \
        --weights yolo26s.pt \
        --data    datasets/pcb-sr-2x/data.yaml \
        --imgsz 1280 --batch 8 --epochs 100 --workers 8

Dependencies (install once in your conda env):
    pip install basicsr facexlib gfpgan
    pip install git+https://github.com/xinntao/Real-ESRGAN.git
"""

import argparse
import shutil
from pathlib import Path

import cv2
import yaml

# ---------------------------------------------------------------------------
# Lazy-import Real-ESRGAN so the script gives a clean error if not installed
# instead of a cryptic AttributeError deep inside the library.
# ---------------------------------------------------------------------------
def _load_realesrgan(device):
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise ImportError(
            "Real-ESRGAN is not installed. Run:\n"
            "  pip install basicsr\n"
            "  pip install git+https://github.com/xinntao/Real-ESRGAN.git"
        )

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        model=model,
        tile=256,          # process in tiles to avoid OOM on large PCB images
        tile_pad=10,
        pre_pad=0,
        half=True,         # fp16 for speed; set False if GPU lacks fp16 support
        device=device,
    )
    return upsampler


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline SR upscaling of PCB training images via Real-ESRGAN x2."
    )
    p.add_argument(
        "--source", type=Path, required=True,
        help="Dataset root containing train/images, train/labels, data.yaml"
    )
    p.add_argument(
        "--dest", type=Path, required=True,
        help="Output dataset root for SR-upscaled training images"
    )
    p.add_argument(
        "--device", default="cuda",
        help="Device for Real-ESRGAN inference: 'cuda' or 'cpu' (default: cuda)"
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip images that already exist in --dest (for resuming interrupted runs)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    src_img_dir = args.source / "train" / "images"
    src_lbl_dir = args.source / "train" / "labels"
    dst_img_dir = args.dest  / "train" / "images"
    dst_lbl_dir = args.dest  / "train" / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
    )
    assert img_files, f"No images found in {src_img_dir}"
    print(f"Found {len(img_files)} training images in {src_img_dir}")

    print("Loading Real-ESRGAN x2 model...")
    upsampler = _load_realesrgan(args.device)
    print("Model ready.")

    n_processed = 0
    n_skipped = 0

    for img_path in img_files:
        out_path = dst_img_dir / f"{img_path.stem}.png"

        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [WARN] Could not read {img_path}, skipping.")
            continue

        # Real-ESRGAN expects BGR numpy array; returns BGR numpy array
        sr_img, _ = upsampler.enhance(img, outscale=2)
        cv2.imwrite(str(out_path), sr_img)

        # Copy label file unchanged (YOLO coords are normalized, resolution-agnostic)
        lbl_src = src_lbl_dir / f"{img_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy(lbl_src, dst_lbl_dir / lbl_src.name)

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  Processed {n_processed}/{len(img_files)}...")

    print(f"\nDone. Upscaled {n_processed} images ({n_skipped} skipped).")

    # -----------------------------------------------------------------------
    # Write new data.yaml:
    #   train -> our SR-upscaled images
    #   val   -> original source val (absolute path, evaluation stays fair)
    #   test  -> original source test (absolute path)
    # -----------------------------------------------------------------------
    src_yaml_path = args.source / "data.yaml"
    assert src_yaml_path.exists(), f"data.yaml not found at {src_yaml_path}"
    with open(src_yaml_path) as f:
        src_cfg = yaml.safe_load(f)

    new_cfg = {
        "path":  str(args.dest.resolve()),
        "train": "train/images",
        "val":   str((args.source / "valid" / "images").resolve()),
        "test":  str((args.source / "test"  / "images").resolve()),
        "nc":    src_cfg["nc"],
        "names": src_cfg["names"],
    }
    with open(args.dest / "data.yaml", "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)

    print(f"New data.yaml written to: {args.dest / 'data.yaml'}")
    print("\nNext step — submit training job:")
    print("  python train.py \\")
    print(f"      --run-key yolov26s_sr_preprocess \\")
    print(f"      --weights yolo26s.pt \\")
    print(f"      --data {args.dest / 'data.yaml'} \\")
    print("      --imgsz 1280 --batch 8 --epochs 100 --workers 8")


if __name__ == "__main__":
    main()
