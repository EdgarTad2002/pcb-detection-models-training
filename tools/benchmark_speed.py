#!/usr/bin/env python3
"""
Dedicated Speed Benchmarking Suite for PCB YOLO Models.

Runs all candidate checkpoints on the EXACT SAME GPU under controlled conditions:
1. GPU warmup (to stabilize CUDA context and GPU boost clocks).
2. RAM pre-caching of inputs (eliminates cluster disk I/O noise).
3. PyTorch CUDA event synchronization (torch.cuda.synchronize / torch.cuda.Event).
4. Measures pure inference latency, end-to-end latency, and consistent FPS.
5. Optionally updates results/*.json so comparison tables have consistent speed numbers.

Usage:
    python tools/benchmark_speed.py --device 0 --update-results
"""

from typing import Any
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is required. Run: pip install ultralytics")
    sys.exit(1)


# Baseline and ablation models to benchmark
CANDIDATE_MODELS = [
    # 1. Standard 640px Baselines
    ("yolov5s", "runs/yolov5s/pcb-filtered/weights/best.pt", 640),
    ("yolov8s", "runs/yolov8s/pcb-filtered/weights/best.pt", 640),
    ("yolov9s", "runs/yolov9s/pcb-filtered/weights/best.pt", 640),
    ("yolov10s", "runs/yolov10s/pcb-filtered/weights/best.pt", 640),
    ("yolov11s", "runs/yolov11s/pcb-filtered/weights/best.pt", 640),
    ("yolov12s", "runs/yolov12s/pcb-filtered/weights/best.pt", 640),
    ("yolov26s", "runs/yolov26s/pcb-filtered/weights/best.pt", 640),

    # 2. Fair 640px Techniques on YOLO26
    ("yolov26s_loss_reweight", "runs/yolov26s_loss_reweight/pcb-filtered/weights/best.pt", 640),
    ("yolov26s_bbox_cappaste", "runs/yolov26s_bbox_cappaste/pcb-filtered/weights/best.pt", 640),
    ("yolov26s_spectral_640", "runs/yolov26s_spectral_640/pcb-filtered/weights/best.pt", 640),
    ("yolov26s_physics_spectral_640", "runs/yolov26s_physics_spectral_640/pcb-filtered/weights/best.pt", 640),
    ("yolov26s_geo_hsv_aug", "runs/yolov26s_geo_hsv_aug/pcb-filtered/weights/best.pt", 640),
    ("yolov26s_p2_combined_v2", "runs/yolov26s_p2_combined_v2/pcb-filtered/weights/best.pt", 640),

    # 3. Native-Resolution Discussion Ablations
    ("yolov26s_native_res", "runs/yolov26s_native_res/pcb-filtered/weights/best.pt", 1280),
    ("yolov26s_focal_loss_native", "runs/yolov26s_focal_loss_native/pcb-filtered/weights/best.pt", 1280),
]


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark model inference speed on a single GPU")
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("PCB_PROJECT_ROOT", ".")),
        help="Project root directory",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory to update (defaults to <project-root>/results)",
    )
    p.add_argument(
        "--test-images-dir",
        type=Path,
        default=None,
        help="Path to directory with real test images",
    )
    p.add_argument("--device", default="0", help="CUDA device index, e.g. 0 or cpu")
    p.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    p.add_argument("--num-iters", type=int, default=100, help="Number of timed benchmark passes")
    p.add_argument(
        "--update-results",
        action="store_true",
        help="Update results/<model>.json with the newly benchmarked speed and FPS",
    )
    return p.parse_args()


def get_benchmark_inputs(
    images_dir: Optional[Path], imgsz: int, num_samples: int, device: str
) -> List[Any]:
    """Loads pre-cached images in RAM or creates dummy tensors if images are unavailable."""
    sample_inputs = []
    if images_dir and images_dir.exists():
        img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        if img_files:
            print(f"Loading {min(len(img_files), num_samples)} real test images into RAM...")
            for p in img_files[:num_samples]:
                with Image.open(p) as im:
                    sample_inputs.append(im.convert("RGB"))
            return sample_inputs

    # Fallback to in-memory tensor inputs
    print(f"Using {num_samples} in-memory synthetic inputs ({imgsz}x{imgsz}) to test pure latency...")
    is_cuda = device != "cpu" and torch.cuda.is_available()
    dev = torch.device(f"cuda:{device}" if is_cuda and device.isdigit() else device)
    for _ in range(num_samples):
        sample_inputs.append(torch.zeros((1, 3, imgsz, imgsz), device=dev))
    return sample_inputs


def benchmark_single_model(
    model: YOLO,
    inputs: List[Any],
    imgsz: int,
    warmup: int,
    device: str,
) -> Tuple[float, float, float]:
    """
    Benchmarks inference latency and FPS with CUDA event timing.

    Returns:
        (pure_inference_ms, total_e2e_ms, fps)
    """
    is_cuda = device != "cpu" and torch.cuda.is_available()

    # 1. Warmup
    for i in range(warmup):
        inp = inputs[i % len(inputs)]
        _ = model.predict(inp, imgsz=imgsz, conf=0.25, verbose=False)

    if is_cuda:
        torch.cuda.synchronize()

    # 2. Timed inference
    latencies_ms = []
    for inp in inputs:
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            res = model.predict(inp, imgsz=imgsz, conf=0.25, verbose=False)[0]

            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_event.elapsed_time(end_event))
        else:
            t0 = time.perf_counter()
            res = model.predict(inp, imgsz=imgsz, conf=0.25, verbose=False)[0]
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_e2e_ms = float(np.mean(latencies_ms))
    # Ultralytics internal inference time (pure forward pass without NMS/resizing)
    pure_inf_ms = getattr(res, "speed", {}).get("inference", avg_e2e_ms)

    fps = 1000.0 / avg_e2e_ms if avg_e2e_ms > 0 else 0.0
    return pure_inf_ms, avg_e2e_ms, fps


def main():
    args = parse_args()
    results_dir = args.results_dir or (args.project_root / "results")

    # Detect GPU properties
    if torch.cuda.is_available() and args.device != "cpu":
        gpu_idx = int(args.device) if args.device.isdigit() else 0
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        print(f"\n======================================================================")
        print(f"🚀 Benchmarking Environment: GPU {gpu_idx} - {gpu_name}")
        print(f"   PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
        print(f"   Warmup: {args.warmup} passes | Timed: {args.num_iters} passes")
        print(f"======================================================================\n")
    else:
        gpu_name = "CPU"
        print(f"\n🚀 Benchmarking Environment: CPU\n")

    # Locate test images if available
    test_img_dir = args.test_images_dir
    if test_img_dir is None:
        candidate_paths = [
            args.project_root / "datasets" / "pcb-filtered-yolov8" / "test" / "images",
            args.project_root / "datasets" / "pcb-filtered-yolov8" / "valid" / "images",
            Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/test/images"),
        ]
        for cp in candidate_paths:
            if cp.exists():
                test_img_dir = cp
                break

    benchmark_records = []

    print(f"{'Model':<30} {'Resolution':<12} {'Pure Inf (ms)':<15} {'Total Latency':<15} {'FPS':<10}")
    print("-" * 85)

    for run_key, rel_weight_path, default_imgsz in CANDIDATE_MODELS:
        weight_path = args.project_root / rel_weight_path
        if not weight_path.exists():
            # Check cluster path if not in current project root
            alt_path = Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training") / rel_weight_path
            if alt_path.exists():
                weight_path = alt_path
            else:
                continue

        try:
            model = YOLO(str(weight_path))
            inputs = get_benchmark_inputs(test_img_dir, default_imgsz, args.num_iters, args.device)
            pure_ms, e2e_ms, fps = benchmark_single_model(
                model, inputs, default_imgsz, args.warmup, args.device
            )

            print(f"{run_key:<30} {default_imgsz:<12} {pure_ms:<15.2f} {e2e_ms:<15.2f} {fps:<10.1f}")

            record = {
                "model": run_key,
                "weights": str(weight_path),
                "imgsz": default_imgsz,
                "gpu": gpu_name,
                "pure_inference_ms": round(pure_ms, 2),
                "total_time_ms": round(e2e_ms, 2),
                "fps": round(fps, 1),
            }
            benchmark_records.append(record)

            # Update results/<model>.json if requested
            if args.update_results:
                target_json = results_dir / f"{run_key}.json"
                if target_json.exists():
                    with open(target_json, "r") as f:
                        data = json.load(f)
                    data["total_time_ms"] = round(e2e_ms, 2)
                    data["fps"] = round(fps, 1)
                    data["gpu_benchmark"] = gpu_name
                    with open(target_json, "w") as f:
                        json.dump(data, f, indent=2)

        except Exception as e:
            print(f"❌ Error benchmarking {run_key}: {e}")

    print("\n" + "=" * 85)
    print(f"✅ Successfully benchmarked {len(benchmark_records)} models on {gpu_name}!")

    if args.update_results:
        print(f"📊 Updated speed and FPS entries in: {results_dir}")
        # Run aggregate_results to refresh CSV/Markdown/Excel
        agg_script = args.project_root / "aggregate_results.py"
        if agg_script.exists():
            print("🔄 Refreshing comparison_table.csv, .md, and .xlsx...")
            os.system(f"python3 {agg_script} --results-dir {results_dir} --out-csv {args.project_root}/comparison_table.csv")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
