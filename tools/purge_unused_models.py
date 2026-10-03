#!/usr/bin/env python3
"""
purge_unused_models.py
======================
Permanently purges redundant duplicates, buggy runs, and sub-optimal models
both locally and on the remote cluster (YSU).

Preserves:
- All generational rectified YOLOs (v5s, v8s, v9s, v10s, v11s, v12s, v26s)
- Top Champions: superyolo26s_rectified_1280, rectified_yolov26s_unified_1280,
  yolov26s_native_res_rectified_1280, yolov26s_native_res_rectified_loss_reweight_1280
- Winning CLAHE: yolov26s_rectified_clahe_s1.0_g0.6_640
- Best Native Spectral: rectified_yolov26s_native_res_physics_spectral_a0.0_1280
- SAHI Champions: sahi_hybrid_1280_640, sahi_hybrid_dual_640, sahi_tile_640
- Omni-Scale Champion: omni_scale_champion_640
- Visual Mamba models (as requested by user): vmamba_standalone_640_v3, vmamba_standalone_1280_v3
- Ongoing clean pretrained LUMA-YOLO models
"""

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. Results JSON files to delete
RESULTS_TO_DELETE = [
    # Category 1: Exact Duplicates & Redundant Baseline Alias Files
    "yolov26s_rectified_640.json",
    "yolov26s_rectified_loss_reweight.json",
    "yolov26s_rectified_1280.json",
    "yolov26s_rectified_loss_reweight_1280.json",
    "superyolo26s_rectified_640.json",

    # Category 2: Buggy & Dead-End Architectures
    "rectified_yolov26s_luma_pcb_640.json",  # Flawed un-pretrained run (0.3849 mAP)
    "rectified_yolov26s_physics_spectral_linear_a0.25_640.json",  # Linear interpolation bug
    "rectified_yolov26s_physics_spectral_a0.25_640.json",  # 640px spectral downsampling artifact
    "kd_distill_640.json",  # Neck feature MSE distillation (13.6s/img, 0.0 gain)
    "kd_distill_640_ablation.json",
    "rectified_yolov26s_p2_head.json",  # Stride-4 head dropped mAP to 0.5156 (-6.56 pp)
    "rectified_yolov26s_focal_loss_640.json",  # Class gradient starvation dropped mAP to 0.5511
    "rectified_yolov26s_bbox_cappaste.json",  # Copy-paste artifacts dropped mAP to 0.5248
    "rectified_yolov26s_bbox_cappaste_with_loss_reweight.json",
    "rectified_yolov26s_bbox_cappaste_640matched.json",
    "yolov26s_rectified_bilateral_640.json",  # Bilateral filter blurred micro-caps
    "yolov26s_rectified_bilateral_1280.json",
    "true_superyolo26s_rectified_640.json",  # Sub-pixel SR ablation dead-end
    "true_superyolo26s_ablation_no_sr.json",

    # Category 3: Sub-optimal Hyperparameter Sweeps
    "yolov26s_rectified_clahe_s0.5_g0.3_640.json",
    "yolov26s_rectified_clahe_s1.0_g0.3_640.json",
    "yolov26s_rectified_clahe_s1.5_g0.4_640.json",
    "rectified_yolov26s_native_res_physics_spectral_a0.25_1280.json",
    "rectified_yolov26s_native_res_physics_spectral_a0.50_1280.json",
    "rectified_yolov26s_native_res_physics_spectral_a1.00_1280.json",
]

# 2. Run directories to delete
RUNS_TO_DELETE = [
    "yolov26s_rectified_640",
    "yolov26s_rectified_loss_reweight",
    "yolov26s_rectified_1280",
    "yolov26s_rectified_loss_reweight_1280",
    "superyolo26s_rectified_640",
    "rectified_yolov26s_luma_pcb_640",
    "rectified_yolov26s_physics_spectral_linear_a0.25_640",
    "rectified_yolov26s_physics_spectral_a0.25_640",
    "kd_distill_640",
    "kd_distill_640_ablation",
    "rectified_yolov26s_p2_head",
    "rectified_yolov26s_focal_loss_640",
    "rectified_yolov26s_bbox_cappaste",
    "rectified_yolov26s_bbox_cappaste_with_loss_reweight",
    "rectified_yolov26s_bbox_cappaste_640matched",
    "yolov26s_rectified_bilateral_640",
    "yolov26s_rectified_bilateral_1280",
    "true_superyolo26s_rectified_640",
    "true_superyolo26s_ablation_no_sr",
    "yolov26s_rectified_clahe_s0.5_g0.3_640",
    "yolov26s_rectified_clahe_s1.0_g0.3_640",
    "yolov26s_rectified_clahe_s1.5_g0.4_640",
    "rectified_yolov26s_native_res_physics_spectral_a0.25_1280",
    "rectified_yolov26s_native_res_physics_spectral_a0.50_1280",
    "rectified_yolov26s_native_res_physics_spectral_a1.00_1280",
]

# 3. Specific obsolete sbatch scripts to delete
SCRIPTS_TO_DELETE = [
    "sbatch/train_yolov26s_rectified_640.sh",
    "sbatch/train_yolov26s_rectified_1280.sh",
    "sbatch/train_yolov26s_rectified_loss_reweight_640.sh",
    "sbatch/train_yolov26s_rectified_loss_reweight_1280.sh",
    "sbatch/train_superyolo26s_rectified.sh",
    "sbatch/train_kd_distill_640.sh",
    "sbatch/train_kd_distill_640_ablation.sh",
    "sbatch/train_rectified_yolov26s_p2_head.sh",
    "sbatch/train_rectified_yolov26s_focal_loss_640.sh",
    "sbatch/train_rectified_yolov26s_bbox_cappaste.sh",
    "sbatch/train_rectified_yolov26s_cappaste_640matched.sh",
    "sbatch/train_yolov26s_rectified_bilateral_640.sh",
    "sbatch/train_yolov26s_rectified_bilateral_1280.sh",
    "sbatch/train_true_superyolo26s_rectified.sh",
    "sbatch/train_true_superyolo26s_ablation.sh",
    "sbatch/train_rectified_physics_spectral.sh",
]

def run_remote(cmd):
    full_cmd = f"ssh etadevosyan@cluster.ysu.am '{cmd}'"
    print(f"📡 Remote execution: {cmd}")
    res = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0 and "No such file" not in res.stderr:
        print(f"⚠️  Remote warning: {res.stderr.strip()}")
    return res

def main():
    print("=" * 70)
    print("🧹 Purging unused, buggy, and sub-optimal models (Keeping Mamba)")
    print("=" * 70)

    # 1. Delete Results JSONs locally
    del_res = 0
    for r in RESULTS_TO_DELETE:
        p = ROOT / "results" / r
        if p.exists():
            p.unlink()
            print(f"🗑️  [Local] Deleted result: results/{r}")
            del_res += 1
    print(f"✅ [Local] Deleted {del_res} results JSON files.")

    # 2. Delete Runs directories locally
    del_runs = 0
    for run in RUNS_TO_DELETE:
        p = ROOT / "runs" / run
        if p.exists():
            shutil.rmtree(p)
            print(f"🗑️  [Local] Deleted run dir: runs/{run}")
            del_runs += 1
    print(f"✅ [Local] Deleted {del_runs} run directories.")

    # 3. Delete obsolete sbatch scripts locally
    del_scripts = 0
    for s in SCRIPTS_TO_DELETE:
        p = ROOT / s
        if p.exists():
            p.unlink()
            print(f"🗑️  [Local] Deleted script: {s}")
            del_scripts += 1
    print(f"✅ [Local] Deleted {del_scripts} scripts.")

    # 4. Remote cleanup on YSU cluster
    print("\n🌐 Synchronizing deletion with remote cluster...")
    remote_res_paths = " ".join([f"/mnt/weka/etadevosyan/pcb-yolo/results/{r}" for r in RESULTS_TO_DELETE] +
                                [f"/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/results/{r}" for r in RESULTS_TO_DELETE])
    run_remote(f"rm -f {remote_res_paths}")

    remote_run_paths = " ".join([f"/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/runs/{r}" for r in RUNS_TO_DELETE])
    run_remote(f"rm -rf {remote_run_paths}")

    remote_script_paths = " ".join([f"/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/{s}" for s in SCRIPTS_TO_DELETE])
    run_remote(f"rm -f {remote_script_paths}")

    print("\n📊 Regenerating comparison table locally...")
    subprocess.run(f"python3 {ROOT}/aggregate_results.py --results-dir {ROOT}/results", shell=True)
    subprocess.run(f"cp {ROOT}/results/comparison_table.* {ROOT}/", shell=True)

    print("\n📊 Regenerating comparison table remotely...")
    run_remote("python3 /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/aggregate_results.py --results-dir /mnt/weka/etadevosyan/pcb-yolo/results")
    run_remote("cp /mnt/weka/etadevosyan/pcb-yolo/results/comparison_table.* /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/")

    print("\n🎉 Purge complete! Publication-grade leaderboard updated.")

if __name__ == "__main__":
    main()
