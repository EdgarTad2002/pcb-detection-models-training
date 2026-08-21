#!/bin/bash
#SBATCH --job-name=run_e_res960
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# imgsz matched to Run D's 960px so the two are a clean architecture-only
# comparison (Run D = P2 + 960px, Run E = 960px only, no P2) -- the earlier
# 832px result stays under yolov26s_res832_only, untouched, since this uses
# a different run-key.
python train.py \
    --run-key yolov26s_res960_only \
    --weights yolo26s.pt \
    --epochs 100 --imgsz 960 --batch 16 --workers 8
