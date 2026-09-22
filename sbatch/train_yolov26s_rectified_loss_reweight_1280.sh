#!/bin/bash
#SBATCH --job-name=yolo26s_rect1280_loss
#SBATCH --partition=research
#SBATCH --mem=48G
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

# Train YOLO26s with Loss Reweighting on the Rectified Unified 4-Class Dataset (1280px Native Resolution)
# Boost classification loss (cls=1.5) and tune box loss (box=5.0)
python train.py \
    --run-key yolov26s_rectified_loss_reweight_1280 \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --cls 1.5 \
    --box 5.0 \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8
