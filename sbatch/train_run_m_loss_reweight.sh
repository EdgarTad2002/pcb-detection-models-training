#!/bin/bash
#SBATCH --job-name=yolo26s_loss_reweight
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

# Run Research #3: Loss Reweighting on standard pcb-filtered-yolov8 dataset (no extra augmentation)
# Boost classification loss (cls=1.5), tune box loss (box=5.0), enable Focal Loss (fl_gamma=1.5)
python train.py \
    --run-key yolov26s_loss_reweight \
    --weights yolo26s.pt \
    --data datasets/pcb-filtered-yolov8/data.yaml \
    --cls 1.5 \
    --box 5.0 \
    --fl-gamma 1.5 \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
