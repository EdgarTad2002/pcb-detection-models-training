#!/bin/bash
#SBATCH --job-name=yolo26s_p2
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_p2_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# Train YOLO26s with P2 High-Resolution Head (Stride 4, 160x160 feature map) for microscopic 0402 SMD capacitors
python train.py \
    --run-key yolov26s_p2_head \
    --weights yolo26s-p2.yaml \
    --pretrained-weights yolo26s.pt \
    --epochs 100 --imgsz 640 --batch 8 --workers 8 \
    --eval-conf 0.001
