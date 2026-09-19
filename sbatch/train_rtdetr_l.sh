#!/bin/bash
#SBATCH --job-name=rtdetr_l
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_rtdetr_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# Train RT-DETR-L (Real-Time Detection Transformer) with Deformable Attention for anchor-free global context
python train.py \
    --run-key rtdetr_l_pcb \
    --weights rtdetr-l.pt \
    --epochs 100 --imgsz 640 --batch 8 --workers 8 \
    --eval-conf 0.001
