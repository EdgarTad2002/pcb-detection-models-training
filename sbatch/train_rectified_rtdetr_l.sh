#!/bin/bash
#SBATCH --job-name=rect_rtdetr_l
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

# RT-DETR-L (Real-Time Detection Transformer) on rectified 4-class dataset
python train.py \
    --run-key rectified_rtdetr_l \
    --weights rtdetr-l.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 --imgsz 640 --batch 8 --workers 8 \
    --eval-conf 0.001
