#!/bin/bash
#SBATCH --job-name=mf_yolo26s_640
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_mf_640_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

python matched_filter_yolo26.py \
    --run-key rectified_yolov26s_matched_filter_640 \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001
