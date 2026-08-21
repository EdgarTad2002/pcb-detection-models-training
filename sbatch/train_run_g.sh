#!/bin/bash
#SBATCH --job-name=run_g_geo_hsv
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

python train.py \
    --run-key yolov26s_geo_hsv_aug \
    --weights yolo26s.pt \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --scale 0.8 --translate 0.1 --degrees 10.0 --shear 0.0 --perspective 0.0 \
    --hsv-h 0.015 --hsv-s 0.7 --hsv-v 0.4 \
    --mosaic 1.0 --close-mosaic 15 \
    --fliplr 0.5 --flipud 0.0
