#!/bin/bash
#SBATCH --job-name=run_q_focal_loss
#SBATCH --partition=research
#SBATCH --mem=40G
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

# Train YOLO26s with Distribution Focal Loss (dfl=2.5, cls=2.0, label_smoothing=0.1) on Native 1280px Dataset
python train.py \
    --run-key yolov26s_focal_loss_native \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res/data.yaml \
    --dfl 2.5 --cls 2.0 --label-smoothing 0.1 \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001
