#!/bin/bash
#SBATCH --job-name=vmamba_tiny
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_vmamba_%j.out

# 1. Configure storage and cache variables
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

# 2. Activate Conda environment from shared cache
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# 3. Navigate to remote workspace
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# 4. Train Standalone VMamba Object Detector (Non-YOLO State Space Model)
echo "=========================================================================="
echo "🚀 Training Standalone VMamba Object Detector on NVIDIA H100 (YSU Cluster)"
echo "=========================================================================="
python train_mamba.py \
    --run-key vmamba_standalone_tiny \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --lr 0.0001 \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50

# 5. Automatically refresh benchmark comparison tables
python aggregate_results.py
