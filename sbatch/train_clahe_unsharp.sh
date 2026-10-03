#!/bin/bash
#SBATCH --job-name=yolo26s_clahe_unsharp
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_clahe_%j.out

# ==============================================================================
# Parameterized CLAHE + Unsharp-Masked YOLO26s Training Script
# Usage:
#   sbatch sbatch/train_clahe_unsharp.sh <SIGMA> <GAMMA> [CLIP_LIMIT]
# Examples:
#   sbatch sbatch/train_clahe_unsharp.sh 0.5 0.3 1.5
#   sbatch sbatch/train_clahe_unsharp.sh 1.0 0.3 1.5
#   sbatch sbatch/train_clahe_unsharp.sh 1.0 0.6 1.5
#   sbatch sbatch/train_clahe_unsharp.sh 1.5 0.4 1.5
# ==============================================================================

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# --- Parameters ---
SIGMA="${1:-0.5}"
GAMMA="${2:-0.3}"
CLIP="${3:-1.5}"

RUN_KEY="yolov26s_rectified_clahe_s${SIGMA}_g${GAMMA}_640"
DEST_DATASET="datasets/pcb-rectified-clahe-s${SIGMA}-g${GAMMA}-640"

echo "=========================================================="
echo "Training CLAHE + Unsharp-Masked YOLO26s (640px)"
echo "Parameters:  sigma=$SIGMA, gamma=$GAMMA, clipLimit=$CLIP"
echo "Dataset:     $DEST_DATASET"
echo "Run Key:     $RUN_KEY"
echo "=========================================================="

# 1. Build CLAHE + Unsharp-masked dataset if not already present
if [ ! -f "$DEST_DATASET/data.yaml" ]; then
    echo "Generating dataset $DEST_DATASET from datasets/pcb-unified-4class..."
    python tools/build_clahe_unsharp_dataset.py \
        --source datasets/pcb-unified-4class \
        --dest "$DEST_DATASET" \
        --sigma "$SIGMA" \
        --gamma "$GAMMA" \
        --clip-limit "$CLIP" \
        --workers 8
fi

# 2. Train YOLO26s warm-started from the rectified loss-reweighted checkpoint
python train.py \
    --run-key "$RUN_KEY" \
    --weights runs/yolov26s_rectified_loss_reweight/pcb-filtered/weights/best.pt \
    --data "$DEST_DATASET/data.yaml" \
    --cls 1.5 \
    --box 5.0 \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001

echo "Training finished for $RUN_KEY."
