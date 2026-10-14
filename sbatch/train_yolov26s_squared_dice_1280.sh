#!/bin/bash
#SBATCH --job-name=y26s_sdice_1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_y26s_sdice_1280_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

SEED=${1:-0}
LAMBDA_DICE=${2:-0.5}

DATA="datasets/pcb-unified-4class-cleaned/data.yaml"
RUN_KEY="rectified_yolov26s_squared_dice_s${SEED}_1280"

echo "=========================================================================="
echo "🎯 TRAINING YOLO26s WITH BCE + SQUARED-DICE LOSS (1280px)"
echo "    Data:         $DATA"
echo "    Run Key:      $RUN_KEY"
echo "    Seed:         $SEED"
echo "    Lambda Dice:  $LAMBDA_DICE"
echo "    Loss:         BCE + Lambda*Squared_Dice (Eq. 3/4) + Squared-IoU"
echo "=========================================================================="

python train.py \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$DATA" \
    --squared-dice \
    --lambda-dice "$LAMBDA_DICE" \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --seed "$SEED" \
    --max-det 1000 \
    --eval-conf 0.001

echo "=========================================================================="
echo "✅ Squared-Dice YOLO26s 1280 Training Complete: $RUN_KEY"
echo "=========================================================================="
