#!/bin/bash
#SBATCH --job-name=sdice_param_sweep
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_sdice_sweep_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

EXP_NAME="$1"
LAMBDA_DICE="$2"
SMOOTH_DICE="$3"
STAGE_WEIGHTS="$4"

echo "=========================================================="
echo "Running Squared-Dice Exploration: $EXP_NAME"
echo "  lambda_dice   = $LAMBDA_DICE"
echo "  smooth_dice   = $SMOOTH_DICE"
echo "  stage_weights = $STAGE_WEIGHTS"
echo "=========================================================="

STAGE_ARGS=""
if [ -n "$STAGE_WEIGHTS" ]; then
    STAGE_ARGS="--stage-weights $STAGE_WEIGHTS"
fi

python train.py \
    --model yolo26s \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-cleaned-1280/data.yaml \
    --imgsz 1280 \
    --batch 8 \
    --epochs 100 \
    --agwd-ota \
    --squared-dice \
    --lambda-dice "$LAMBDA_DICE" \
    --smooth-dice "$SMOOTH_DICE" \
    $STAGE_ARGS \
    --copy-paste 0.15 \
    --max-det 1000 \
    --run-key "$EXP_NAME" \
    --device 0

echo "Finished $EXP_NAME"
