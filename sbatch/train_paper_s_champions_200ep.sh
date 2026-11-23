#!/bin/bash
#SBATCH --job-name=pcb_s_paper
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_s_paper_%A_%a.out

set -euo pipefail

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

DATA_YAML="datasets/pcb-native-reviewed-r4-mixed-crops/data.yaml"
EPOCHS=200
BATCH=8
IMGSZ=1280
WORKERS=8

COMMON_AUG="--mosaic 1.0 --close-mosaic 15 --scale 0.5 --translate 0.1 --fliplr 0.5 --flipud 0.5"

# Task dispatch
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # 1. YOLO26s-P2 Micro-Head + AGWD-OTA + Squared-Dice (Novel Architecture)
    RUN_KEY="r4_s_yolo26s_p2_crop_specialist_200ep"
    MODEL_CMD="python train.py \
        --model yolo26s-p2 \
        --weights yolo26s-p2.yaml \
        --pretrained-weights yolo26s.pt \
        --data $DATA_YAML \
        --eval-data $DATA_YAML \
        --eval-split val \
        --imgsz $IMGSZ \
        --batch $BATCH \
        --epochs $EPOCHS \
        --workers $WORKERS \
        $COMMON_AUG \
        --squared-dice --lambda-dice 0.50 --smooth-dice 0.01 \
        --agwd-ota --agwd-kappa 12.8 --agwd-tau 576.0 \
        --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
        --max-det 1000 --eval-conf 0.001 \
        --run-key $RUN_KEY --device 0"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    # 2. YOLO26s Crop Specialist (AGWD-OTA + Squared-Dice)
    RUN_KEY="r4_s_yolo26s_crop_specialist_200ep"
    MODEL_CMD="python train.py \
        --model yolo26s \
        --weights yolo26s.pt \
        --data $DATA_YAML \
        --eval-data $DATA_YAML \
        --eval-split val \
        --imgsz $IMGSZ \
        --batch $BATCH \
        --epochs $EPOCHS \
        --workers $WORKERS \
        $COMMON_AUG \
        --squared-dice --lambda-dice 0.50 --smooth-dice 0.01 \
        --agwd-ota --agwd-kappa 12.8 --agwd-tau 576.0 \
        --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
        --max-det 1000 --eval-conf 0.001 \
        --run-key $RUN_KEY --device 0"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    # 3. PhysOT-YOLO26s (Physics-guided Optical Transport / Weber Stem / CSFA Neck / ACA SupCon)
    RUN_KEY="r4_s_physot_yolo26s_mixed_200ep"
    MODEL_CMD="python train.py \
        --model physot-yolo26s \
        --weights yolo26s.pt \
        --data $DATA_YAML \
        --eval-data $DATA_YAML \
        --eval-split val \
        --imgsz $IMGSZ \
        --batch $BATCH \
        --epochs $EPOCHS \
        --workers $WORKERS \
        $COMMON_AUG \
        --squared-dice --lambda-dice 0.50 --smooth-dice 0.01 \
        --agwd-ota --use-igr-stem --use-fg-csfa --use-aca-supcon --agwd-kappa 12.8 --agwd-tau 576.0 \
        --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
        --max-det 1000 --eval-conf 0.001 \
        --run-key $RUN_KEY --device 0"

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    # 4. Standard Stock YOLO11s Control (Ordinary Baseline to prove our techniques beat it)
    RUN_KEY="r4_s_stock_yolo11s_baseline_200ep"
    MODEL_CMD="python train.py \
        --model yolo11s \
        --weights yolo11s.pt \
        --data $DATA_YAML \
        --eval-data $DATA_YAML \
        --eval-split val \
        --imgsz $IMGSZ \
        --batch $BATCH \
        --epochs $EPOCHS \
        --workers $WORKERS \
        $COMMON_AUG \
        --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
        --max-det 1000 --eval-conf 0.001 \
        --run-key $RUN_KEY --device 0"
fi

echo "=========================================================="
echo "Starting Slurm Task $SLURM_ARRAY_TASK_ID: $RUN_KEY"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Command: $MODEL_CMD"
echo "=========================================================="

eval "$MODEL_CMD"

echo "Finished Task $SLURM_ARRAY_TASK_ID: $RUN_KEY"
