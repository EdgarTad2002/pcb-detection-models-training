#!/bin/bash
#SBATCH --job-name=yolo12_paper
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_yolo12_paper_%j.out

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
RUN_KEY="r4_s_yolov12s_agwd_sdice_mixed_200ep"

COMMON_AUG="--mosaic 1.0 --close-mosaic 15 --scale 0.5 --translate 0.1 --fliplr 0.5 --flipud 0.5"

echo "=========================================================="
echo "Starting YOLOv12s with Full Innovation Suite (200 Epochs)"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Dataset: $DATA_YAML | Run Key: $RUN_KEY"
echo "=========================================================="

python train.py \
    --model yolo12s \
    --weights yolo12s.pt \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --eval-split val \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --epochs "$EPOCHS" \
    --workers "$WORKERS" \
    $COMMON_AUG \
    --squared-dice --lambda-dice 0.50 --smooth-dice 0.01 \
    --agwd-ota --agwd-kappa 12.8 --agwd-tau 576.0 \
    --optimizer SGD --lr0 0.01 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --max-det 1000 --eval-conf 0.001 \
    --run-key "$RUN_KEY" --device 0

echo "Finished training for YOLOv12s Flagship: $RUN_KEY"
