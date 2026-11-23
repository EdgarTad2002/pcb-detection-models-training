#!/bin/bash
#SBATCH --job-name=nms_sweep
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_nms_sweep_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

WEIGHTS="${1:-runs/yolo26s_native_cleaned_agwd_cappaste_1280/pcb-filtered/weights/best.pt}"
OUTPUT_KEY="${2:-yolo26s_agwd}"

echo "Starting NMS IoU Sweep on $WEIGHTS"

python tools/sweep_nms_iou.py \
    --weights "$WEIGHTS" \
    --data datasets/pcb-native-res-cleaned-1280/data.yaml \
    --split test \
    --imgsz 1280 \
    --conf 0.001 \
    --max-det 1000 \
    --iou-values 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 \
    --output-json "/mnt/weka/etadevosyan/pcb-yolo/results/${OUTPUT_KEY}_nms_sweep.json"

echo "NMS Sweep Finished."
