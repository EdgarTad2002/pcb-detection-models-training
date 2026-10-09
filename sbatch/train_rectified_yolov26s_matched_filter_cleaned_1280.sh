#!/bin/bash
#SBATCH --job-name=mf_clean_1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_mf_clean_1280_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "🚀 Training Matched-Filter YOLO26s on ULTRA-CLEAN DATASET (1280px)..."
python matched_filter_yolo26.py \
    --run-key rectified_yolov26s_matched_filter_cleaned_1280 \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class-cleaned/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001

echo "✅ Matched-Filter YOLO26s Cleaned 1280 Training Complete!"
