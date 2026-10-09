#!/bin/bash
#SBATCH --job-name=y26x_clean_1280
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_y26x_clean_1280_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "🚀 Training Flagship YOLO26x (59M Params) on CLEANED RECTIFIED DATASET (1280px)..."
python train.py \
    --run-key rectified_yolov26x_cleaned_1280 \
    --weights yolo26x.pt \
    --data datasets/pcb-unified-4class-cleaned/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001

echo "✅ Training and Evaluation on Cleaned Dataset Complete!"
