#!/bin/bash
#SBATCH --job-name=audit_labels
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_audit_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "🔍 Running Full Train Split Missing-Label Auditor..."
python tools/audit_missing_labels.py \
    --images-dir datasets/pcb-unified-4class/train/images \
    --labels-dir datasets/pcb-unified-4class/train/labels \
    --weights runs/rectified_yolov26s_matched_filter_1280/pcb-filtered/weights/best.pt \
    --out-dir results/audit_train_missing_labels \
    --conf 0.25 \
    --corr-thresh 0.45 \
    --imgsz 1280

echo "✅ Audit completed! Results saved to results/audit_train_missing_labels/"
