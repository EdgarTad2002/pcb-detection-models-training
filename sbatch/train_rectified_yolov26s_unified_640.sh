#!/bin/bash
#SBATCH --job-name=rect_yolo26s_uni640
#SBATCH --partition=research
#SBATCH --mem=36G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_unified640_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "=========================================================="
echo "Training Rectified Unified Champion YOLO26s (640px)"
echo "Combines: Clean BBox Copy-Paste + Loss Reweight + Label Smoothing"
echo "=========================================================="

BANK_DIR="capacitor_bank_rectified"
DEST_DATASET="datasets/pcb-rectified-cappaste-640"

# 1. Build capacitor crop bank from rectified dataset
if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "📦 Building rectified capacitor crop bank..."
    python build_capacitor_bank.py \
        --source datasets/pcb-unified-4class \
        --dest "$BANK_DIR" \
        --class-id 0 \
        --margin 0.15
fi

# 2. Build augmented copy-paste dataset
if [ ! -f "$DEST_DATASET/data.yaml" ]; then
    echo "🎨 Building augmented copy-paste dataset at 640px..."
    python bbox_copy_paste.py \
        --source datasets/pcb-unified-4class \
        --bank "$BANK_DIR" \
        --dest "$DEST_DATASET" \
        --class-id 0 \
        --paste-prob 0.7 --min-pastes 1 --max-pastes 4
fi

# 3. Train Unified Champion Model (640px)
python train.py \
    --run-key rectified_yolov26s_unified_640 \
    --weights yolo26s.pt \
    --data "$DEST_DATASET/data.yaml" \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001
