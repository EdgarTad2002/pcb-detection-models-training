#!/bin/bash
#SBATCH --job-name=yolo26n_gurgen
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_yolo26n_gurgen_%j.out

set -e

# 1. Configure storage and cache variables
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

# 2. Activate Conda environment from shared cache
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# 3. Dedicated Workspace on Weka for YOLO26n Gurgen
WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Copy training scripts from source repository if not already present or out of date
SRC_DIR="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training"
if [ -d "$SRC_DIR" ]; then
    cp "$SRC_DIR/train.py" . 2>/dev/null || true
    cp "$SRC_DIR/aggregate_results.py" . 2>/dev/null || true
fi

# 4. Link datasets if not present in the workspace
DATASET_SOURCE="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets"
if [ ! -d "datasets" ] && [ -d "$DATASET_SOURCE" ]; then
    echo "🔗 Symlinking PCB datasets from $DATASET_SOURCE..."
    ln -s "$DATASET_SOURCE" datasets
fi

# Verify cleaned 4-class dataset
DATA_PATH="datasets/pcb-unified-4class-cleaned/data.yaml"
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Cleaned dataset config not found at $DATA_PATH!"
    echo "Checking available datasets:"
    ls -la datasets/ 2>/dev/null || ls -la "$DATASET_SOURCE" 2>/dev/null || true
    exit 1
fi

echo "=========================================================================="
echo "🚀 Training YOLO26n on NVIDIA H100 (Cleaned 4-Class Dataset)"
echo "   Workspace: $WORKSPACE_DIR"
echo "   Data:      $DATA_PATH"
echo "   Run Key:   yolo26n_unified_cleaned_640"
echo "=========================================================================="

# 5. Train YOLO26n (Ultralytics)
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key yolo26n_unified_cleaned_640 \
    --weights yolo26n.pt \
    --data "$DATA_PATH" \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50

# 6. Aggregate results into the shared master leaderboard
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    echo "📊 Aggregating all models into master table in: $SHARED_RESULTS_DIR..."
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    mkdir -p results
    cp "$SHARED_RESULTS_DIR"/comparison_table.* results/ 2>/dev/null || true
else
    python aggregate_results.py
fi

echo "✅ YOLO26n training and evaluation finished successfully!"
