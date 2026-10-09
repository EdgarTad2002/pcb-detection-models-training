#!/bin/bash
#SBATCH --job-name=v26s_ret640
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_yolov26s_retinex_640_%j.out

set -e

# 1. Storage and cache variables
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

# 2. Activate Conda environment
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# 3. Enter dedicated workspace
WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Link datasets from shared location if not present
DATASET_SOURCE="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets"
if [ ! -d "datasets" ] && [ -d "$DATASET_SOURCE" ]; then
    echo "🔗 Symlinking shared PCB datasets..."
    ln -s "$DATASET_SOURCE" datasets
fi

# Determine source dataset
if [ -f "datasets/pcb-unified-4class-cleaned/data.yaml" ]; then
    SRC_DATA="datasets/pcb-unified-4class-cleaned"
elif [ -f "datasets/pcb-unified-4class/data.yaml" ]; then
    SRC_DATA="datasets/pcb-unified-4class"
else
    echo "❌ ERROR: No 4-class unified dataset found!"
    exit 1
fi

# 4. Generate Retinex + Canny Enhanced Dataset (640px)
ENHANCED_DATA="datasets/pcb-retinex-enhanced-640/data.yaml"
if [ ! -f "$ENHANCED_DATA" ]; then
    echo "=========================================================================="
    echo "🎨 Building Retinex Material Reflectance + Canny Dataset (640px)..."
    echo "   Source: $SRC_DATA"
    echo "   Dest:   datasets/pcb-retinex-enhanced-640"
    echo "   Alpha (Reflectance): 0.30 | Beta (Edge): 0.10"
    echo "=========================================================================="
    python tools/enhance_pcb_retinex.py \
        --source "$SRC_DATA" \
        --dest datasets/pcb-retinex-enhanced-640 \
        --alpha 0.30 \
        --beta 0.10 \
        --workers 8
fi

echo "=========================================================================="
echo "🚀 Training YOLO26s (Champion 640px) on Retinex + Canny Dataset (NVIDIA H100)"
echo "   Run Key: yolov26s_retinex_enhanced_640"
echo "   Data:    $ENHANCED_DATA"
echo "=========================================================================="

# 5. Train YOLO26s (640px)
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key yolov26s_retinex_enhanced_640 \
    --weights yolo26s.pt \
    --data "$ENHANCED_DATA" \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50

# 6. Aggregate into shared master leaderboard
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    echo "📊 Aggregating all models into master table in: $SHARED_RESULTS_DIR..."
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    mkdir -p results
    cp "$SHARED_RESULTS_DIR"/comparison_table.* results/ 2>/dev/null || true
else
    python aggregate_results.py
fi

echo "✅ YOLO26s (640px) training and evaluation completed!"
