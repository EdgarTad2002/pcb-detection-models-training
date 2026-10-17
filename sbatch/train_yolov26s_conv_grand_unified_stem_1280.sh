#!/bin/bash
#SBATCH --job-name=v26s_cg_stem
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_yolov26s_conv_grand_unified_stem_1280_%j.out

# ==============================================================================
# YOLO26s with Learnable Convolutional Grand Unified Stem (Zero Attention)
# Combines:
#   - 15x15 Gaussian Conv for Illumination (L)
#   - Agaian Adaptive Gamma Correction (AGC) Eq. 23 on L
#   - Log-space Reflectance: log(R) = log(I) - log(L)
#   - Sargis Weber Contrast on Reflectance via local min/max pooling (alpha=0.20)
#   - Pure 2D Convolutional adapter residual into Layer 0 Conv
#   - Champion SGD recipe: cls=1.5, box=5.0, dfl=2.0, ls=0.10, max_det=1000 @ 1280px
# ==============================================================================

set -e

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

# Locate source dataset
if [ -f "datasets/pcb-unified-4class-cleaned/data.yaml" ]; then
    SRC_DATA="datasets/pcb-unified-4class-cleaned"
elif [ -f "/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-unified-4class-cleaned/data.yaml" ]; then
    SRC_DATA="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-unified-4class-cleaned"
else
    SRC_DATA="datasets/pcb-unified-4class"
fi

# Step 1: Ensure capacitor crop bank exists
BANK_DIR="capacitor_bank_cleaned_1280"
if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "=========================================================================="
    echo "📦 Building High-Resolution Capacitor Crop Bank..."
    echo "=========================================================================="
    python build_capacitor_bank.py \
        --source "$SRC_DATA" \
        --dest "$BANK_DIR" \
        --class-id 0 \
        --margin 0.15
fi

# Step 2: Ensure Copy-Paste Augmented Dataset exists
AUG_DATA="datasets/pcb-unified-4class-cappaste-1280"
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Copy-Paste Augmented Dataset..."
    echo "=========================================================================="
    python bbox_copy_paste.py \
        --source "$SRC_DATA" \
        --bank "$BANK_DIR" \
        --dest "$AUG_DATA" \
        --class-id 0 \
        --paste-prob 0.7 \
        --min-pastes 1 \
        --max-pastes 4
fi

RUN_KEY="yolov26s_conv_grand_unified_stem_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "🚀 Training YOLO26s with Convolutional Grand Unified Stem on NVIDIA H100"
echo "   Run Key:       $RUN_KEY"
echo "   Data:          $AUG_DATA/data.yaml"
echo "   Stem:          Pure 2D Convolutions (Gaussian + Agaian AGC + Sargis Weber)"
echo "   Resolution:    1280px Native Input, max_det = 1000"
echo "=========================================================================="

python models/train_conv_grand_unified_stem.py \
    --run-key "$RUN_KEY" \
    --weights "yolo26s.pt" \
    --data "$AUG_DATA/data.yaml" \
    --optimizer SGD \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --device 0 \
    --eval-conf 0.001 \
    --eval-iou 0.50 \
    --eval-split "test" \
    --max-det 1000

# Update Leaderboard
if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR" || true
    echo ""
    echo "=========================================================================="
    echo "🏆 Comparison with Previous Models:"
    grep -E "^\| ($RUN_KEY|yolov26s_grand_unified_1280|yolov26s_ultimate_retinex_copypaste_1280_maxdet1000) \|" \
        "$SHARED_RESULTS_DIR/comparison_table.md" || true
    echo "=========================================================================="
fi

echo "=========================================================================="
echo "✅ Convolutional Grand Unified Stem Training & Evaluation Complete!"
echo "=========================================================================="
