#!/bin/bash
#SBATCH --job-name=grand_unified
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_grand_unified_superyolo_1280_%j.out

# ==============================================================================
# GRAND UNIFIED TRAINING PIPELINE:
# 1. CTDN / Physical Retinex Decomposition (Material R vs Lighting L)
# 2. Agaian Adaptive Gamma Correction (IEEE 10365161 Eq. 23)
# 3. Sargis Weber-Fechner Edge Contrast on Reflectance R (alpha=0.20, window 3x3)
# 4. SuperYOLO26s Architecture at 1280px Native Resolution
# 5. Group Multi-Stage Loss: BCE + Squared-Dice Loss across P1, P2, P3, P4
# 6. Copy-Paste Micro-Capacitor Augmentation (Balanced Class Distributions)
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

# Step 1: Generate Grand Unified Dataset
GU_DATA="datasets/pcb-grand-unified-1280"
BANK_DIR="capacitor_bank_grand_unified_1280"
AUG_DATA="datasets/pcb-grand-unified-cappaste-1280"

# Auto-heal: If previous run created incomplete dataset without valid/images, wipe and rebuild
if [ -d "$GU_DATA" ] && { [ ! -d "$GU_DATA/valid/images" ] || [ $(ls -1 "$GU_DATA/valid/images" 2>/dev/null | wc -l) -eq 0 ]; }; then
    echo "⚠️ Incomplete dataset detected (missing valid split). Re-generating cleanly..."
    rm -rf "$GU_DATA" "$BANK_DIR" "$AUG_DATA"
fi

if [ ! -f "$GU_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Grand Unified Dataset (Retinex R + Agaian AGC + Sargis Weber)..."
    echo "   Source: $SRC_DATA"
    echo "   Dest:   $GU_DATA"
    echo "=========================================================================="
    python tools/generate_grand_unified_dataset.py \
        --src "$SRC_DATA" \
        --dest "$GU_DATA" \
        --alpha 0.20 \
        --ksize 3 \
        --workers 8
fi

# Step 2: Build Micro-Capacitor Bank
BANK_DIR="capacitor_bank_grand_unified_1280"
if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "=========================================================================="
    echo "📦 Building High-Resolution Grand Unified Capacitor Crop Bank..."
    echo "=========================================================================="
    python build_capacitor_bank.py \
        --source "$GU_DATA" \
        --dest "$BANK_DIR" \
        --class-id 0 \
        --margin 0.15
fi

# Step 3: Copy-Paste Augment Grand Unified Dataset
AUG_DATA="datasets/pcb-grand-unified-cappaste-1280"
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Grand Unified + Bbox Copy-Paste Augmented Dataset..."
    echo "=========================================================================="
    python bbox_copy_paste.py \
        --source "$GU_DATA" \
        --bank "$BANK_DIR" \
        --dest "$AUG_DATA" \
        --class-id 0 \
        --paste-prob 0.7 \
        --min-pastes 1 \
        --max-pastes 4
fi

# Step 4: Train SuperYOLO26s with Group Multi-Stage Squared-Dice Loss
RUN_KEY="superyolo26s_grand_unified_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "🚀 Training Grand Unified SuperYOLO26s (BCE + Squared-Dice Loss) on H100"
echo "   Run Key:       $RUN_KEY"
echo "   Data:          $AUG_DATA/data.yaml"
echo "   Auxiliary SR:  lambda_sr = 0.10"
echo "   Multi-Stage:   lambda_group = 0.15 (Squared-Dice + BCE)"
echo "   Target:        Break 66%+ mAP50 barrier with enhanced capacitor recall!"
echo "=========================================================================="

python models/superyolo_multistage.py \
    --run-key "$RUN_KEY" \
    --weights "yolo26s.pt" \
    --data "$AUG_DATA/data.yaml" \
    --lambda-sr 0.10 \
    --lambda-group 0.15 \
    --use-squared-dice \
    --sr-scale 2 \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --device 0 \
    --eval-conf 0.001 \
    --eval-iou 0.50 \
    --eval-split "test" \
    --max-det 1000

# Step 5: Aggregate results into shared leaderboard
if [ -f "aggregate_results.py" ]; then
    python aggregate_results.py \
        --results-dir "$SHARED_RESULTS_DIR" \
        --output-dir "$SHARED_RESULTS_DIR" || true
fi

echo "=========================================================================="
echo "✅ Grand Unified SuperYOLO Training & Evaluation Complete!"
echo "=========================================================================="
