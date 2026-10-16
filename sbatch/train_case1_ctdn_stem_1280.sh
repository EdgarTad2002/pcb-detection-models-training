#!/bin/bash
#SBATCH --job-name=case1_ctdn
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_case1_ctdn_stem_1280_%j.out

# ==============================================================================
# CASE 1: YOLO26s with In-Network CTDN Feature Stem & Group Multi-Stage SDice Loss
# - Deep latent-space decomposition replacing static conv stem
# - Cross-Attention(Illumination, Reflectance) + Self-Attention
# - Differentiable Agaian Adaptive Gamma Correction in latent space
# - Group Multi-Stage Loss: BCE + Squared-Dice Loss across P1, P2, P3, P4
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
    echo "🎨 Building Copy-Paste Augmented Dataset for Case 1..."
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

# Step 3: Train Case 1 YOLO26s CTDN Stem Model on H100
RUN_KEY="yolov26s_case1_ctdn_stem_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "🚀 Training Case 1: YOLO26s with In-Network CTDN Feature Stem on H100"
echo "   Run Key:       $RUN_KEY"
echo "   Data:          $AUG_DATA/data.yaml"
echo "   Multi-Stage:   lambda_group = 0.15 (BCE + Squared-Dice Loss)"
echo "   Resolution:    1280px Native Input"
echo "=========================================================================="

python models/train_case1_ctdn_stem.py \
    --run-key "$RUN_KEY" \
    --weights "yolo26s.pt" \
    --data "$AUG_DATA/data.yaml" \
    --lambda-group 0.15 \
    --use-squared-dice \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --device 0 \
    --eval-conf 0.001 \
    --eval-iou 0.50 \
    --eval-split "test" \
    --max-det 1000

# Step 4: Aggregate results into shared leaderboard
if [ -f "aggregate_results.py" ]; then
    python aggregate_results.py \
        --results-dir "$SHARED_RESULTS_DIR" \
        --output-dir "$SHARED_RESULTS_DIR" || true
fi

echo "=========================================================================="
echo "✅ Case 1 In-Network CTDN Stem Training & Evaluation Complete!"
echo "=========================================================================="
