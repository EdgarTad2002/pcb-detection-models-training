#!/bin/bash
#SBATCH --job-name=sy_c2_ret
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_superyolo26s_case2_agaian_retinex_1280_%j.out

# ==============================================================================
# CASE 2: SuperYOLO26s with Multi-Stage SDice on Agaian Fractional Retinex
# Combines:
#   - Agaian Generalized Fractional Retinex (lambda=1.20, vartheta=0.50)
#   - Sargis Weber Contrast Guidance on R (alpha=0.20, 3x3)
#   - Agaian Adaptive Gamma Correction on L (IEEE Access 10365161 Eq. 23)
#   - SuperYOLO Auxiliary SR Head (lambda_sr = 0.10)
#   - Group Multi-Stage BCE + Normalized Squared-Dice Loss (lambda_group = 0.15)
#   - Evaluation: native 1280px, max_det = 1000 on test split
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

# Step 1: Ensure Case 2 Dataset Exists
AGAIAN_DATA="datasets/pcb-agaian-retinex-1280"
if [ ! -f "$AGAIAN_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🔬 Generating Case 2: Agaian Generalized Fractional Retinex Dataset..."
    echo "=========================================================================="
    python tools/generate_case2_agaian_dataset.py \
        --source "$SRC_DATA" \
        --dest "$AGAIAN_DATA" \
        --lambda-exp 1.20 \
        --vartheta-exp 0.50 \
        --alpha-weber 0.20 \
        --ksize 3 \
        --weber-weight 0.25 \
        --workers 8
fi

# Step 2: Ensure Capacitor Bank Exists
BANK_DIR="capacitor_bank_agaian_retinex_1280"
if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "=========================================================================="
    echo "📦 Building High-Resolution Capacitor Bank for Case 2..."
    echo "=========================================================================="
    python build_capacitor_bank.py \
        --source "$AGAIAN_DATA" \
        --dest "$BANK_DIR" \
        --class-id 0 \
        --margin 0.15
fi

# Step 3: Ensure Copy-Paste Augmented Dataset Exists
AUG_DATA="datasets/pcb-agaian-retinex-cappaste-1280"
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Copy-Paste Augmented Dataset for Case 2..."
    echo "=========================================================================="
    python bbox_copy_paste.py \
        --source "$AGAIAN_DATA" \
        --bank "$BANK_DIR" \
        --dest "$AUG_DATA" \
        --class-id 0 \
        --paste-prob 0.7 \
        --min-pastes 1 \
        --max-pastes 4
fi

# Step 4: Train SuperYOLO26s with Multi-Stage SDice on H100
RUN_KEY="superyolo26s_case2_agaian_retinex_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "🚀 Training SuperYOLO26s on Case 2 Agaian Fractional Retinex on NVIDIA H100"
echo "   Run Key:       $RUN_KEY"
echo "   Data:          $AUG_DATA/data.yaml"
echo "   Auxiliary SR:  lambda_sr = 0.10"
echo "   Multi-Stage:   lambda_group = 0.15 (Normalized Squared-Dice + BCE)"
echo "   Resolution:    1280px Native Input, max_det = 1000"
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

# Step 5: Update Leaderboard
if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR" || true
    echo ""
    echo "=========================================================================="
    echo "🏆 Comparison with Previous Champions:"
    grep -E "^\| ($RUN_KEY|yolov26s_case2_agaian_retinex_1280|yolov26s_grand_unified_1280|superyolo26s_grand_unified_1280) \|" \
        "$SHARED_RESULTS_DIR/comparison_table.md" || true
    echo "=========================================================================="
fi

echo "=========================================================================="
echo "✅ Case 2 SuperYOLO26s Training & Evaluation Complete!"
echo "=========================================================================="
