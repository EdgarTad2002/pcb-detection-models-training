#!/bin/bash
#SBATCH --job-name=v26s_weber
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_yolov26s_weber_a0.20_copypaste_1280_%j.out

# ==============================================================================
# YOLO26s Champion Trained on Sargis's Winning Weber Contrast Representation
# Parameters proven by 10-board empirical ablation:
#   - Exponent: alpha = 0.20 (cuts substrate noise by 57%, CNR = 106.66)
#   - Window:   3x3 (sub-pixel sharpness, avoids capacitor smearing)
#   - Res:      1280px native input
#   - Recipe:   SGD, cls=1.5, box=5.0, dfl=2.0, ls=0.10, copy-paste (p=0.7)
#   - Eval:     max_det = 1000 at 1280px
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

# Locate native resolution source dataset (audited gold-standard ground truth)
if [ -f "datasets/pcb-unified-4class-cleaned/data.yaml" ]; then
    SRC_DATA="datasets/pcb-unified-4class-cleaned"
elif [ -f "/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-unified-4class-cleaned/data.yaml" ]; then
    SRC_DATA="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-unified-4class-cleaned"
else
    SRC_DATA="datasets/pcb-unified-4class"
fi

# Step 1: Generate Sargis Weber-Contrast Enhanced Dataset (alpha=0.20, window 3x3)
WEBER_DATA="datasets/pcb-weber-a0.20-1280"
if [ ! -f "$WEBER_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Sargis Weber-Contrast Dataset (alpha=0.20, window 3x3)..."
    echo "   Source: $SRC_DATA"
    echo "   Dest:   $WEBER_DATA"
    echo "   Formula: f(L) = 300 * ((Lmax - Lmin) / (Lmin + 1))^0.20 * log(1 + ratio)"
    echo "=========================================================================="
    python tools/enhance_pcb_sargis_weber.py \
        --source "$SRC_DATA" \
        --dest "$WEBER_DATA" \
        --alpha-exp 0.20 \
        --ksize 3 \
        --w-orig 0.70 \
        --w-r 0.30 \
        --w-edge 0.10 \
        --workers 8
fi

# Step 2: Build Micro-Capacitor Crop Bank from Weber Dataset
BANK_DIR="capacitor_bank_weber_a0.20_1280"
if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "=========================================================================="
    echo "📦 Building High-Resolution Weber (alpha=0.20) Capacitor Crop Bank..."
    echo "=========================================================================="
    python build_capacitor_bank.py \
        --source "$WEBER_DATA" \
        --dest "$BANK_DIR" \
        --class-id 0 \
        --margin 0.15
fi

# Step 3: Copy-Paste Augment Weber Dataset
AUG_DATA="datasets/pcb-weber-a0.20-cappaste-1280"
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Weber (alpha=0.20) + Bbox Copy-Paste Augmented Dataset..."
    echo "=========================================================================="
    python bbox_copy_paste.py \
        --source "$WEBER_DATA" \
        --bank "$BANK_DIR" \
        --dest "$AUG_DATA" \
        --class-id 0 \
        --paste-prob 0.7 \
        --min-pastes 1 \
        --max-pastes 4
fi

# Step 4: Train YOLO26s Champion on NVIDIA H100 (1280px Native Resolution)
RUN_KEY="yolov26s_weber_a0.20_copypaste_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "🚀 Training YOLO26s CHAMPION on Sargis Weber (alpha=0.20) on NVIDIA H100"
echo "   Run Key: $RUN_KEY"
echo "   Data:    $AUG_DATA/data.yaml"
echo "   Loss:    cls=1.5, box=5.0, dfl=2.0, label_smoothing=0.10"
echo "   Target:  Beat Canny Champion (62.83% mAP50, 38.58% Capacitor AP)!"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$AUG_DATA/data.yaml" \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --label-smoothing 0.1 \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50 \
    --max-det 1000

# Step 5: Update Master Leaderboard
if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    echo ""
    echo "=========================================================================="
    echo "🏆 Comparison with Previous Champions:"
    grep -E "^\| (yolov26s_weber_a0.20_copypaste_1280|yolov26s_ultimate_retinex_copypaste_1280_maxdet1000|superyolo26s_rectified_1280) \|" \
        "$SHARED_RESULTS_DIR/comparison_table.md" || true
    echo "=========================================================================="
fi

echo "✅ YOLO26s Weber (alpha=0.20) Training & Evaluation Complete."
