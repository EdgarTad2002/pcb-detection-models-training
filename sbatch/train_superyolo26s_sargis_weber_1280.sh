#!/bin/bash
#SBATCH --job-name=sy_weber
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_superyolo26s_sargis_weber_1280_%j.out

# ==============================================================================
# SuperYOLO Trained on Sargis's Local Contrast (Weber-Fechner) Edge Enhancement
# Replacing non-differentiable Canny binary spikes with smooth, continuous edges
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

# Locate native resolution source dataset (prefer audited gold-standard cleaned dataset)
if [ -f "datasets/pcb-unified-4class-cleaned/data.yaml" ]; then
    SRC_DATA="datasets/pcb-unified-4class-cleaned"
elif [ -f "datasets/pcb-native-res-unified-4class/data.yaml" ]; then
    SRC_DATA="datasets/pcb-native-res-unified-4class"
else
    SRC_DATA="datasets/pcb-unified-4class"
fi

# Step 1: Generate Sargis Weber-Contrast Enhanced Dataset (1280px Native)
WEBER_DATA="datasets/pcb-weber-enhanced-1280"
# Clean up if previously aborted with empty images
if [ -d "$WEBER_DATA/train/images" ]; then
    IMG_COUNT=$(ls "$WEBER_DATA/train/images" 2>/dev/null | wc -l)
    if [ "$IMG_COUNT" -lt 10 ]; then
        echo "Removing incomplete $WEBER_DATA (only $IMG_COUNT images)..."
        rm -rf "$WEBER_DATA"
    fi
fi

if [ ! -f "$WEBER_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Sargis Weber-Contrast Enhanced PCB Dataset (1280px)..."
    echo "   Source: $SRC_DATA"
    echo "   Dest:   $WEBER_DATA"
    echo "   Formula: f(L) = 300 * ((Lmax - Lmin) / (Lmin + 1))^0.10 * log(1 + ratio)"
    echo "   Window: 3x3 | alpha=0.10"
    echo "=========================================================================="
    python tools/enhance_pcb_sargis_weber.py \
        --source "$SRC_DATA" \
        --dest "$WEBER_DATA" \
        --alpha-exp 0.10 \
        --ksize 3 \
        --w-orig 0.70 \
        --w-r 0.30 \
        --w-edge 0.10 \
        --workers 8
fi

# Step 2: Build Micro-Capacitor Crop Bank from Weber Dataset
BANK_DIR="capacitor_bank_weber_1280"
# Clean up if bank has no image crops
if [ -d "$BANK_DIR" ]; then
    CROP_COUNT=$(ls "$BANK_DIR"/*.png 2>/dev/null | wc -l)
    if [ "$CROP_COUNT" -eq 0 ]; then
        echo "Removing empty $BANK_DIR..."
        rm -rf "$BANK_DIR"
    fi
fi

if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "=========================================================================="
    echo "📦 Building High-Resolution Weber Capacitor Crop Bank..."
    echo "=========================================================================="
    python build_capacitor_bank.py \
        --source "$WEBER_DATA" \
        --dest "$BANK_DIR" \
        --class-id 0 \
        --margin 0.15
fi

# Step 3: Copy-Paste Augment Weber Dataset
AUG_DATA="datasets/pcb-weber-cappaste-1280"
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Weber + Bbox Copy-Paste Augmented Dataset..."
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

# Step 4: Train SuperYOLO on the Weber-Enhanced Dataset
RUN_KEY="superyolo26s_sargis_weber_copypaste_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "🚀 Training SuperYOLO on Sargis Weber Contrast Dataset (NVIDIA H100)"
echo "   Run Key: $RUN_KEY"
echo "   Arch:    SuperYOLO (1280px HR -> 640px LR + SR Head)"
echo "   Data:    $AUG_DATA/data.yaml"
echo "   Advantage: Continuous, differentiable edge maps replace noisy Canny!"
echo "=========================================================================="

python superyolo_pcb.py \
    --project-root "$WORKSPACE_DIR" \
    --results-dir "$SHARED_RESULTS_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$AUG_DATA/data.yaml" \
    --lambda-sr 0.10 \
    --edge-weight 0.5 \
    --cls-weight 1.5 \
    --box-weight 5.0 \
    --dfl-weight 2.0 \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001

# Step 5: Evaluate at max_det=1000
W="runs/$RUN_KEY/pcb-filtered/weights/best.pt"
if [ -f "$W" ]; then
    echo ""
    echo "Evaluating at max_det=1000..."
    python train.py \
        --project-root "$WORKSPACE_DIR" \
        --run-key "${RUN_KEY}_maxdet1000" \
        --weights "$W" \
        --data "$AUG_DATA/data.yaml" \
        --skip-train \
        --eval-weights "$W" \
        --imgsz 640 --batch 8 --epochs 100 \
        --eval-conf 0.001 --eval-iou 0.50 --max-det 1000
fi

# Step 6: Update Master Leaderboard
if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    echo ""
    echo "=========================================================================="
    echo "Updated Leaderboard comparison:"
    grep -E "^\| (superyolo26s_[a-z0-9_]*|yolov26s_ultimate_retinex_copypaste_1280_maxdet1000) \|" \
        "$SHARED_RESULTS_DIR/comparison_table.md" || true
    echo "=========================================================================="
fi

echo "✅ SuperYOLO Weber Contrast Job Complete."
