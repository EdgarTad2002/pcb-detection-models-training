#!/bin/bash
#SBATCH --job-name=sahi_champ
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_eval_sahi_champion_%j.out

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

# Copy latest updated scripts if present in source repo
SRC_DIR="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training"
if [ -d "$SRC_DIR" ]; then
    cp -r "$SRC_DIR/tools" . 2>/dev/null || true
    cp "$SRC_DIR/train.py" . 2>/dev/null || true
    cp "$SRC_DIR/aggregate_results.py" . 2>/dev/null || true
fi

# 4. Identify champion weights
CHAMPION_RUN="yolov26s_ultimate_retinex_copypaste_1280"
WEIGHTS_PATH="runs/$CHAMPION_RUN/pcb-filtered/weights/best.pt"

if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "❌ Error: Champion weights not found at $WEIGHTS_PATH"
    echo "   Checking alternative checkpoint paths..."
    WEIGHTS_PATH=$(find runs -name "best.pt" | grep -i "ultimate" | head -n 1)
    if [ -z "$WEIGHTS_PATH" ] || [ ! -f "$WEIGHTS_PATH" ]; then
        echo "❌ Could not find any champion weights matching 'ultimate'."
        exit 1
    fi
    echo "   Found alternative weights: $WEIGHTS_PATH"
fi

# 5. Identify test dataset
if [ -f "datasets/pcb-retinex-cappaste-1280/data.yaml" ]; then
    DATA_YAML="datasets/pcb-retinex-cappaste-1280/data.yaml"
elif [ -f "datasets/pcb-retinex-enhanced-1280/data.yaml" ]; then
    DATA_YAML="datasets/pcb-retinex-enhanced-1280/data.yaml"
elif [ -f "datasets/pcb-native-res-unified-4class/data.yaml" ]; then
    DATA_YAML="datasets/pcb-native-res-unified-4class/data.yaml"
else
    DATA_YAML="datasets/pcb-unified-4class/data.yaml"
fi

echo "=========================================================================="
echo "🎯 EVALUATING CHAMPION MODEL WITH SAHI SLICED HYPER-INFERENCE"
echo "   Model:       $CHAMPION_RUN"
echo "   Weights:     $WEIGHTS_PATH"
echo "   Dataset:     $DATA_YAML"
echo "   Technique:   Dual-pass Full-frame (1280px) + Slices with DIoU-NMS Fusion"
echo "=========================================================================="

# Pass 1: SAHI with 640px slices (2x magnification over full board)
echo ""
echo ">>> [1/2] Running SAHI Benchmark with 640px slices..."
python tools/eval_sahi_benchmark.py \
    --weights "$WEIGHTS_PATH" \
    --run-key "$CHAMPION_RUN" \
    --data "$DATA_YAML" \
    --split test \
    --imgsz 1280 \
    --slice-size 640 \
    --overlap 0.20 \
    --conf 0.01 \
    --iou 0.50 \
    --nms-type diou \
    --device 0

# Pass 2: SAHI with 480px slices (2.67x magnification for ultra-dense micro-capacitors)
echo ""
echo ">>> [2/2] Running SAHI Benchmark with 480px slices..."
python tools/eval_sahi_benchmark.py \
    --weights "$WEIGHTS_PATH" \
    --run-key "$CHAMPION_RUN" \
    --data "$DATA_YAML" \
    --split test \
    --imgsz 1280 \
    --slice-size 480 \
    --overlap 0.20 \
    --conf 0.01 \
    --iou 0.50 \
    --nms-type diou \
    --device 0

# 6. Aggregate into master comparison table
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    echo "=========================================================================="
    echo "📊 Updating Master Comparison Leaderboard: $SHARED_RESULTS_DIR..."
    echo "=========================================================================="
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    mkdir -p results
    cp "$SHARED_RESULTS_DIR"/comparison_table.* results/ 2>/dev/null || true
    echo ""
    echo "🏆 Top Leaderboard Entries:"
    head -n 25 "$SHARED_RESULTS_DIR/comparison_table.md"
fi

echo "=========================================================================="
echo "✅ SAHI Benchmark Evaluation Completed Successfully!"
echo "=========================================================================="
