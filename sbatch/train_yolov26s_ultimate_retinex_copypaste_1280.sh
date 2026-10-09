#!/bin/bash
#SBATCH --job-name=v26s_ult1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_yolov26s_ultimate_retinex_copypaste_1280_%j.out

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
    cp "$SRC_DIR/build_capacitor_bank.py" . 2>/dev/null || true
    cp "$SRC_DIR/bbox_copy_paste.py" . 2>/dev/null || true
    cp "$SRC_DIR/train.py" . 2>/dev/null || true
    cp "$SRC_DIR/aggregate_results.py" . 2>/dev/null || true
fi

# 4. Step A: Build Crop Bank of Retinex-Enhanced Capacitors (if not present)
RETINEX_DATA="datasets/pcb-retinex-enhanced-1280"
BANK_DIR="capacitor_bank_retinex_1280"
if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "=========================================================================="
    echo "📦 Building High-Resolution Retinex Capacitor Crop Bank..."
    echo "=========================================================================="
    python build_capacitor_bank.py \
        --source "$RETINEX_DATA" \
        --dest "$BANK_DIR" \
        --class-id 0 \
        --margin 0.15
fi

# 4. Step B: Synthetically Augment Training Boards with Copy-Paste (if not present)
AUG_DATA="datasets/pcb-retinex-cappaste-1280"
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "=========================================================================="
    echo "🎨 Building Retinex + Bbox Copy-Paste Augmented Dataset..."
    echo "=========================================================================="
    python bbox_copy_paste.py \
        --source "$RETINEX_DATA" \
        --bank "$BANK_DIR" \
        --dest "$AUG_DATA" \
        --class-id 0 \
        --paste-prob 0.7 \
        --min-pastes 1 \
        --max-pastes 4
fi

echo "=========================================================================="
echo "🚀 Training ULTIMATE CHAMPION MODEL on NVIDIA H100 (1280px)"
echo "   Techniques Fused:"
echo "   1. Retinex Material Reflectance + Canny Edge Enhancement"
echo "   2. Synthetic Micro-Capacitor Copy-Paste Augmentation"
echo "   3. Balanced Loss Reweighting (cls=1.5, box=5.0, dfl=2.0, ls=0.10)"
echo "   Run Key: yolov26s_ultimate_retinex_copypaste_1280"
echo "   Data:    $AUG_DATA/data.yaml"
echo "   Target:  Break All-Time Leaderboard Record (>65% mAP50)!"
echo "=========================================================================="

# 5. Train YOLO26s (1280px Native Resolution)
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key yolov26s_ultimate_retinex_copypaste_1280 \
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

echo "✅ Ultimate Champion Model Training & Evaluation Completed!"
