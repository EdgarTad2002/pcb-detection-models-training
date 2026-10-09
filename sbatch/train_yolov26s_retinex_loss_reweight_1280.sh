#!/bin/bash
#SBATCH --job-name=v26s_ret_rw1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_yolov26s_retinex_loss_reweight_1280_%j.out

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

# 4. Use the Retinex-Enhanced 1280px Dataset
ENHANCED_DATA="datasets/pcb-retinex-enhanced-1280/data.yaml"
if [ ! -f "$ENHANCED_DATA" ]; then
    echo "❌ ERROR: Enhanced dataset not found at $ENHANCED_DATA!"
    exit 1
fi

echo "=========================================================================="
echo "🚀 Training YOLO26s: Retinex Material + Canny + Loss Reweighting (1280px)"
echo "   Workspace: $WORKSPACE_DIR"
echo "   Data:      $ENHANCED_DATA"
echo "   Weights:   yolo26s.pt"
echo "   Loss Regimen: cls=1.5, box=5.0, dfl=2.0, label_smoothing=0.10"
echo "   Target:    New All-Time Project Leaderboard Record (>64% mAP50)"
echo "=========================================================================="

# 5. Train YOLO26s with Champion Loss Reweighting at 1280px
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key yolov26s_retinex_loss_reweight_1280 \
    --weights yolo26s.pt \
    --data "$ENHANCED_DATA" \
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

echo "✅ All-Time Record Attempt Training & Evaluation Completed!"
