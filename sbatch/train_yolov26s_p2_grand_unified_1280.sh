#!/bin/bash
#SBATCH --job-name=v26s_p2_gu
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_yolov26s_p2_grand_unified_1280_%j.out

# ==============================================================================
# CASE 3: YOLO26s-P2 Architecture on Grand Unified Dataset
# Architecture:
#   - 4 Detection Heads: P2, P3, P4, P5
#   - P2 Head: Stride 4 (320x320 grid @ 1280px native input)
#   - Directly resolves tiny 0201/0402 SMD capacitors (4x6 pixels)
#   - Warm-started from yolo26s.pt
#   - Optimizer: SGD, cls=1.5, box=5.0, dfl=2.0, ls=0.10, max_det=1000 @ 1280px
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

AUG_DATA="datasets/pcb-grand-unified-cappaste-1280"
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "⚠️ $AUG_DATA/data.yaml not found. Generating dataset..."
    bash sbatch/train_grand_unified_superyolo_1280.sh || true
fi

RUN_KEY="yolov26s_p2_grand_unified_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "🚀 Training YOLO26s-P2 (4 Detection Heads) on Grand Unified on NVIDIA H100"
echo "   Run Key:       $RUN_KEY"
echo "   Architecture:  yolo26s-p2.yaml (Strides 4, 8, 16, 32)"
echo "   Data:          $AUG_DATA/data.yaml"
echo "   Heads:         P2 (320x320 grid), P3, P4, P5"
echo "   Resolution:    1280px Native Input, max_det = 1000"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s-p2.yaml \
    --pretrained-weights yolo26s.pt \
    --data "$AUG_DATA/data.yaml" \
    --optimizer SGD \
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
    --eval-split test \
    --max-det 1000

# Update Leaderboard
if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR" || true
    echo ""
    echo "=========================================================================="
    echo "🏆 Comparison with Previous Models:"
    grep -E "^\| ($RUN_KEY|yolov26s_grand_unified_1280|superyolo26s_grand_unified_1280) \|" \
        "$SHARED_RESULTS_DIR/comparison_table.md" || true
    echo "=========================================================================="
fi

echo "=========================================================================="
echo "✅ Case 3 YOLO26s-P2 Grand Unified Training & Evaluation Complete!"
echo "=========================================================================="
