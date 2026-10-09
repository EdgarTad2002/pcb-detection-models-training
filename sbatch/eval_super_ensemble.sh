#!/bin/bash
#SBATCH --job-name=super_ens
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_eval_super_ensemble_%j.out

set -e

# 1. Environment & Caching
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

# 2. Checkpoints
W_CHAMP="runs/yolov26s_ultimate_retinex_copypaste_1280/pcb-filtered/weights/best.pt"
W_STEM="runs/yolov26s_retinex_stem_copypaste_1280/pcb-filtered/weights/best.pt"
W_P2="runs/yolov26s_p2_warmstart_retinex_copypaste_1280/pcb-filtered/weights/best.pt"
W_LOSS="runs/yolov26s_retinex_loss_reweight_1280/pcb-filtered/weights/best.pt"

# Verify champion exists
[ -f "$W_CHAMP" ] || { echo "❌ Missing $W_CHAMP"; exit 1; }

# Locate test dataset
if [ -f "datasets/pcb-retinex-cappaste-1280/data.yaml" ]; then
    DATA_YAML="datasets/pcb-retinex-cappaste-1280/data.yaml"
elif [ -f "datasets/pcb-retinex-enhanced-1280/data.yaml" ]; then
    DATA_YAML="datasets/pcb-retinex-enhanced-1280/data.yaml"
else
    DATA_YAML="datasets/pcb-native-res-unified-4class/data.yaml"
fi

echo "=========================================================================="
echo "🎯 RUNNING HIGH-RESOLUTION WEIGHTED BOX FUSION (WBF) SUPER-ENSEMBLES"
echo "   Dataset: $DATA_YAML"
echo "=========================================================================="

# Ensemble 1: Top-2 Dual Specialists (Micro-Capacitor Champion + Connector Specialist)
if [ -f "$W_STEM" ]; then
    echo ""
    echo ">>> [1/3] Running Top-2 Ensemble: Retinex Champion + Learnable Stem..."
    python tools/eval_super_ensemble.py \
        --data "$DATA_YAML" \
        --split test \
        --imgsz 1280 \
        --weights "$W_CHAMP" "$W_STEM" \
        --model-weights 1.1 1.0 \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "ensemble_wbf_top2_specialists_1280" \
        --device 0
fi

# Ensemble 2: Top-3 Tri-Model (Champion + Stem + P2 Head)
if [ -f "$W_STEM" ] && [ -f "$W_P2" ]; then
    echo ""
    echo ">>> [2/3] Running Top-3 Ensemble: Champion + Stem + P2 Head..."
    python tools/eval_super_ensemble.py \
        --data "$DATA_YAML" \
        --split test \
        --imgsz 1280 \
        --weights "$W_CHAMP" "$W_STEM" "$W_P2" \
        --model-weights 1.2 1.0 0.9 \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "ensemble_wbf_top3_champions_1280" \
        --device 0
fi

# Ensemble 3: Top-4 Quad Consensus (Champion + Stem + P2 + Loss Reweight)
if [ -f "$W_STEM" ] && [ -f "$W_P2" ] && [ -f "$W_LOSS" ]; then
    echo ""
    echo ">>> [3/3] Running Top-4 Ensemble: Quad Consensus..."
    python tools/eval_super_ensemble.py \
        --data "$DATA_YAML" \
        --split test \
        --imgsz 1280 \
        --weights "$W_CHAMP" "$W_STEM" "$W_P2" "$W_LOSS" \
        --model-weights 1.2 1.0 0.9 1.0 \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "ensemble_wbf_top4_quad_1280" \
        --device 0
fi

# 3. Update Master Leaderboard
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    echo ""
    echo "=========================================================================="
    echo "📊 Updating Master Leaderboard..."
    echo "=========================================================================="
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    mkdir -p results
    cp "$SHARED_RESULTS_DIR"/comparison_table.* results/ 2>/dev/null || true
    echo ""
    echo "🏆 Top Leaderboard Entries:"
    head -n 25 "$SHARED_RESULTS_DIR/comparison_table.md"
fi

echo "=========================================================================="
echo "✅ Super-Ensemble Evaluation Completed Successfully!"
echo "=========================================================================="
