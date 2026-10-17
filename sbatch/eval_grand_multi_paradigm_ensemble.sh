#!/bin/bash
#SBATCH --job-name=ens_grand
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_eval_grand_multi_paradigm_ensemble_%j.out

# ==============================================================================
# Grand Multi-Paradigm Weighted Box Fusion (WBF) Ensemble
# Combines the complementary strengths of our top specialized models:
#   1. Retinex Champion: Capacitor Specialist (38.58% AP@50)
#   2. Case 2 Agaian Retinex YOLO26s: Connector Specialist (66.38% AP@50)
#   3. Grand Unified YOLO26s: Electrolytic Capacitor Specialist (76.43% AP@50, 40.61% mAP50-95)
#   4. Grand Unified SuperYOLO26s: IC Specialist (62.80% AP@50)
#   5. Case 2 SuperYOLO26s: Maximum Precision (70.77%)
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

# Model Checkpoints
W_CHAMP="runs/yolov26s_ultimate_retinex_copypaste_1280/pcb-filtered/weights/best.pt"
W_GU_YOLO="runs/yolov26s_grand_unified_1280/pcb-filtered/weights/best.pt"
W_GU_SUPER="runs/detect/runs/superyolo26s_grand_unified_1280/pcb-filtered/weights/best.pt"
W_C2_YOLO="runs/yolov26s_case2_agaian_retinex_1280/pcb-filtered/weights/best.pt"
W_C2_SUPER="runs/detect/runs/superyolo26s_case2_agaian_retinex_1280/pcb-filtered/weights/best.pt"

# Locate evaluation test split
if [ -f "datasets/pcb-grand-unified-cappaste-1280/data.yaml" ]; then
    DATA_YAML="datasets/pcb-grand-unified-cappaste-1280/data.yaml"
elif [ -f "datasets/pcb-retinex-cappaste-1280/data.yaml" ]; then
    DATA_YAML="datasets/pcb-retinex-cappaste-1280/data.yaml"
else
    DATA_YAML="datasets/pcb-unified-4class-cleaned/data.yaml"
fi

echo "=========================================================================="
echo "🎯 EVALUATING GRAND MULTI-PARADIGM SUPER-ENSEMBLES (WBF @ 1280px)"
echo "   Dataset: $DATA_YAML (test split, max_det=1000)"
echo "=========================================================================="

# ------------------------------------------------------------------------------
# Ensemble 1: Tri-Specialist Consensus (Capacitor + Connector + El-Cap)
# ------------------------------------------------------------------------------
if [ -f "$W_CHAMP" ] && [ -f "$W_C2_YOLO" ] && [ -f "$W_GU_YOLO" ]; then
    echo ""
    echo ">>> [1/3] Running Tri-Specialist Consensus (Champion + Case2 YOLO + GrandUnified YOLO)..."
    python tools/eval_super_ensemble.py \
        --data "$DATA_YAML" \
        --split test \
        --imgsz 1280 \
        --weights "$W_CHAMP" "$W_C2_YOLO" "$W_GU_YOLO" \
        --model-weights 1.2 1.0 1.0 \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "ensemble_wbf_tri_specialists_1280" \
        --device 0
fi

# ------------------------------------------------------------------------------
# Ensemble 2: Quad-Paradigm Specialist Fusion (Capacitor + Connector + El-Cap + IC)
# ------------------------------------------------------------------------------
if [ -f "$W_CHAMP" ] && [ -f "$W_C2_YOLO" ] && [ -f "$W_GU_YOLO" ] && [ -f "$W_GU_SUPER" ]; then
    echo ""
    echo ">>> [2/3] Running Quad-Paradigm Specialist Fusion..."
    python tools/eval_super_ensemble.py \
        --data "$DATA_YAML" \
        --split test \
        --imgsz 1280 \
        --weights "$W_CHAMP" "$W_C2_YOLO" "$W_GU_YOLO" "$W_GU_SUPER" \
        --model-weights 1.25 1.0 1.0 1.0 \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "ensemble_wbf_quad_specialists_1280" \
        --device 0
fi

# ------------------------------------------------------------------------------
# Ensemble 3: Penta-Model Full Multi-Paradigm WBF
# ------------------------------------------------------------------------------
CANDIDATE_LIST=()
WEIGHT_LIST=()

[ -f "$W_CHAMP" ] && CANDIDATE_LIST+=("$W_CHAMP") && WEIGHT_LIST+=(1.25)
[ -f "$W_GU_YOLO" ] && CANDIDATE_LIST+=("$W_GU_YOLO") && WEIGHT_LIST+=(1.05)
[ -f "$W_C2_YOLO" ] && CANDIDATE_LIST+=("$W_C2_YOLO") && WEIGHT_LIST+=(1.0)
[ -f "$W_GU_SUPER" ] && CANDIDATE_LIST+=("$W_GU_SUPER") && WEIGHT_LIST+=(0.95)
[ -f "$W_C2_SUPER" ] && CANDIDATE_LIST+=("$W_C2_SUPER") && WEIGHT_LIST+=(0.90)

if [ ${#CANDIDATE_LIST[@]} -ge 3 ]; then
    echo ""
    echo ">>> [3/3] Running Full Multi-Paradigm Ensemble (${#CANDIDATE_LIST[@]} models)..."
    python tools/eval_super_ensemble.py \
        --data "$DATA_YAML" \
        --split test \
        --imgsz 1280 \
        --weights "${CANDIDATE_LIST[@]}" \
        --model-weights "${WEIGHT_LIST[@]}" \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "ensemble_wbf_penta_all_approaches_1280" \
        --device 0
fi

# ------------------------------------------------------------------------------
# Update Leaderboard
# ------------------------------------------------------------------------------
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    echo ""
    echo "=========================================================================="
    echo "📊 Updating Master Leaderboard..."
    echo "=========================================================================="
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR" || true
    echo ""
    echo "🏆 Top Leaderboard Entries:"
    head -n 30 "$SHARED_RESULTS_DIR/comparison_table.md" || true
fi

echo "=========================================================================="
echo "✅ Grand Multi-Paradigm Ensemble Evaluation Complete!"
echo "=========================================================================="
