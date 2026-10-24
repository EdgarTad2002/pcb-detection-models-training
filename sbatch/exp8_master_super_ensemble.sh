#!/bin/bash
#SBATCH --job-name=ens_master
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_exp8_master_super_ensemble_%j.out

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

EVAL_DATA="datasets/pcb-native-res-cleaned-1280/data.yaml"

echo "=========================================================================="
echo "🏆 [Exp 8] Running Master Multi-Paradigm Super-Ensemble (WBF @ 1280px)"
echo "   Canonical Benchmark: $EVAL_DATA (test split, max_det=1000)"
echo "=========================================================================="

# Candidate Checkpoints from Newly Trained Ablation Matrix
CANDIDATES=()
WEIGHTS=()

add_checkpoint() {
    local path="$1"
    local w="$2"
    for candidate in "$path" "runs/$path" "runs/detect/$path" "runs/detect/runs/$path"; do
        if [ -f "$candidate" ]; then
            CANDIDATES+=("$candidate")
            WEIGHTS+=("$w")
            echo "  [+] Included in Ensemble: $candidate (weight=$w)"
            return
        fi
    done
    echo "  [-] Checkpoint not yet found: $path"
}

echo "Discovering trained checkpoints..."
# Baseline Champions (65.2% & 64.5%)
add_checkpoint "yolo26s_exp1_baseline_1280/pcb-filtered/weights/best.pt" 1.30
add_checkpoint "yolo12s_exp1_baseline_1280/pcb-filtered/weights/best.pt" 1.25

# Loss-Only Multi-Stage SDice Specialists
add_checkpoint "yolo26s_exp4_loss_only_sdice_1280/pcb-filtered/weights/best.pt" 1.20
add_checkpoint "yolo12s_exp4_loss_only_sdice_1280/pcb-filtered/weights/best.pt" 1.15

# YOLO26s Architecture Specialists
add_checkpoint "yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt" 1.25
add_checkpoint "yolo26s_exp3_grand_unified_1280/pcb-filtered/weights/best.pt" 1.10
add_checkpoint "yolo26s_exp2_agaian_retinex_1280/pcb-filtered/weights/best.pt" 1.05
add_checkpoint "yolo26s_exp5_conv_stem_1280/pcb-filtered/weights/best.pt" 1.10
add_checkpoint "yolo26s_exp6_ctdn_stem_1280/pcb-filtered/weights/best.pt" 1.15

# YOLO12s Architecture Specialists
add_checkpoint "yolo12s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt" 1.20
add_checkpoint "yolo12s_exp3_grand_unified_1280/pcb-filtered/weights/best.pt" 1.05
add_checkpoint "yolo12s_exp2_agaian_retinex_1280/pcb-filtered/weights/best.pt" 1.00
add_checkpoint "yolo12s_exp5_conv_stem_1280/pcb-filtered/weights/best.pt" 1.05
add_checkpoint "yolo12s_exp6_ctdn_stem_1280/pcb-filtered/weights/best.pt" 1.10

# Historical Reference Champion (if present)
add_checkpoint "yolov26s_ultimate_retinex_copypaste_1280/pcb-filtered/weights/best.pt" 1.15

echo ""
if [ ${#CANDIDATES[@]} -ge 2 ]; then
    echo "🎯 Running Weighted Box Fusion on ${#CANDIDATES[@]} models..."
    python tools/eval_super_ensemble.py \
        --data "$EVAL_DATA" \
        --split test \
        --imgsz 1280 \
        --weights "${CANDIDATES[@]}" \
        --model-weights "${WEIGHTS[@]}" \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "ensemble_wbf_master_all_approaches_1280" \
        --device 0
else
    echo "⚠️ Need at least 2 checkpoints to ensemble. Found ${#CANDIDATES[@]}."
fi

SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    echo ""
    echo "=========================================================================="
    echo "📊 Updating Comparison Leaderboard..."
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR" || true
    head -n 35 "$SHARED_RESULTS_DIR/comparison_table.md" || true
fi

echo "✅ Exp 8 Ensemble Evaluation Complete!"
