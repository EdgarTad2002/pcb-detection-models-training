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

# Resolve 4-class pcb-filtered-yolov8 dataset
if [ -f "/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/data.yaml" ]; then
    EVAL_DATA="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/data.yaml"
elif [ -f "datasets/pcb-filtered-yolov8/data.yaml" ]; then
    EVAL_DATA="datasets/pcb-filtered-yolov8/data.yaml"
else
    EVAL_DATA="data/pcb_filtered_4class.yaml"
fi

IMGSZ=${PCB_IMGSZ:-640}
RUN_KEY="ensemble_wbf_master_filtered_4class"

echo "=========================================================================="
echo "🏆 [Exp 8] Master Super-Ensemble (Weighted Box Fusion on 4-Class pcb-filtered)"
echo "   Benchmark:     $EVAL_DATA (test split, max_det=1000)"
echo "   Resolution:    ${IMGSZ}px"
echo "   Run Key:       $RUN_KEY"
echo "=========================================================================="

# Candidate Checkpoints from Newly Trained Filtered Matrix & Prior Runs
CANDIDATES=()
WEIGHTS=()

add_checkpoint() {
    local path="$1"
    local w="$2"
    for candidate in "$path" "runs/$path" "runs/detect/$path" "runs/detect/runs/$path" "/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen/runs/$path" "/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen/runs/detect/runs/$path"; do
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
# Filtered Exp 7: All Approaches
add_checkpoint "yolo26s_filtered_exp7_all_approaches/pcb-filtered/weights/best.pt" 1.35
add_checkpoint "yolo12s_filtered_exp7_all_approaches/pcb-filtered/weights/best.pt" 1.30

# Filtered Exp 1: Baseline
add_checkpoint "yolo26s_filtered_exp1_baseline/pcb-filtered/weights/best.pt" 1.25
add_checkpoint "yolo12s_filtered_exp1_baseline/pcb-filtered/weights/best.pt" 1.20

# Filtered Exp 4: Multi-Stage SDice Loss
add_checkpoint "yolo26s_filtered_exp4_loss_only_sdice/pcb-filtered/weights/best.pt" 1.25
add_checkpoint "yolo12s_filtered_exp4_loss_only_sdice/pcb-filtered/weights/best.pt" 1.20

# Filtered Exp 5 & 6: In-Network Stems
add_checkpoint "yolo26s_filtered_exp6_ctdn_stem/pcb-filtered/weights/best.pt" 1.20
add_checkpoint "yolo12s_filtered_exp6_ctdn_stem/pcb-filtered/weights/best.pt" 1.15
add_checkpoint "yolo26s_filtered_exp5_conv_stem/pcb-filtered/weights/best.pt" 1.15
add_checkpoint "yolo12s_filtered_exp5_conv_stem/pcb-filtered/weights/best.pt" 1.10

# Filtered Exp 3 & 2: Grand Unified & Agaian Retinex
add_checkpoint "yolo26s_filtered_exp3_grand_unified/pcb-filtered/weights/best.pt" 1.15
add_checkpoint "yolo12s_filtered_exp3_grand_unified/pcb-filtered/weights/best.pt" 1.10
add_checkpoint "yolo26s_filtered_exp2_agaian_retinex/pcb-filtered/weights/best.pt" 1.10
add_checkpoint "yolo12s_filtered_exp2_agaian_retinex/pcb-filtered/weights/best.pt" 1.05

# Cross-Resolution & Prior Champions
add_checkpoint "yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt" 1.25
add_checkpoint "yolo12s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt" 1.20
add_checkpoint "yolov26s_ultimate_retinex_copypaste_1280/pcb-filtered/weights/best.pt" 1.15

echo ""
if [ ${#CANDIDATES[@]} -ge 2 ]; then
    echo "🎯 Running Weighted Box Fusion on ${#CANDIDATES[@]} models..."
    python tools/eval_super_ensemble.py \
        --data "$EVAL_DATA" \
        --split test \
        --imgsz "$IMGSZ" \
        --weights "${CANDIDATES[@]}" \
        --model-weights "${WEIGHTS[@]}" \
        --iou-thr 0.55 \
        --max-det 1000 \
        --run-key "$RUN_KEY" \
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
