#!/bin/bash
#SBATCH --job-name=sahi_eval
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_sahi_eval_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

WEIGHTS="${1:-runs/yolo26s_native_cleaned_agwd_cappaste_1280/pcb-filtered/weights/best.pt}"
MODEL_KEY="${2:-yolo26s_agwd}"

echo "=========================================================="
echo "Running Apple-to-Apple Standard vs Sliced Evaluation"
echo "Weights: $WEIGHTS"
echo "=========================================================="

# 1. Un-sliced standard baseline through THIS EXACT EVALUATION SCRIPT
echo "--- 1. Evaluating Standard Un-sliced Baseline ---"
python tools/eval_sahi_native_test.py \
    --weights "$WEIGHTS" \
    --no-slices \
    --conf 0.001 \
    --iou 0.50 \
    --nms-iou 0.50 \
    --max-det 1000 \
    --output-json "/mnt/weka/etadevosyan/pcb-yolo/results/${MODEL_KEY}_apples_standard_test.json"

# 2. Sliced SAHI with Global-Local Fusion + IoS Seam Merging
echo "--- 2. Evaluating Sliced SAHI (Global-Local Fusion) ---"
python tools/eval_sahi_native_test.py \
    --weights "$WEIGHTS" \
    --slice-size 1280 \
    --overlap 0.20 \
    --ios-thresh 0.65 \
    --conf 0.001 \
    --iou 0.50 \
    --nms-iou 0.50 \
    --max-det 1000 \
    --output-json "/mnt/weka/etadevosyan/pcb-yolo/results/${MODEL_KEY}_sahi_globallocal_test.json"

# 3. Sliced SAHI Ablation without Full Frame (Slices Only)
echo "--- 3. Evaluating Sliced SAHI Ablation (Slices Only, No Global Pass) ---"
python tools/eval_sahi_native_test.py \
    --weights "$WEIGHTS" \
    --slice-size 1280 \
    --overlap 0.20 \
    --ios-thresh 0.65 \
    --no-full-frame \
    --conf 0.001 \
    --iou 0.50 \
    --nms-iou 0.50 \
    --max-det 1000 \
    --output-json "/mnt/weka/etadevosyan/pcb-yolo/results/${MODEL_KEY}_sahi_slicesonly_test.json"

echo "All 3 Evaluations Complete."
