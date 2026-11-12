#!/bin/bash
#SBATCH --job-name=r4_apex_sota
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=logs/r4_apex_sota_%j.out
#SBATCH --error=logs/r4_apex_sota_%j.err

set -euo pipefail

echo "================================================================================"
echo "  APEX-PCB-YOLO: Next-Gen SOTA Training & 3-Tier Evaluation on R4-Final"
echo "  Host: $(hostname) | Date: $(date)"
echo "================================================================================"

mkdir -p logs runs/r4_apex_sota_suite

# Activate Conda environment
if [ -f "/mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh" ]; then
    source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4

nvidia-smi || true

# ------------------------------------------------------------------------------
# STEP 1: Build Multi-Class Feathered BBox Copy-Paste Dataset from R4-Final
# ------------------------------------------------------------------------------
AUG_YAML="datasets/pcb-r4-final-apex-copypaste/data.yaml"
RAW_YAML="datasets/pcb-native-reviewed-r4-final/data.yaml"

if [ ! -f "$AUG_YAML" ]; then
    echo ""
    echo "[Step 1/5] Building Multi-Class Feathered Copy-Paste Dataset from R4-Final..."
    python build_r4_multiclass_copypaste.py \
        --src-dir datasets/pcb-native-reviewed-r4-final \
        --dst-dir datasets/pcb-r4-final-apex-copypaste \
        --aug-prob 0.88 \
        --max-iou 0.03 \
        --seed 42
else
    echo "[Step 1/5] Augmented dataset already exists at $AUG_YAML"
fi

# ------------------------------------------------------------------------------
# STEP 2: Train Champion 1 — APEX-YOLO11m @ 1536px + Greedy Validation Soup
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/5] Training Champion 1: r4_apex_yolo11m_1536 (imgsz=1536, 65 epochs)..."
python apex_r4_model.py \
    --model yolo11m.pt \
    --data "$AUG_YAML" \
    --eval-data "$RAW_YAML" \
    --project runs/r4_apex_sota_suite \
    --name r4_apex_yolo11m_1536 \
    --epochs 65 \
    --imgsz 1536 \
    --batch 4 \
    --lr0 0.0005 \
    --lrf 0.05 \
    --close-mosaic 15 \
    --patience 25 \
    --aux-mask-weight 0.16 \
    --sr-weight 0.08 \
    --lambda-dice 0.22 \
    --inner-ratio 1.25 \
    --tau-kl 0.28 \
    --soup-top-k 5 \
    --device 0 \
    --workers 6 \
    --seed 42

# ------------------------------------------------------------------------------
# STEP 3: Train Champion 2 — APEX-YOLO26s @ 1536px + Greedy Validation Soup
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/5] Training Champion 2: r4_apex_yolo26s_1536 (imgsz=1536, 65 epochs)..."
python apex_r4_model.py \
    --model yolo26s.pt \
    --data "$AUG_YAML" \
    --eval-data "$RAW_YAML" \
    --project runs/r4_apex_sota_suite \
    --name r4_apex_yolo26s_1536 \
    --epochs 65 \
    --imgsz 1536 \
    --batch 4 \
    --lr0 0.0006 \
    --lrf 0.05 \
    --close-mosaic 15 \
    --patience 25 \
    --aux-mask-weight 0.18 \
    --sr-weight 0.08 \
    --lambda-dice 0.22 \
    --inner-ratio 1.25 \
    --tau-kl 0.28 \
    --soup-top-k 5 \
    --device 0 \
    --workers 6 \
    --seed 42

# ------------------------------------------------------------------------------
# STEP 4: Train Champion 3 — APEX-YOLO26x @ 1280px + Greedy Validation Soup
# ------------------------------------------------------------------------------
echo ""
echo "[Step 4/5] Training Champion 3: r4_apex_yolo26x_1280 (imgsz=1280, 60 epochs)..."
python apex_r4_model.py \
    --model yolo26x.pt \
    --data "$AUG_YAML" \
    --eval-data "$RAW_YAML" \
    --project runs/r4_apex_sota_suite \
    --name r4_apex_yolo26x_1280 \
    --epochs 60 \
    --imgsz 1280 \
    --batch 4 \
    --lr0 0.0004 \
    --lrf 0.05 \
    --close-mosaic 15 \
    --patience 25 \
    --aux-mask-weight 0.15 \
    --sr-weight 0.08 \
    --lambda-dice 0.20 \
    --inner-ratio 1.25 \
    --tau-kl 0.28 \
    --soup-top-k 5 \
    --device 0 \
    --workers 6 \
    --seed 42

# ------------------------------------------------------------------------------
# STEP 5: Run 3-Tier SOTA Evaluation (Single + Per-Class WBF + Quadrant WBF)
# ------------------------------------------------------------------------------
echo ""
echo "[Step 5/5] Running 3-Tier APEX Benchmark on R4-Final Test Split..."
python evaluate_apex_r4_suite.py \
    --data "$RAW_YAML" \
    --project runs/r4_apex_sota_suite \
    --device 0 \
    --include-legacy \
    --out-json runs/r4_apex_sota_suite/r4_apex_sota_benchmark.json

echo ""
echo "================================================================================"
echo "  ALL APEX-PCB-YOLO EXPERIMENTS & 3-TIER EVALUATIONS COMPLETED SUCCESSFULLY!"
echo "  Results JSON: runs/r4_apex_sota_suite/r4_apex_sota_benchmark.json"
echo "================================================================================"
