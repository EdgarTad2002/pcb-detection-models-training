#!/bin/bash
#SBATCH --job-name=sota_60plus
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sota_60plus_%j.out
#SBATCH --error=logs/sota_60plus_%j.err

set -euo pipefail

echo "================================================================================"
echo "  FINAL SOTA 60%+ PCB DETECTOR SUITE (1536px Micro-Head + Macro-Boosted WBF)"
echo "  Host: $(hostname) | Date: $(date)"
echo "================================================================================"

mkdir -p logs runs/r4_apex_sota_suite results

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4

nvidia-smi || true

AUG_YAML="datasets/pcb-r4-final-apex-copypaste/data.yaml"
RAW_YAML="datasets/pcb-native-reviewed-r4-final/data.yaml"

# ------------------------------------------------------------------------------
# STEP 1: Macro-Boosted APEX-YOLO11m @ 1536px (Target: Boost ElCap to 55%+)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 1/3] Training APEX-YOLO11m Macro-Boost @ 1536px (35 epochs)..."
export APEX_MACRO_BOOST=1
python apex_r4_model.py \
    --model runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt \
    --data "$AUG_YAML" \
    --eval-data "$RAW_YAML" \
    --project runs \
    --name r4_apex_yolo11m_1536_macroboost \
    --epochs 35 \
    --imgsz 1536 \
    --batch 4 \
    --lr0 0.0025 \
    --lrf 0.02 \
    --close-mosaic 10 \
    --patience 20 \
    --aux-mask-weight 0.16 \
    --sr-weight 0.08 \
    --lambda-dice 0.22 \
    --inner-ratio 1.25 \
    --tau-kl 0.28 \
    --soup-top-k 5 \
    --device 0 \
    --workers 6 \
    --seed 42
unset APEX_MACRO_BOOST

# ------------------------------------------------------------------------------
# STEP 2: YOLO26s-P2 Stride-4 Micro-Head @ 1536px (Target: Boost Capacitor to 55%+)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 2/3] Training YOLO26s-P2 Stride-4 Micro-Head @ 1536px (45 epochs)..."
python train.py \
    --model yolo26s-p2.yaml \
    --weights runs/r4_s_yolo26s_p2_crop_specialist_250ep/pcb-filtered/weights/best.pt \
    --data "$AUG_YAML" \
    --eval-data "$RAW_YAML" \
    --eval-split val \
    --imgsz 1536 \
    --batch 4 \
    --epochs 45 \
    --workers 6 \
    --mosaic 1.0 \
    --close-mosaic 12 \
    --patience 20 \
    --scale 0.5 \
    --translate 0.1 \
    --fliplr 0.5 \
    --flipud 0.5 \
    --squared-dice --lambda-dice 0.40 --smooth-dice 0.01 \
    --agwd-ota --agwd-kappa 12.8 --agwd-tau 576.0 \
    --optimizer SGD --lr0 0.003 --lrf 0.01 --momentum 0.937 --weight-decay 0.0005 \
    --max-det 1000 --eval-conf 0.001 \
    --run-key r4_s_yolo26s_p2_1536 \
    --device 0

# ------------------------------------------------------------------------------
# STEP 3: Run Unified 3-Tier SOTA Benchmark on Held-Out Test Split (Target: 60%+ mAP)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Step 3/3] Running Unified 3-Tier APEX Super-Ensemble on Held-Out Test Split..."
python evaluate_apex_r4_suite.py \
    --data "$RAW_YAML" \
    --project runs \
    --device 0 \
    --top-k 8 \
    --include-legacy \
    --out-json results/r4_final_60plus_sota_summary.json

echo ""
echo "================================================================================"
echo "  FINAL SOTA 60%+ SUITE COMPLETED SUCCESSFULLY!"
echo "  Summary JSON: results/r4_final_60plus_sota_summary.json"
echo "================================================================================"
