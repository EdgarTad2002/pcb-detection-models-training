#!/bin/bash
#SBATCH --job-name=unified_gamma_1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_unified_gamma_1280_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

ALPHA=${1:-0.35}
AGC_ALPHA=${2:-0.50}

BASE_CAPPASTE="datasets/pcb-native-res-rectified-cappaste-1280"
DST_DATA="datasets/pcb-unified-cappaste-adaptive-gamma-1280"
RUN_KEY="rectified_yolov26s_unified_adaptive_gamma_1280"

echo "=========================================================================="
echo "☀️  STEP 1: Building Unified Copy-Paste + Adaptive Gamma Dataset..."
echo "    Base Dataset: $BASE_CAPPASTE"
echo "    Destination:  $DST_DATA"
echo "    Parameters:   Alpha=$ALPHA, AGC_Alpha=$AGC_ALPHA (Zero Canny)"
echo "=========================================================================="

if [ ! -f "$DST_DATA/data.yaml" ]; then
    python tools/enhance_pcb_adaptive_gamma.py \
        --source "$BASE_CAPPASTE" \
        --dest "$DST_DATA" \
        --method agaian2023 \
        --alpha "$ALPHA" \
        --workers 8
else
    echo ">>> Reusing existing dataset: $DST_DATA"
fi

echo "=========================================================================="
echo "🚀 STEP 2: Training Unified Champion (Model 6) with Adaptive Gamma (1280px)..."
echo "    Run Key:      $RUN_KEY"
echo "=========================================================================="

python train.py \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$DST_DATA/data.yaml" \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --max-det 1000 \
    --eval-conf 0.001

echo "=========================================================================="
echo "✅ Unified Adaptive Gamma YOLO26s 1280 Training Complete: $RUN_KEY"
echo "=========================================================================="
