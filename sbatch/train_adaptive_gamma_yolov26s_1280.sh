#!/bin/bash
#SBATCH --job-name=adapt_gamma_y26s_1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_adapt_gamma_y26s_1280_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

SEED=${1:-0}
ALPHA=${2:-0.35}
AGC_ALPHA=${3:-0.50}

SRC_DATA="datasets/pcb-unified-4class-cleaned"
DST_DATA="datasets/pcb-adaptive-gamma-a${ALPHA}-1280"
RUN_KEY="rectified_yolov26s_adaptive_gamma_s${SEED}_1280"

echo "=========================================================================="
echo "☀️  STEP 1: Building Adaptive Gamma Dataset (ZERO CANNY EDGES)..."
echo "    Source:       $SRC_DATA"
echo "    Destination:  $DST_DATA"
echo "    Alpha:        $ALPHA (AGCWD alpha: $AGC_ALPHA)"
echo "=========================================================================="

if [ ! -f "$DST_DATA/data.yaml" ]; then
    python tools/enhance_pcb_adaptive_gamma.py \
        --source "$SRC_DATA" \
        --dest "$DST_DATA" \
        --method agaian2023 \
        --alpha "$ALPHA" \
        --workers 8
else
    echo ">>> Reusing existing dataset: $DST_DATA"
fi

echo "=========================================================================="
echo "🚀 STEP 2: Training YOLO26s on Adaptive Gamma Dataset (1280px)..."
echo "    Run Key:      $RUN_KEY"
echo "    Seed:         $SEED"
echo "=========================================================================="

python train.py \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$DST_DATA/data.yaml" \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --seed "$SEED" \
    --max-det 1000 \
    --eval-conf 0.001

echo "=========================================================================="
echo "✅ Adaptive Gamma YOLO26s 1280 Training Complete: $RUN_KEY"
echo "=========================================================================="
