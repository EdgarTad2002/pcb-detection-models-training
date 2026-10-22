#!/bin/bash
#SBATCH --job-name=v26s_gu
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_exp3_grand_unified_yolo26s_%j.out

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

TRAIN_DATA="datasets/pcb-grand-unified-cappaste-1280/data.yaml"
EVAL_DATA="datasets/pcb-native-res-cleaned-1280/data.yaml"
RUN_KEY="yolo26s_exp3_grand_unified_1280"

echo "=========================================================================="
echo "🚀 [Exp 3] Training YOLO26s on Grand Unified Dataset (Retinex + AGC + Weber)"
echo "   Run Key:       $RUN_KEY"
echo "   Weights:       yolo26s.pt"
echo "   Train Data:    $TRAIN_DATA"
echo "   Eval Data:     $EVAL_DATA"
echo "   Resolution:    1280px Native Input, max_det=1000"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --optimizer SGD \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --label-smoothing 0.10 \
    --copy-paste 0.30 \
    --mosaic 1.0 \
    --mixup 0.15 \
    --max-det 1000 \
    --eval-split test \
    --device 0

echo "✅ Exp 3 (YOLO26s Grand Unified) Finished Successfully!"
