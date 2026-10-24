#!/bin/bash
#SBATCH --job-name=v26s_ctdn
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_exp6_ctdn_stem_yolo26s_%j.out

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

TRAIN_DATA="datasets/pcb-native-res-cleaned-1280/data.yaml"
EVAL_DATA="datasets/pcb-native-res-cleaned-1280/data.yaml"
RUN_KEY="yolo26s_exp6_ctdn_stem_1280"

echo "=========================================================================="
echo "🧠 [Exp 6] Training YOLO26s with In-Network CTDN Transformer Stem"
echo "   Run Key:       $RUN_KEY"
echo "   Architecture:  CTDN Attention Stem (Cross-Attention + Self-Attention)"
echo "   Weights:       yolo26s.pt"
echo "   Train Data:    $TRAIN_DATA (Raw Cleaned Native)"
echo "   Eval Data:     $EVAL_DATA"
echo "   Resolution:    1280px Native Input, max_det=1000"
echo "=========================================================================="

python models/train_case1_ctdn_stem.py \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --lambda-group 0.15 \
    --use-squared-dice \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --optimizer SGD \
    --eval-split test \
    --max-det 1000 \
    --device 0

echo "✅ Exp 6 (YOLO26s CTDN Stem) Finished Successfully!"
