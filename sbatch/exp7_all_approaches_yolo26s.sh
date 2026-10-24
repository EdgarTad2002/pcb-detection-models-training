#!/bin/bash
#SBATCH --job-name=v26s_all
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_exp7_all_approaches_yolo26s_%j.out

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
    CANONICAL_DATA="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/data.yaml"
elif [ -f "datasets/pcb-filtered-yolov8/data.yaml" ]; then
    CANONICAL_DATA="datasets/pcb-filtered-yolov8/data.yaml"
else
    CANONICAL_DATA="data/pcb_filtered_4class.yaml"
fi

# Training uses Grand Unified enhanced data if available, else canonical
if [ -f "datasets/pcb-grand-unified-filtered/data.yaml" ]; then
    TRAIN_DATA="datasets/pcb-grand-unified-filtered/data.yaml"
elif [ -f "datasets/pcb-agaian-retinex-filtered/data.yaml" ]; then
    TRAIN_DATA="datasets/pcb-agaian-retinex-filtered/data.yaml"
else
    TRAIN_DATA="$CANONICAL_DATA"
fi
EVAL_DATA="$CANONICAL_DATA"

IMGSZ=${PCB_IMGSZ:-640}
BATCH=${PCB_BATCH:-16}
RUN_KEY="yolo26s_filtered_exp7_all_approaches"

echo "=========================================================================="
echo "🔥 [Exp 7] Training YOLO26s: ALL APPROACHES TOGETHER (4-Class pcb-filtered)"
echo "   Run Key:       $RUN_KEY"
echo "   Weights:       yolo26s.pt"
echo "   Train Data:    $TRAIN_DATA"
echo "   Eval Data:     $EVAL_DATA (Canonical 4-Class)"
echo "   Loss:          DualCENet Multi-Stage SDice (Eq. 8, Scale-Aware Weights)"
echo "   Resolution:    ${IMGSZ}px, Batch: $BATCH, max_det=1000"
echo "=========================================================================="

python models/superyolo_multistage.py \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --lambda-sr 0.10 \
    --lambda-group 0.15 \
    --use-squared-dice \
    --epochs 100 \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --workers 8 \
    --eval-split test \
    --max-det 1000 \
    --device 0

echo "✅ Exp 7 (YOLO26s All Approaches on pcb-filtered) Finished Successfully!"
