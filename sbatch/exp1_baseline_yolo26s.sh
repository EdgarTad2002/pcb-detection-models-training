#!/bin/bash
#SBATCH --job-name=v26s_base
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_exp1_baseline_yolo26s_%j.out

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
    DATA_YAML="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/data.yaml"
elif [ -f "datasets/pcb-filtered-yolov8/data.yaml" ]; then
    DATA_YAML="datasets/pcb-filtered-yolov8/data.yaml"
else
    DATA_YAML="data/pcb_filtered_4class.yaml"
fi

IMGSZ=${PCB_IMGSZ:-640}
BATCH=${PCB_BATCH:-16}
RUN_KEY="yolo26s_filtered_exp1_baseline"

echo "=========================================================================="
echo "🚀 [Exp 1] Training Baseline YOLO26s on 4-Class pcb-filtered-yolov8"
echo "   Run Key:       $RUN_KEY"
echo "   Weights:       yolo26s.pt"
echo "   Data:          $DATA_YAML"
echo "   Resolution:    ${IMGSZ}px, Batch: $BATCH, max_det=1000"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --epochs 100 \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
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

echo "✅ Exp 1 (YOLO26s Baseline on pcb-filtered) Finished Successfully!"
