#!/bin/bash
#SBATCH --job-name=v26s_loss
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_exp4_loss_only_yolo26s_%j.out

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
RUN_KEY="yolo26s_filtered_exp4_loss_only_sdice"

echo "=========================================================================="
echo "🎯 [Exp 4] Training YOLO26s Multi-Stage SDice Loss on 4-Class pcb-filtered"
echo "   Run Key:       $RUN_KEY"
echo "   Weights:       yolo26s.pt"
echo "   Data:          $DATA_YAML"
echo "   Loss:          DualCENet Multi-Stage SDice (Eq. 8, Scale-Aware Weights)"
echo "   Resolution:    ${IMGSZ}px, Batch: $BATCH, max_det=1000"
echo "=========================================================================="

python models/superyolo_multistage.py \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
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

echo "✅ Exp 4 (YOLO26s Loss Only on pcb-filtered) Finished Successfully!"
