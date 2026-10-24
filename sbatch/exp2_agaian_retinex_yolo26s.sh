#!/bin/bash
#SBATCH --job-name=v26s_c2
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_exp2_agaian_retinex_yolo26s_%j.out

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

# Base 4-class pcb-filtered-yolov8 dataset
if [ -f "/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/data.yaml" ]; then
    SRC_DATA="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8"
elif [ -f "datasets/pcb-filtered-yolov8/data.yaml" ]; then
    SRC_DATA="datasets/pcb-filtered-yolov8"
else
    SRC_DATA="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8"
fi

EVAL_DATA="$SRC_DATA/data.yaml"
AGAIAN_DATA="datasets/pcb-agaian-retinex-filtered"

if [ ! -f "$AGAIAN_DATA/data.yaml" ]; then
    echo "⚡ Generating Case 2 Agaian Fractional Retinex dataset from $SRC_DATA..."
    python tools/generate_case2_agaian_dataset.py \
        --source "$SRC_DATA" \
        --dest "$AGAIAN_DATA" \
        --workers 8
fi

IMGSZ=${PCB_IMGSZ:-640}
BATCH=${PCB_BATCH:-16}
RUN_KEY="yolo26s_filtered_exp2_agaian_retinex"

echo "=========================================================================="
echo "🔬 [Exp 2] Training YOLO26s on Agaian Retinex (Filtered 4-Class)"
echo "   Run Key:       $RUN_KEY"
echo "   Weights:       yolo26s.pt"
echo "   Train Data:    $AGAIAN_DATA/data.yaml"
echo "   Eval Data:     $EVAL_DATA"
echo "   Resolution:    ${IMGSZ}px, Batch: $BATCH, max_det=1000"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$AGAIAN_DATA/data.yaml" \
    --eval-data "$EVAL_DATA" \
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

echo "✅ Exp 2 (YOLO26s Agaian Retinex) Finished Successfully!"
