#!/bin/bash
#SBATCH --job-name=yolo26s_rect_bilateral_1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "=========================================================="
echo "Training YOLO26s on Rectified Bilateral-Filtered Dataset (1280px)"
echo "Source: datasets/pcb-unified-4class -> datasets/pcb-rectified-bilateral-640"
echo "Warm-Start Checkpoint: yolov26s_rectified_loss_reweight_1280"
echo "Loss Tuning: --cls 1.5 --box 5.0"
echo "=========================================================="

# 1. Build bilateral dataset from rectified 4-class dataset if not already present
if [ ! -f "datasets/pcb-rectified-bilateral-640/data.yaml" ]; then
    echo "Generating bilateral-filtered dataset from datasets/pcb-unified-4class..."
    python tools/build_bilateral_dataset.py \
        --source datasets/pcb-unified-4class \
        --dest datasets/pcb-rectified-bilateral-640 \
        --d 5 --sigma-color 75 --sigma-space 75 \
        --workers 8
fi

# 2. Train YOLO26s at 1280px warm-started from the #1 champion checkpoint
python train.py \
    --run-key yolov26s_rectified_bilateral_1280 \
    --weights runs/yolov26s_rectified_loss_reweight_1280/pcb-filtered/weights/best.pt \
    --data datasets/pcb-rectified-bilateral-640/data.yaml \
    --cls 1.5 \
    --box 5.0 \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001
