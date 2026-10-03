#!/bin/bash
#SBATCH --job-name=luma_pt_640
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_luma_pt_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "=========================================================================="
echo "🚀 Training YOLO26s-LUMA-PCB with Full Pretrained COCO Transfer (86.3% Params)"
echo "Features: LIAM (Identity Init) + AConv (Anti-Aliased) + SimAM + SPPELAN"
echo "Dataset:  datasets/pcb-unified-4class/data.yaml (640px, 100 epochs)"
echo "=========================================================================="

echo "📦 1. Building / verifying clean pretrained LUMA checkpoint..."
python tools/build_luma_pretrained_checkpoint.py

echo ""
echo "🔥 2. Starting training from clean pretrained checkpoint..."
python train.py \
    --run-key rectified_yolov26s_luma_pretrained_640 \
    --weights weights/yolo26s_luma_pretrained.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --eval-conf 0.001

echo ""
echo "📊 Updating comparison table..."
python aggregate_results.py

echo "✅ YOLO26s-LUMA-Pretrained training completed!"
