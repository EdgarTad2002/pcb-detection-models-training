#!/bin/bash
#SBATCH --job-name=bilat_ret_y26s_1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_bilat_ret_y26s_1280_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "=========================================================================="
echo "☀️  STEP 1: Building Edge-Preserving Bilateral Retinex Residual Dataset..."
echo "=========================================================================="
python tools/enhance_pcb_retinex.py \
    --source datasets/pcb-unified-4class-cleaned \
    --dest datasets/pcb-unified-4class-bilateral-retinex \
    --alpha 0.30 \
    --beta 0.10 \
    --workers 8

echo "=========================================================================="
echo "🚀 STEP 2: Training YOLO26s on Bilateral Retinex Fused Dataset (1280px)..."
echo "=========================================================================="
python train.py \
    --run-key rectified_yolov26s_bilateral_retinex_1280 \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class-bilateral-retinex/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001

echo "=========================================================================="
echo "✅ Bilateral Retinex YOLO26s 1280 Training & Evaluation Complete!"
echo "=========================================================================="
