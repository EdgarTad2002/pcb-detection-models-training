#!/bin/bash
#SBATCH --job-name=retinex_y26s_1280
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_retinex_y26s_1280_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "=========================================================================="
echo "☀️  STEP 1: Building Retinex-Rectified Dataset (Fu et al., CVPR 2016)..."
echo "=========================================================================="
python tools/build_retinex_cleaned_dataset.py \
    --src-dir datasets/pcb-unified-4class-cleaned \
    --dst-dir datasets/pcb-unified-4class-cleaned-retinex \
    --device cuda \
    --c1 0.01 \
    --c2 0.1 \
    --lambd 1.0 \
    --gamma 2.2 \
    --max-iter 15 \
    --eps1 1e-3 \
    --eps2 1e-3 \
    --batch-size 4 \
    --overwrite

echo "=========================================================================="
echo "🚀 STEP 2: Training YOLO26s on Retinex-Rectified Cleaned Dataset (1280px)..."
echo "=========================================================================="
python train.py \
    --run-key rectified_yolov26s_retinex_cleaned_1280 \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class-cleaned-retinex/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001

echo "=========================================================================="
echo "✅ Retinex-YOLO26s Cleaned 1280 Training & Evaluation Complete!"
echo "=========================================================================="
