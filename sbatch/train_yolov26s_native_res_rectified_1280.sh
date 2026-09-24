#!/bin/bash
#SBATCH --job-name=y26_nat_rect1280
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

# Train YOLO26s on the Genuine Native-Resolution Rectified 4-Class Dataset (1280px)
# Uses true uncompressed camera captures (~1500px wide) with rectified 4-class taxonomy
python train.py \
    --run-key yolov26s_native_res_rectified_1280 \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-unified-4class/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001
