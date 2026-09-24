#!/bin/bash
#SBATCH --job-name=superyolo26s
#SBATCH --partition=research
#SBATCH --mem=32G
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

# Train SuperYOLO (YOLO26s with Auxiliary PixelShuffle Super-Resolution Head)
# Uses genuine native-resolution rectified dataset with 640px input and lambda_sr=0.10
python superyolo_pcb.py \
    --run-key superyolo26s_rectified_640 \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-unified-4class/data.yaml \
    --lambda-sr 0.10 \
    --cls-weight 1.5 \
    --box-weight 5.0 \
    --dfl-weight 2.0 \
    --label-smoothing 0.10 \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001
