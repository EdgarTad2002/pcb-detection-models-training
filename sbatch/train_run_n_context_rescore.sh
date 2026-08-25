#!/bin/bash
#SBATCH --job-name=run_n_context_rescore
#SBATCH --partition=research
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

python context_rescore.py \
    --weights runs/yolov26s_native_res/pcb-filtered/weights/best.pt \
    --run-key yolov26s_native_res_contextrescore \
    --anchor-conf 0.5 --radius-frac 0.15 --boost-factor 2.5
