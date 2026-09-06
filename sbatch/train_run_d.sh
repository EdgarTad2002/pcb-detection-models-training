#!/bin/bash
#SBATCH --job-name=run_d_p2_hires
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

# Fair 640px benchmark: P2 head with COCO pretrained weights transferred,
# trained for 100 epochs at 640px, keeping results in runs/yolov26s_p2_combined_v2
python train.py \
    --run-key yolov26s_p2_combined_v2 \
    --weights yolo26s-p2.yaml \
    --pretrained-weights yolo26s.pt \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
