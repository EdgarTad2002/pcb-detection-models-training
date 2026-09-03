#!/bin/bash
#SBATCH --job-name=run_t_spectral_yolo
#SBATCH --partition=research
#SBATCH --mem=40G
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

# Run T: Spectral-Enhanced YOLO26s (31-band HSI Reconstructor + Learned 1x1 Adapter)
# Trained on standard 640px dataset (pcb-filtered-yolov8) for direct, fair comparison
# with YOLO26s baseline, evaluating at academic benchmark conf=0.001
python spectral_yolo26.py \
    --run-key yolov26s_spectral_640 \
    --weights yolo26s.pt \
    --data datasets/pcb-filtered-yolov8/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --results-dir /mnt/weka/etadevosyan/pcb-yolo/results \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001

