#!/bin/bash
#SBATCH --job-name=superyolo_v5s
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

# Build paired SR dataset if not present
if [ ! -f "datasets/pcb-sr-640/data.yaml" ]; then
    python build_sr_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest datasets/pcb-sr-640 \
        --degradation-factor 4
fi

# Train Native SuperYOLO on YOLOv5s (the exact architecture recommended by Prof. Agaian)
# - imgsz 640, 100 epochs, batch 16, eval-conf 0.001
python sr_yolo26.py \
    --run-key yolov5s_superyolo_640 \
    --weights yolov5s.pt \
    --data datasets/pcb-sr-640/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 --sr-lambda 0.5 \
    --sr-target-imgsz 640 --eval-conf 0.001
