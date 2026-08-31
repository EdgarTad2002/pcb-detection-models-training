#!/bin/bash
#SBATCH --job-name=run_s_superyolo_native
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

# 1. Build the TRUE LR/HR pair from the 640px dataset and Native dataset.
if [ ! -f "datasets/pcb-sr-native/data.yaml" ]; then
    python build_sr_dataset_v2.py \
        --source-lr datasets/pcb-filtered-yolov8 \
        --source-hr datasets/pcb-native-res \
        --dest datasets/pcb-sr-native
fi

# 2. Train YOLO26s with the auxiliary SuperYOLO-style SR branch using genuine high-res targets
python sr_yolo26.py \
    --run-key yolov26s_superyolo_native \
    --data datasets/pcb-sr-native/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 --sr-lambda 1.0
