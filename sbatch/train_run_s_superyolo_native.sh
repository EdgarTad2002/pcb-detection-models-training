#!/bin/bash
#SBATCH --job-name=run_s_superyolo_native
#SBATCH --partition=research
#SBATCH --mem=64G
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
#    - imgsz 640: detector trains on 640px input (matching source-lr)
#    - sr-target-imgsz 1280: auxiliary branch reconstructs native 1280px targets from intermediate layer
#    - sr-lambda 100.0: scales L1 loss to balance with detection loss
#    - eval-conf 0.001: evaluates across full PR curve for benchmark table reproduction
python sr_yolo26.py \
    --run-key yolov26s_superyolo_native_v2 \
    --data datasets/pcb-sr-native/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 --sr-lambda 100.0 \
    --sr-target-imgsz 1280 --eval-conf 0.001

