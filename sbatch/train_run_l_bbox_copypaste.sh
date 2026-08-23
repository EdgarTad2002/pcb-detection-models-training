#!/bin/bash
#SBATCH --job-name=run_l_bbox_cappaste
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

# 1. Build the capacitor crop bank (only once)
if [ ! -d "capacitor_bank" ] || [ -z "$(ls -A capacitor_bank 2>/dev/null)" ]; then
    python build_capacitor_bank.py \
        --source datasets/pcb-native-res \
        --dest capacitor_bank \
        --margin 0.15
fi

# 2. Build the augmented dataset (only once)
if [ ! -f "datasets/pcb-native-res-cappaste/data.yaml" ]; then
    python bbox_copy_paste.py \
        --source datasets/pcb-native-res \
        --bank capacitor_bank \
        --dest datasets/pcb-native-res-cappaste \
        --paste-prob 0.7 --min-pastes 1 --max-pastes 4
fi

# 3. Train standard YOLO26s on the augmented data -- reuses train.py
#    unmodified, same imgsz/epochs as your baseline for a fair comparison.
python train.py \
    --run-key yolov26s_bbox_cappaste \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-cappaste/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
