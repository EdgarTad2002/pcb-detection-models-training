#!/bin/bash
#SBATCH --job-name=run_r_superyolo
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

# 1. Build the LR/HR pair from the STANDARD 640px dataset only -- no
#    native-res source needed. LR = degraded (downsample+upsample), HR =
#    original sharp 640x640 image (SR reconstruction target only).
if [ ! -f "datasets/pcb-sr-640/data.yaml" ]; then
    python build_sr_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest datasets/pcb-sr-640 \
        --degradation-factor 4
fi

# 2. Sanity check: confirm layer 4 is really a 256-channel stride-8 block
#    for the installed Ultralytics version before committing to a full run.
python -c "
from ultralytics import YOLO
m = YOLO('yolo26s.pt')
layer = m.model.model[4]
print('Layer 4:', layer)
"

# 3. Train YOLO26s with the auxiliary SuperYOLO-style SR branch
python sr_yolo26.py \
    --run-key yolov26s_superyolo_640 \
    --data datasets/pcb-sr-640/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 --sr-lambda 1.0
