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

# 2. Sanity check: confirm layer 2 is really a 128-channel stride-4 block
#    for the installed Ultralytics version before committing to a full run.
python -c "
from ultralytics import YOLO
m = YOLO('yolo26s.pt')
layer = m.model.model[2]
print('Layer 2 (SR Hook):', layer)
"

# 3. Train YOLO26s with the auxiliary SuperYOLO-style SR branch
#    - imgsz 640: detector trains on degraded 640px input
#    - sr-target-imgsz 640: auxiliary branch reconstructs original 640px sharp image
#    - sr-lambda 100.0: scales L1 loss to balance with detection loss
#    - eval-conf 0.001: evaluates across full PR curve for benchmark table reproduction
python sr_yolo26.py \
    --run-key yolov26s_superyolo_640 \
    --data datasets/pcb-sr-640/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 --sr-lambda 100.0 \
    --sr-target-imgsz 640 --eval-conf 0.001
