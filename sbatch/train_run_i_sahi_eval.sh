#!/bin/bash
#SBATCH --job-name=run_i_sahi_eval
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

# Run this AFTER train_run_i_sahi.sh has finished, once best.pt exists.
# Uses the ORIGINAL (not native-res) test images for eval, matching every
# other model's evaluation source in this project.
python eval_sahi.py \
    --run-key yolov26s_sahi_tiles_native_slicedinfer \
    --weights runs/yolov26s_sahi_tiles_native/pcb-filtered/weights/best.pt \
    --test-images datasets/pcb-filtered-yolov8/test/images \
    --test-labels datasets/pcb-filtered-yolov8/test/labels \
    --slice-size 640 --overlap 0.2 \
    --conf 0.25 --iou 0.5
