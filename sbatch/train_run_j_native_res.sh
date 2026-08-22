#!/bin/bash
#SBATCH --job-name=run_j_native_res
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

# Trains directly on the NATIVE-RESOLUTION source (no tiling, no P2 head,
# no augmentation changes) -- isolates whether genuine extra detail + a
# higher imgsz helps, separate from Run E's earlier test (which only
# upscaled the already-downsampled 640px images -- no new information).
# Also sidesteps Roboflow v3's "stretch to 640x640" aspect-ratio distortion,
# since Ultralytics' own letterbox resize preserves aspect ratio.
python train.py \
    --run-key yolov26s_native_res \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8
