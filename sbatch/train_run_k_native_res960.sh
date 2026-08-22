#!/bin/bash
#SBATCH --job-name=run_k_native_res960
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

# Same native-res source as Run J (yolov26s_native_res), but at a more
# deployable imgsz=960 instead of 1280. Tests how much of Run J's Capacitor/
# mAP gain survives at meaningfully better FPS -- Run J hit 8.45 FPS, too
# slow for the paper's "industrial conveyor belt" framing.
python train.py \
    --run-key yolov26s_native_res960 \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res/data.yaml \
    --epochs 100 --imgsz 960 --batch 16 --workers 8
