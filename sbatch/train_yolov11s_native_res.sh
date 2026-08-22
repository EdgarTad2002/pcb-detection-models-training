#!/bin/bash
#SBATCH --job-name=run_yolov11s_native_res
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

# Same native-resolution ablation as yolov26s_native_res, but on YOLOv11s --
# v11s was the strongest baseline at 640px (mAP50 50.3%, best precision+recall
# combo of all 7 baselines), so this checks whether it also benefits from
# genuine extra detail + higher imgsz the way v26s did (Capacitor AP50 jumped
# from 2.7% -> 6.6% under the same treatment).
python train.py \
    --run-key yolov11s_native_res \
    --weights yolo11s.pt \
    --data datasets/pcb-native-res/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8
