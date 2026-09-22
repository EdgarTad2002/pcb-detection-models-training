#!/bin/bash
#SBATCH --job-name=eval_sahi
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

echo "=========================================================="
echo "Evaluating SAHI Sliced Inference on Champion Model"
echo "Model: yolov26s_rectified_loss_reweight_1280"
echo "=========================================================="

python tools/eval_sahi_benchmark.py \
    --weights runs/yolov26s_rectified_loss_reweight_1280/pcb-filtered/weights/best.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --split test \
    --imgsz 1280 \
    --slice-size 480 \
    --overlap 0.20 \
    --nms-type diou \
    --device 0

echo "=========================================================="
echo "Evaluating SAHI Sliced Inference on 640px Model"
echo "Model: yolov26s_rectified_640"
echo "=========================================================="

python tools/eval_sahi_benchmark.py \
    --weights runs/yolov26s_rectified_640/pcb-filtered/weights/best.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --split test \
    --imgsz 640 \
    --slice-size 360 \
    --overlap 0.20 \
    --nms-type diou \
    --device 0
