#!/bin/bash
#SBATCH --job-name=run_w_bilateral
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
echo "Running Run W: Bilateral Edge-Preserving Filter on YOLO26s"
echo "Filter Parameters: d=5, sigmaColor=75, sigmaSpace=75"
echo "Dataset: datasets/pcb-bilateral-640"
echo "Direct Baseline Comparison: yolov26s (51.68% mAP@0.5)"
echo "=========================================================="

# 1. Build bilateral dataset if not already present
if [ ! -f "datasets/pcb-bilateral-640/data.yaml" ]; then
    echo "Generating bilateral-filtered dataset from datasets/pcb-filtered-yolov8..."
    python tools/build_bilateral_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest datasets/pcb-bilateral-640 \
        --workers 8
fi

# 2. Train standard YOLO26s (100% matched hyperparameters to baseline)
python train.py \
    --run-key yolov26s_bilateral_filter_640 \
    --weights yolo26s.pt \
    --data datasets/pcb-bilateral-640/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --results-dir /mnt/weka/etadevosyan/pcb-yolo/results \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001
