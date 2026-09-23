#!/bin/bash
#SBATCH --job-name=vmamba_tiny
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_vmamba_%j.out

# 1. Configure storage and cache variables
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

# 2. Activate Conda environment from shared cache
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# 3. Dedicated Isolated Workspace on Weka
# By default, uses an isolated folder to prevent touching or overwriting other codes/runs
WORKSPACE_DIR=${PCB_MAMBA_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-mamba-standalone"}
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# 4. Link shared datasets if not present in the isolated workspace
DATASET_SOURCE="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets"
if [ ! -d "datasets" ] && [ -d "$DATASET_SOURCE" ]; then
    echo "🔗 Symlinking shared PCB datasets from $DATASET_SOURCE..."
    ln -s "$DATASET_SOURCE" datasets
fi

# 5. Train Standalone VMamba Object Detector (Non-YOLO State Space Model)
echo "=========================================================================="
echo "🚀 Training Standalone VMamba Object Detector on NVIDIA H100 (YSU Cluster)"
echo "   Workspace: $WORKSPACE_DIR"
echo "=========================================================================="
python train_mamba.py \
    --run-key vmamba_standalone_tiny \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --lr 0.0001 \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50

# 6. Aggregate results within the workspace
if [ -f "aggregate_results.py" ]; then
    python aggregate_results.py
fi
