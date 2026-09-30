#!/bin/bash
#SBATCH --job-name=vmamba_640
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_vmamba_640_%j.out

set -e

# 1. Configure storage and cache variables
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$YOLO_CONFIG_DIR"

# 2. Activate Conda environment from shared cache
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# 3. Dedicated Isolated Workspace on Weka
WORKSPACE_DIR=${PCB_MAMBA_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-mamba-standalone"}
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# 4. Link shared datasets if not present in the isolated workspace
DATASET_SOURCE="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets"
if [ ! -d "datasets" ] && [ -d "$DATASET_SOURCE" ]; then
    echo "🔗 Symlinking shared PCB datasets from $DATASET_SOURCE..."
    ln -s "$DATASET_SOURCE" datasets
fi

# Enforce canonical 640px dataset specified by project guidelines
DATA_PATH="datasets/pcb-unified-4class/data.yaml"
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Canonical 640px dataset not found at $DATA_PATH"
    exit 1
fi
DATA_ARG="--data $DATA_PATH"

# 5. Train Standalone VMamba Object Detector (640px Best-of-Both-Worlds)
echo "=========================================================================="
echo "🚀 Training Standalone VMamba Object Detector (640px) on NVIDIA H100"
echo "   Workspace: $WORKSPACE_DIR"
echo "   Data Arg:  $DATA_ARG"
echo "   Run Key:   vmamba_standalone_640_v3"
echo "=========================================================================="
python train_mamba.py \
    --run-key vmamba_standalone_640_v3 \
    $DATA_ARG \
    --backbone-depths 2 2 2 2 \
    --stage-types conv conv mamba mamba \
    --no-checkpoint \
    --epochs 100 \
    --imgsz 640 \
    --batch 8 \
    --grad-accum 2 \
    --lr 0.0001 \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50

# 6. Aggregate results into the shared project results directory
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    echo "📊 Aggregating all models into master table in: $SHARED_RESULTS_DIR..."
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    mkdir -p results
    cp "$SHARED_RESULTS_DIR"/comparison_table.* results/ 2>/dev/null || true
else
    python aggregate_results.py
fi
