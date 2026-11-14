#!/bin/bash
#SBATCH --job-name=wbf_r4_v43
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wbf_v43_cpu_%j.out
#SBATCH --error=logs/wbf_v43_cpu_%j.err

set -euo pipefail

mkdir -p logs results

if [ -f "/mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh" ]; then
    source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export LRU_CACHE_CAPACITY=16

if [ ! -f "runs/r4_yolo_roi_verifier/roi_verifier.pt" ]; then
    echo "========================================================================"
    echo "[Step 1/2] Train Stage-2 High-Res YOLO ROI Specialist Verifier"
    echo "========================================================================"
    /mnt/weka/etadevosyan/.conda/envs/pcb-yolo/bin/python train_yolo_roi_verifier.py
fi

echo "========================================================================"
echo "[Step 2/2] Quick WBF v4.3 (17-Model Cached Pool + Per-Class Peak Lock + Stage-2 ROI Verifier)"
echo "========================================================================"
/mnt/weka/etadevosyan/.conda/envs/pcb-yolo/bin/python quick_wbf.py
