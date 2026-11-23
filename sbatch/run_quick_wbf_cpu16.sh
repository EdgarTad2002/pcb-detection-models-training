#!/bin/bash
#SBATCH --job-name=super_ensemble_sota
#SBATCH --partition=research_cpu
#SBATCH --gres=cpuonly:16
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/super_ensemble_%j.out
#SBATCH --error=logs/super_ensemble_%j.err

set -euo pipefail

mkdir -p logs results

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export LRU_CACHE_CAPACITY=16

echo "========================================================================"
echo "RUNNING QUICK WBF v4 ON DEDICATED research_cpu PARTITION"
echo "Host: $(hostname) | Date: $(date)"
echo "========================================================================"

python quick_wbf.py

echo "========================================================================"
echo "WBF v4 COMPLETED SUCCESSFULLY!"
echo "========================================================================"
