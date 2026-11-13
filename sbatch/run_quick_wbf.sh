#!/bin/bash
#SBATCH --job-name=wbf_r4_eval
#SBATCH --partition=research
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wbf_r4_eval_%j.out
#SBATCH --error=logs/wbf_r4_eval_%j.err

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

/mnt/weka/etadevosyan/.conda/envs/pcb-yolo/bin/python quick_wbf.py
