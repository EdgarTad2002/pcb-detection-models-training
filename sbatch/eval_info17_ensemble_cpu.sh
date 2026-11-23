#!/bin/bash
#SBATCH --job-name=info17_ens_cpu
#SBATCH --partition=research_cpu
#SBATCH --gres=cpuonly:16
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/info17_ens_cpu_%j.out
#SBATCH --error=logs/info17_ens_cpu_%j.err

set -euo pipefail

mkdir -p logs results
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics

echo "Running info17 ensemble audit on $(hostname) at $(date)"
python tools/eval_info17_ensemble_cpu.py
echo "Completed at $(date)"
