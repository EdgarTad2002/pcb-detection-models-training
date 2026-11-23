#!/bin/bash
#SBATCH --job-name=r3_old_vs_new
#SBATCH --partition=research
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_reviewed_validation_%j.out
set -euo pipefail
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/reviewed_validation_${SLURM_JOB_ID}"
python tools/evaluate_reviewed_validation.py run --output "$1" --device 0
