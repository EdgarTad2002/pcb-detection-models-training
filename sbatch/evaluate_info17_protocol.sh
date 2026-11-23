#!/bin/bash
#SBATCH --job-name=info17_protocol
#SBATCH --partition=research
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_info17_protocol_%j.out
set -euo pipefail
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/info17_protocol_${SLURM_JOB_ID}"
python tools/evaluate_info17_protocol.py run --output "$1" --device 0
