#!/bin/bash
#SBATCH --job-name=pcb_original_top2
#SBATCH --partition=research
#SBATCH --array=0-5
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_original_top2_%A_%a.out
set -euo pipefail
PLAN="$(realpath "$1")"
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/pcb_mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo
OUTPUT="$(dirname "$PLAN")"
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training
python "$OUTPUT/source_snapshot/tools/train_original_top2.py" run \
  --plan "$PLAN" --index "$SLURM_ARRAY_TASK_ID" --device 0
