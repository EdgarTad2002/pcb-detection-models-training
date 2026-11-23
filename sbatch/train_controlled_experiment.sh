#!/bin/bash
#SBATCH --job-name=pcb_controlled
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_controlled_%j.out
set -euo pipefail

PLAN="$(realpath "$1")"
JOB_KEY="$2"
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo
PROJECT_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$PLAN")"
cd "$PROJECT_ROOT"
python tools/controlled_experiments.py run --plan "$PLAN" --job-key "$JOB_KEY" --device 0
