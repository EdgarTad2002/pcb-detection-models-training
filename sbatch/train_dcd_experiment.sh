#!/bin/bash
#SBATCH --job-name=pcb_dcd
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_dcd_%j.out
set -euo pipefail
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate "${DCD_ENV_PATH:-/mnt/weka/etadevosyan/.conda/envs/pcb-dcd}"
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training
python tools/run_dcd_experiment.py --action train "$@"
