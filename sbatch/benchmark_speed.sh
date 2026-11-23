#!/bin/bash
#SBATCH --job-name=bench_speed
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bench_%j.out

set -euo pipefail

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# Default: the 21 completed clean-main checkpoints, using clean validation boards.
# Pass --weights or --plan explicitly to select other runs. No accuracy evaluation
# or historical result updates occur unless the separate accuracy flag is given.
python tools/benchmark_speed.py \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --device 0 \
    --max-det 1000 \
    --warmup 30 \
    --num-iters 100 "$@"
