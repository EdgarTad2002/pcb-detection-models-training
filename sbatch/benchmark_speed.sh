#!/bin/bash
#SBATCH --job-name=bench_speed
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_bench_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "======================================================================"
echo "🚀 Starting Controlled GPU Speed Benchmark on $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python tools/benchmark_speed.py \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --results-dir /mnt/weka/etadevosyan/pcb-yolo/results \
    --device 0 \
    --warmup 25 \
    --num-iters 100 \
    --update-results

echo "======================================================================"
echo "✅ Speed Benchmarking Complete!"
echo "======================================================================"
