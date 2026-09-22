#!/bin/bash
# ==============================================================================
# submit_all_rectified.sh - Submit all 10 rectified retraining jobs to Slurm
# ==============================================================================
# Usage: ssh etadevosyan@cluster.ysu.am "cd /mnt/weka/.../pcb-detection-models-training && bash sbatch/submit_all_rectified.sh"

set -e

echo "=========================================================="
echo "Submitting 10 Rectified 4-Class Retraining Jobs"
echo "Dataset: datasets/pcb-unified-4class/data.yaml"
echo "=========================================================="

# 1. Baseline YOLO generations (7 models)
sbatch sbatch/train_rectified_yolov5s.sh
sbatch sbatch/train_rectified_yolov8s.sh
sbatch sbatch/train_rectified_yolov9s.sh
sbatch sbatch/train_rectified_yolov10s.sh
sbatch sbatch/train_rectified_yolov11s.sh
sbatch sbatch/train_rectified_yolov12s.sh
sbatch sbatch/train_rectified_yolov26s.sh

# 2. Loss reweighting variants (2 models)
sbatch sbatch/train_rectified_yolov26s_loss_reweight.sh
sbatch sbatch/train_rectified_yolov9s_loss_reweight.sh

# 3. Transformer (1 model)
sbatch sbatch/train_rectified_rtdetr_l.sh

echo ""
echo "✅ All 10 jobs submitted! Monitor with: squeue -u etadevosyan"
