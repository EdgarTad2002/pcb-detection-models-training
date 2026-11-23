#!/bin/bash
#SBATCH --job-name=eval_matrix
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_eval_matrix_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

GLOBAL_WEIGHTS="runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt"
CROP_WEIGHTS="runs/yolo26s_crop_specialist_mixed1280/pcb-filtered/weights/best.pt"
VAL_IMAGES="datasets/pcb-native-res-cleaned-1280/valid/images"
VAL_LABELS="datasets/pcb-native-res-cleaned-1280/valid/labels"

echo "=========================================================="
echo "Running Controlled Validation Matrix"
echo "  Global Weights: $GLOBAL_WEIGHTS"
echo "  Crop Weights:   $CROP_WEIGHTS"
echo "  Val Images:     $VAL_IMAGES"
echo "=========================================================="

python tools/run_val_comparison_matrix.py \
    --global-weights "$GLOBAL_WEIGHTS" \
    --crop-weights "$CROP_WEIGHTS" \
    --images-dir "$VAL_IMAGES" \
    --labels-dir "$VAL_LABELS" \
    --out-dir "runs/dual_branch_eval_val" \
    --device 0

echo "Validation comparison matrix complete!"
