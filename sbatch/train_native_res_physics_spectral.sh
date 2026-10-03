#!/bin/bash
#SBATCH --job-name=nat_phy_spec
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_nat_spectral_%j.out

# ==============================================================================
# Train Physics-Informed Spectral YOLO26s on Rectified Native Resolution 4-Class Dataset (1280px)
# Usage:
#   sbatch sbatch/train_native_res_physics_spectral.sh [ALPHA]
# Example:
#   sbatch sbatch/train_native_res_physics_spectral.sh 0.25
# ==============================================================================

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# 1. Ensure physical optical contrast priors are computed
if [ ! -f "data/pcb_spectral_priors.json" ]; then
    python tools/extract_pcb_vision_spectrum.py --output data/pcb_spectral_priors.json
fi

# 2. Configure alpha parameter (pass as $1, defaults to 0.25)
ALPHA="${1:-0.25}"
RUN_KEY="rectified_yolov26s_native_res_physics_spectral_a${ALPHA}_1280"

echo "=========================================================="
echo "Running Rectified Native-Res Physics-Spectral YOLO26s with alpha = $ALPHA"
echo "Dataset: datasets/pcb-native-res-unified-4class/data.yaml"
echo "Run Key: $RUN_KEY"
echo "Resolution: 1280px, Batch: 8, Epochs: 100"
echo "=========================================================="

python physics_spectral_yolo26.py \
    --run-key "$RUN_KEY" \
    --alpha "$ALPHA" \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-unified-4class/data.yaml \
    --priors-path data/pcb_spectral_priors.json \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --results-dir /mnt/weka/etadevosyan/pcb-yolo/results \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001

echo "Training complete for $RUN_KEY"
