#!/bin/bash
#SBATCH --job-name=run_o_feature_stack
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# 1. Build the feature-stack dataset (only once). Replaces RGB with
#    [CLAHE-contrast, Saturation, Sobel-edge-magnitude] -- a deterministic
#    recombination of the same captured pixels, NOT synthesized infrared/
#    hyperspectral data (that's not physically recoverable from RGB alone).
#    train/valid/test are ALL transformed identically, since this changes
#    the input domain itself -- unlike tiling, where test stayed untouched.
if [ ! -f "datasets/pcb-native-res-featstack/data.yaml" ]; then
    python build_feature_stack.py \
        --source datasets/pcb-native-res \
        --dest datasets/pcb-native-res-featstack
fi

# 2. Train standard YOLO26s -- no architecture change needed, since this
#    stays a valid 3-channel image, just with different channel content.
python train.py \
    --run-key yolov26s_feature_stack \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-featstack/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
