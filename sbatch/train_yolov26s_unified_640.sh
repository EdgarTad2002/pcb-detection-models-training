#!/bin/bash
#SBATCH --job-name=yolo26s_uni640
#SBATCH --partition=research
#SBATCH --mem=40G
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

# ==============================================================================
# Unified Champion Model (640px Real-Time Edge Variant)
# Combines:
# 1. Micro-Capacitor Copy-Paste Data Augmentation (balances sparse PCB boards)
# 2. Loss Reweighting (cls=1.5, box=5.0, dfl=2.0, label_smoothing=0.1)
# ==============================================================================

# 1. Build augmented copy-paste dataset if missing
if [ ! -f "datasets/pcb-native-res-cappaste/data.yaml" ]; then
    if [ ! -f "capacitor_bank/metadata.json" ]; then
        echo "📦 Building capacitor crop bank..."
        python build_capacitor_bank.py \
            --source datasets/pcb-native-res \
            --dest capacitor_bank \
            --margin 0.15
    fi
    echo "🎨 Building copy-paste augmented dataset..."
    python bbox_copy_paste.py \
        --source datasets/pcb-native-res \
        --bank capacitor_bank \
        --dest datasets/pcb-native-res-cappaste \
        --paste-prob 0.7 --min-pastes 1 --max-pastes 4
fi

# 3. Train Unified Champion Model (640px)
python train.py \
    --run-key yolov26s_unified_640 \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-cappaste/data.yaml \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --label-smoothing 0.1 \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001
