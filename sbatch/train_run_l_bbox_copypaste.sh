#!/bin/bash
#SBATCH --job-name=run_l_bbox_cappaste
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

# 1. Build the capacitor crop bank (rebuilds if metadata.json is missing)
if [ ! -f "capacitor_bank/metadata.json" ]; then
    echo "Building capacitor crop bank with relative bounding box metadata..."
    rm -rf capacitor_bank
    python build_capacitor_bank.py \
        --source datasets/pcb-native-res \
        --dest capacitor_bank \
        --margin 0.15
fi

# 2. Build the augmented dataset (rebuilds if bank metadata is newer than data.yaml)
if [ ! -f "datasets/pcb-native-res-cappaste/data.yaml" ] || [ "capacitor_bank/metadata.json" -nt "datasets/pcb-native-res-cappaste/data.yaml" ]; then
    echo "Building augmented copy-paste dataset with accurate capacitor labels..."
    rm -rf datasets/pcb-native-res-cappaste
    python bbox_copy_paste.py \
        --source datasets/pcb-native-res \
        --bank capacitor_bank \
        --dest datasets/pcb-native-res-cappaste \
        --paste-prob 0.7 --min-pastes 1 --max-pastes 4
fi

# 3. Train standard YOLO26s on the augmented data -- reuses train.py
#    unmodified, same imgsz/epochs as your baseline for a fair comparison,
#    with --eval-conf 0.001 for benchmark consistency.
python train.py \
    --run-key yolov26s_bbox_cappaste \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-cappaste/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001

