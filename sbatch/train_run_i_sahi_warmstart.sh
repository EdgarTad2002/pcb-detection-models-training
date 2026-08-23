#!/bin/bash
#SBATCH --job-name=run_i_sahi_warm
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

# Same tiled dataset as the cold-start Run I -- reuses it if already built,
# builds it if this runs first.
if [ ! -f "datasets/pcb-tiled-native-640/data.yaml" ]; then
    python slice_dataset.py \
        --source datasets/pcb-native-res \
        --dest datasets/pcb-tiled-native-640 \
        --tile-size 640 \
        --overlap 0.2 \
        --min-visibility 0.3 \
        --keep-empty-prob 0.1
fi

# WARM START: fine-tunes from your already-trained native-res YOLO26s
# checkpoint (currently your best individual model, 50.91% mAP50) instead
# of the generic COCO-pretrained yolo26s.pt. Combines both improvements
# (native-res detail + tiling) rather than learning them independently.
NATIVE_RES_CKPT="runs/yolov26s_native_res/pcb-filtered/weights/best.pt"

if [ ! -f "$NATIVE_RES_CKPT" ]; then
    echo "ERROR: $NATIVE_RES_CKPT not found. Run Run J (train_run_j_native_res.sh) first."
    exit 1
fi

python train.py \
    --run-key yolov26s_sahi_tiles_native_warmstart \
    --weights "$NATIVE_RES_CKPT" \
    --data datasets/pcb-tiled-native-640/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
