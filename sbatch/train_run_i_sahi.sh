#!/bin/bash
#SBATCH --job-name=run_i_sahi_train
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

# 1. Build the tiled dataset from the NATIVE-RES source (not the old
#    degraded 640px export) -- combines both benefits: real extra detail
#    AND fewer competing objects per training crop. Skips re-tiling if the
#    output already exists.
if [ ! -f "datasets/pcb-tiled-native-640/data.yaml" ]; then
    python slice_dataset.py \
        --source datasets/pcb-native-res \
        --dest datasets/pcb-tiled-native-640 \
        --tile-size 640 \
        --overlap 0.2 \
        --min-visibility 0.3 \
        --keep-empty-prob 0.1
fi

# 2. Train a standard YOLO26s on the tiled data -- reuses train.py
#    unmodified. train.py's own end-of-training eval runs model.val()
#    directly on the ORIGINAL full-size test images (slice_dataset.py
#    points "test" at them), giving a free "tile-trained, naive whole-image
#    inference" number to compare against the real SAHI sliced-inference
#    result from eval_sahi.py (run separately, see train_run_i_sahi_eval.sh).
python train.py \
    --run-key yolov26s_sahi_tiles_native \
    --weights yolo26s.pt \
    --data datasets/pcb-tiled-native-640/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
