#!/bin/bash
#SBATCH --job-name=v26s_p2ws
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_yolov26s_p2_warmstart_retinex_1280_%j.out

# Step 2 (Option A): P2 detection head (stride 4 -> 320x320 grid at 1280px),
# WARM-STARTED from the champion checkpoint instead of from scratch.
#
# Why the previous P2 run failed (50.47% @640): new head trained from scratch,
# 640px, 100 epochs, batch 8. Here: 1280px, champion backbone/neck transferred,
# 200 epochs, same Retinex + copy-paste data and loss weights as the champion.
#
# Check the log for "Transferred N/M items from pretrained weights" -- N should
# be a large fraction of M (backbone + most of the neck).

set -e

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"

CHAMPION_W="runs/yolov26s_ultimate_retinex_copypaste_1280/pcb-filtered/weights/best.pt"
AUG_DATA="datasets/pcb-retinex-cappaste-1280"
[ -f "$CHAMPION_W" ] || { echo "Missing $CHAMPION_W"; exit 1; }
[ -f "$AUG_DATA/data.yaml" ] || { echo "Missing $AUG_DATA"; exit 1; }

EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-8}
RUN_KEY="yolov26s_p2_warmstart_retinex_copypaste_1280"

echo "=========================================================================="
echo "Training $RUN_KEY"
echo "  Arch:       yolo26s-p2.yaml (P2/P3/P4/P5 heads)"
echo "  Warm-start: $CHAMPION_W"
echo "  Data:       $AUG_DATA  | imgsz 1280 | epochs $EPOCHS | batch $BATCH"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s-p2.yaml \
    --pretrained-weights "$CHAMPION_W" \
    --data "$AUG_DATA/data.yaml" \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --label-smoothing 0.1 \
    --epochs "$EPOCHS" \
    --imgsz 1280 \
    --batch "$BATCH" \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50

# Extra eval at max_det=1000 (P2 produces many more small-object proposals)
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "${RUN_KEY}_maxdet1000" \
    --weights yolo26s-p2.yaml \
    --data "$AUG_DATA/data.yaml" \
    --skip-train \
    --eval-weights "runs/$RUN_KEY/pcb-filtered/weights/best.pt" \
    --imgsz 1280 --batch "$BATCH" --epochs "$EPOCHS" \
    --eval-conf 0.001 --eval-iou 0.50 --max-det 1000

SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
[ -d "$SHARED_RESULTS_DIR" ] && python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
echo "Done."
