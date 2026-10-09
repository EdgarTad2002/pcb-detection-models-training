#!/bin/bash
#SBATCH --job-name=v26s_optauto
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_yolov26s_ultimate_optauto_1280_%j.out

# Step 1 (optimizer check): identical to the champion run EXCEPT the optimizer.
# train.py defaults to --optimizer SGD; here we let Ultralytics pick its own
# default for YOLO26 (optimizer=auto). The log line "optimizer: ..." shows
# which optimizer and lr it chose.

set -e

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"

SRC_DIR="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training"
if [ -d "$SRC_DIR" ]; then
    cp "$SRC_DIR/train.py" . 2>/dev/null || true
    cp "$SRC_DIR/aggregate_results.py" . 2>/dev/null || true
fi

AUG_DATA="datasets/pcb-retinex-cappaste-1280"
[ -f "$AUG_DATA/data.yaml" ] || { echo "Missing $AUG_DATA (run train_yolov26s_ultimate_retinex_copypaste_1280.sh first)"; exit 1; }

RUN_KEY="yolov26s_ultimate_retinex_copypaste_optauto_1280"
echo "=========================================================================="
echo "Training $RUN_KEY  (champion recipe + optimizer=auto)"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$AUG_DATA/data.yaml" \
    --optimizer auto \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --label-smoothing 0.1 \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --eval-conf 0.001 \
    --eval-iou 0.50

# Extra eval at max_det=1000 (separate run key, original JSON untouched)
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "${RUN_KEY}_maxdet1000" \
    --weights yolo26s.pt \
    --data "$AUG_DATA/data.yaml" \
    --skip-train \
    --eval-weights "runs/$RUN_KEY/pcb-filtered/weights/best.pt" \
    --imgsz 1280 --batch 8 --epochs 100 \
    --eval-conf 0.001 --eval-iou 0.50 --max-det 1000

SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
[ -d "$SHARED_RESULTS_DIR" ] && python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
echo "Done."
