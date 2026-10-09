#!/bin/bash
#SBATCH --job-name=sy_maxdet
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_eval_superyolo_maxdet1000_%j.out

# Zero-training re-evaluation of the existing SuperYOLO checkpoint
# (superyolo26s_rectified_1280, 63.04% @ max_det=300) with max_det=1000.
#
# SuperYOLO trains on 1280px HR images but its backbone only ever sees the
# 640px downsample, so its native detection resolution is 640px.
#   [1] 640px  + max_det=1000  -> the fair, like-for-like number
#   [2] 1280px + max_det=1000  -> exploratory probe only (scale mismatch vs training)
# Original JSON is untouched (new run keys).

set -e

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

# The original SuperYOLO job ran from the other clone; check both locations.
SY_W=""
for cand in \
    "runs/superyolo26s_rectified_1280/pcb-filtered/weights/best.pt" \
    "/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/runs/superyolo26s_rectified_1280/pcb-filtered/weights/best.pt"; do
    if [ -f "$cand" ]; then SY_W="$cand"; break; fi
done
[ -n "$SY_W" ] || { echo "Could not find superyolo26s_rectified_1280 best.pt"; exit 1; }

# SuperYOLO was trained/evaluated on the RAW native-res 4-class data
DATA_YAML="datasets/pcb-native-res-unified-4class/data.yaml"
[ -f "$DATA_YAML" ] || { echo "Missing $DATA_YAML"; exit 1; }

echo "=========================================================================="
echo "SuperYOLO checkpoint: $SY_W"
echo "Data:                 $DATA_YAML"
echo "=========================================================================="

echo ">>> [1/2] 640px, max_det=1000 (native detection resolution)"
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key superyolo26s_rectified_1280_maxdet1000 \
    --weights "$SY_W" \
    --data "$DATA_YAML" \
    --skip-train \
    --eval-weights "$SY_W" \
    --imgsz 640 --batch 8 --epochs 100 \
    --eval-conf 0.001 --eval-iou 0.50 --max-det 1000

echo ">>> [2/2] 1280px, max_det=1000 (exploratory: backbone trained on 640px inputs)"
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key superyolo26s_rectified_1280_eval1280_maxdet1000 \
    --weights "$SY_W" \
    --data "$DATA_YAML" \
    --skip-train \
    --eval-weights "$SY_W" \
    --imgsz 1280 --batch 8 --epochs 100 \
    --eval-conf 0.001 --eval-iou 0.50 --max-det 1000

SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    echo ""
    echo "SuperYOLO rows + current champion:"
    grep -E "^\| (superyolo26s_rectified_1280[a-z0-9_]*|yolov26s_ultimate_retinex_copypaste_1280_maxdet1000) \|" \
        "$SHARED_RESULTS_DIR/comparison_table.md" || true
fi
echo "Done."
