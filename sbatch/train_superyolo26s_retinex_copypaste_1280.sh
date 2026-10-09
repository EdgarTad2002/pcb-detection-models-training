#!/bin/bash
#SBATCH --job-name=sy_retinex
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_superyolo26s_retinex_copypaste_1280_%j.out

# SuperYOLO (HR 1280 -> LR 640 detection + auxiliary SR head) trained on the
# champion's data: Retinex reflectance + Canny enhanced boards with micro-
# capacitor copy-paste.
#
# Everything else is IDENTICAL to the original superyolo26s_rectified_1280 run
# (yolo26s.pt, lambda_sr 0.10, edge 0.5, cls 1.5 / box 5.0 / dfl 2.0,
# 100 epochs, batch 8), so the data change is the only variable.
#
# Note: the SR head reconstructs the 1280px *Retinex-enhanced* image, so the
# auxiliary loss now also pushes the backbone to preserve the injected
# reflectance/edge detail.

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

AUG_DATA="datasets/pcb-retinex-cappaste-1280"
[ -f "$AUG_DATA/data.yaml" ] || { echo "Missing $AUG_DATA (built by the ultimate champion job)"; exit 1; }

RUN_KEY="superyolo26s_retinex_copypaste_1280"
SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"

echo "=========================================================================="
echo "Training $RUN_KEY"
echo "  Arch: SuperYOLO (YOLO26s, HR 1280 -> LR 640 + SR head, lambda_sr=0.10)"
echo "  Data: $AUG_DATA (Retinex + Canny + capacitor copy-paste)"
echo "=========================================================================="

python superyolo_pcb.py \
    --project-root "$WORKSPACE_DIR" \
    --results-dir "$SHARED_RESULTS_DIR" \
    --run-key "$RUN_KEY" \
    --weights yolo26s.pt \
    --data "$AUG_DATA/data.yaml" \
    --lambda-sr 0.10 \
    --edge-weight 0.5 \
    --cls-weight 1.5 \
    --box-weight 5.0 \
    --dfl-weight 2.0 \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --eval-conf 0.001

# superyolo_pcb.py evaluates at max_det=300; add the max_det=1000 numbers
W="runs/$RUN_KEY/pcb-filtered/weights/best.pt"
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "${RUN_KEY}_maxdet1000" \
    --weights "$W" \
    --data "$AUG_DATA/data.yaml" \
    --skip-train \
    --eval-weights "$W" \
    --imgsz 640 --batch 8 --epochs 100 \
    --eval-conf 0.001 --eval-iou 0.50 --max-det 1000

if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    echo ""
    echo "SuperYOLO variants + current champion:"
    grep -E "^\| (superyolo26s_[a-z0-9_]*|yolov26s_ultimate_retinex_copypaste_1280_maxdet1000) \|" \
        "$SHARED_RESULTS_DIR/comparison_table.md" || true
fi
echo "Done."
