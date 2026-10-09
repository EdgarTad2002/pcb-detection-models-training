#!/bin/bash
#SBATCH --job-name=cheapchk
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_eval_cheap_checks_champion_%j.out

# Step 1 of the architecture plan: cheap checks on the current champion.
#   [1] Does max_det=300 cap recall on dense boards?  (audit + re-eval at 1000)
#   [2] Is --dfl actually doing anything for YOLO26?  (results.csv audit)
# The original champion JSON is NOT overwritten (new run key *_maxdet1000).

set -e

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"

CHAMPION_RUN="yolov26s_ultimate_retinex_copypaste_1280"
RUN_DIR="runs/$CHAMPION_RUN/pcb-filtered"
WEIGHTS="$RUN_DIR/weights/best.pt"
DATA_YAML="datasets/pcb-retinex-cappaste-1280/data.yaml"
[ -f "$WEIGHTS" ] || { echo "Missing $WEIGHTS"; exit 1; }
[ -f "$DATA_YAML" ] || { echo "Missing $DATA_YAML"; exit 1; }

echo "=========================================================================="
echo "[1+2] Auditing GT density vs max_det and DFL loss usage"
echo "=========================================================================="
python tools/audit_eval_settings.py \
    --data "$DATA_YAML" \
    --split test \
    --run-dir "$RUN_DIR" \
    --max-det 300

echo "=========================================================================="
echo "[1b] Re-evaluating champion with max_det=1000 (same weights, same test set)"
echo "=========================================================================="
python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "${CHAMPION_RUN}_maxdet1000" \
    --weights "$WEIGHTS" \
    --data "$DATA_YAML" \
    --skip-train \
    --eval-weights "$WEIGHTS" \
    --imgsz 1280 \
    --batch 8 \
    --epochs 100 \
    --eval-conf 0.001 \
    --eval-iou 0.50 \
    --max-det 1000

SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
if [ -d "$SHARED_RESULTS_DIR" ]; then
    python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
    echo ""
    echo "Champion @max_det=300 vs @max_det=1000:"
    grep -E "^\| ${CHAMPION_RUN}(_maxdet1000)? \|" "$SHARED_RESULTS_DIR/comparison_table.md" || true
fi
echo "Done."
