#!/bin/bash
#SBATCH --job-name=v26s_rstem
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_yolov26s_retinex_stem_1280_%j.out

# Step 3 (Option B): learnable in-network Retinex-Edge stem (retinex_stem.py).
# Trains on RAW native-res images (+ raw capacitor copy-paste); the Retinex
# decomposition happens inside the network, after augmentation, and is learned.
#
# Usage:
#   sbatch sbatch/train_yolov26s_retinex_stem_1280.sh
#       -> 3-head YOLO26s from COCO yolo26s.pt (clean ablation vs offline Retinex champion)
#   sbatch sbatch/train_yolov26s_retinex_stem_1280.sh yolo26s-p2.yaml runs/<p2_run>/pcb-filtered/weights/best.pt
#       -> stem on top of the P2 model from step 2 (use once step 2 has finished)

set -e

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"
[ -e datasets ] || { echo "Missing datasets/ symlink in $WORKSPACE_DIR"; exit 1; }

MODEL_CFG=${1:-yolo26s.pt}
PRETRAINED=${2:-}
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-8}

RUN_KEY="yolov26s_retinex_stem_copypaste_1280"
[[ "$MODEL_CFG" == *p2* ]] && RUN_KEY="yolov26s_p2_retinex_stem_copypaste_1280"

# 0. Self-test the stem (fails fast in seconds if anything is wrong)
echo "=========================================================================="
echo "[0] retinex_stem.py self-test"
echo "=========================================================================="
python retinex_stem.py

# 1. Locate RAW native-res source (same lookup order as the offline Retinex job,
#    so the test images are the same boards the champion was evaluated on)
if [ -f "datasets/pcb-native-res-unified-4class/data.yaml" ]; then
    SRC_DATA="datasets/pcb-native-res-unified-4class"
elif [ -f "datasets/pcb-native-res/data.yaml" ]; then
    SRC_DATA="datasets/pcb-native-res"
elif [ -f "datasets/pcb-unified-4class-cleaned/data.yaml" ]; then
    SRC_DATA="datasets/pcb-unified-4class-cleaned"
else
    SRC_DATA="datasets/pcb-unified-4class"
fi

# 2. Raw capacitor bank + copy-paste dataset (same settings as the champion)
BANK_DIR="capacitor_bank_raw_1280"
AUG_DATA="datasets/pcb-native-cappaste-1280"
if [ ! -f "$BANK_DIR/metadata.json" ]; then
    echo "[2a] Building RAW capacitor crop bank from $SRC_DATA..."
    python build_capacitor_bank.py --source "$SRC_DATA" --dest "$BANK_DIR" --class-id 0 --margin 0.15
fi
if [ ! -f "$AUG_DATA/data.yaml" ]; then
    echo "[2b] Building RAW copy-paste dataset $AUG_DATA..."
    python bbox_copy_paste.py --source "$SRC_DATA" --bank "$BANK_DIR" --dest "$AUG_DATA" \
        --class-id 0 --paste-prob 0.7 --min-pastes 1 --max-pastes 4
fi

PRETRAIN_ARGS=()
if [ -n "$PRETRAINED" ]; then
    [ -f "$PRETRAINED" ] || { echo "Missing pretrained weights $PRETRAINED"; exit 1; }
    PRETRAIN_ARGS=(--pretrained-weights "$PRETRAINED")
fi

echo "=========================================================================="
echo "[3] Training $RUN_KEY"
echo "  Arch:       $MODEL_CFG + learnable Retinex-Edge stem"
echo "  Pretrained: ${PRETRAINED:-<from MODEL_CFG>}"
echo "  Data:       $AUG_DATA (RAW images) | imgsz 1280 | epochs $EPOCHS | batch $BATCH"
echo "=========================================================================="

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "$RUN_KEY" \
    --weights "$MODEL_CFG" \
    "${PRETRAIN_ARGS[@]}" \
    --retinex-stem \
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

python train.py \
    --project-root "$WORKSPACE_DIR" \
    --run-key "${RUN_KEY}_maxdet1000" \
    --weights "$MODEL_CFG" \
    --data "$AUG_DATA/data.yaml" \
    --skip-train \
    --eval-weights "runs/$RUN_KEY/pcb-filtered/weights/best.pt" \
    --imgsz 1280 --batch "$BATCH" --epochs "$EPOCHS" \
    --eval-conf 0.001 --eval-iou 0.50 --max-det 1000

SHARED_RESULTS_DIR="/mnt/weka/etadevosyan/pcb-yolo/results"
[ -d "$SHARED_RESULTS_DIR" ] && python aggregate_results.py --results-dir "$SHARED_RESULTS_DIR"
echo "Done."
