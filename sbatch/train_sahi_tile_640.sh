#!/bin/bash
#SBATCH --job-name=sahi_tile_640
#SBATCH --partition=research
#SBATCH --mem=48G
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

echo "=========================================================="
echo "SAHI Tile Training: native-res images → 640px tiles"
echo ""
echo "Problem: at 640px, capacitors are ~7px (4x downsampled from 2500px)"
echo "Fix: slice 2500px images into 640px tiles → capacitors remain at 30px"
echo "Train on tiles, evaluate with SAHI slicing inference"
echo "=========================================================="

TILE_DATASET="datasets/pcb-sahi-tile-640"

# ── Step 1: Build tile dataset (only if not already done) ──────────────────
if [ ! -f "$TILE_DATASET/data.yaml" ]; then
    echo ""
    echo "🔪 Building SAHI tile dataset from native-res images..."
    python tools/build_sahi_tile_dataset.py \
        --source  datasets/pcb-native-res-unified-4class \
        --dest    "$TILE_DATASET" \
        --tile-size 640 \
        --stride  320 \
        --min-visibility 0.25 \
        --val-ratio 0.10 \
        --jpeg-quality 95 \
        --workers 8
    echo "✅ Tile dataset built."
else
    echo "ℹ️  Tile dataset already exists at $TILE_DATASET, skipping build."
fi

# ── Step 2: Train standard YOLOv26s on tile dataset ───────────────────────
# No custom trainer needed — tiles are just 640px images with YOLO labels.
# The model specialises in tile-level detection (native component density).
echo ""
echo "🚀 Training YOLOv26s on tile dataset..."
python train.py \
    --run-key   sahi_tile_640 \
    --weights   yolo26s.pt \
    --data      "$TILE_DATASET/data.yaml" \
    --epochs    100 \
    --imgsz     640 \
    --batch     16 \
    --workers   8 \
    --eval-conf 0.001

# ── Step 3: SAHI inference evaluation on full test images ──────────────────
# The tiled model + SAHI inference closely matches the training view.
# Also run standard (non-SAHI) eval for fair comparison to other 640px models.
echo ""
echo "📊 Running SAHI benchmark evaluation..."
WEIGHTS="runs/sahi_tile_640/pcb-filtered/weights/best.pt"
python tools/eval_sahi_benchmark.py \
    --weights  "$WEIGHTS" \
    --data     "$TILE_DATASET/data.yaml" \
    --split    test \
    --slice-size 640 \
    --overlap  0.25 \
    --conf     0.001 \
    --run-key  sahi_tile_640_sahi_eval
