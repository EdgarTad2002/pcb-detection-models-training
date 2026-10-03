#!/bin/bash
#SBATCH --job-name=omni_champ
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_omni_champ_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

echo "=========================================================================="
echo "👑 Omni-Scale PCB-YOLO Champion Training Pipeline"
echo "Unifying: Rectified Unified 4-Class + CLAHE + Multi-Scale Slicing + Dual SAHI"
echo "=========================================================================="

DATASET="datasets/pcb-omni-scale-clahe"

# ── Step 1: Build Omni-Scale CLAHE Dataset ────────────────────────────────────
if [ ! -f "$DATASET/data.yaml" ]; then
    echo ""
    echo "🛠️  Step 1: Building Omni-Scale Compound Dataset..."
    python tools/build_omni_scale_dataset.py \
        --source datasets/pcb-native-res-unified-4class \
        --dest "$DATASET" \
        --tile-size 640 \
        --stride 320 \
        --sigma 1.0 \
        --gamma 0.6 \
        --clip-limit 1.5 \
        --workers 8
    echo "✅ Dataset generated."
else
    echo "ℹ️  Dataset already cached at $DATASET"
fi

# ── Step 2: Train YOLOv26s on Compound Dataset ────────────────────────────────
echo ""
echo "🚀 Step 2: Training Omni-Scale YOLOv26s Champion (100 Epochs)..."
python train.py \
    --run-key omni_scale_champion_640 \
    --weights yolo26s.pt \
    --data "$DATASET/data.yaml" \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --eval-conf 0.001

# ── Step 3: Run Dual-Stream SAHI Hyper-Inference Evaluation ───────────────────
echo ""
echo "📊 Step 3: Running Dual-Stream SAHI Evaluation..."
WEIGHTS="runs/omni_scale_champion_640/pcb-filtered/weights/best.pt"

python tools/eval_sahi_benchmark.py \
    --weights "$WEIGHTS" \
    --data "$DATASET/data.yaml" \
    --split test \
    --slice-size 640 \
    --overlap 0.25 \
    --conf 0.001 \
    --nms-type diou \
    --run-key omni_scale_champion_sahi \
    --save-vis results/visuals_omni_scale_champion \
    --max-vis 10

# ── Step 4: Update Aggregate Tables ───────────────────────────────────────────
echo ""
echo "📈 Step 4: Aggregating Leaderboard..."
python aggregate_results.py

echo ""
echo "🎉 Omni-Scale Champion Training & Benchmark Complete!"
