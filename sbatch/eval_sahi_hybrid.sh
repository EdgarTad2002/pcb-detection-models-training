#!/bin/bash
#SBATCH --job-name=sahi_hybrid
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_sahi_hybrid_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

TILE_WEIGHTS="runs/sahi_tile_640/pcb-filtered/weights/best.pt"
STD_640_WEIGHTS="runs/rectified_yolov26s/pcb-filtered/weights/best.pt"
CHAMP_1280_WEIGHTS="runs/yolov26s_native_res_rectified_1280/pcb-filtered/weights/best.pt"
DATA="datasets/pcb-sahi-tile-640/data.yaml"

echo "=========================================================================="
echo "🚀 1. Dual-Stream Hybrid: Whole-image 640px + Tile-trained 640px"
echo "Macro context: $STD_640_WEIGHTS"
echo "Micro detail : $TILE_WEIGHTS"
echo "=========================================================================="
python tools/eval_sahi_benchmark.py \
    --weights "$TILE_WEIGHTS" \
    --full-frame-weights "$STD_640_WEIGHTS" \
    --full-frame-imgsz 640 \
    --data "$DATA" \
    --split test \
    --slice-size 640 \
    --overlap 0.25 \
    --conf 0.001 \
    --nms-type diou \
    --run-key sahi_hybrid_dual_640 \
    --save-vis results/visuals_sahi_hybrid_dual_640 \
    --max-vis 10

echo ""
echo "=========================================================================="
echo "🏆 2. Champion Hybrid: Native 1280px + Tile-trained 640px"
echo "Macro context: $CHAMP_1280_WEIGHTS (at 1280px)"
echo "Micro detail : $TILE_WEIGHTS (at 640px)"
echo "=========================================================================="
python tools/eval_sahi_benchmark.py \
    --weights "$TILE_WEIGHTS" \
    --full-frame-weights "$CHAMP_1280_WEIGHTS" \
    --full-frame-imgsz 1280 \
    --data "$DATA" \
    --split test \
    --slice-size 640 \
    --overlap 0.25 \
    --conf 0.001 \
    --nms-type diou \
    --run-key sahi_hybrid_1280_640 \
    --save-vis results/visuals_sahi_hybrid_1280_640 \
    --max-vis 10

echo ""
echo "✅ Both Hybrid SAHI evaluations completed."
