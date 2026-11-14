#!/bin/bash
#SBATCH --job-name=apex_v3_triad
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/r4_apex_v3_triad_%j.out
#SBATCH --error=logs/r4_apex_v3_triad_%j.err

set -euo pipefail

mkdir -p logs results

if [ -f "/mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh" ]; then
    source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

PY="/mnt/weka/etadevosyan/.conda/envs/pcb-yolo/bin/python"
DATA_YAML="datasets/pcb-native-reviewed-r4-final/data.yaml"

echo "========================================================================"
echo "[Step 1/4] Train Family 1: APEXv3_YOLO26s_STAL_1536 (Preserved STAL + ProgLoss + 5-Ch ElCap Head)"
echo "========================================================================"
$PY apex_r4_model.py \
    --run-key r4_apex_v3_yolo26s_stal_1536 \
    --weights yolo26s.pt \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --imgsz 1536 --batch 4 --epochs 80 --patience 25 --lr0 0.0035

echo "========================================================================"
echo "[Step 2/4] Train Family 2: APEXv3_YOLO11m_5Ch_1536 (5-Ch ElCap Radial Disk + Mutual Exclusion)"
echo "========================================================================"
$PY apex_r4_model.py \
    --run-key r4_apex_v3_yolo11m_1536 \
    --weights yolo11m.pt \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --imgsz 1536 --batch 4 --epochs 80 --patience 25 --lr0 0.0035

echo "========================================================================"
echo "[Step 3/4] Train Family 3: APEXv3_YOLOv12s_5Ch_1536 (Area-Attention A2 + 5-Ch ElCap Radial Disk)"
echo "========================================================================"
$PY apex_r4_model.py \
    --run-key r4_apex_v3_yolov12s_1536 \
    --weights yolo12s.pt \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --imgsz 1536 --batch 4 --epochs 80 --patience 25 --lr0 0.0035

echo "========================================================================"
echo "[Step 4/4] Run Stage-2 YOLO ROI Verifier + Quick WBF v4.4 Final Evaluation"
echo "========================================================================"
$PY train_yolo_roi_verifier.py --device 0 --crop-size 192 --epochs 40
$PY quick_wbf.py
