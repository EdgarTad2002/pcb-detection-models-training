#!/bin/bash
#SBATCH --job-name=pcb_r4_sota
#SBATCH --partition=research
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/weka/etadevosyan/pcb-yolo/logs/r4_sota_champions_%j.out
#SBATCH --error=/mnt/weka/etadevosyan/pcb-yolo/logs/r4_sota_champions_%j.err

set -euo pipefail

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

WORKDIR="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="/mnt/weka/etadevosyan/.config/Ultralytics"
mkdir -p "$YOLO_CONFIG_DIR" /mnt/weka/etadevosyan/pcb-yolo/logs results

DATA_YAML="$WORKDIR/datasets/pcb-native-reviewed-r4-final/data.yaml"

echo "=================================================================="
echo "STAGE 1: YOLO11m + Retinex-Sargis-Edge + AGWD-OTA + Reg-SDice @ 1536px"
echo "=================================================================="
python train_sota_r4.py \
    --use-retinex-sargis-edge \
    --model yolo11m \
    --weights yolo11m.pt \
    --run-key "r4_sota_yolo11m_retinex_agwd_1536" \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --eval-split val \
    --epochs 65 \
    --imgsz 1536 \
    --batch 4 \
    --nbs 64 \
    --optimizer SGD \
    --lr0 0.005 \
    --lrf 0.01 \
    --cos-lr \
    --momentum 0.937 \
    --weight-decay 0.001 \
    --warmup-epochs 3.0 \
    --label-smoothing 0.05 \
    --copy-paste 0.30 \
    --mixup 0.10 \
    --mosaic 1.0 \
    --close-mosaic 15 \
    --scale 0.60 \
    --fliplr 0.5 \
    --flipud 0.5 \
    --agwd-ota \
    --agwd-kappa 14.4 \
    --agwd-tau 829.0 \
    --agwd-alpha-kld 1.0 \
    --tal-beta 3.0 \
    --dynamic-k-max 12 \
    --squared-dice \
    --lambda-dice 0.25 \
    --smooth-dice 0.01 \
    --stage-weights uniform \
    --max-det 1000 \
    --seed 42

echo "=================================================================="
echo "STAGE 2: YOLO26x + Retinex-Sargis-Edge + AGWD-OTA + Reg-SDice @ 1280px"
echo "=================================================================="
python train_sota_r4.py \
    --use-retinex-sargis-edge \
    --model yolo26x \
    --weights yolo26x.pt \
    --run-key "r4_sota_yolo26x_retinex_agwd_1280" \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --eval-split val \
    --epochs 60 \
    --imgsz 1280 \
    --batch 4 \
    --nbs 64 \
    --optimizer SGD \
    --lr0 0.004 \
    --lrf 0.01 \
    --cos-lr \
    --momentum 0.937 \
    --weight-decay 0.001 \
    --warmup-epochs 3.0 \
    --label-smoothing 0.05 \
    --copy-paste 0.30 \
    --mixup 0.15 \
    --mosaic 1.0 \
    --close-mosaic 15 \
    --scale 0.60 \
    --fliplr 0.5 \
    --flipud 0.5 \
    --agwd-ota \
    --agwd-kappa 12.0 \
    --agwd-tau 576.0 \
    --agwd-alpha-kld 1.0 \
    --tal-beta 3.0 \
    --dynamic-k-max 12 \
    --squared-dice \
    --lambda-dice 0.25 \
    --smooth-dice 0.01 \
    --stage-weights uniform \
    --max-det 1000 \
    --seed 42

echo "=================================================================="
echo "STAGE 3: YOLO26s + Retinex-Sargis-Edge + AGWD-OTA + Reg-SDice @ 1536px"
echo "=================================================================="
python train_sota_r4.py \
    --use-retinex-sargis-edge \
    --model yolo26s \
    --weights yolo26s.pt \
    --run-key "r4_sota_yolo26s_retinex_agwd_1536" \
    --data "$DATA_YAML" \
    --eval-data "$DATA_YAML" \
    --eval-split val \
    --epochs 65 \
    --imgsz 1536 \
    --batch 4 \
    --nbs 64 \
    --optimizer SGD \
    --lr0 0.005 \
    --lrf 0.01 \
    --cos-lr \
    --momentum 0.937 \
    --weight-decay 0.001 \
    --warmup-epochs 3.0 \
    --label-smoothing 0.05 \
    --copy-paste 0.30 \
    --mixup 0.10 \
    --mosaic 1.0 \
    --close-mosaic 15 \
    --scale 0.60 \
    --fliplr 0.5 \
    --flipud 0.5 \
    --agwd-ota \
    --agwd-kappa 14.4 \
    --agwd-tau 829.0 \
    --agwd-alpha-kld 1.0 \
    --tal-beta 3.0 \
    --dynamic-k-max 12 \
    --squared-dice \
    --lambda-dice 0.25 \
    --smooth-dice 0.01 \
    --stage-weights uniform \
    --max-det 1000 \
    --seed 42

echo "=================================================================="
echo "STAGE 4: Full Test-Split Benchmark & Multi-Model WBF Fusion"
echo "=================================================================="
python evaluate_r4_sota_and_wbf.py
