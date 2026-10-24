#!/bin/bash
#SBATCH --job-name=v12s_all
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=slurm_exp7_all_approaches_yolo12s_%j.out

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

TRAIN_DATA="datasets/pcb-native-res-cleaned-1280/data.yaml"
EVAL_DATA="datasets/pcb-native-res-cleaned-1280/data.yaml"
RUN_KEY="yolo12s_exp7_all_approaches_together_1280"

echo "=========================================================================="
echo "🔥 [Exp 7] Training YOLO12s: AS-PGMS-YOLO (All Approaches Upgraded) on H100"
echo "   Run Key:       $RUN_KEY"
echo "   Weights:       yolo12s.pt"
echo "   Train Data:    $TRAIN_DATA (Canonical Clean 1280px)"
echo "   Eval Data:     $EVAL_DATA (Canonical Clean Benchmark)"
echo "   Innovations:   Fractional AMSR + Soft-AGC + True Sargis Weber + SPD P2->P3 Bridge"
echo "                  + 3-Ch Scale-Decoupled SDice + Scale-Adaptive NWD Box Loss"
echo "   Resolution:    1280px Native Input, max_det=1000"
echo "=========================================================================="

python models/as_pgms_yolo.py \
    --run-key "$RUN_KEY" \
    --weights yolo12s.pt \
    --data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --lambda-sr 0.10 \
    --lambda-group 0.20 \
    --use-squared-dice \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --optimizer SGD \
    --cls 1.5 \
    --box 5.0 \
    --dfl 2.0 \
    --label-smoothing 0.10 \
    --copy-paste 0.30 \
    --mosaic 1.0 \
    --mixup 0.15 \
    --eval-split test \
    --max-det 1000 \
    --device 0

echo "✅ Exp 7 (YOLO12s AS-PGMS-YOLO All Approaches Upgraded) Finished Successfully!"
