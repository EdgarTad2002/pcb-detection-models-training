#!/bin/bash
#SBATCH --job-name=eval_ctdn
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_eval_ctdn_%j.out

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

EVAL_DATA="datasets/pcb-native-res-cleaned-1280/data.yaml"

echo "=========================================================="
echo "Evaluating YOLO26s Exp 6 CTDN Stem..."
echo "=========================================================="
python models/train_case1_ctdn_stem.py \
    --run-key yolo26s_exp6_ctdn_stem_1280 \
    --weights yolo26s.pt \
    --eval-data "$EVAL_DATA" \
    --skip-train

echo "=========================================================="
echo "Evaluating YOLO12s Exp 6 CTDN Stem..."
echo "=========================================================="
python models/train_case1_ctdn_stem.py \
    --run-key yolo12s_exp6_ctdn_stem_1280 \
    --weights yolo12s.pt \
    --eval-data "$EVAL_DATA" \
    --skip-train

echo "=========================================================="
echo "Aggregating all results..."
echo "=========================================================="
python aggregate_results.py --results-dir /mnt/weka/etadevosyan/pcb-yolo/results
