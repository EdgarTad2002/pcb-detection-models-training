#!/bin/bash
#SBATCH --job-name=bilat_seed_ens
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_bilat_seed_ens_%j.out

# WBF ensemble of the 3 bilateral-Retinex seeds. Run AFTER:  bash sbatch/submit_retinex_study.sh core
# All members were trained on the SAME bilateral images, so they are evaluated on the
# bilateral test split (no train/test preprocessing mismatch). Do NOT mix in models trained
# on other preprocessing (e.g. raw-image models) with this --data: they would see the wrong inputs.

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo
export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

W=runs/sweep_bilat_a0.30_b0.10
for S in 0 1 2; do
    test -f ${W}_s${S}_hv0.4_yolo26s/pcb-filtered/weights/best.pt || { echo "missing seed $S weights -- run the core tier first"; exit 1; }
done

python tools/eval_super_ensemble.py \
    --data datasets/pcb-bilat-a0.30-b0.10/data.yaml \
    --split test --imgsz 1280 --max-det 1000 \
    --weights ${W}_s0_hv0.4_yolo26s/pcb-filtered/weights/best.pt \
              ${W}_s1_hv0.4_yolo26s/pcb-filtered/weights/best.pt \
              ${W}_s2_hv0.4_yolo26s/pcb-filtered/weights/best.pt \
    --run-key ensemble_bilat_3seed_wbf_1280 \
    --results-dir /mnt/weka/etadevosyan/pcb-yolo/results
