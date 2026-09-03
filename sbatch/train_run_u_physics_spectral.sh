#!/bin/bash
#SBATCH --job-name=run_u_physics_spectral
#SBATCH --partition=research
#SBATCH --mem=32G
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

# 1. Ensure laboratory optical contrast priors are computed
if [ ! -f "data/pcb_spectral_priors.json" ]; then
    python tools/extract_pcb_vision_spectrum.py --output data/pcb_spectral_priors.json
fi

# 2. Run U: Physics-Informed Spectral YOLO26s
#    - Integrates PCB-Vision optical reflectance signatures into 31-band adapter
#    - Trained on standard 640px dataset for direct, fair comparison with baselines
#    - Evaluated at academic benchmark conf=0.001
python physics_spectral_yolo26.py \
    --run-key yolov26s_physics_spectral_640 \
    --weights yolo26s.pt \
    --data datasets/pcb-filtered-yolov8/data.yaml \
    --priors-path data/pcb_spectral_priors.json \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --results-dir /mnt/weka/etadevosyan/pcb-yolo/results \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001
