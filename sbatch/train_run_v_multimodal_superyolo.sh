#!/bin/bash
#SBATCH --job-name=run_v_multimodal_superyolo
#SBATCH --partition=research
#SBATCH --mem=40G
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

# 1. Prepare the multimodal dataset structure and clean any stale data.yaml
if [ ! -f "datasets/pcb-vision-multimodal/data.yaml" ] || grep -q "2: Capacitor" "datasets/pcb-vision-multimodal/data.yaml" 2>/dev/null; then
    rm -f "datasets/pcb-vision-multimodal/data.yaml"
    python tools/prepare_pcb_vision_multimodal.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest datasets/pcb-vision-multimodal
fi


# 2. Run V: Multimodal SuperYOLO26s (RGB + Physical NIR + Auxiliary SR Branch)
#    - 4-channel input: [R, G, B, NIR] with COCO weight transfer
#    - Auxiliary 8x SR reconstruction head to 1280px targets during training
#    - Evaluated at standard benchmark conf=0.001
python multimodal_superyolo26.py \
    --run-key yolov26s_multimodal_superyolo \
    --weights yolo26s.pt \
    --data datasets/pcb-vision-multimodal/data.yaml \
    --project-root /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training \
    --results-dir /mnt/weka/etadevosyan/pcb-yolo/results \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --sr-lambda 100.0 --sr-target-imgsz 1280 --eval-conf 0.001
