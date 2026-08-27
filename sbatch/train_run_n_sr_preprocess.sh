#!/bin/bash
#SBATCH --job-name=run_n_sr_preprocess
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

# Stop immediately if any command fails -- prevents training from starting
# if SR upscaling or installation failed silently.
set -e

# 1. Install Real-ESRGAN into the ACTIVE conda python (not system python).
#    Using `python -m pip` ensures packages land in the right env.
python -m pip install --quiet basicsr
python -m pip install --quiet git+https://github.com/xinntao/Real-ESRGAN.git

# Verify installation succeeded before proceeding.
python -c "from basicsr.archs.rrdbnet_arch import RRDBNet; print('basicsr OK')"
python -c "from realesrgan import RealESRGANer; print('realesrgan OK')"

# 2. Build the SR-upscaled dataset (only once -- skip existing images on reruns).
#    Uses the standard pcb-filtered-yolov8 dataset as source so the evaluation
#    protocol (val/test splits) stays identical to all other runs.
if [ ! -f "datasets/pcb-sr-2x/data.yaml" ]; then
    python sr_upscale_dataset.py \
        --source datasets/pcb-filtered-yolov8 \
        --dest   datasets/pcb-sr-2x \
        --device cuda
fi

# 3. Train YOLO26s on the SR-upscaled training images.
#    imgsz=1280 matches the 2x upscaled image dimensions so YOLO uses the
#    extra resolution rather than downsampling it back to 640.
#    batch=8 because 1280px images use ~4x the GPU memory vs 640px.
python train.py \
    --run-key yolov26s_sr_preprocess \
    --weights yolo26s.pt \
    --data datasets/pcb-sr-2x/data.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8
