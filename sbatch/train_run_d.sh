#!/bin/bash
#SBATCH --job-name=run_d_p2_hires
#SBATCH --partition=research
#SBATCH --mem=48G
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

# NOTE: copy_paste is intentionally NOT set here. It requires segmentation
# masks to know what to cut and paste, and this dataset is bbox-only -- so
# the copy_paste=0.3 used in the original Colab Run D almost certainly had
# no real effect. This re-run is therefore genuinely "P2 head + high-res
# only" (yolov26s_p2_combined_v2 name kept for continuity with the earlier
# result, but interpret it as P2+960px, not P2+960px+copy_paste).
#
# On Colab this OOM'd at batch=8/imgsz=960 and needed batch=4/imgsz=832.
# On a cluster GPU with real headroom, try batch=16/imgsz=960 first --
# drop back only if this still OOMs.
python train.py \
    --run-key yolov26s_p2_combined_v2 \
    --weights yolo26s-p2.yaml \
    --epochs 150 --imgsz 960 --batch 16 --workers 8
