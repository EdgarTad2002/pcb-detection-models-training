#!/bin/bash
#SBATCH --job-name=yolov10s
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

source /home/etadevosyan/miniconda3/etc/profile.d/conda.sh
conda activate pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/Ensemble-methods-of-YOLO-models-for-PCB-detection

python train.py \
    --run-key yolov10s \
    --weights yolov10s.pt \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
