#!/bin/bash
#SBATCH --job-name=run_h_p2_only
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

source /home/etadevosyan/miniconda3/etc/profile.d/conda.sh
conda activate pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/Ensemble-methods-of-YOLO-models-for-PCB-detection

python train.py \
    --run-key yolov26s_p2_only \
    --weights yolo26s-p2.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
