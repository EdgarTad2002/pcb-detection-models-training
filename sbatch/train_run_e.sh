#!/bin/bash
#SBATCH --job-name=run_e_res832
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

source /home/etadevosyan/miniconda3/etc/profile.d/conda.sh
conda activate pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/Ensemble-methods-of-YOLO-models-for-PCB-detection

python train.py \
    --run-key yolov26s_res832_only \
    --weights yolo26s.pt \
    --epochs 100 --imgsz 832 --batch 16 --workers 8
