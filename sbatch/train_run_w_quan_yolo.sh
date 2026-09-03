#!/bin/bash
#SBATCH --job-name=run_w_quan_yolo
#SBATCH --partition=research
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_quan_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# Point to QUAN's local codebase so we do NOT overwrite your official ultralytics environment
export PYTHONPATH=/mnt/weka/etadevosyan/pcb-yolo/QUAN_ultralytics:$PYTHONPATH

cd /mnt/weka/etadevosyan/pcb-yolo/QUAN_ultralytics

echo "=== Running QUAN (Quaternion Approximate Network) Training for PCB Detection ==="

# Train QUAN-YOLO11s (small model, matching YOLO26s scale)
python -c "
import jsoncd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training
git pull origin main

# Submit the QUAN training job
sbatch sbatch/train_run_w_quan_yolo.sh

import time
from pathlib import Path
from ultralytics import YOLO

data_yaml = '/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8/data.yaml'
project_dir = '/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/runs/yolo11s_quan'
results_dir = Path('/mnt/weka/etadevosyan/pcb-yolo/results')
results_dir.mkdir(parents=True, exist_ok=True)

# 1. Initialize QUAN-YOLO11s model
model = YOLO('yolo11s-quan.yaml')

# 2. Train on PCB dataset
trainer = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    classes=[2, 4, 7, 9],
    device=0,
    optimizer='SGD',
    project=project_dir,
    name='pcb-filtered',
    exist_ok=True
)

# 3. Standardized test evaluation at conf=0.001
best_weights = Path(project_dir) / 'pcb-filtered' / 'weights' / 'best.pt'
clean_model = YOLO(str(best_weights))
metrics = clean_model.val(
    data=data_yaml,
    split='test',
    classes=[2, 4, 7, 9],
    conf=0.001,
    iou=0.5,
    device=0
)

# 4. Save results to comparison table JSON
speed = metrics.speed
total_time_ms = speed.get('preprocess', 0.0) + speed.get('inference', 0.0) + speed.get('postprocess', 0.0)
fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
class_names = ['Capacitor', 'Connector', 'Electrolytic Capacitor', 'IC']
per_class_ap = {name: float(ap) for name, ap in zip(class_names, metrics.box.ap50)}

summary = {
    'model': 'yolo11s_quan',
    'weights': str(best_weights),
    'mAP50': float(metrics.box.map50),
    'mAP50_95': float(metrics.box.map),
    'precision': float(metrics.box.p.mean()),
    'recall': float(metrics.box.r.mean()),
    'total_time_ms': float(total_time_ms),
    'fps': float(fps),
    'per_class_ap50': per_class_ap,
    'eval_conf': 0.001,
    'eval_iou': 0.5,
    'eval_split': 'test',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

out_json = results_dir / 'yolo11s_quan.json'
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)

print('Successfully saved QUAN evaluation to:', out_json)
"
