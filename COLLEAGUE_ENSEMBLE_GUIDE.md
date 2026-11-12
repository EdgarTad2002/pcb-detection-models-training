# 🛠️ Colleague's Quick-Start Guide: Top 2 PCB Models & Ensemble Integration

> **Target Audience:** Collaborators / Researchers on the YSU High-Performance Computing Cluster (`cluster.ysu.am`).  
> **Purpose:** Step-by-step instructions to locate existing weights, retrain the two highest-performing models on the original dataset, and plug them into an ensemble pipeline.

---

## 📌 Executive Summary of the Two Top Models

| Feature | **Model 1: `yolo26s_agwd_sdice_smooth0.01_uniform`** | **Model 2: `yolo26s_exp7_all_approaches_together_1280`** |
| :--- | :--- | :--- |
| **Role** | **#1 Single-Model Champion** across 134 cluster runs | **#1 Multi-Paradigm Physics & Loss Champion** |
| **mAP@50 (Overall)** | **65.78%** | **63.47%** |
| **Micro-Capacitor AP** | **43.59%** (Best single-model capacitor score) | **36.14%** |
| **Connector AP** | **69.52%** | **72.48%** (Wins on metallic connectors) |
| **Electrolytic Cap AP** | **~76.0%** | **85.10%** (Highest single-model score) |
| **IC AP** | **73.24%** | **70.69%** |
| **Key Innovation** | **Anisotropic Gaussian Wasserstein Distance (`AGWD`)** + Symmetrized Jeffreys KL-Divergence + **Dynamic-$k$ Optimal Transport (`SimOTA`)** + Uniform Multi-Stage **Squared-Dice (`SDice`)** loss (`smooth=0.01`) | **Grand Unified Physics** (Bilateral Retinex + Agaian AGC + Sargis Weber contrast) + Multi-Stage SDice + Auxiliary SR head |
| **Result when Ensembled** | 🏆 **68.77% mAP@50** (All-Time Project Record via Weighted Box Fusion) |

---

## 🖥️ 1. Cluster Environment & Setup

Connect to the cluster and activate the standardized Conda environment:

```bash
ssh <your_username>@cluster.ysu.am

# Source conda & activate shared project environment
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# Set working directory to project workspace
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen
# (or /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training)

# Ensure proper cache & path exports
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

---

## 💾 2. Where the Checkpoints are Stored on the Cluster

If you want to use the **already-trained weights** immediately without retraining:

### Checkpoint Paths:
1. **Model 1 (`yolo26s_agwd_sdice_smooth0.01_uniform`)**:
   ```bash
   /mnt/weka/etadevosyan/pcb-yolo/runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt
   # or
   runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt
   ```
2. **Model 2 (`yolo26s_exp7_all_approaches_together_1280`)**:
   ```bash
   /mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen/runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt
   # or
   runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt
   ```

### Quick Verification Command (Paste into terminal):
```bash
python -c "
from pathlib import Path
candidates = [
    'runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt',
    '/mnt/weka/etadevosyan/pcb-yolo/runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt',
    'runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt',
    '/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen/runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt'
]
for c in candidates:
    p = Path(c)
    print(f'[{ \"EXISTS\" if p.exists() else \"MISSING\" }] {p.resolve()}')
"
```

---

## 🚀 3. How to Retrain on the "Original" Datasets

There are two versions of the original Zhou & Agaian (2026) dataset:
- **Original 640px Baseline:** `datasets/pcb-unified-4class/data.yaml`  
- **Original 1280px Native Resolution:** `datasets/pcb-native-res-unified-4class/data.yaml`

### Retraining Model 1: `yolo26s_agwd_sdice_smooth0.01_uniform`
Implemented in `/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/agwd_ota_loss.py` and invoked via `train.py`. It combines:
1. **Scale-Adaptive Normalized Wasserstein Distance ($W_2^2$) + Symmetrized Jeffreys KL-Divergence ($D_J$)** for anisotropic bounding boxes (`--agwd-ota --agwd-kappa 12.0 --agwd-tau 576.0 --agwd-alpha-kld 1.0`).
2. **Dynamic-$k$ SimOTA Label Assigner** (`--tal-beta 3.0 --dynamic-k-max 12`) to prevent positive-anchor starvation on sub-8px capacitors.
3. **Multi-Stage Squared-Dice Loss** across pyramid heads (`--squared-dice --lambda-dice 0.50 --smooth-dice 0.01 --stage-weights uniform`).

```bash
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# Retrain on 1280px dataset (or pass --data datasets/pcb-native-reviewed-r4-final/data.yaml):
python train.py \
    --model yolo26s \
    --weights yolo26s.pt \
    --run-key "yolo26s_retrain_agwd_sdice_original_1280" \
    --data datasets/pcb-native-res-cleaned-1280/data.yaml \
    --eval-data datasets/pcb-native-res-cleaned-1280/data.yaml \
    --eval-split val \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --optimizer SGD \
    --lr0 0.01 \
    --lrf 0.01 \
    --momentum 0.937 \
    --weight-decay 0.0005 \
    --warmup-epochs 3.0 \
    --agwd-ota \
    --agwd-kappa 12.0 \
    --agwd-tau 576.0 \
    --agwd-alpha-kld 1.0 \
    --tal-beta 3.0 \
    --dynamic-k-max 12 \
    --squared-dice \
    --lambda-dice 0.50 \
    --smooth-dice 0.01 \
    --stage-weights uniform \
    --seed 42
```

### Retraining Model 2: `yolo26s_exp7_all_approaches_together_1280`
Uses the Grand Unified Physics pipeline (Retinex reflection + Agaian AGC + Sargis Weber edges) + Auxiliary Super-Resolution + Multi-Stage SDice:

```bash
# On 1280px Native Original Data:
python models/superyolo_multistage.py \
    --run-key "yolo26s_retrain_exp7_all_approaches_original_1280" \
    --weights yolo26s.pt \
    --data datasets/pcb-native-res-unified-4class/data.yaml \
    --eval-data datasets/pcb-native-res-unified-4class/data.yaml \
    --smooth 1.0 \
    --stage-weights-mode calibrated \
    --use-squared-dice \
    --lambda-group 0.15 \
    --lambda-sr 0.10 \
    --cls 1.5 --box 5.0 --dfl 2.0 --label-smoothing 0.10 \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --workers 8 \
    --device 0
```

---

## 🤝 4. How to Use Both Models in an Ensemble Pipeline

### Method A: Built-in Weighted Box Fusion (WBF) Evaluator
Run the standalone super-ensemble evaluation tool (`tools/eval_super_ensemble.py`):

```bash
# Run on GPU (inside an sbatch script or srun session):
python tools/eval_super_ensemble.py \
    --data datasets/pcb-native-res-unified-4class/data.yaml \
    --split test \
    --imgsz 1280 \
    --weights \
        runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt \
        runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt \
    --model-weights 1.30 1.25 \
    --iou-thr 0.55 \
    --max-det 1000 \
    --run-key "my_colleague_top2_ensemble" \
    --device 0

# Or test directly on the Login Node CPU by adding --device cpu:
# python tools/eval_super_ensemble.py ... --device cpu
```

---

### Method B: Python API for Your Custom Ensemble Script
If you are writing your own custom voting, NMS, or stacking script, here is how to extract raw boxes, scores, and class predictions:

```python
from pathlib import Path
import torch
from ultralytics import YOLO

# 1. Load checkpoints
w1 = "runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt"
w2 = "runs/detect/runs/yolo26s_exp7_all_approaches_together_1280/pcb-filtered/weights/best.pt"

model1 = YOLO(w1)
model2 = YOLO(w2)

image_path = "datasets/pcb-native-res-unified-4class/test/images/sample.jpg"

# 2. Run inference at test resolution (conf=0.001 to get dense candidates for fusion)
res1 = model1.predict(image_path, imgsz=1280, conf=0.001, max_det=1000, device=0)[0]
res2 = model2.predict(image_path, imgsz=1280, conf=0.001, max_det=1000, device=0)[0]

# 3. Extract bounding boxes (normalized [x1, y1, x2, y2]), scores, and class labels
# Model 1 predictions:
boxes1 = res1.boxes.xyxyn.cpu().numpy()  # normalized [0, 1]
scores1 = res1.boxes.conf.cpu().numpy()
classes1 = res1.boxes.cls.cpu().numpy().astype(int)

# Model 2 predictions:
boxes2 = res2.boxes.xyxyn.cpu().numpy()
scores2 = res2.boxes.conf.cpu().numpy()
classes2 = res2.boxes.cls.cpu().numpy().astype(int)

# 4. Feed into your custom ensemble method (e.g. ensemble-boxes WBF):
from ensemble_boxes import weighted_boxes_fusion

boxes_list = [boxes1, boxes2]
scores_list = [scores1, scores2]
labels_list = [classes1, classes2]
weights = [1.30, 1.25]  # Model 1 has slightly higher weight due to higher single mAP

fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    weights=weights,
    iou_thr=0.55,
    skip_box_thr=0.001
)

print(f"Fused {len(fused_boxes)} candidate detections!")
```

---

## 📊 5. Reference Metrics to Verify Against

When your evaluation runs correctly, you should observe numbers in these ranges:

| Class | Class ID | Expected AP@50 Range | Key Characteristic |
| :--- | :---: | :---: | :--- |
| **Capacitor** | `0` | **38.0% – 46.5%** | Highly sensitive to resolution and Squared-Dice smoothing |
| **Connector** | `1` | **69.0% – 81.5%** | Highly sensitive to Sargis Weber / Retinex edge enhancement |
| **Electrolytic Capacitor** | `2` | **72.0% – 85.0%** | Large distinct cylindrical components |
| **IC** | `3` | **70.0% – 79.5%** | Consistent rectangular quad-pack / DIP geometry |
| **Overall mAP@50** | — | **63.5% – 68.8%** | Single Model: ~65.8% \| WBF Ensemble: ~68.8% |

---

## 📬 Support & Collaboration Notes
- All experiment outputs and JSON benchmarks land automatically in `/mnt/weka/etadevosyan/pcb-yolo/results/`.
- To re-generate the latest global leaderboard markdown table, run:
  ```bash
  python aggregate_results.py --results-dir /mnt/weka/etadevosyan/pcb-yolo/results
  head -n 25 /mnt/weka/etadevosyan/pcb-yolo/results/comparison_table.md
  ```
