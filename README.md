# PCB Component Detection — Benchmarking & Improving YOLO Architectures

Extending [Zhou & Agaian (2026), *"Ensemble Learning Using YOLO Models for Semiconductor E-Waste Recycling"*](https://doi.org/10.3390/info17040322) with a reproduction of their baselines, the introduction of **YOLO26** (released after the paper), a diagnosis of a critical small-object failure mode, and a controlled ablation study to fix it — all trained on the **YSU Cluster**.

---

## Why this project

The original paper benchmarks YOLOv5s → YOLOv12s on PCB component detection and shows that ensembling multiple YOLO architectures beats any single model. This repo:

1. **Reproduces their six baselines** under an identical, fixed evaluation protocol
2. **Introduces YOLO26s** — Ultralytics' newest release, not covered in the paper — and finds it wins decisively on speed but has a severe accuracy gap on one class
3. **Diagnoses the "Capacitor Crisis"**: YOLO26s collapses to ~1% AP / <1% recall on the Capacitor class specifically, despite performing competitively on the other three
4. **Runs a controlled ablation study** to isolate exactly which architectural or training change actually fixes it — resolution, augmentation, and a small-object (P2) detection head, tested independently rather than bundled together

The goal isn't to "beat" the paper's numbers — it's to understand *why* a state-of-the-art model fails on this specific class, and what actually recovers it.

---

## The four PCB component classes

| Class | Description | Detection difficulty |
|---|---|---|
| IC | Integrated circuits | Moderate — consistent rectangular shape |
| Connector | Board/edge connectors | Moderate — varying types and orientations |
| Electrolytic Capacitor | Cylindrical capacitors | Easy — distinctive shape and size |
| **Capacitor** | Small surface-mount capacitors | **Hard** — tiny, dense (66.7% of all instances), visually similar to solder pads/vias |

---

## Repository structure

```
.
├── train.py                  # Shared, parameterized training + evaluation script
│                              # (every model/run below is just different CLI flags
│                              #  against this one script — no duplicated logic)
├── aggregate_results.py      # Combines all per-run result JSONs into one comparison table
├── environment.yml           # Conda environment spec
├── sbatch/                   # One Slurm submission script per model/experiment
│   ├── train_yolov5s.sh
│   ├── train_yolov8s.sh
│   ├── train_yolov9s.sh
│   ├── train_yolov10s.sh
│   ├── train_yolov11s.sh
│   ├── train_yolov12s.sh
│   ├── train_yolov26s.sh
│   ├── train_run_d.sh        # Ablation: P2 head + high-res (960px)
│   ├── train_run_e.sh        # Ablation: resolution only (960px, standard head)
│   ├── train_run_g.sh        # Ablation: geometric + color augmentation only
│   └── train_run_h.sh        # Ablation: P2 detection head only (standard res)
├── pcb_colab/                # Dataset download + config helpers
│   └── colab_helpers.py
├── ensemble/                 # Original paper's NMS / voting / WBF fusion code
│   ├── nms.py
│   └── voting_methods/
├── datasets/                 # NOT committed — see Setup below
└── runs/                     # NOT committed — training outputs land here
```

---

## Setup

### 1. Clone and create the environment

```bash
git clone https://github.com/EdgarTad2002/pcb-detection-models-training.git
cd pcb-detection-models-training
conda env create -f environment.yml
conda activate pcb-yolo
```

### 2. Get the dataset

Not included in this repo (large binary files don't belong in git, and the source is already reproducibly hosted). Download via Roboflow:

```python
from pcb_colab.colab_helpers import download_roboflow_dataset
from pathlib import Path

download_roboflow_dataset(api_key="YOUR_ROBOFLOW_KEY", project_root=Path("."))
```

Source: [Printed Circuit Board dataset](https://universe.roboflow.com/roboflow-100/printed-circuit-board) (Roboflow 100, v3, CC BY 4.0)

Confirm it landed correctly:
```bash
ls datasets/pcb-filtered-yolov8/data.yaml
```

### 3. Evaluation protocol (fixed across every model in this repo)

| Setting | Value |
|---|---|
| Confidence threshold | `0.25` |
| IoU threshold | `0.50` |
| Classes | Capacitor, Connector, Electrolytic Capacitor, IC (indices `2, 4, 7, 9`) |
| Split | `test` (held out, distinct from `valid`) |

Matched to the original paper's methodology so results are directly comparable.

---

## Usage

### Train any single model

```bash
python train.py \
    --run-key yolov26s \
    --weights yolo26s.pt \
    --epochs 100 --imgsz 640 --batch 16 --workers 8
```

Every run writes its evaluation summary to `results/<run-key>.json` — no shared file gets read-modified-written, so concurrent Slurm jobs never race each other.

### On the YSU Cluster

```bash
sbatch sbatch/train_yolov8s.sh
squeue -u <your-username>          # check queue status
tail -f slurm_<jobid>.out           # watch live output
```

All jobs are independent — submit as many as your allocation allows in parallel.

### Aggregate results

```bash
python aggregate_results.py
```

Builds one comparison CSV from whatever `results/*.json` files currently exist — safe to run at any point, even mid-experiment.

---

## Roadmap

- [ ] Complete Run D / E(960px) / H and publish results
- [ ] **SAHI tiled training/inference** — slice board images so small capacitors occupy proportionally more pixels; the most direct lever on the actual root cause
- [ ] **Hand-rolled bbox-based copy-paste** — replicate the density benefit of Ultralytics' segmentation-based `copy_paste` using bbox crops instead
- [ ] **Relation/context-based re-scoring** — boost low-confidence Capacitor detections that sit near confidently-detected ICs, based on spatial co-occurrence priors
- [ ] Fold the best-performing YOLO26 variant back into the paper's own NMS / Consensus-voting / WBF ensemble pipeline (`ensemble/`)

---

## Acknowledgments

- Baseline methodology, dataset filtering, and ensemble fusion code adapted from Zhou, X. & Agaian, S. (2026). *Ensemble Learning Using YOLO Models for Semiconductor E-Waste Recycling.* **Information**, 17(4), 322. https://doi.org/10.3390/info17040322
- Dataset: [Roboflow 100 — Printed Circuit Board](https://universe.roboflow.com/roboflow-100/printed-circuit-board) (CC BY 4.0)
- Model architectures: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Training infrastructure: YSU Cluster
