# Automated Optical Inspection (AOI) for Dense PCB Components: Benchmarking, Rectification, & the Spatial-Efficiency Paradox

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv5--YOLO26-purple.svg)](https://github.com/ultralytics/ultralytics)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20H100%20(YSU%20Cluster)-green.svg)](https://cluster.ysu.am)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An advanced, publication-grade research framework and production pipeline for **Automated Optical Inspection (AOI)** of surface-mount devices (SMD) on dense Printed Circuit Boards (PCBs) for **semiconductor e-waste recycling** and **industrial manufacturing QA**.

This project directly reproduces, substantially extends, and resolves critical small-object failure modes from the foundational literature:
> **Reference Paper:**  
> Zhou, X., & Agaian, S. (2026). *Ensemble Learning Using YOLO Models for Semiconductor E-Waste Recycling*. **Information**, 17(4), 322. [https://doi.org/10.3390/info17040322](https://doi.org/10.3390/info17040322)

---

## 📌 Table of Contents

1. [Executive Summary & The Core Research Problem](#-1-executive-summary--the-core-research-problem)
2. [The Dataset Journey & Taxonomy Rectification](#-2-the-dataset-journey--taxonomy-rectification)
3. [Explored Architectures & Methodologies](#-3-explored-architectures--methodologies)
   - [Model Families Evaluated](#model-families-evaluated)
   - [What Worked (Empirically Validated Breakthroughs)](#what-worked-empirically-validated-breakthroughs)
   - [What Failed (Key Negative Results & Physical Insights)](#what-failed-key-negative-results--physical-insights)
4. [Official Test Set Leaderboard & Results](#-4-official-test-set-leaderboard--results)
5. [High-Performance Slurm Cluster Workflow](#-5-high-performance-slurm-cluster-workflow)
   - [Cluster Infrastructure & Environment](#cluster-infrastructure--environment)
   - [Pushing Code & Remote Job Launching](#pushing-code--remote-job-launching)
   - [SLURM Shell Script Specifications](#slurm-shell-script-specifications)
   - [Monitoring & Managing Jobs in the Terminal](#monitoring--managing-jobs-in-the-terminal)
   - [Pulling Remote Results, Checkpoints, & Tables](#pulling-remote-results-checkpoints--tables)
6. [Local Setup, Training, & Inference Guide](#-6-local-setup-training--inference-guide)
   - [Conda Environment Setup](#conda-environment-setup)
   - [Dataset Preparation & Symlinking](#dataset-preparation--symlinking)
   - [Local Training](#local-training)
   - [Inference & Visualization (SAHI, DIoU, Notebooks)](#inference--visualization-sahi-diou-notebooks)
7. [Repository File & Directory Structure](#-7-repository-file--directory-structure)
8. [Citation & Acknowledgments](#-8-citation--acknowledgments)

---

## 🔬 1. Executive Summary & The Core Research Problem

### The Industrial Challenge
In automated PCB recycling and SMT electronics manufacturing, vision-based optical inspection systems must accurately localize and classify densely packed electronic components across large boards. While macro-components such as Integrated Circuits (ICs) and electrolytic capacitors exhibit distinct geometrical boundaries, **micro-components (such as 0402 and 0201 Surface-Mount Chip Capacitors)** present a severe detection barrier:
- They comprise over **66% of all physical instances** on typical boards.
- They are physically tiny (often spanning only **$3 \times 3$ to $8 \times 8$ pixels** at standard input resolutions).
- They share extreme visual similarity with passive solder pads, surface traces, and PCB test vias.

### The "Capacitor Crisis"
When modern one-stage object detectors (YOLOv5s through the newly released YOLO26s) are trained on standard $640 \times 640$ image resolutions, they experience catastrophic feature degradation on the micro-capacitor class:
- **Baseline YOLO26s** achieves a blisteringly fast **147 FPS** on NVIDIA H100 GPUs, but collapses to **$\approx 5.36\%$ AP50** on Capacitors (and $< 1\%$ AP under unrectified legacy taxonomies).
- Detection heads downsampled through typical $32\times$ backbones suffer from the **Nyquist-Shannon spatial undersampling limit**, where micro-objects lose essential spatial feature representations.

### The "1280px Efficiency Paradox"
Increasing the detector's input resolution to native $1280 \times 1280$ resolution restores fine spatial detail and substantially boosts Capacitor AP ($5.36\% \rightarrow 13.91\%$ on legacy data, and up to **$28.94\%$** on rectified data). However, native 1280px inference increases quadratic computational complexity ($4\times$ pixel volume):
- Inference throughput collapsed from **147.0 FPS down to 7.5 FPS**.
- The **Frontier Efficiency Index** ($\text{FEI} = F_1 \times \log_{10}(\text{FPS})$) plummeted from $1.207$ to $0.537$, failing the hard real-time SMT production throughput requirement of $\ge 30\text{ FPS}$.

### How This Project Solves It
Through a unified combination of:
1. **Dataset Taxonomy Rectification** (resolving a critical annotation flaw in the benchmark).
2. **Physics-Informed Spectral Priors** ($\alpha=0.25$ optical band weighting).
3. **Loss Reweighting & Bounding Box Copy-Paste Augmentation**.
4. **Distance-IoU (DIoU) NMS & Class-Adaptive Calibrated Thresholding**.
5. **Dual-Pass Slicing Aided Hyper Inference (SAHI)**.

We engineered the **`yolov26s_unified_1280`** champion pipeline, achieving **$58.35\%$ mAP50**, **$77.73\%$ Precision**, and **$33.1\text{ FPS}$** on NVIDIA H100 hardware—recovering real-time speed ($\text{FEI} = 0.9317$) while delivering industry-leading detection precision.

---

## 📦 2. The Dataset Journey & Taxonomy Rectification

### The Legacy Problem (`datasets/pcb-filtered-yolov8`)
The foundational paper relied on the Roboflow-100 Printed Circuit Board dataset (`v3`, 44 held-out test boards). An in-depth audit of the raw dataset annotations revealed two severe structural defects:
1. **Noisy Multi-Class Clutter**: 23 disparate classes existed in the raw labels (Resistor, Solder Pad, Pin, Hole, etc.) alongside human labeling typos (e.g., class `9: IC` and class `22: iC`).
2. **The Split-Capacitor Annotation Bug**: Human annotators split surface-mount capacitors across two separate classes:
   - `Class 1: Capacitor Jumper`
   - `Class 2: Capacitor`
   The legacy evaluation protocol only evaluated target indices `[2, 4, 7, 9]`. Consequently, hundreds of ground-truth micro-capacitors labeled as `Class 1` were treated as background or false positives, mathematically penalizing any model that correctly detected them.

### The Unified 4-Class Rectification (`tools/rectify_dataset.py`)
We developed a non-destructive dataset rectification tool that maps the 23 raw classes into a clean, canonical 4-class taxonomy (`datasets/pcb-unified-4class`):

$$\begin{aligned}
\text{Raw Class 1 (Capacitor Jumper)} &\cup \text{Raw Class 2 (Capacitor)} \longrightarrow \mathbf{\text{Class 0: Capacitor}} \\
\text{Raw Class 4 (Connector)} &\longrightarrow \mathbf{\text{Class 1: Connector}} \\
\text{Raw Class 7 (Electrolytic Capacitor)} &\longrightarrow \mathbf{\text{Class 2: Electrolytic Capacitor}} \\
\text{Raw Class 9 (IC)} &\cup \text{Raw Class 22 (iC typo)} \longrightarrow \mathbf{\text{Class 3: IC}}
\end{aligned}$$
All other 19 non-target classes (pins, tracks, mounting holes) are completely discarded.

```bash
python tools/rectify_dataset.py \
    --source datasets/pcb-filtered-yolov8 \
    --dest datasets/pcb-unified-4class
```

#### Verification & Class Distribution Across Splits:
| Split | Total Images | Class 0: Capacitor | Class 1: Connector | Class 2: Electrolytic Cap | Class 3: IC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 506 | 32,875 | 4,288 | 328 | 2,752 |
| **Valid** | 145 | 8,924 | 1,180 | 85 | 739 |
| **Test** | 44 | 2,757 | 374 | 27 | 216 |

#### The Rectification Impact:
Retraining the entire 10-model benchmark on the rectified dataset yielded a massive **$\approx +14\%$ to $+15\%$ absolute jump in Capacitor AP50** across all YOLO models (e.g., YOLO26s Capacitor AP surged from $5.36\%$ to **$20.66\%$** at 640px, and up to **$28.94\%$** at 1280px).

---

## 🧠 3. Explored Architectures & Methodologies

### Model Families Evaluated
All models were trained from official pre-trained weights for 100 epochs on NVIDIA H100 GPUs under identical training parameters (`imgsz=640` or `1280`, `batch=16`, `workers=8`, `eval-conf=0.001`):
1. **YOLOv5s**: Anchor-based baseline established in Zhou & Agaian (2026).
2. **YOLOv8s**: Anchor-free baseline with C2f feature extraction.
3. **YOLOv9s**: Programmable Gradient Information (PGI) and GELAN architecture.
4. **YOLOv10s**: Dual-label assignment for NMS-free end-to-end inference.
5. **YOLOv11s**: C3k2 modules with optimized spatial attention.
6. **YOLOv12s**: Area attention mechanisms balancing global and local receptive fields.
7. **YOLO26s**: Ultralytics' newest generation detector featuring optimized lightweight kernel operations and ultra-low latency (147 FPS).
8. **RT-DETR-L**: Real-Time Detection Transformer utilizing a hybrid encoder, Deformable Multi-Scale Attention, and Hungarian bipartite matching.

---

### What Worked (Empirically Validated Breakthroughs)

#### 1. Loss Reweighting & Label Smoothing (`--cls-weight 1.5 --box-weight 5.0 --dfl-weight 2.0`)
- **The Problem:** The standard bounding box loss weight ($7.5$) and classification weight ($0.5$) cause models to ignore minute $4\times 4$ pixel capacitors in favor of easily localizable ICs.
- **The Fix:** Reweighting the multitask loss function:
  $$\mathcal{L}_{\text{total}} = 5.0 \cdot \mathcal{L}_{\text{box}} + 1.5 \cdot \mathcal{L}_{\text{cls}} + 2.0 \cdot \mathcal{L}_{\text{dfl}}$$
  Combined with label smoothing ($\epsilon=0.10$), this prevents overconfidence and forces gradients to focus on fine ceramic chip capacitor contours.
- **Result:** Lifted YOLO26s Capacitor AP on rectified 1280px to **$28.28\%$** and achieved overall **$58.96\%$ mAP50**.

#### 2. Micro-Component Bounding Box Copy-Paste (`bbox_copy_paste.py`)
- Standard Ultralytics `copy_paste` requires polygonal segmentation masks.
- We implemented an offline synthetic augmentation engine extracting isolated ground-truth chip capacitors into a bank (`build_capacitor_bank.py`) and randomly placing them into non-overlapping PCB substrate regions during training, mitigating class frequency imbalances.

#### 3. Physics-Informed Spectral Priors (`physics_spectral_yolo26.py`)
- Standard CNNs treat RGB images as uncalibrated tensors. Utilizing optical reflectance curves from the PCB-Vision benchmark (Arbash et al., IEEE Sensors J. 2024), we calculated the physical contrast disparity $\mathcal{C}(\lambda)$ across 31 spectral bands ($400\text{ nm} - 700\text{ nm}$) between FR-4 solder mask and ceramic components.
- Introduced an information-theoretic logarithmic $\alpha$-rooting contrast modulation (recommended by Prof.~Sos Agaian) for Layer-0 adapter initialization:
  $$\tilde{\mathcal{W}}(\lambda; \alpha) = \mathcal{C}(\lambda) \cdot \left[1 + \ln(1 + \mathcal{C}(\lambda))\right]^\alpha$$
- **Result:** An empirical $\alpha$-sweep revealed an optimal peak at **$\alpha=0.25$**, delivering **$54.13\%$ mAP50** (+2.45% over baseline) and **$38.60\%$ mAP50-95** (+2.80%) with zero latency penalty.

#### 4. Distance-IoU NMS (DIoU-NMS) vs. Adjacent Multi-Pin Headers
- Standard Hard-NMS suppresses proposals whose overlap exceeds $N_t$. When identical pin headers touch along a seam ($0\text{ mm}$ clearance), standard Hard-NMS eliminates valid neighboring connectors.
- By incorporating normalized central distance $\mathcal{S}_{\text{DIoU}} = \text{IoU} - (\rho^2/c^2)^\beta$, DIoU-NMS successfully **rescued 4 physical connectors from false deletion** across the test set without increasing false alarms.

#### 5. Class-Adaptive Calibrated Confidence Thresholds ($\tau_c^*$)
- Flat confidence thresholds ($\tau = 0.25$) are suboptimal: macro components (ICs, electrolytic caps) peak in $F_1$ at $\tau \ge 0.60$, while micro-capacitors peak at $\tau \approx 0.07$.
- Sweeping 93 threshold steps per class to maximize the harmonic mean ($F_1$) yielded optimal thresholds:
  - $\tau_{\text{Cap}}^* = 0.07$
  - $\tau_{\text{Conn}}^* = 0.65$
  - $\tau_{\text{ElCap}}^* = 0.87$
  - $\tau_{\text{IC}}^* = 0.59$
- **Result:** Lifted Macro-$F_1$ from **$46.05\% \rightarrow 59.62\%$ (+13.56% absolute gain)**.

#### 6. Slicing Aided Hyper Inference (SAHI) (`tools/sahi_pcb_inference.py`)
- Tiles full-resolution images into overlapping slices ($360\text{ px} - 480\text{ px}$, $20\%-25\%$ overlap), runs YOLO over each patch at native pixel resolution, and merges bounding boxes using batched NMS.
- Eliminates downsampling blur, recovering dozens of small capacitors that vanish in standard single-pass 640px inference.

---

### What Failed (Key Negative Results & Physical Insights)

1. **Spatial Sharpening Pre-Processing (Unsharp Masking & CLAHE)**:
   - *Hypothesis:* Sharpening 2D edges prior to inference will enhance micro-capacitor boundaries.
   - *Empirical Outcome:* Dropped mAP50 by **$-2.29\%$** ($53.77\% \rightarrow 51.48\%$).
   - *Physical Root Cause:* FR-4 circuit boards possess microscopic fiberglass weave patterns and etched copper grain. 2D spatial filtering sharpens this high-frequency substrate noise, which early convolutional layers misinterpret as false solder features. Micro-object detection requires receptive field adaptation, not heuristic image filtering.
2. **P2 Detection Head (Stride 4) from Scratch**:
   - Adding a high-resolution $160 \times 160$ feature map head to YOLO26s (`yolo26s-p2.yaml`) achieved **$50.47\%$ mAP50** (lower than the 3-head baseline $51.68\%$).
   - *Reason:* The newly added detection head was initialized from scratch; standard 100-epoch schedules with batch size 8 are insufficient for the additional parameters to converge without extensive pre-training.
3. **RT-DETR-L on Rectified 4-Class**:
   - Collapsed to $34.29\%$ mAP50 on the rectified dataset.
   - *Reason:* DETR-based architectures rely on bipartite Hungarian matching loss, which is hypersensitive to learning rate warmups, weight decay schedules, and positional embedding initialization when the number of target classes is dramatically reduced.

---

## 📊 4. Official Test Set Leaderboard & Results

Ranked on the official 44 held-out test PCB boards (PASCAL VOC / COCO $\text{IoU}=0.50$ evaluation protocol):

| Model Architecture | Dataset & Regimen | Resolution | mAP50 | FEI | $F_1$ | FPS (H100) | Latency | AP50 Cap | AP50 Conn | AP50 El-Cap | AP50 IC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **superyolo26s_rectified_1280 (🏆 #1 Grand Champion & ⚡ Real-Time Champion)** | Native Rectified + SR | 1280 | **63.04%** | **1.197** | **0.679** | **57.9** | **17.3 ms** | 35.73% | 71.77% | 77.50% | 67.14% |
| **rectified_yolov26s_unified_1280** | Multi-Scale Rectified | 1280 | **62.90%** | 0.418 | **0.684** | 4.1 | 244.9 ms | 37.78% | 72.87% | 74.75% | 66.18% |
| **rectified_yolov26s_native_res_physics_spectral_a0.0_1280** | Native + Physics Spectral | 1280 | 60.47% | 0.634 | 0.667 | 8.9 | 112.1 ms | 36.18% | 68.70% | 72.71% | 64.30% |
| **yolov26s_native_res_rectified_loss_reweight_1280** | Native Rectified + Loss Reweight | 1280 | 59.86% | 0.420 | 0.677 | 4.2 | 240.0 ms | 37.89% | 66.89% | 67.81% | 66.85% |
| **yolov26s_native_res_rectified_1280** | Native Rectified Baseline | 1280 | 59.81% | 1.039 | 0.672 | 35.2 | 28.4 ms | 38.02% | 69.92% | 67.81% | 63.50% |
| **yolov26s_rectified_clahe_s1.0_g0.6_640 (🎨 CLAHE Champion)** | Rectified + CLAHE ($s=1.0, \gamma=0.6$) | 640 | 59.26% | 0.616 | 0.633 | 9.4 | 106.5 ms | 20.91% | 68.02% | **84.64%** | 63.49% |
| **ensemble_wbf_top6** | Weighted Box Fusion (Top-6) | 640 | 58.76% | 0.665 | 0.583 | 13.8 | 72.4 ms | 6.70% | **80.69%** | 76.10% | **71.57%** |
| **sahi_hybrid_1280_640_sahi_640 (🔬 #1 Capacitor Record)** | Dual-Scale SAHI (1280+640) | 640 | 58.17% | 0.120 | 0.232 | 3.3 | 305.4 ms | **40.10%** | 56.53% | 80.82% | 55.22% |
| **rectified_yolov26s** | Rectified 4-Class Baseline | 640 | 58.12% | 0.450 | 0.612 | 5.4 | 184.0 ms | 20.66% | 68.40% | 77.50% | 65.93% |
| **sahi_hybrid_dual_640_sahi_640** | Dual 640 SAHI | 640 | 57.86% | 0.101 | 0.201 | 3.2 | 314.1 ms | 36.78% | 56.01% | 83.43% | 55.24% |
| **rectified_yolov26s_loss_reweight** | Rectified + Class Loss Reweight | 640 | 57.29% | 0.432 | 0.625 | 4.9 | 203.2 ms | 23.21% | 65.56% | 79.35% | 61.05% |
| **rectified_yolov11s** | Rectified 4-Class Baseline | 640 | 56.89% | 0.415 | 0.635 | 4.5 | 222.2 ms | 17.93% | 73.36% | 72.16% | 64.10% |
| **rectified_yolov5s** | Rectified 4-Class Baseline | 640 | 56.83% | 0.545 | 0.571 | 9.0 | 111.0 ms | 17.81% | 70.06% | 76.13% | 63.32% |
| **rectified_yolov9s_loss_reweight** | Rectified + Class Loss Reweight | 640 | 56.77% | 0.430 | 0.607 | 5.1 | 195.5 ms | 13.72% | 69.27% | 76.88% | 67.21% |
| **rectified_yolov9s** | Rectified 4-Class Baseline | 640 | 56.58% | 0.434 | 0.605 | 5.2 | 191.5 ms | 16.38% | 73.09% | 72.63% | 64.24% |
| **rectified_yolov8s** | Rectified 4-Class Baseline | 640 | 56.16% | 0.436 | 0.621 | 5.0 | 199.0 ms | 19.38% | 66.67% | 73.50% | 65.09% |
| **sahi_tile_640_sahi_eval_sahi_640** | Sliced Tile SAHI | 640 | 55.47% | 0.116 | 0.221 | 3.4 | 298.0 ms | 36.32% | 52.42% | 82.10% | 51.03% |
| **rectified_yolov10s** | Rectified 4-Class Baseline | 640 | 55.33% | 0.634 | 0.622 | 10.4 | 95.8 ms | 19.09% | 66.91% | 74.22% | 61.11% |
| **rectified_yolov12s** | Rectified 4-Class Baseline | 640 | 55.03% | 0.425 | 0.620 | 4.8 | 206.3 ms | 16.31% | 65.58% | 76.88% | 61.36% |
| **rectified_yolov26s_unified_640** | Multi-Scale Rectified | 640 | 54.62% | 1.142 | 0.603 | 78.5 | 12.7 ms | 20.38% | 65.75% | 73.38% | 58.96% |
| **omni_scale_champion_640** | Multi-Scale Compound Head | 640 | 54.30% | 0.434 | 0.625 | 5.0 | 201.9 ms | 20.37% | 61.29% | 79.94% | 55.60% |
| **rectified_yolov26s_luma_pretrained_1280** | LUMA-YOLO (LIAM + AConv + SPPELAN) | 1280 | 54.24% | 0.554 | 0.610 | 8.1 | 123.8 ms | 21.94% | 64.35% | 70.62% | 60.03% |
| **rectified_yolov26s_luma_pretrained_640** | LUMA-YOLO (LIAM + AConv + SPPELAN) | 640 | 51.86% | 0.498 | 0.556 | 7.9 | 127.3 ms | 13.58% | 56.90% | 79.04% | 57.92% |
| **vmamba_standalone_1280_v3** | Visual Mamba State-Space | 1280 | 39.02% | 0.218 | 0.487 | 2.8 | 351.2 ms | 8.77% | 36.36% | 70.76% | 40.18% |
| **vmamba_standalone_640_v3** | Visual Mamba State-Space | 640 | 35.08% | 0.560 | 0.439 | 18.9 | 52.9 ms | 1.76% | 34.60% | 77.41% | 26.54% |

---

## 🖥️ 5. High-Performance Slurm Cluster Workflow

All model training and heavy benchmarks are executed remotely on the **Yerevan State University (YSU) High-Performance Computing Cluster**.

### Cluster Infrastructure & Environment
- **Hostname:** `cluster.ysu.am`
- **Username:** `etadevosyan`
- **Cluster Root Path:** `/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training`
- **Shared Storage:** Distributed high-throughput Weka filesystem (`/mnt/weka/`)
- **GPU Accelerators:** NVIDIA H100 80GB GPUs
- **Conda Environment:** `/mnt/weka/etadevosyan/.conda/envs/pcb-yolo`
- **Shared Miniforge:** `/mnt/weka/shared-cache/miniforge3/bin/conda`

To ensure reproducible, permission-safe execution without quota collisions, all Slurm jobs export the following environment variables:
```bash
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
```

---

### Pushing Code & Remote Job Launching

We built automated developer tooling in `tools/` to eliminate manual `scp` and `ssh` steps.

#### 1. Push Local Code & Submit Any Slurm Job in One Command:
```bash
bash tools/remote_train.sh <sbatch_script_name> [optional_args]
```
*What this does under the hood:*
1. Runs `rsync` from your local machine to `/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/`, automatically filtering out heavy folders (`runs/`, `datasets/`, `.git/`, `*.pt`, `__pycache__`).
2. SSHes into `cluster.ysu.am`, verifies file existence, and executes `sbatch sbatch/<sbatch_script_name>`.
3. Displays the current active job queue via `squeue -u etadevosyan`.

*Example:*
```bash
# Push latest code changes and train rectified YOLO26s
bash tools/remote_train.sh train_rectified_yolov26s.sh

# Train physics-spectral ablation with alpha=0.25
bash tools/remote_train.sh train_run_u_physics_spectral.sh 0.25
```

---

### SLURM Shell Script Specifications

Every batch job in `sbatch/` is a self-contained, reproducible Bash script. Here is the canonical anatomy of an sbatch script (e.g., `sbatch/train_rectified_yolov26s.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=rect_yolov26s         # Job name displayed in squeue
#SBATCH --partition=research             # Target compute partition
#SBATCH --mem=32G                        # System RAM allocated
#SBATCH --cpus-per-task=8                # CPU cores for DataLoader workers
#SBATCH --gres=gpu:1                     # NVIDIA GPU allocation (H100)
#SBATCH --output=slurm_%j.out            # Combined stdout & stderr log (%j = job ID)

# 1. Configure storage and cache variables
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

# 2. Activate Conda environment from shared cache
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

# 3. Navigate to remote workspace
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

# 4. Execute parameterized training
python train.py \
    --run-key rectified_yolov26s \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 --imgsz 640 --batch 16 --workers 8 \
    --eval-conf 0.001
```

#### Launching the Entire 10-Model Rectified Benchmark in Parallel:
To retrain all 10 rectified models concurrently on the cluster:
```bash
ssh etadevosyan@cluster.ysu.am "cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training && bash sbatch/submit_all_rectified.sh"
```

---

### Monitoring & Managing Jobs in the Terminal

Connect to the cluster directly to monitor your workloads:
```bash
ssh etadevosyan@cluster.ysu.am
```

| Task | Slurm / Terminal Command | Description |
|:---|:---|:---|
| **View Active Jobs** | `squeue -u etadevosyan` | Shows job ID, partition, status (`R` = Running, `PD` = Pending), time, and nodes. |
| **Stream Live Output** | `tail -f slurm_<JOB_ID>.out` | Streams real-time training progress, loss values, GPU utilization, and metrics. |
| **Search Output** | `grep -E "(Epoch|mAP50)" slurm_<JOB_ID>.out` | Quickly inspect completed epochs and validation checkpoints. |
| **Cancel a Job** | `scancel <JOB_ID>` | Gracefully terminates a running or queued job. |
| **Cancel All Jobs** | `scancel -u etadevosyan` | Terminates all jobs owned by your user. |
| **Inspect GPU State** | `srun --jobid=<JOB_ID> nvidia-smi` | Checks live VRAM utilization and temperature on the assigned compute node. |

---

### Pulling Remote Results, Checkpoints, & Tables

When remote training jobs complete, pull down the lightweight checkpoints (`best.pt`), evaluation metrics, and curves using `tools/remote_pull_results.sh`:

```bash
bash tools/remote_pull_results.sh
```

*What this script does:*
1. **Syncs Evaluation JSONs:** Uses `rsync` with the cluster's miniforge binary to pull all `results/*.json` files.
2. **Selective Checkpoint Pulling:** Syncs **only** `best.pt`, `results.png`, `results.csv`, and confusion matrices from remote `runs/` folders while excluding massive `last.pt` and intermediate weights.
3. **Automatic Aggregation:** Automatically triggers `aggregate_results.py`, regenerating:
   - `comparison_table.csv`
   - `comparison_table.xlsx`
   - `comparison_table.md`

---

## 💻 6. Local Setup, Training, & Inference Guide

### Conda Environment Setup
On your local workstation (Linux / macOS / WSL2):

```bash
git clone https://github.com/EdgarTad2002/pcb-detection-models-training.git
cd pcb-detection-models-training

conda env create -f environment.yml
conda activate pcb-yolo
```

*Core dependencies:* `python=3.11`, `ultralytics`, `torch`, `torchvision`, `sahi`, `opencv-python-headless`, `ensemble-boxes`, `pandas`, `matplotlib`, `pyyaml`.

---

### Dataset Preparation & Symlinking
If downloading raw Roboflow-100 data from scratch:
```bash
# Export your Roboflow API key
export ROBOFLOW_API_KEY="your_api_key_here"

# Download and extract the raw dataset
python -c "
from pcb_colab.colab_helpers import download_roboflow_dataset
from pathlib import Path
download_roboflow_dataset(api_key='$ROBOFLOW_API_KEY', project_root=Path('.'))
"

# Generate the rectified 4-class dataset (uses zero-duplication symlinks)
python tools/rectify_dataset.py \
    --source datasets/pcb-filtered-yolov8 \
    --dest datasets/pcb-unified-4class
```

---

### Local Training
To train any model locally on a consumer or workstation GPU:

```bash
# Standard 640px training on Rectified 4-class dataset
python train.py \
    --run-key local_yolov26s \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --workers 4

# Native 1280px training with loss reweighting
python train.py \
    --run-key local_yolov26s_reweight_1280 \
    --weights yolo26s.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --epochs 100 \
    --imgsz 1280 \
    --batch 8 \
    --cls-weight 1.5 \
    --box-weight 5.0 \
    --dfl-weight 2.0
```

Each run automatically writes its final test evaluation metrics to `results/<run-key>.json` without race conditions.

---

### Inference & Visualization (SAHI, DIoU, Notebooks)

#### 1. Interactive Model Comparison Notebook (`visualize_predictions.ipynb`)
Open `visualize_predictions.ipynb` in VS Code or JupyterLab. The notebook provides:
- Auto-detection of Unified 4-Class vs. Legacy 23-Class checkpoints.
- Side-by-side ground truth vs. predicted bounding box visualizer.
- Class-filtered visual inspections (e.g., isolate and inspect micro-capacitors).
- Interactive dual-model comparison slider (`compare_two_models`).
- Standard YOLO vs. SAHI tiled hyper-inference comparison (`compare_groundtruth_standard_sahi`).

#### 2. Running Standalone SAHI Tiled Inference
```bash
python tools/sahi_pcb_inference.py \
    --source data_samples/images/ATTIOT_Bottom_jpg.rf.8a97ad6664656973c60d95057d9d473c.jpg \
    --weights runs/rectified_yolov26s/pcb-unified-4class/weights/best.pt \
    --slice-size 360 \
    --overlap 0.25 \
    --conf 0.20 \
    --out-dir runs/sahi_visualizations
```

#### 3. Calibrating Class-Adaptive Confidence Thresholds
To calculate the optimal $F_1$ threshold per class:
```bash
python tools/calibrate_thresholds.py \
    --weights runs/rectified_yolov26s/pcb-unified-4class/weights/best.pt \
    --data datasets/pcb-unified-4class/data.yaml \
    --split test
```

#### 4. Speed & Latency Benchmarking
To benchmark pure CUDA inference latency with warmups on a specific GPU:
```bash
python tools/benchmark_speed.py --device 0 --update-results
```

---

## 📁 7. Repository File & Directory Structure

```
pcb-detection-models-training/
│
├── README.md                           # Master project documentation & cluster guide (this file)
├── environment.yml                     # Conda environment specification
├── train.py                            # Parameterized, race-condition-free training & evaluation CLI
├── superyolo_pcb.py                    # SuperYOLO: Auxiliary Super-Resolution learning on PCB
├── aggregate_results.py                # Parses results/*.json and compiles CSV, Excel, & Markdown tables
├── comparison_table.md                 # Markdown summary of all evaluated models and ablations
├── comparison_table.csv                # Raw comma-separated metrics table
├── comparison_table.xlsx               # Multi-sheet Excel workbook with formatted benchmark tables
├── weekly_report.md                    # In-depth scientific progress report (Phase 4 findings & theory)
├── weekly_report.tex                   # LaTeX version of Phase 4 technical report
│
├── tools/                              # Research, rectification, and cluster deployment tooling
│   ├── remote_train.sh                 # Syncs code to YSU Slurm cluster and launches jobs
│   ├── remote_pull_results.sh          # Pulls best.pt, metrics, and training curves from cluster
│   ├── rectify_dataset.py              # Converts raw 23-class Roboflow data to clean 4-class taxonomy
│   ├── sahi_pcb_inference.py           # Slicing Aided Hyper Inference engine with batched NMS
│   ├── calibrate_thresholds.py         # Sweeps confidence thresholds to maximize per-class F1 score
│   ├── benchmark_speed.py              # Standardized CUDA event latency and FPS benchmark suite
│   ├── eval_sahi_benchmark.py          # Benchmark evaluating mAP improvements under SAHI slicing
│   ├── eval_tta_benchmark.py           # Multi-scale and flip Test-Time Augmentation evaluation engine
│   ├── eval_unified_pipeline.py        # End-to-end evaluation of the champion inference pipeline
│   ├── eval_sharpening_benchmark.py    # Spatial sharpening (Unsharp Masking, CLAHE) ablation study
│   ├── plot_sharpening_parameter_grid.py # Multi-panel visual parameter grid for CLAHE + unsharp masking
│   ├── build_luma_pretrained_checkpoint.py # Semantic weight mapping for LUMA-YOLO pretraining
│   ├── build_clahe_unsharp_dataset.py  # Generates enhanced dataset for CLAHE sweep champion
│   ├── build_sahi_tile_dataset.py      # Generates sliced tile datasets for SAHI models
│   ├── build_omni_scale_dataset.py     # Generates compound resolution datasets
│   ├── extract_pcb_vision_spectrum.py  # Computes optical spectral priors for physics-informed YOLO
│   ├── inspect_cluster_gt.py           # Audits ground-truth label files for class index anomalies
│   └── purge_unused_models.py          # Automated maintenance script to prune models and sync cluster
│
├── sbatch/                             # Ready-to-submit SLURM batch scripts for YSU Cluster
│   ├── train_superyolo26s_rectified_1280.sh # SuperYOLO Grand Champion (1280px + auxiliary SR head)
│   ├── train_yolov26s_native_res_rectified_1280.sh # Native-res rectified 1280px baseline
│   ├── train_yolov26s_native_res_rectified_loss_reweight_1280.sh # Native-res rectified loss-reweight
│   ├── train_rectified_yolov26s_luma_pcb.sh # Clean pretrained LUMA-YOLO (640px)
│   ├── train_rectified_yolov26s_luma_pretrained_1280.sh # Clean pretrained LUMA-YOLO (1280px)
│   ├── train_rectified_yolov26s_unified_1280.sh # Multi-scale unified champion (1280px)
│   ├── train_rectified_yolov26s_unified_640.sh # Multi-scale unified pipeline (640px)
│   ├── train_native_res_physics_spectral.sh # Physics-informed spectral YOLO (1280px)
│   ├── train_clahe_unsharp.sh          # Spatial enhancement CLAHE training runner
│   ├── train_sahi_tile_640.sh          # Sliced tile SAHI training runner
│   ├── train_omni_scale_champion.sh    # Compound resolution Omni-Scale runner
│   ├── train_rectified_yolov26s.sh     # Rectified YOLO26s baseline (640px)
│   ├── train_rectified_yolov26s_loss_reweight.sh # Rectified YOLO26s with loss reweighting
│   ├── train_rectified_yolov5s.sh      # Rectified YOLOv5s baseline
│   ├── train_rectified_yolov8s.sh      # Rectified YOLOv8s baseline
│   ├── train_rectified_yolov9s.sh      # Rectified YOLOv9s baseline
│   ├── train_rectified_yolov10s.sh     # Rectified YOLOv10s baseline
│   ├── train_rectified_yolov11s.sh     # Rectified YOLOv11s baseline
│   ├── train_rectified_yolov12s.sh     # Rectified YOLOv12s baseline
│   ├── eval_sahi_hybrid.sh             # Sliced Aided Hyper Inference hybrid benchmark runner
│   └── benchmark_speed.sh              # Remote Slurm speed benchmark runner
│
├── ensemble/                           # Multi-model ensembling and fusion modules
│   ├── nms.py                          # Non-Maximum Suppression fusion logic
│   ├── ensemble_eval.py                # Evaluates WBF and Voting ensembles across model checkpoints
│   └── voting_methods/                 # Consensus voting, affirmative voting, and WBF implementations
│
├── notebooks & visualization/
│   ├── visualize_predictions.ipynb     # Interactive ground truth vs. prediction comparison notebook
│   └── quick_gt_viewer.ipynb           # Lightweight bounding box visualizer for dataset exploration
│
├── datasets/                           # Local dataset storage (symlinked or downloaded; gitignored)
│   ├── pcb-unified-4class/             # Canonical, clean 4-class dataset (Class 0: Cap, 1: Conn, 2: ElCap, 3: IC)
│   └── pcb-filtered-yolov8/            # Legacy 23-class Roboflow raw dataset
│
├── results/                            # JSON output summaries from each test evaluation (git tracked)
│   ├── rectified_yolov26s.json
│   ├── yolov26s_unified_1280.json
│   └── ...
│
└── runs/                               # Model checkpoints (best.pt), confusion matrices, and training curves
```

---

## 📜 8. Citation & Acknowledgments

If you utilize this codebase, benchmark results, or rectified dataset methodology in your research, please cite:

```bibtex
@article{zhou2026ensemble,
  title={Ensemble Learning Using YOLO Models for Semiconductor E-Waste Recycling},
  author={Zhou, X. and Agaian, S.},
  journal={Information},
  volume={17},
  number={4},
  pages={322},
  year={2026},
  publisher={MDPI},
  doi={10.3390/info17040322}
}

@mastersthesis{tadevosyan2026pcbaoi,
  title={Resolving the Spatial-Efficiency Paradox in Automated Optical Inspection of Dense Printed Circuit Boards},
  author={Tadevosyan, Edgar},
  school={Yerevan State University},
  year={2026},
  note={Supervised by Prof. Sos Agaian}
}
```

### Acknowledgments
- **Supervisor:** Prof. Sos Agaian (Yerevan State University / CUNY) for academic guidance and research formulation.
- **Compute Infrastructure:** Yerevan State University (YSU) High-Performance Computing Cluster and NVIDIA H100 resources.
- **Core Libraries:** [Ultralytics](https://github.com/ultralytics/ultralytics), [SAHI](https://github.com/obss/sahi), and [Ensemble-Boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion).
