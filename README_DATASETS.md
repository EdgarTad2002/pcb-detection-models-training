# PCB Benchmark Dataset Catalog & Reference Manual

> **Permanent Documentation for all Active Datasets on the YSU-HPC Cluster (`cluster.ysu.am`)**  
> **Directory Path on Cluster:** `/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets`  
> **Total Active Datasets:** 8 verified datasets

This document provides a permanent, comprehensive reference for each of the 8 remaining datasets on the cluster: how each dataset was created, its input resolution, which models were trained on it, and its exact role in our research paper.

---

## 📌 Master Overview Table

| # | Dataset Folder Name | Resolution | Annotation State | Preprocessing / Augmentation | Models Trained on It | Role in Research Paper |
|:-:|:---|:---:|:---:|:---|:---|:---|
| **1** | `pcb-unified-4class` | 640px | Original (With Errors) | Clean 4-Class ID mapping | Baseline YOLOv5s–v26s (640px) | **Core Baseline from Paper (640px)** |
| **2** | `pcb-native-res-unified-4class` | 1280px | Original (With Errors) | Native high-res camera captures | `superyolo26s`, `physics_spectral`, `native_res_1280` | **Core Baseline from Paper (1280px)** |
| **3** | `pcb-filtered-yolov8` | 640px | Original (With Errors) | Direct Roboflow export (23-class) | Legacy initial YOLOv8s baselines | **Direct Paper Reproduction Archive** |
| **4** | `pcb-unified-4class-cleaned` | 1280px | **Audited Clean** (103 errors fixed) | Full dataset audit & label correction | Cleaned baselines & controls (`sweep_base_s0..s2`) | **The Gold-Standard Ground Truth** |
| **5** | `pcb-native-res-rectified-cappaste-1280` | 1280px | Audited + Aug | Micro-Capacitor Copy-Paste ($p=0.7$) | `rectified_yolov26s_unified_1280` (62.90%) | **#2 Single-Model Champion** |
| **6** | `pcb-retinex-cappaste-1280` | 1280px | Audited + Aug | Copy-Paste + Bilateral Retinex | `yolov26s_ultimate_retinex_copypaste_1280` | **#1 SOTA Ensemble Backbone (WBF Top-4)** |
| **7** | `pcb-bilat-a0.30-b0.10` | 1280px | Audited Clean | Bilateral Retinex ($\alpha=0.30, \beta=0.10$) | Multi-seed sweep (`s0`, `s1`, `s2`, `yolo26x`) | **Multi-Seed Variance Study (+7.8% Recall)** |
| **8** | `pcb-unified-4class-rgb-variational-retinex` | 1280px | Audited Clean | Fu et al. (CVPR 2016) ADMM Variational | `rectified_yolov26s_rgb_variational_retinex_1280` | **Variational Physics Ablation** |

---

## 🗂️ Detailed Dataset Profiles

---

### 1. `pcb-unified-4class` (The 640px Original Paper Baseline)
* **Cluster Path:** `datasets/pcb-unified-4class`
* **Resolution:** $640 \times 640\text{px}$
* **Classes (nc=4):** `0: Capacitor`, `1: Connector`, `2: Electrolytic Capacitor`, `3: IC`
* **How It Was Created:**
  Created by mapping the 23-class dataset from the reference paper (*Zhou & Agaian, MDPI Information 2026*) into a unified 4-class taxonomy using `tools/rectify_dataset.py`.
* **Important Annotation Detail:**
  **Contains the original dataset errors / missing labels!** It has not been cleaned. This is intentional.
* **Models Trained on It:**
  * `rectified_yolov5s` (56.83% mAP50)
  * `rectified_yolov8s` (56.16% mAP50)
  * `rectified_yolov9s` (56.58% mAP50)
  * `rectified_yolov10s` (55.33% mAP50)
  * `rectified_yolov11s` (56.89% mAP50)
  * `rectified_yolov12s` (55.03% mAP50)
  * `rectified_yolov26s` (58.12% mAP50)
* **Role in the Paper:**
  The central standard-resolution baseline. Provides a fair comparison against prior literature showing how different YOLO generations perform on the exact dataset from the Zhou & Agaian paper.

---

### 2. `pcb-native-res-unified-4class` (The 1280px Original Paper Baseline)
* **Cluster Path:** `datasets/pcb-native-res-unified-4class`
* **Resolution:** Full Native Resolution ($1280 \times 1280\text{px}$ inference crops / high-res camera captures)
* **Classes (nc=4):** `0: Capacitor`, `1: Connector`, `2: Electrolytic Capacitor`, `3: IC`
* **How It Was Created:**
  Extracted from the raw 4K uncompressed PCB images without downsampling to 640px, mapped to the 4-class taxonomy.
* **Important Annotation Detail:**
  **Contains the original paper errors** (missing capacitor boxes, label noise), but preserves high-frequency spatial details.
* **Models Trained on It:**
  * **`superyolo26s_rectified_1280`** (**0.6304 mAP50**, **57.9 FPS**, 17.3 ms — Real-Time Edge Champion!)
  * `yolov26s_native_res_rectified_1280` (59.81% mAP50 baseline)
  * `rectified_yolov26s_native_res_physics_spectral_a0.0_1280` (60.47% mAP50)
  * `yolov26s_retinex_loss_reweight_1280` (59.86% mAP50)
* **Role in the Paper:**
  Proves the **Resolution Scaling Discovery**: demonstrates that increasing input resolution from 640px to 1280px on the *same uncleaned data* boosts Capacitor AP from $20.66\% \rightarrow 38.02\%$ (+17.4% absolute gain) purely by resolving spatial undersampling.

---

### 3. `pcb-filtered-yolov8` (The Direct Roboflow Paper Archive)
* **Cluster Path:** `datasets/pcb-filtered-yolov8`
* **Resolution:** $640 \times 640\text{px}$
* **Classes (nc=23):** Legacy 23-class Roboflow taxonomy (classes 2: Capacitor, 4: Connector, 7: Electrolytic Capacitor, 9: IC).
* **How It Was Created:**
  Direct, unedited raw download from the Roboflow dataset URL cited in Zhou & Agaian (2026).
* **Role in the Paper:**
  Archival baseline proving reproducibility of the original 23-class paper results before 4-class unification.

---

### 4. `pcb-unified-4class-cleaned` (The Audited Ground-Truth Benchmark)
* **Cluster Path:** `datasets/pcb-unified-4class-cleaned`
* **Resolution:** High-Resolution Native ($1280\text{px}$)
* **Classes (nc=4):** Cleaned & verified annotations (4,909 total bounding boxes across train/val/test).
* **How It Was Created:**
  Generated by `tools/export_cleaned_dataset.py` following our comprehensive multi-model discrepancy audit (`tools/audit_missing_labels.py`). Restored **103 missing physical capacitors**, corrected mislabeled IC packages, and eliminated bounding box boundary shifts.
* **Models Trained on It:**
  * `rectified_yolov26s_cleaned_1280` (58.60% mAP50)
  * `rectified_yolov26x_cleaned_1280` (59.13% mAP50)
  * Matched baseline controls across seeds: `sweep_base_s0_hv0.4_yolo26s`, `sweep_base_s1`, `sweep_base_s2`
* **Role in the Paper:**
  The **Dataset Rectification Contribution**: provides the cleaned, noise-free benchmark. Used in the paper's ablation section to contrast models trained on noisy labels vs. verified ground truth.

---

### 5. `pcb-native-res-rectified-cappaste-1280` (The Copy-Paste Single-Model Champion)
* **Cluster Path:** `datasets/pcb-native-res-rectified-cappaste-1280`
* **Resolution:** $1280 \times 1280\text{px}$
* **How It Was Created:**
  Built via `sbatch/train_rectified_yolov26s_unified_1280.sh` using `bbox_copy_paste.py`. Crops isolated chip capacitors into an offline component bank, then synthetically pastes them onto varied PCB substrates during training with paste probability $p=0.70$.
* **Models Trained on It:**
  * **`rectified_yolov26s_unified_1280`** (**0.6290 mAP50**, **0.6837 F1**, Capacitor AP: 37.78%) — *Our #2 best single model overall*.
* **Role in the Paper:**
  Proves that targeted micro-component data augmentation overcomes extreme class imbalance (capacitors making up 66% of components but suffering from background confusion).

---

### 6. `pcb-retinex-cappaste-1280` (The Quad-Ensemble Foundation)
* **Cluster Path:** `datasets/pcb-retinex-cappaste-1280`
* **Resolution:** $1280 \times 1280\text{px}$
* **How It Was Created:**
  Combines capacitor copy-paste augmentation with Bilateral Retinex material reflectance decomposition ($I_{\text{fused}} = 0.70 I + 0.30 R$).
* **Models Trained on It:**
  * `yolov26s_ultimate_retinex_copypaste_1280` (**0.6155 / 0.6283 mAP50**, Capacitor AP: 38.58%)
  * `yolov26s_retinex_stem_copypaste_1280` (59.33% mAP50)
  * `yolov26s_p2_warmstart_retinex_copypaste_1280` (60.31% mAP50)
* **Role in the Paper:**
  The core training dataset for the **#1 All-Time SOTA Ensemble** ([`ensemble_wbf_top4_quad_1280`](file:///home/edgar/pcb-detection-models-training/results/ensemble_wbf_top4_quad_1280.json), **65.60% mAP50**, 41.89% Capacitor AP).

---

### 7. `pcb-bilat-a0.30-b0.10` (The Multi-Seed Variance Benchmark)
* **Cluster Path:** `datasets/pcb-bilat-a0.30-b0.10`
* **Resolution:** $1280 \times 1280\text{px}$
* **How It Was Created:**
  Generated by `tools/enhance_pcb_retinex.py` from `pcb-unified-4class-cleaned`. Decomposes illumination $L$ via bilateral filter, extracts reflectance $R = I / L$, and blends with $\alpha=0.30, \beta=0.10$.
* **Models Trained on It:**
  * `sweep_bilat_a0.30_b0.10_s0_hv0.4_yolo26s` (59.97% mAP50)
  * `sweep_bilat_a0.30_b0.10_s1_hv0.4_yolo26s` (59.94% mAP50)
  * `sweep_bilat_a0.30_b0.10_s2_hv0.4_yolo26s` (57.89% mAP50)
  * `sweep_bilat_a0.30_b0.10_s0_hv0.4_yolo26x` (59.75% mAP50)
  * **`ensemble_bilat_3seed_wbf_1280`** (**0.6191 mAP50**, **0.6336 Recall** — *+7.8% recall gain*)
* **Role in the Paper:**
  The centerpiece of our **Multi-Seed Statistical Significance Study**. Shows that single-seed runs vary by $\pm 0.017$, and multi-seed ensembling over identical initializations converts noise into a $+7.8\%$ recall boost.

---

### 8. `pcb-unified-4class-rgb-variational-retinex` (The Variational Retinex Ablation)
* **Cluster Path:** `datasets/pcb-unified-4class-rgb-variational-retinex`
* **Resolution:** $1280 \times 1280\text{px}$
* **How It Was Created:**
  Generated by `tools/build_retinex_cleaned_dataset.py` using Fu et al. (CVPR 2016) ADMM weighted variational optimization independently across each color channel ($R, G, B$).
* **Models Trained on It:**
  * `rectified_yolov26s_rgb_variational_retinex_1280` (55.91% mAP50)
* **Role in the Paper:**
  Serves as the **Physical Model Ablation**: compares classical Bilateral filtering ($0.17\text{s/image}$) against heavy numerical optimization ($3.3\text{s/image}$), demonstrating that optimization-based illumination decomposition does not justify its $20\times$ computation cost.

---

## 🎯 Quick Decision Guide (Which Dataset Should I Use?)

* **If your professor says: "Run a model on the original paper dataset to compare with Zhou & Agaian":**
  👉 Use `datasets/pcb-unified-4class/data.yaml` (640px) or `datasets/pcb-native-res-unified-4class/data.yaml` (1280px).
* **If you want to train a clean, fair baseline without any fancy augmentations:**
  👉 Use `datasets/pcb-unified-4class-cleaned/data.yaml`.
* **If you are training the new Adaptive Gamma model (Zero Canny):**
  👉 Use `tools/enhance_pcb_adaptive_gamma.py` to build `datasets/pcb-adaptive-gamma-1280` from `pcb-unified-4class-cleaned`.
* **If you want to reproduce the #2 single-model champion (62.90% mAP):**
  👉 Use `datasets/pcb-native-res-rectified-cappaste-1280/data.yaml`.
* **If you want to evaluate the #1 WBF quad-ensemble (65.60% mAP):**
  👉 Use `datasets/pcb-retinex-cappaste-1280/data.yaml`.
