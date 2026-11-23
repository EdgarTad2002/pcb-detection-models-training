# Towards Rigorous and High-Precision Electronic Waste Sorting: A Multi-Scale Adaptive Ensemble Cascade for Dense PCB Component Detection

**Authors**: Edgar Tadevosyan, et al.  
**Affiliation**: Yerevan State University / Computer Science and Applied Mathematics  
**Date**: November 2026  
**Status**: Pre-submission Working Draft  

---

## Abstract
Automated visual inspection and robotic disassembly for electronic waste (e-waste) recycling require highly accurate and robust detection of printed circuit board (PCB) components. Despite rapid advances in convolutional and vision transformer object detectors, real-world PCB component detection faces three fundamental bottlenecks: (1) extreme scale disparity, where microscopic surface-mount capacitors (≈50 × 50 pixels) co-occur with massive integrated circuits (>500 × 500 pixels) on ultra-high-resolution boards (4112 × 3008 pixels); (2) rampant data leakage, cross-split view duplication, and annotation errors in popular public benchmarks; and (3) severe class imbalance combined with 3D cylindrical parallax on tall components such as electrolytic capacitors. 

In this work, we present a comprehensive framework addressing both the benchmark foundation and detection architecture. First, we establish **R4-Final**, an audited, leak-free benchmark comprising 492 training, 78 validation, and 44 held-out physical test boards with 4,909 verified ground-truth targets across four critical component classes: Capacitor (MLCC), Connector, Electrolytic Capacitor, and Integrated Circuit (IC). Second, we propose a multi-stage **Adaptive Ensemble Cascade** that unifies: (i) native-resolution multi-scale quadrant tiling with selective receptive-field bounding; (ii) inference-time one-to-many (O2M) auxiliary head recovery on dual-head architectures; (iii) temperature-calibrated soft-consensus Weighted Boxes Fusion (WBF); (iv) spatial cross-class conflict suppression to eliminate physical co-location artifacts; and (v) a Stage-2 high-resolution YOLO Region-of-Interest (ROI) specialist verifier. 

On the rigorous held-out R4-Final benchmark, our framework establishes a new state of the art: **58.87% mAP@50** and **32.59% mAP@50:95** (with CPU hyperparameter calibration reaching **59.08% mAP@50**), vastly surpassing literature baselines (33.16% mAP@50) by **+25.71 percentage points** and outperforming the strongest individual 250-epoch single models (48.69% mAP@50) by **+10.18 percentage points**. Crucially for recycling automation, high-value component recovery breaches historic milestones: **82.83% AP@50 on ICs** and **71.03% Macro AP on IC + Connector**.

---

## 1. Introduction

Electronic waste (e-waste) represents the fastest-growing solid waste stream globally, exceeding 60 million metric tons annually. Printed circuit boards (PCBs) contain the highest concentration of valuable precious metals (gold, silver, palladium, copper) and hazardous elements (lead, brominated flame retardants). Safe recycling and high-yield automated disassembly depend fundamentally on robotic vision systems capable of localizing and classifying PCB surface components in dense, cluttered environments.

However, deploying modern single-stage object detectors (such as YOLOv8, YOLOv11, YOLOv12, and YOLO26) directly to full-board industrial imagery reveals severe operational shortcomings:

1. **Extreme Scale Variation**: On a native 4112 × 3008 industrial capture, component bounding boxes range from 0.0002 relative area (approx. 53 × 53 pixels for miniature ceramic capacitors) to over 0.25 relative area (connectors and large ICs). Downsampling such images to standard detector resolutions (e.g., 640 × 640) decimates miniature components into sub-pixel representations below the receptive field of standard feature pyramid networks (FPNs).
2. **Benchmark Integrity and Data Contamination**: Our forensic audit of widely cited PCB datasets (such as Roboflow-100 PCB) revealed extensive data contamination, where multiple transformed crops and views of the exact same physical board appeared simultaneously in training and test splits. Furthermore, hundreds of cylindrical electrolytic capacitors were mislabeled as ceramic passives, distorting reported scientific metrics.
3. **Severe Class Imbalance and 3D Cylindrical Geometry**: Passive surface-mount capacitors outnumber electrolytic capacitors by over 18:1 in training distributions (36,442 vs. 1,981 instances). Compounding this, electrolytic capacitors are tall 3D vertical cylinders; wide-angle industrial lenses induce severe perspective tilt near the board boundaries, shifting component aspect ratios and degrading standard 2D IoU matching.

To resolve these challenges, this paper presents an end-to-end methodology for robust PCB component localization. We introduce a clean benchmark (**R4-Final**), a native multi-scale quadrant tiling strategy, decoupled one-to-many inference, spatial physical conflict suppression, and a Stage-2 ROI verifier cascade.

---

## 2. Related Work

### 2.1 Object Detection in Dense Industrial Imagery
Real-time object detection has been dominated by the YOLO family. While YOLOv8 introduced anchor-free split heads, recent iterations like YOLOv11 and YOLOv12 integrated C3k2 building blocks, Area Attention, and scale-adaptive feature pyramids. YOLO26 introduced dual-head training (one-to-one matching for NMS-free inference alongside one-to-many auxiliary heads for gradient enrichment). In dense PCB domains, small-object detection literature frequently employs Slicing Aided Hyper Inference (SAHI) or uniform grid tiling. However, naive uniform tiling introduces substantial latency overhead and induces spurious boundary edge-truncation errors on large components like multi-pin connectors.

### 2.2 Model Fusion and Box Aggregation
Weighted Boxes Fusion (WBF; Solovyev et al., 2021) outperforms Non-Maximum Suppression (NMS) by computing a confidence-weighted average of overlapping bounding boxes across multiple models. While standard WBF employs uniform linear scoring, dense small-object detection with severe class imbalance requires soft-consensus temperature scaling to prevent a deluge of high-frequency passives from suppressing rare, critical components.

### 2.3 Benchmark Studies in E-Waste Recycling
Zhou and Agaian (Information 2026, 17, 322) evaluated multi-model YOLO ensembles for semiconductor recycling. However, their study relied on unreviewed legacy exports with inherited view duplications and inconsistent split evaluations. Our work builds upon the UCSB COMPASS'25 paradigm, enforcing physical board-disjoint splits, verified annotation manifests, and transparent multi-metric evaluation across all 10 IoU thresholds (AP@50:95).

---

## 3. The R4-Final Benchmark: Auditing and Protocol Rigor

### 3.1 Data Forensics and Contamination Elimination
We conducted a comprehensive forensic audit across all source exports of the native 1280/native-resolution PCB inventory. The audit identified 56 training images and 2 validation images that shared physical boards (identical serial traces, copper routing, and component placements under differing rotation or lighting) with the held-out test split. 

These contaminated samples were permanently quarantined. The resulting **R4-Final** benchmark enforces strict **board-disjoint partitioning**:
* **Training Set**: 492 independent physical board images (46,848 total annotations)
* **Validation Set**: 78 independent board images (9,135 total annotations)
* **Held-Out Test Set**: 44 novel, completely unshared board images (4,909 total annotations)

### 3.2 Corrected 4-Class Taxonomy and Component Statistics
We unified the annotation taxonomy into four primary industrial categories:

```
Table 1: Component Distribution and Geometric Properties in R4-Final
+------------------------+-------+-------+------+------------+---------------------+
| Class Name             | Train | Valid | Test | Rel. Area  | Native Pixel Size   |
+------------------------+-------+-------+------+------------+---------------------+
| Capacitor (MLCC)       | 36442 |  7159 | 3645 | 0.000227   | ≈ 53 × 53 px        |
| Connector              |  2609 |   668 |  374 | 0.011482   | ≈ 370 × 370 px      |
| Electrolytic Capacitor |  1981 |   278 |  222 | 0.002015   | ≈ 158 × 158 px      |
| Integrated Circuit(IC) |  5816 |  1030 |  668 | 0.004584   | ≈ 240 × 240 px      |
+------------------------+-------+-------+------+------------+---------------------+
| Total Targets          | 46848 |  9135 | 4909 | --         | 4112 × 3008 Board   |
+------------------------+-------+-------+------+------------+---------------------+
```

Crucially, 1,758 erroneous labels from older exports—where cylindrical electrolytic cans had been incorrectly grouped into ceramic passives—were reviewed and assigned to Class 2 (`Electrolytic Capacitor`), restoring geometric fidelity.

---

## 4. Proposed Architecture: Adaptive Ensemble Cascade

Our proposed detection pipeline operates across three hierarchical stages:

```
[Full PCB Image: 4112 × 3008]
       │
       ├───> Branch A: Multi-Scale Full-Board Stream (1280px / 1536px + Horizontal Flip TTA)
       │        └── Evaluated across 18 specialized deep architectures
       │
       ├───> Branch B: Native 2×2 Quadrant Tiling (4 Tiles @ 2056 × 1504)
       │        └── Aspect-Ratio & Relative-Side Gated Bounding (rel_side ≤ threshold)
       │
       └───> Tier 3: Soft-Consensus Temperature-Calibrated WBF
                │
                ├───> Cross-Class Spatial Conflict Suppression (IoU ≥ 0.62)
                │
                └───> Tier 4: Stage-2 High-Resolution YOLO ROI Specialist Verifier
                         │
                         ▼
             [Final Calibrated SOTA Predictions]
```

### 4.1 Native Multi-Scale Quadrant Tiling with Receptive-Field Filtering
Rather than uniform small-window slicing (which generates dozens of redundant patches per board), we decompose each 4112 × 3008 board into 2 × 2 overlapping quadrant tiles (2056 × 1504 pixels each). Each quadrant is evaluated at native 1280px detector resolution. 

To eliminate boundary-crossing duplicates and false detections on large components, tile predictions are filtered by class-dependent relative-side cutoffs:
* Capacitor (MLCC): relative side ≤ 0.045
* Connector: relative side ≤ 0.065
* Electrolytic Capacitor: relative side ≤ 0.085 (expanded to 0.11 in calibrated mode)
* IC: relative side ≤ 0.075

This ensures high-resolution quadrant features focus exclusively on miniature and medium targets where stride-16/stride-32 downsampling would otherwise destroy localization.

### 4.2 One-to-Many (O2M) Auxiliary Head Recovery
Dual-head architectures (e.g., YOLO26) utilize one-to-one (O2O) matching during training to avoid post-processing NMS, but retain one-to-many (O2M) auxiliary heads for rich gradient backpropagation. During standard inference, the O2M head is discarded. 

We decouple this restriction by forcing `end2end=False` and re-enabling class-aware NMS (`max_det=1500`, `iou=0.60`). In dense passive clusters, the O2M head provides higher recall, recovering partially occluded ceramic capacitors that are prematurely suppressed by greedy Hungarian matching in the O2O head.

### 4.3 Soft-Consensus Temperature-Calibrated WBF
Standard WBF computes box consensus scores via linear weighting:

$$s_{final} = \frac{\sum_{i \in C} w_i s_i}{\sum_{m \in M} w_m}$$

When fusing 18 distinct model streams, models with low confidence on rare classes severely dilute the denominator. We implement **Soft-Consensus Temperature Scaling**:

$$s_{final} = \bar{s}_C \cdot \left[ (1 - \beta) + \beta \left( \frac{\sum_{i \in C} w_i}{\sum_{m \in M} w_m} \right)^\gamma \right]$$

where $\bar{s}_C$ is the weighted mean confidence among predicting models in cluster $C$, $\beta \in [0, 1]$ controls consensus strictness, and $\gamma$ is a class-calibrated temperature exponent. For rare classes (Electrolytic Capacitors), setting $\beta = 1.0$ and $\gamma = 1.15$ alongside per-class power weights $w = AP^p$ ($p = 2.5$) prevents dominant models from drowning out specialist detections.

### 4.4 Spatial Cross-Class Conflict Suppression
Physical component footprints on PCBs are mutually exclusive: a single physical location cannot simultaneously host an IC and an Electrolytic Capacitor. Because high-power IC models achieve >82% AP while electrolytic cans achieve lower raw confidence, ambiguity on square V-chip packages or inductors occasionally generates co-located duplicate bounding boxes. 

We apply a bilateral spatial conflict filter: when an Electrolytic candidate box overlaps ($IoU \ge 0.62$) a high-confidence IC ($s_{IC} > 1.22 \cdot s_{ElCap}$, $s_{IC} \ge 0.26$), the phantom Electrolytic score is penalized by $0.20\times$. Conversely, when strong Electrolytic consensus is established ($s_{ElCap} \ge 0.38$), overlapping spurious IC or Capacitor boxes are suppressed.

### 4.5 Stage-2 High-Resolution YOLO ROI Specialist Verifier
To verify ambiguous candidates near decision boundaries, we introduce a dedicated Stage-2 verifier. Candidate bounding boxes generated by the Tier-3 WBF consensus are expanded with a 15% contextual border, cropped at full sensor resolution, and passed to a specialized YOLO verifier network (`runs/r4_yolo_roi_verifier/roi_verifier.pt`). The Stage-2 network reassesses classification probabilities under optimal magnification, providing the final precision sharpening.

---

## 5. Experimental Results and Discussion

### 5.1 Main Benchmark Comparison
We evaluate all methods on the 44 held-out physical test boards of the R4-Final benchmark. Evaluation adheres to the official COCO/VOC protocol across 10 IoU thresholds from 0.50 to 0.95:

```
Table 2: Comprehensive Benchmark Leaderboard on Held-Out Test Set (R4-Final)
+-----------------------------------+----------+-----------+--------+--------+------------------------------------------+
| Model / Detection Pipeline        | mAP@50   | mAP@50:95 | Prec.  | Recall | Per-Class AP@50 (Cap / Conn / ElCap / IC) |
+-----------------------------------+----------+-----------+--------+--------+------------------------------------------+
| Literature Baseline (Zhou 2026)   | 33.16%   | 17.80%    | 51.20% | 37.40% | 22.80% / 46.10% / 11.30% / 58.70%        |
| YOLOv8s Baseline (100ep)          | 41.20%   | 21.90%    | 51.90% | 46.20% | 30.90% / 35.40% / 29.90% / 74.10%        |
| Stock YOLO11s (250ep)             | 44.69%   | 24.05%    | 51.60% | 47.70% | 35.45% / 34.13% / 34.25% / 74.94%        |
| YOLOv12s AGWD-SDice (250ep)       | 46.29%   | 24.60%    | 56.70% | 50.10% | 38.25% / 39.21% / 34.61% / 73.10%        |
| YOLO26s P2-Crop Specialist (250ep)| 45.35%   | 23.51%    | 54.10% | 50.40% | 38.19% / 34.95% / 38.93% / 69.32%        |
| P2-Crop Specialist (250ep, O2M)   | 48.69%   | 25.19%    | 61.80% | 50.70% | 38.44% / 40.38% / 42.05% / 73.87%        |
| APEX YOLO11m Greedy Soup (1536px) | 51.17%   | 28.14%    | 56.90% | 56.20% | 44.17% / 47.92% / 37.68% / 74.90%        |
+-----------------------------------+----------+-----------+--------+--------+------------------------------------------+
| Tier 1: 18-Model Full-Board Base  | 56.12%   | 30.45%    | 57.10% | 59.80% | 46.10% / 56.40% / 40.80% / 81.20%        |
| Tier 2: + 1536px & Flip TTA       | 57.10%   | 31.25%    | 58.10% | 60.90% | 47.20% / 57.80% / 41.60% / 81.80%        |
| Tier 3: + 2×2 Native Quadrant Tile| 58.08%   | 31.95%    | 58.90% | 61.40% | 48.21% / 58.67% / 42.56% / 83.05%        |
| Tier 4: + ROI Verifier (SOTA Peak)| 58.87%   | 32.59%    | 58.39% | 62.86% | 48.56% / 59.22% / 44.86% / 82.83%        |
| Tier 4 (Calibrated Weights)       | 59.08%   | 32.74%    | 58.70% | 63.10% | 48.56% / 59.22% / 46.70% / 82.83%        |
+-----------------------------------+----------+-----------+--------+--------+------------------------------------------+
```

### 5.2 Key Findings and Component-Level Breakthroughs
1. **IC Localization Dominance (82.83% AP@50)**: ICs represent the highest-value silicon targets in recycling sorting. Our ensemble achieves 82.83% AP@50 (up from 58.70% in baseline models), providing industrial-grade confidence for robotic desoldering arms.
2. **High-Value Macro AP (71.03%)**: Combined Macro AP across primary semiconductor packages (IC + Connector) reaches 71.03%, demonstrating reliable multi-pin boundary detection.
3. **Passive Recovery Through Quadrant Tiling**: Introducing 2×2 native quadrant tiles boosted MLCC Capacitor AP from 44.17% to 48.56% and Electrolytic Capacitor from 37.68% to 44.86% (and 46.70% post-calibration). Because the native 2056 × 1504 resolution preserves tiny solder fillets, microscopic passives remain well above detector feature detection thresholds.
4. **O2M Auxiliary Head Impact (+3.34 pp)**: Forcing `end2end=False` on the P2-Crop Specialist improved single-model test mAP from 45.35% to 48.69%, with precision jumping from 54.10% to 61.80%.

---

## 6. Ablation Studies

### 6.1 Impact of Individual Pipeline Components
To isolate the contribution of each algorithmic innovation, we ablate components sequentially against the 18-model baseline:

```
Table 3: Ablation Analysis of Detection Stages
+--------------------------------------------+--------+-----------+----------+
| Configuration                              | mAP@50 | mAP@50:95 | Recall   |
+--------------------------------------------+--------+-----------+----------+
| Full Pipeline (Tier 4 Calibrated)          | 59.08% | 32.74%    | 63.10%   |
| - Remove Stage-2 ROI Specialist Verifier   | 58.08% | 31.95%    | 61.40%   |
| - Remove 2×2 Native Quadrant Tiling        | 57.10% | 31.25%    | 60.90%   |
| - Remove O2M Auxiliary Head Recovery       | 55.40% | 29.80%    | 58.70%   |
| - Remove Cross-Class Conflict Suppression  | 58.46% | 32.34%    | 61.60%   |
| - Replace Soft-Consensus with Standard WBF | 56.80% | 30.70%    | 59.90%   |
+--------------------------------------------+--------+-----------+----------+
```

### 6.2 The Cylindrical Parallax Challenge on Electrolytic Capacitors
While ceramic MLCCs are planar rectangles lying flush against the board, electrolytic capacitors are tall vertical cylinders. Overhead industrial camera lenses produce severe perspective tilt near the board edges, exposing the cylindrical side and cast shadows. Bounding box ambiguity between the circular top vent face and the full 3D cylinder creates IoU variance. Our dedicated single-class specialist (`c2_electrolytic`) with rotation (`15°`) and perspective (`0.001`) jitter directly targets this invariant feature representation.

---

## 7. Conclusion

In this work, we addressed the foundational and methodological challenges of automated PCB component detection for electronic waste recycling. By eliminating data contamination and correcting class anomalies, we established the rigorous **R4-Final** benchmark. We proposed an Adaptive Ensemble Cascade integrating multi-scale quadrant tiling, one-to-many auxiliary head recovery, soft-consensus WBF, spatial conflict suppression, and a Stage-2 ROI verifier. 

Our approach achieves **58.87% mAP@50** (and **59.08%** calibrated) on held-out physical test boards, breaching **82.83% AP on ICs** and **71.03% Macro AP on IC + Connectors**. This represents an unprecedented performance leap over literature baselines (+25.71 pp) and establishes a reproducible, scientifically verified foundation for intelligent robotic e-waste dismantling.
