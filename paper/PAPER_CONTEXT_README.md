# PCB Detection Paper Writing Context README

This document summarizes the finalized datasets, models, verified NVIDIA H100 benchmark results, mathematical formulations, editorial/supervisor rules, and key repository paths for writing the manuscript.

> **Important Constraint:** Do **NOT** modify existing `.tex` files under `paper/mdpi_manuscript/` until explicitly instructed by the user.

---

## 1. Editorial & Supervisor Rules

- **Target Manuscript Scope:**
  - Focus **strictly** on `D_clean` (`1280 px`, `62.18%–69.16% mAP₅₀`) and `D_orig` (`640 px`, 3-seed comparison with Zhou & Agaian's papers).
  - Do **NOT** include `D_V4` (`r4`) or external datasets (FICS-PCB, FPIC, PCB-Vision, PCB-DSLR).
- **Writing Style & Formatting:**
  - Simple, student-friendly English in first-person active voice.
  - Title must start with **"Electronic Waste Component Detection..."** (or equivalent phrasing starting with *Electronic Waste Component Detection*).
  - Section order: **Introduction → Methodology → Dataset → Results → Conclusions**.
  - Expand `P` and `R` to **Precision** and **Recall** in all tables.
  - **Author Contributions:** Edgar (`E.T.`) leads Author Contributions.

---

## 2. Datasets

We evaluate on two curated four-class datasets (`Capacitor`, `Connector`, `Electrolytic Capacitor`, `IC`) derived from the raw `pcb-filtered-yolov8` Roboflow release.

### 2.1 `Capacitor` vs. `Capacitor Jumper` Reconciliation (Raw `pcb-filtered-yolov8`)
- `192` distinct board stems contain `Capacitor` in one copy and `Capacitor Jumper` in another.
- `256` images use `Capacitor Jumper` only.
- Merging `32,564` `Capacitor` boxes + `19,034` `Capacitor Jumper` boxes yields `51,598` total boxes before cleaning/deduplication.

### 2.2 `D_clean` (`pcb-native-res-cleaned-1280`, 1280 px)
| Split | Images | Capacitor | Connector | Electrolytic Capacitor | IC |
|---|---:|---:|---:|---:|---:|
| Train | 548 | 40,484 | 3,451 | 672 | 6,027 |
| Val | 80 | 7,259 | 714 | 166 | 1,030 |
| Test | 44 | 3,849 | 374 | 18 | 664 |
| **Total** | **672** | **51,592** | **4,539** | **856** | **7,721** |

### 2.3 `D_orig` (`info17-original-fourclass-v1`, 640 px, 3-seed protocol)
| Split | Images | Capacitor | Connector | Electrolytic Capacitor | IC |
|---|---:|---:|---:|---:|---:|
| Train | 543 | 27,014 | 3,470 | 672 | 5,998 |
| Val | 77 | 3,630 | 714 | 166 | 1,032 |
| Test | 41 | 1,920 | 374 | 18 | 574 |
| **Total** | **661** | **32,564** | **4,558** | **856** | **7,604** |

---

## 3. Finalized Models & Key Methodology

### 3.1 Model Roster (`D_clean` 1280 px Champions)
- **`M₁`: `YOLOv12s` (`D_CP`)** — Trained with copy-paste augmentation.
- **`M₂`: `Crop YOLO26s` (`D_mix`)** — Trained with crop/mixed-scale strategy.
- **`M₃`: `YOLO26s AGWD+SDice` (`D_clean`)** — Trained with Adaptive Gradient-Weighted Detection (AGWD) and Soft-Dice (`SDice`) loss.
- **`M₄`: `PhysOT-YOLO26s` (`D_CP`)** — YOLO26s with the in-network **Illumination-Guided Retinex (`IGR`)** stem at Layer 0 (`Conv₀`) and Physics-informed Optimal Transport (`PhysOT`) / copy-paste.
- **Ensembles via Weighted Boxes Fusion (WBF):**
  - `Two-Model WBF (M₁ + M₂)`
  - `Four-Model WBF (M₁–M₄)`
  - `Five-Model WBF (M₁–M₅)†` (uses offline cached `M₅` predictions)

### 3.2 Image Enhancement & Retinex / `IGR` Formulations
- **Retinex Decomposition:** `I(x, y) = L(x, y) ⊙ R(x, y)` where `L` is illumination and `R` is reflectance.
  - Initial illumination estimate: `L₀(x, y) = max_{c ∈ {R, G, B}} I_c(x, y)`
  - Edge-aware weight (Sobel gradient `∇L₀`): `w_edge = exp(-|∇L₀|² / (2σ_r²))`
  - Multi-scale Gaussian smoothing (`k ∈ {5, 15, 31}`): `L_k = (G_k * (w_edge ⊙ L₀)) / (G_k * w_edge + ε_d)`, combined via learnable weights `π₁, π₂, π₃`: `L = π₁·L₅ + π₂·L₁₅ + π₃·L₃₁`
  - Log-reflectance: `log R = log(I + ε) - log(L + ε)` (`ε = 10⁻³`)
- **Offline Bilateral Retinex (`retinex_stem.py`):**
  - `I_enh = (1 - α)I + αR + βE`, with `α = 0.30`, `β = 0.10`, and `E = ||∇R||`.
  - Achieved `59.97%`, `59.94%`, `57.89%` across three seeds (`61.91%` with 3-seed WBF), outperforming ADMM variational Retinex (`55.91%`).
- **Online In-Network `IGR` Block in `M₄` (`igr_stem.py`):**
  - Normalized reflectance: `R̃ = σ(a ⊙ log R + b)`
  - Shadow & specular glare masks:
    `M_shadow = L² / (L² + τ_shadow²)`, `M_spec = ReLU(L - τ_glare) ⊙ ||∇R̃||`
  - Spatially varying adaptive gamma (`γ₀ = 1.0`, `Δγ = 0.5`):
    `γ(x, y) = γ₀ + Δγ · tanh(g_θ([L, L - L̄, ||∇L₀||]))`, `I_γ = I^{γ(x, y)}`
  - FiLM-modulated fusion at Layer 0:
    `F_out = Conv₀(I) + (1 + tanh(α_L)) ⊙ F_R + β_L`
  - Zero-initialized projections (`γ = 1, F_R = 0, α_L = 0, β_L = 0`) ensure `M₄` starts identically to pretrained YOLO26s at step 0 and is trained end-to-end with `L_total`.

---

## 4. Verified Results (Accuracy & NVIDIA H100 FP32 Batch-1 Speed)

### 4.1 `D_clean` (`1280 px`) Main Results (`results/champion_audit_v1_20261030/summary.csv` & `04_results.tex`)

| Model | mAP₅₀ (%) | mAP₅₀:₉₅ (%) | Precision (%) | Recall (%) | F1 (%) | Cap (%) | Conn (%) | ElCap (%) | IC (%) | Latency (ms) | FPS | Params Active (Ckpt) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M₁: YOLOv12s` (`D_CP`) | 66.16 | 45.45 | 61.70 | 49.42 | 54.88 | 43.43 | 77.88 | 74.50 | 68.85 | 14.33 | 69.79 | 9.23M (9.25M) |
| `M₂: Crop YOLO26s` (`D_mix`) | 66.56 | 45.48 | 54.61 | 51.23 | 52.87 | 39.67 | 70.94 | 83.33 | 72.30 | 11.11 | 89.99 | 9.47M (9.95M) |
| `M₃: YOLO26s AGWD+SDice` (`D_clean`) | 64.99 | 43.93 | 53.42 | 54.80 | 54.10 | 44.12 | 68.63 | 74.51 | 72.69 | 11.13 | 89.85 | 9.47M (9.95M) |
| `M₄: PhysOT-YOLO26s` (`D_CP`) | 62.18 | 42.87 | 59.87 | 49.48 | 54.18 | 43.68 | 65.81 | 71.02 | 68.21 | 30.75 | 32.52 | 10.43M (10.92M) |
| `Two-Model WBF (M₁ + M₂)` | 68.49 | 47.51 | 63.27 | 51.31 | 56.67 | 45.10 | 76.95 | 78.19 | 73.74 | 47.94 | 20.86 | 18.70M (19.21M) |
| `Four-Model WBF (M₁–M₄)` | 68.74 | 47.66 | 65.20 | 51.82 | 57.75 | 47.04 | 78.85 | 73.31 | 75.77 | 115.97 | 8.62 | 38.60M (40.08M) |
| `Five-Model WBF (M₁–M₅)†` | 69.16 | 48.05 | 67.47 | 49.34 | 56.99 | 47.41 | 80.51 | 71.67 | 77.06 | — | — | — |

### 4.2 Additional `1280 px` H100 Control / Comparison Speeds (Cluster Job `397722`)
- **`YOLOv10s Control (1280 px, Table 9)`:** `25.61 ms`, `39.04 FPS`, `7.22M` active (`8.07M` checkpoint).
- **`YOLO26x (1280 px, Table 10)`:** `71.61 ms`, `13.97 FPS`, `58.16M` active (`61.33M` checkpoint).

### 4.3 `D_orig` (`640 px`, 3-Seed Benchmark) (`results/info17_speed_v1_20261031/benchmark.json` & `04_results.tex` Table 7)

| Model | mAP₅₀ (%) | Latency (ms) | FPS | Params |
|---|---:|---:|---:|---:|
| `YOLOv8s Baseline` | 52.34 ± 1.38 | 6.27 ± 0.12 | 159.50 | 11.14M |
| `YOLOv12s Baseline` | 53.61 ± 2.39 | 9.92 ± 0.07 | 100.79 | 9.25M |
| `YOLO26s Baseline` | 49.94 ± 1.85 | 7.77 ± 0.22 | 128.69 | 9.95M |
| `YOLO26s AGWD + SDice` | 53.23 ± 1.40 | 7.77 ± 0.18 | 128.77 | 9.95M |
| `PhysOT-YOLO26s` | 52.27 ± 1.79 | 11.78 ± 0.09 | 84.91 | 10.92M |
| `Crop + AGWD/SDice (YOLO26s)` | 55.08 ± 0.89 | 7.70 ± 0.03 | 129.80 | 9.95M |

---

## 5. Key Repository File Paths

- **Manuscript Directory:** `/home/edgar/pcb-detection-models-training/paper/mdpi_manuscript/`
  - Results section: `/home/edgar/pcb-detection-models-training/paper/mdpi_manuscript/sections/04_results.tex`
- **Benchmark Results:**
  - `1280 px` Champion Audit: `/home/edgar/pcb-detection-models-training/results/champion_audit_v1_20261030/summary.csv`
  - `640 px` 3-Seed Speed Benchmark: `/home/edgar/pcb-detection-models-training/results/info17_speed_v1_20261031/benchmark.json`
- **Enhancement & Stem Implementations:**
  - Offline Bilateral Retinex: `/home/edgar/pcb-detection-models-training/retinex_stem.py`
  - In-Network `IGR` Block: `/home/edgar/pcb-detection-models-training/igr_stem.py`
