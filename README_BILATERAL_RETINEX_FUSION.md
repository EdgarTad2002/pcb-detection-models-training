# Method 1: Bilateral Retinex Residual Fusion

Offline preprocessing that blends each image with an estimate of its illumination-free reflectance
and a Canny edge map. Implemented in [`tools/enhance_pcb_retinex.py`](tools/enhance_pcb_retinex.py).

## Idea

Retinex models an image as `I(x,y) = R(x,y) · L(x,y)`: `L` is smooth illumination, `R` is surface
reflectance. If we can estimate `L`, we can divide it out.

## Algorithm (exactly what the code does)

1. `L_init = max(R, G, B)` per pixel.
2. `L = BilateralFilter(L_init, d=5, σ_color=75, σ_space=75)` (edge-preserving smoothing).
3. `R_c = clip(I_c / (L + 1e-4), 0, 1)` for each colour channel.
4. Canny edges on a lightly bilateral-smoothed grayscale of `R` (Otsu-derived thresholds).
5. Output `I_out = (1-α)·I + α·R + β·Edge`, defaults `α = 0.30`, `β = 0.10`.

Labels are copied unchanged, so image geometry must (and does) stay identical.

## Example (Arty_Top board, downscaled to 1280 px)

![Method 1 panel](results/retinex_fusion_presentation/method1_bilateral_retinex_panel.png)

![Zoom comparison](results/retinex_fusion_presentation/microscopic_zoom_comparison.png)

What the crops actually show: Method 1 raises the brightness of large dark components (the BGA
package turns mid-gray), i.e. it **reduces** their contrast against the board. This board is evenly
lit, so there are no deep shadows to "recover". Whether this helps a detector is an empirical
question, answered below.

## Measured results (YOLO26s, 1280 px, 100 epochs, test split)

Single seed each. The test split has 44 images, 2 of which Ultralytics drops as corrupt
(labels mixing polygon and box rows), so 42 are evaluated.

| Training data | mAP50 | mAP50-95 | Recall | Capacitor AP50 | Connector AP50 |
| --- | --- | --- | --- | --- | --- |
| Cleaned RGB (baseline) | 0.5860 | 0.4259 | 0.5830 | 0.2919 | 0.6522 |
| **Bilateral fusion (this method)** | 0.5864 | 0.4231 | 0.6008 | 0.2812 | 0.6971 |
| Variational fusion (Method 2) | 0.5591 | 0.4076 | 0.5716 | 0.2673 | 0.6668 |

**Reading:** Method 1 is indistinguishable from the baseline on mAP50 (+0.0004) and slightly lower on
mAP50-95 and Capacitor AP. The recall and Connector differences are within what a single run on 42
images can produce. No claim that this method improves small-object detection is supported yet. A
multi-seed study is set up in [`sbatch/submit_retinex_study.sh`](sbatch/submit_retinex_study.sh) to decide this.

## Measured cost

- 0.17 s per 1280-px image on one CPU core (measured in `tools/demo_retinex_fusions.py`).
- Building the 672-image dataset on the cluster took 11.5 s with 8 workers (from the Slurm log).

## Reproduce

```bash
python tools/enhance_pcb_retinex.py --source datasets/pcb-unified-4class-cleaned \
    --dest datasets/pcb-bilat-a0.30-b0.10 --alpha 0.30 --beta 0.10 --workers 8
bash tools/remote_train.sh train_bilateral_sweep.sh bilateral 0.30 0.10 0 0.4 yolo26s.pt
```
