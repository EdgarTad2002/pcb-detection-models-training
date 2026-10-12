# Retinex preprocessing for PCB detection: what we tried and what we know

Two offline "illumination-normalising" preprocessors, each blending the original image with an estimated
reflectance map, were compared against plain cleaned RGB for YOLO26s at 1280 px.

| Method | Doc | One-line description |
| --- | --- | --- |
| 1. Bilateral fusion | [README_BILATERAL_RETINEX_FUSION.md](README_BILATERAL_RETINEX_FUSION.md) | Bilateral-filter illumination, `0.7·I + 0.3·R + 0.1·Canny(R)` |
| 2. Variational fusion | [README_VARIATIONAL_RETINEX_FUSION.md](README_VARIATIONAL_RETINEX_FUSION.md) | Fu et al. CVPR 2016 solved per RGB channel, `0.7·I + 0.3·R` |

## Results so far (single seed, 42 evaluated test images)

| Training data | mAP50 | mAP50-95 | Recall | Capacitor AP50 |
| --- | --- | --- | --- | --- |
| Cleaned RGB (baseline) | 0.5860 | 0.4259 | 0.5830 | 0.2919 |
| Method 1: bilateral | 0.5864 | 0.4231 | 0.6008 | 0.2812 |
| Method 2: variational | 0.5591 | 0.4076 | 0.5716 | 0.2673 |

## What this does and does not show

- Neither method has beaten the baseline. Method 1 ties it; Method 2 is lower.
- With one seed and a 42-image test set, differences under about 1 point cannot be distinguished from noise.
- The hypothesis that edge injection helps small classes is **not** supported by these numbers.
- Earlier explanations for the ranking (e.g. the L1 prior flattening small parts) were guesses, not findings.
- On this evenly lit board the visual effect is modest; Method 1 mainly lowers the contrast of dark packages.

## Follow-up experiments (prepared, not yet run)

`sbatch/submit_retinex_study.sh` defines tiers that address the open questions:

- **core**: baseline vs bilateral over 3 seeds each (is there any real effect?), plus `hsv_v=0.1`
  (default `0.4` re-randomises brightness after normalisation).
- **grid**: `β=0` (do edges matter?), stronger/weaker `α`.
- **extras**: capacitor copy-paste, a learnable in-network Retinex-Edge stem, YOLO26x, and a P2 head.

All follow-ups use `--max-det 1000` (dense boards can exceed Ultralytics' default of 300 detections).
Aggregate with `python tools/summarize_study.py`, which reports mean ± std across seeds and states
whether a gain exceeds run-to-run noise.

## Documented pitfalls found along the way

- A copied `data.yaml` kept `path:` pointing at the source dataset, so one run silently trained on the
  wrong images. Fixed in `tools/enhance_pcb_retinex.py` (regression test in `tests/`).
- `ensemble_eval.py` still uses legacy 23-class IDs; use `tools/eval_super_ensemble.py` for 4-class data.
