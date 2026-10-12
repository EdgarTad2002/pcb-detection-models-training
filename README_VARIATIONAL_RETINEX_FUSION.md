# Method 2: Weighted-Variational Retinex (Fu et al., CVPR 2016), per-channel fusion

Implementation of the weighted variational Retinex model of Fu et al. (CVPR 2016) in PyTorch
([`tools/weighted_variational_retinex.py`](tools/weighted_variational_retinex.py)), applied to all
three RGB channels and blended back with the original image.

This README describes what the code does. For the exact energy functional and derivation, refer to the
paper itself; formulas are not restated here from memory.

## Algorithm as implemented

Work in the log domain (`s = log S`, `r = log R`, `l = log L`) and alternate (ADMM-style):

1. **d-update:** soft-threshold `d = shrink(R·∇r + b, 1/(2λ))`.
2. **r-update:** closed-form FFT solve (periodic boundaries), then clamp `r ≤ 0`.
3. **l-update:** closed-form FFT solve, then clamp `l ≥ s`.
4. Multiplier update `b ← b + R·∇r − d`.
5. Stop when `‖r_k − r_{k−1}‖/‖r_{k−1}‖` and the same for `l` are both ≤ 1e-3, or at `max_iter`.

Defaults: `c1=0.01, c2=0.1, λ=1, γ=2.2, max_iter=15`.

**Documented approximation:** the spatially varying weights `R^{k-1}`, `L^{k-1}` in the FFT denominators
are replaced by their per-image scalar means (`use_mean_weights=True`). A spatially varying weight does
not diagonalise under the FFT; this keeps each solve O(N log N). This is an interpretation, not
something the paper states.

Fusion step (`--mode rgb_fusion`): each channel is decomposed independently (the 3 channels are run as
a batch of 3), then `I_out = (1-α)·I + α·R`, `α = 0.30`. The other mode (`--mode v_channel`) processes
only the HSV V channel and outputs `R · L^{1/γ}` with `L` normalised to (0, 1].

## Correctness checks (unit tests in `tests/`)

- FFT derivative kernels equal the forward-difference operators used in the spatial code
  (an earlier version had a backward-difference kernel; fixed and tested).
- Adjoint identity `⟨∇x, y⟩ = ⟨x, ∇ᵀy⟩` holds.
- Batched `(B,H,W)` results match per-image results.

## Example (Arty_Top, downscaled to 1280 px)

![Method 2 panel](results/retinex_fusion_presentation/method2_variational_retinex_panel.png)

![Zoom comparison](results/retinex_fusion_presentation/microscopic_zoom_comparison.png)

In the crops the Method 2 output is visually very close to the original (more so than Method 1).

## Measured results (YOLO26s, 1280 px, 100 epochs, test split, single seed)

| Training data | mAP50 | mAP50-95 | Recall | Capacitor AP50 |
| --- | --- | --- | --- | --- |
| Cleaned RGB (baseline) | 0.5860 | 0.4259 | 0.5830 | 0.2919 |
| Variational fusion (this method) | 0.5591 | 0.4076 | 0.5716 | 0.2673 |

It scored lower than the baseline in this run. I do not have a tested explanation: the idea that the
L1 prior flattens tiny components is a hypothesis only. Seed-to-seed variance is not yet measured, so
even this gap is not established. See `sbatch/submit_retinex_study.sh`.

## Convergence caveat

On the Arty_Top V channel the stopping rule was **not** met within 15 iterations
(final `ε_r = 0.038`, `ε_l = 0.0030`), so outputs reflect the iteration cap, not full convergence.

## Measured cost

- CPU, 3 channels at 1280 px: 3.3 s per image (measured in `tools/demo_retinex_fusions.py`).
- H100, whole dataset build (672 images, batch 4, including disk I/O): about 46 images/s (Slurm log).

## Reproduce

```bash
python tools/build_retinex_cleaned_dataset.py --src-dir datasets/pcb-unified-4class-cleaned \
    --dst-dir datasets/pcb-unified-4class-rgb-variational-retinex --device cuda \
    --mode rgb_fusion --alpha 0.30 --max-iter 15 --batch-size 4 --overwrite
```
