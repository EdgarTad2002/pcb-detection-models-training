#!/usr/bin/env python3
"""
Learnable Retinex-Edge Stem for YOLO26 (in-network replacement for the offline
Retinex + Canny preprocessing in tools/enhance_pcb_retinex.py).

Why:
    The offline pipeline bakes  I_enh = 0.7*I + 0.3*R + 0.1*Canny(R)  into 3
    JPEG channels. The detector can't separate color / reflectance / edges,
    can't re-weight them, and Ultralytics' HSV/mosaic augmentations are
    applied AFTER the enhancement (distorting the physical signal).

What this module does (all on GPU, differentiable, trained end-to-end):
    1. Illumination  L = learnable large-kernel conv over max(R,G,B)
                         (Gaussian-initialised -> starts as the classic
                         smooth-illumination Retinex estimate)
    2. Reflectance   logR = log(I + eps) - log(L + eps)   (single-scale Retinex)
    3. Edges         E = |Sobel(mean_c logR)|             (soft, threshold-free
                         replacement for Canny, computed on reflectance so
                         shadows / glare do not create edges)
    4. Adapter       concat[I, logR, E] (7 ch) -> 1x1 conv MLP -> 3 ch residual
                     The last conv is ZERO-initialised, so at step 0 the model
                     is exactly the pretrained detector (cannot start worse).
    5. Output        x + residual  ->  original pretrained Layer-0 Conv

The wrapper copies Ultralytics' graph attributes (f, i, type, np) so the
DetectionModel forward loop is unaffected. Same pattern as the proven
PhysicsSpectralInputBlock in physics_spectral_yolo26.py.

This file must stay importable as `retinex_stem` (checkpoints pickle the class
path), so keep it at the project root next to train.py.

Self-test (runs on CPU or GPU, no dataset needed):
    python retinex_stem.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g2 = torch.outer(g1, g1)
    return g2 / g2.sum()


class LearnableRetinexEdge(nn.Module):
    """Produces a 3-channel residual from Retinex reflectance + edge features."""

    def __init__(self, ksize: int = 15, sigma: float = 4.0, eps: float = 1e-3, hidden: int = 16):
        super().__init__()
        self.eps = eps
        self.ksize = ksize

        # 1. Learnable illumination estimator (Gaussian init)
        self.illum = nn.Conv2d(1, 1, ksize, padding=ksize // 2, bias=False, padding_mode="replicate")
        with torch.no_grad():
            self.illum.weight.copy_(_gaussian_kernel(ksize, sigma)[None, None])

        # 3. Fixed Sobel operators (buffers -> move with .to(device), saved in state_dict)
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        self.register_buffer("sobel_x", sx[None, None].clone())
        self.register_buffer("sobel_y", sx.t()[None, None].clone())

        # 4. Adapter: 7 -> hidden -> 3, last layer zero-init (identity at start)
        self.adapter = nn.Sequential(
            nn.Conv2d(7, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.adapter[-1].weight)

    def features(self, x: torch.Tensor):
        """x: (B,3,H,W) in [0,1]. Returns (L, logR, E) in float32."""
        x = x.float()
        p = self.ksize // 2
        maxc = x.max(dim=1, keepdim=True)[0]
        # explicit fp32 weights: module may have been .half()'d for inference
        L = F.conv2d(F.pad(maxc, (p, p, p, p), mode="replicate"), self.illum.weight.float())
        L = L.clamp(min=self.eps)
        log_r = torch.log(x + self.eps) - torch.log(L + self.eps)
        log_r = log_r.clamp(-4.0, 2.0)
        gray = log_r.mean(dim=1, keepdim=True)
        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), self.sobel_x.float())
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), self.sobel_y.float())
        E = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return L, log_r, E

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        dev_type = "cuda" if x.is_cuda else "cpu"
        # Retinex division/log is numerically fragile in fp16 -> force fp32
        with torch.autocast(device_type=dev_type, enabled=False):
            x32 = x.float()
            _, log_r, E = self.features(x32)
            feats = torch.cat([x32, log_r, E], dim=1)
            # adapter params may be fp16 after .half() at inference -> match them
            w_dtype = self.adapter[0].weight.dtype
            residual = self.adapter(feats.to(w_dtype))
        return residual.to(in_dtype)


class RetinexStemInputBlock(nn.Module):
    """Wraps YOLO Layer 0: x -> x + LearnableRetinexEdge(x) -> orig_conv."""

    def __init__(self, orig_conv: nn.Module, ksize: int = 15, sigma: float = 4.0, hidden: int = 16):
        super().__init__()
        self.orig_conv = orig_conv
        # Ultralytics graph bookkeeping attributes
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "Conv")
        self.np = getattr(orig_conv, "np", sum(p.numel() for p in orig_conv.parameters()))
        self.retinex = LearnableRetinexEdge(ksize=ksize, sigma=sigma, hidden=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orig_conv(x + self.retinex(x))


def wrap_model_with_retinex_stem(det_model: nn.Module) -> nn.Module:
    """Idempotently wraps det_model.model[0] (an Ultralytics DetectionModel)."""
    if not isinstance(det_model.model[0], RetinexStemInputBlock):
        det_model.model[0] = RetinexStemInputBlock(det_model.model[0])
    return det_model


def get_retinex_stem_trainer():
    """Returns a DetectionTrainer subclass that injects the stem into Layer 0.

    Imported lazily so this module can be unpickled without ultralytics'
    trainer stack being initialised.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class RetinexStemDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            wrap_model_with_retinex_stem(model)
            # If warm-starting from a checkpoint that already has a stem, transfer it too
            if weights is not None:
                model.load(weights, verbose=False)
            if verbose:
                n = sum(p.numel() for p in model.model[0].retinex.parameters())
                print(f"[RetinexStem] Injected learnable Retinex-Edge stem into Layer 0 (+{n} params, zero-init residual)")
            return model

    return RetinexStemDetectionTrainer


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _flatten_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        out = []
        for o in obj:
            out.extend(_flatten_tensors(o))
        return out
    return []


def _selftest():
    torch.manual_seed(0)
    from pathlib import Path
    from ultralytics import YOLO

    if Path("yolo26s.pt").exists():
        print("[selftest] Loading yolo26s.pt...")
        base = YOLO("yolo26s.pt").model
        wrapped = YOLO("yolo26s.pt").model
    elif Path("yolov8s.pt").exists():
        print("[selftest] Loading yolov8s.pt...")
        base = YOLO("yolov8s.pt").model
        wrapped = YOLO("yolov8s.pt").model
    else:
        from ultralytics.nn.tasks import DetectionModel
        cfg = "yolov8s.yaml"
        print(f"[selftest] Building {cfg} (nc=4)...")
        base = DetectionModel(cfg, nc=4, verbose=False)
        wrapped = DetectionModel(cfg, nc=4, verbose=False)
        wrapped.load_state_dict(base.state_dict())
    wrap_model_with_retinex_stem(wrapped)

    base.eval()
    wrapped.eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        yb = _flatten_tensors(base(x))
        yw = _flatten_tensors(wrapped(x))
    assert len(yb) == len(yw) and len(yb) > 0, "unexpected model output structure"
    diff = max((a.float() - b.float()).abs().max().item() for a, b in zip(yb, yw))
    print(f"[selftest] max |base - wrapped| at init = {diff:.3e} (expected ~0)")
    assert diff < 1e-4, "zero-init residual should make the wrapped model identical at init"

    # Gradient check: residual output conv must receive gradient
    wrapped.train()
    stem = wrapped.model[0].retinex
    feats = torch.cat([x, *stem.features(x)[1:]], dim=1)
    out = stem.adapter(feats)
    out.sum().backward()
    g = stem.adapter[-1].weight.grad
    assert g is not None and g.abs().sum().item() > 0, "adapter output layer must receive gradient"
    assert torch.isfinite(g).all(), "non-finite gradient in stem"
    print("[selftest] gradient flows into zero-init adapter: OK")

    # fp16 path (inference after .half())
    if torch.cuda.is_available():
        m = wrapped.cuda().half().eval()
        with torch.no_grad():
            ys = _flatten_tensors(m(x.cuda().half()))
        assert all(torch.isfinite(t.float()).all() for t in ys), "non-finite output in fp16"
        print("[selftest] CUDA fp16 forward: OK")

    n = sum(p.numel() for p in stem.parameters())
    print(f"[selftest] stem params: {n} | PASSED")


if __name__ == "__main__":
    _selftest()
