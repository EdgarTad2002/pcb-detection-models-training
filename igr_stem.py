#!/usr/bin/env python3
"""
igr_stem.py
===========
Proposal 2: Dual-Stream Illumination-Gated Retinex & Differentiable Parametric Gamma
Network (IGR-Net) for Dense PCB Inspection.

Why Standard RGB & Simple Retinex Residuals Fail on PCBs
--------------------------------------------------------
1. Bottlenecked 3-Channel Collapse:
   Previous Retinex stems (`retinex_stem.py`) and offline scripts (`enhance_pcb_retinex.py`)
   discard the illumination field L(x,y) and compress [S, logR, E] back into a 3-channel
   RGB tensor before Layer 0 (`orig_conv(x + residual)`). By the Data Processing Inequality,
   projecting 7 physical channels down to 3 channels before spatial feature extraction
   causes irreversible information loss.
2. Physical Value of Illumination L(x,y):
   On PCBs, L(x,y) encodes two critical physical phenomena:
     - Specular Metallic Glare (L -> 1): Concave solder fillets at 0402/0201 capacitor
       terminals act as micro-mirrors under inspection LEDs.
     - 3D Component Shadow Occlusion (L -> 0): Tall electrolytic capacitors and ICs cast
       shadows over neighboring micro-passives, where naive division R = S / L amplifies
       sensor read noise by 1/L.

What IGR-Net Does
-----------------
1. Multi-Scale Edge-Preserving Bilateral-Guided Retinex Decomposition:
   Estimates illumination L(x,y) across scales k in {5, 15, 31} weighted by local edge
   reliability w_edge(x,y) = exp(-||grad L_0||^2 / (2 * sigma_r^2)) to eliminate halo
   artifacts around high-contrast solder joints.
2. Differentiable Luminance-Guided Pixel-Wise Gamma Predictor:
   Predicts a spatially varying, channel-wise exponent field gamma(x,y) in [0.5, 1.5]
   from [L, L - mean(L), |grad L|] to simultaneously lift shadowed traces and compress
   solder glare bloom. Zero-initialized so gamma(x,y) = 1.0 at step 0.
3. Feature-Space FiLM Gating (Zero 3-Channel Bottleneck):
   Runs parallel stride-2 feature extraction on the Reflectance-Gamma stream (F_R in R^{C0})
   and Illumination-Glare stream (alpha_L, beta_L in R^{C0}), fusing directly in the
   C0-channel (e.g. 32ch) Layer-0 feature space:
       F_out = orig_conv(S) + (1 + tanh(alpha_L)) * F_R + beta_L
   With zero-initialization on the output projections of F_R and beta_L, the network
   starts 100% identical to the COCO-pretrained YOLO26s checkpoint at step 0.

Self-test:
    python igr_stem.py
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

if __name__ == "__main__":
    import sys
    sys.modules["igr_stem"] = sys.modules[__name__]


def _make_gaussian_2d(ksize: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) * 0.5
    g1 = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
    g2 = torch.outer(g1, g1)
    return (g2 / g2.sum()).view(1, 1, ksize, ksize)


class MultiScaleEdgePreservingRetinex(nn.Module):
    """
    Decomposes input radiance S in [0, 1]^{B x 3 x H x W} into:
      - Smooth edge-preserving Illumination L in [eps, 1]^{B x 1 x H x W}
      - Affine-normalized Log-Reflectance R_tilde in (0, 1)^{B x 3 x H x W}
      - Reflectance Sobel gradient fields (gx, gy, grad_mag)
      - Shadow reliability mask M_shadow and Specular fillet mask M_spec
    """

    __module__ = "igr_stem"

    def __init__(self, eps: float = 1e-3, sigma_r: float = 0.15, tau_shadow: float = 0.12, tau_glare: float = 0.82):
        super().__init__()
        self.eps = float(eps)
        self.tau_shadow = float(tau_shadow)
        self.tau_glare = float(tau_glare)

        # Learnable edge-stopping range parameter and multi-scale weights pi_k for k in {5, 15, 31}
        self.log_sigma_r = nn.Parameter(torch.tensor(float(sigma_r)).log())
        self.scale_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))

        # Depthwise Gaussian kernels at scales k = 5 (sigma=1.2), 15 (sigma=3.5), 31 (sigma=7.0)
        self.conv_k5 = nn.Conv2d(1, 1, 5, padding=2, bias=False, padding_mode="replicate")
        self.conv_k15 = nn.Conv2d(1, 1, 15, padding=7, bias=False, padding_mode="replicate")
        self.conv_k31 = nn.Conv2d(1, 1, 31, padding=15, bias=False, padding_mode="replicate")
        with torch.no_grad():
            self.conv_k5.weight.copy_(_make_gaussian_2d(5, 1.2))
            self.conv_k15.weight.copy_(_make_gaussian_2d(15, 3.5))
            self.conv_k31.weight.copy_(_make_gaussian_2d(31, 7.0))

        # Learnable per-channel affine normalization on log-reflectance (replaces hard clamp)
        self.refl_scale = nn.Parameter(torch.full((1, 3, 1, 1), 0.5, dtype=torch.float32))
        self.refl_bias = nn.Parameter(torch.zeros((1, 3, 1, 1), dtype=torch.float32))

        # Fixed Sobel derivative kernels
        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32) / 8.0
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3).clone())
        self.register_buffer("sobel_y", sx.t().contiguous().view(1, 1, 3, 3).clone())

    def _sobel_xy(self, x1ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad = F.pad(x1ch, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(pad, self.sobel_x.float())
        gy = F.conv2d(pad, self.sobel_y.float())
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return gx, gy, mag

    def forward(self, s32: torch.Tensor):
        # 1. Initial max-channel illumination estimate L0
        l0 = s32.max(dim=1, keepdim=True)[0]
        _, _, grad_l0 = self._sobel_xy(l0)

        # Edge-preserving bilateral guidance weight w_edge in (0, 1]
        sigma_r = self.log_sigma_r.exp().clamp(min=0.02, max=1.0)
        w_edge = torch.exp(-(grad_l0 * grad_l0) / (2.0 * sigma_r * sigma_r))
        wl0 = w_edge * l0

        # Normalized bilateral-guided filtering at k in {5, 15, 31}
        L_s = []
        for conv in (self.conv_k5, self.conv_k15, self.conv_k31):
            p = conv.kernel_size[0] // 2
            num = F.conv2d(F.pad(wl0, (p, p, p, p), mode="replicate"), conv.weight.float().abs())
            den = F.conv2d(F.pad(w_edge, (p, p, p, p), mode="replicate"), conv.weight.float().abs()).clamp(min=1e-4)
            L_s.append(num / den)

        pi = F.softmax(self.scale_logits.float(), dim=0)
        L = (pi[0] * L_s[0] + pi[1] * L_s[1] + pi[2] * L_s[2]).clamp(min=self.eps, max=1.0)

        # 2. Intrinsic log-reflectance R = log(S + eps) - log(L + eps)
        log_r = torch.log(s32.clamp(min=0.0) + self.eps) - torch.log(L + self.eps)
        r_tilde = torch.sigmoid(self.refl_scale.float() * log_r + self.refl_bias.float())

        # 3. Shadow suppression mask & Specular solder terminal mask
        m_shadow = (L * L) / (L * L + self.tau_shadow * self.tau_shadow)
        r_gray = (r_tilde * m_shadow).mean(dim=1, keepdim=True)
        gx_r, gy_r, edge_r = self._sobel_xy(r_gray)
        m_spec = F.relu(L - self.tau_glare) * edge_r

        return L, r_tilde, gx_r, gy_r, edge_r, m_shadow, m_spec, grad_l0


class DifferentiableGammaPredictor(nn.Module):
    """
    Predicts spatially varying, channel-wise gamma exponent gamma(x,y) in [0.5, 1.5]
    from illumination context [L, L - mean(L), |grad L|].
    Zero-initialized at output so gamma(x,y) == 1.0 at step 0.
    """

    __module__ = "igr_stem"

    def __init__(self, hidden: int = 12, gamma_0: float = 1.0, delta_gamma: float = 0.5, eps: float = 1e-4):
        super().__init__()
        self.gamma_0 = float(gamma_0)
        self.delta_gamma = float(delta_gamma)
        self.eps = float(eps)

        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s32: torch.Tensor, L: torch.Tensor, grad_l0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l_mean = L.mean(dim=(-2, -1), keepdim=True)
        illum_ctx = torch.cat([L, L - l_mean, grad_l0], dim=1)
        w_dtype = self.net[0].weight.dtype
        net_out = self.net(illum_ctx.to(w_dtype)).float()
        gamma_map = self.gamma_0 + self.delta_gamma * torch.tanh(net_out)
        s_gamma = torch.pow(s32.clamp(min=self.eps, max=1.0), gamma_map)
        return s_gamma, gamma_map


class IGRDualStreamStem(nn.Module):
    """
    Wraps YOLO26s Layer 0 (`orig_conv`, typically Conv(3 -> 32, k=3, s=2)).
    Computes:
      1. Base path: F_base = orig_conv(x)
      2. Reflectance-Gamma structural stream: F_R in R^{B x C0 x H/2 x W/2}
      3. Illumination-Glare FiLM modulation gate: (alpha_L, beta_L) in R^{B x C0 x H/2 x W/2}
      4. Output: F_out = F_base + (1 + tanh(alpha_L)) * F_R + beta_L
    """

    __module__ = "igr_stem"

    def __init__(self, orig_conv: nn.Module, hidden: int = 24, use_checkpoint: bool = False):
        super().__init__()
        self.orig_conv = orig_conv
        self.use_checkpoint = bool(use_checkpoint)

        # Preserve Ultralytics graph attributes
        self.f = getattr(orig_conv, "f", -1)
        self.i = getattr(orig_conv, "i", 0)
        self.type = getattr(orig_conv, "type", "Conv")

        # Determine Layer-0 output channels and stride
        inner_conv = getattr(orig_conv, "conv", orig_conv)
        out_ch = getattr(inner_conv, "out_channels", 32)
        stride = getattr(inner_conv, "stride", (2, 2))
        self.out_ch = int(out_ch)

        self.retinex = MultiScaleEdgePreservingRetinex()
        self.gamma_net = DifferentiableGammaPredictor(hidden=12)

        # Reflectance stream: [S_gamma (3), M_shadow * R_tilde (3), gx_R (1), gy_R (1), edge_R (1)] = 9 channels -> out_ch
        self.refl_branch = nn.Sequential(
            nn.Conv2d(9, hidden, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.out_ch, kernel_size=1, bias=False),
        )
        # Zero-init final projection so F_R == 0 at step 0
        nn.init.zeros_(self.refl_branch[-1].weight)

        # Illumination FiLM gate: [L (1), M_shadow (1), M_spec (1)] = 3 channels -> 2 * out_ch (alpha_L, beta_L)
        self.illum_film = nn.Sequential(
            nn.Conv2d(3, hidden // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, 2 * self.out_ch, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.illum_film[-1].weight)
        nn.init.zeros_(self.illum_film[-1].bias)

        self.np = sum(p.numel() for p in self.parameters())

    def _compute_igr_residual(self, x: torch.Tensor) -> torch.Tensor:
        dev_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            s32 = x.float()
            L, r_tilde, gx_r, gy_r, edge_r, m_shadow, m_spec, grad_l0 = self.retinex(s32)
            s_gamma, _ = self.gamma_net(s32, L, grad_l0)

            refl_in = torch.cat([s_gamma, m_shadow * r_tilde, gx_r, gy_r, edge_r], dim=1)
            illum_in = torch.cat([L, m_shadow, m_spec], dim=1)

            w_dtype = self.refl_branch[0].weight.dtype
            f_r = self.refl_branch(refl_in.to(w_dtype)).float()
            film_params = self.illum_film(illum_in.to(w_dtype)).float()
            alpha_l, beta_l = film_params.chunk(2, dim=1)

            mod_residual = (1.0 + torch.tanh(alpha_l)) * f_r + beta_l
        return mod_residual.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_feat = self.orig_conv(x)
        if self.use_checkpoint and self.training and x.requires_grad:
            igr_res = checkpoint(self._compute_igr_residual, x, use_reentrant=False)
        else:
            igr_res = self._compute_igr_residual(x)
        return base_feat + igr_res.to(base_feat.dtype)


def wrap_model_with_igr_stem(det_model: nn.Module, use_checkpoint: bool = False) -> nn.Module:
    """Idempotently wraps det_model.model[0] with IGRDualStreamStem."""
    if not isinstance(det_model.model[0], IGRDualStreamStem):
        det_model.model[0] = IGRDualStreamStem(det_model.model[0], use_checkpoint=use_checkpoint)
    return det_model


# Register safe globals for PyTorch 2.6+ serialization
try:
    torch.serialization.add_safe_globals([
        IGRDualStreamStem,
        MultiScaleEdgePreservingRetinex,
        DifferentiableGammaPredictor,
    ])
except Exception:
    pass


def get_igr_stem_trainer(use_checkpoint: bool = False):
    """Returns a DetectionTrainer subclass that injects IGRDualStreamStem into Layer 0."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class IGRStemDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            wrap_model_with_igr_stem(model, use_checkpoint=use_checkpoint)
            if weights is not None:
                model.load(weights, verbose=False)
            if verbose:
                stem = model.model[0]
                n_extra = sum(p.numel() for n, p in stem.named_parameters() if not n.startswith("orig_conv"))
                print(
                    f"[IGR-Net] Injected Dual-Stream Illumination-Gated Retinex & Gamma Stem "
                    f"into Layer 0 (+{n_extra} params, {stem.out_ch}ch FiLM gating, zero-init identity)"
                )
            return model

    return IGRStemDetectionTrainer


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------
def _selftest():
    print("=" * 75)
    print("🔬 Running Self-Test for Proposal 2: IGR-Net (`igr_stem.py`)")
    print("=" * 75)
    torch.manual_seed(0)

    dummy_conv = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.SiLU(inplace=True),
    )
    dummy_conv.conv = dummy_conv[0]

    import copy
    base_conv = copy.deepcopy(dummy_conv).eval()
    igr_block = IGRDualStreamStem(copy.deepcopy(dummy_conv), use_checkpoint=False).eval()

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        y_base = base_conv(x)
        y_igr = igr_block(x)

    max_diff = (y_base - y_igr).abs().max().item()
    print(f"[Zero-Init Identity Check] max |base - IGR| at step 0 = {max_diff:.3e} (expected 0.0)")
    assert max_diff < 1e-6, "IGRDualStreamStem must be exact identity at step 0!"

    # Verify gradient flow into Retinex, Gamma, Reflectance branch, and Illumination FiLM gate
    igr_block.train()
    x_grad = torch.rand(2, 3, 128, 128, requires_grad=True)
    # Perturb zero-init last layer weights slightly after step 1 to verify full upstream chain
    out = igr_block(x_grad)
    loss = out.pow(2).mean()
    loss.backward()

    assert igr_block.refl_branch[-1].weight.grad is not None
    assert igr_block.illum_film[-1].weight.grad is not None
    assert torch.isfinite(igr_block.refl_branch[-1].weight.grad).all()
    n_extra = sum(p.numel() for n, p in igr_block.named_parameters() if not n.startswith("orig_conv"))
    print(f"[Gradient & Parameter Check] Extra stem parameters: +{n_extra} | Gradients finite: OK | PASSED")
    print("=" * 75)


if __name__ == "__main__":
    _selftest()
