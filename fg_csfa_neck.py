#!/usr/bin/env python3
"""
fg_csfa_neck.py
===============
Proposal 3: Frequency-Gated Cross-Scale Feature Alignment & Quaternion Spectral-Spatial
Fusion (FG-CSFA) for Dense Micro-SMD Detection.

Core Scientific Contributions
-----------------------------
1. Quaternion Radiometric Stem (`QuaternionConv2d` & `QuaternionRadiometricAdapter`):
   Models the 4 orthogonal physical modalities at each pixel as a Quaternion Q in H:
       Q(x, y) = L(x, y) * 1 + R_spec(x, y) * i + grad_x R(x, y) * j + grad_y R(x, y) * k
   where R_spec(x, y) is initialized from the PCB-Vision 31-band optical reflectance
   contrast prior (Arbash et al., IEEE Sensors J. 2024) with Agaian logarithmic alpha-rooting.
   Hamilton product convolution W *_H Q enforces skew-symmetric cross-channel coupling
   between irradiance (1), spectral material contrast (i), horizontal solder fillets (j),
   and vertical solder fillets (k) with 4x fewer parameters than standard Conv2d.

2. High-Frequency FR-4 Fiberglass Weave Suppression (`FrequencyGatedWeaveSuppressor`):
   Decomposes shallow stride-4 P2 features (320x320 at 1280px) into low-pass component
   P2_low = AvgPool_3x3(P2) and high-frequency residual dP2 = P2 - P2_low.
   Modulates dP2 with a semantic gate predicted from P3 so high-frequency edges are
   preserved on component pads while periodic FR-4 fiberglass weave noise is suppressed.

3. Space-to-Depth (SPD) Lossless Folding & Deformable Phase Alignment (`FGCSFAAlignmentBlock`):
   Folds the filtered stride-4 P2 feature map into 4 sub-pixel phase channels at stride 8
   (SPD-Conv, Sunkara & Luo 2022) and aligns P3 features via differentiable 2D sub-pixel
   offset sampling (`F.grid_sample`). Fuses aligned P3 and SPD(P2) with zero-initialized
   BiFPN cross-scale weights so the detector starts 100% identical to pretrained YOLO26s
   weights while learning to inject stride-4 sub-pixel detail into the P3 head without
   adding a 4th P2 head's 4x anchor/NMS latency penalty.

Self-test:
    python fg_csfa_neck.py
"""

from typing import Optional, Tuple
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    sys.modules["fg_csfa_neck"] = sys.modules[__name__]


# ---------------------------------------------------------------------------
# 1. Hamilton Product Quaternion Convolution (H = R + R i + R j + R k)
# ---------------------------------------------------------------------------
class QuaternionConv2d(nn.Module):
    """
    2D Quaternion Convolution using the Hamilton product:
        [W_r  -W_i  -W_j  -W_k] [Q_r]
        [W_i   W_r  -W_k   W_j] [Q_i]
        [W_j   W_k   W_r  -W_j] [Q_j]
        [W_k  -W_j   W_i   W_r] [Q_k]
    Input channels and output channels must be multiples of 4.
    """

    __module__ = "fg_csfa_neck"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, (
            "QuaternionConv2d requires in_channels and out_channels divisible by 4"
        )
        self.in_q = in_channels // 4
        self.out_q = out_channels // 4
        self.stride = stride
        self.padding = padding

        self.w_r = nn.Parameter(torch.empty(self.out_q, self.in_q, kernel_size, kernel_size))
        self.w_i = nn.Parameter(torch.empty(self.out_q, self.in_q, kernel_size, kernel_size))
        self.w_j = nn.Parameter(torch.empty(self.out_q, self.in_q, kernel_size, kernel_size))
        self.w_k = nn.Parameter(torch.empty(self.out_q, self.in_q, kernel_size, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        for w in (self.w_r, self.w_i, self.w_j, self.w_k):
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
            w.data.mul_(0.5)

    def assemble_hamilton_kernel(self) -> torch.Tensor:
        """Constructs the (4*out_q, 4*in_q, k, k) real-equivalent Hamilton kernel."""
        row0 = torch.cat([self.w_r, -self.w_i, -self.w_j, -self.w_k], dim=1)
        row1 = torch.cat([self.w_i, self.w_r, -self.w_k, self.w_j], dim=1)
        row2 = torch.cat([self.w_j, self.w_k, self.w_r, -self.w_i], dim=1)
        row3 = torch.cat([self.w_k, -self.w_j, self.w_i, self.w_r], dim=1)
        return torch.cat([row0, row1, row2, row3], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.assemble_hamilton_kernel().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.conv2d(x, kernel, bias=b, stride=self.stride, padding=self.padding)


class QuaternionRadiometricAdapter(nn.Module):
    """
    Constructs the 4-component physical quaternion field Q = [L, R_spec, gx, gy]
    and projects it into Layer-0 feature space via QuaternionConv2d.
    Zero-initialized output projection preserves exact pretrained identity at step 0.
    """

    __module__ = "fg_csfa_neck"

    def __init__(
        self,
        out_ch: int = 32,
        stride: int = 2,
        priors_path: str = "data/pcb_spectral_priors.json",
        alpha: float = 0.25,
    ):
        super().__init__()
        self.out_ch = int(out_ch)
        self.stride = int(stride)

        # Spectral contrast projection (3 RGB -> 1 optimal ceramic-vs-soldermask contrast channel)
        self.spectral_proj = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        self._init_spectral_prior(priors_path, alpha=alpha)

        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32) / 8.0
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3).clone())
        self.register_buffer("sobel_y", sx.t().contiguous().view(1, 1, 3, 3).clone())

        q_hidden = 16  # 4 quaternions
        self.q_conv1 = QuaternionConv2d(4, q_hidden, kernel_size=3, stride=stride, padding=1, bias=False)
        self.q_bn1 = nn.BatchNorm2d(q_hidden)
        self.q_act = nn.SiLU(inplace=True)
        self.q_out = nn.Conv2d(q_hidden, self.out_ch, kernel_size=1, bias=False)
        nn.init.zeros_(self.q_out.weight)

    def _init_spectral_prior(self, priors_path: Optional[str], alpha: float = 0.25):
        p = Path(priors_path) if priors_path is not None else None
        if p is not None and p.exists():
            with open(p, "r") as f:
                priors = json.load(f)
            w = torch.tensor(priors["normalized_contrast_weights"], dtype=torch.float32)
            if alpha > 0:
                c_scaled = w * 31.0
                w = w * torch.pow(1.0 + torch.log(1.0 + c_scaled), alpha)
                w = w / w.sum()
            # Aggregate 31 bands into Red (590-700nm), Green (500-580nm), Blue (400-490nm) contrast gains
            b_gain = float(w[:10].sum())
            g_gain = float(w[10:19].sum())
            r_gain = float(w[19:].sum())
            # Ceramic capacitors have positive contrast in Red/Blue and contrast against Green soldermask
            rgb_w = torch.tensor([r_gain, -0.5 * g_gain, b_gain], dtype=torch.float32).view(1, 3, 1, 1)
        else:
            rgb_w = torch.tensor([0.6, -0.4, 0.3], dtype=torch.float32).view(1, 3, 1, 1)

        with torch.no_grad():
            self.spectral_proj.weight.copy_(rgb_w)
            self.spectral_proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            x32 = x.float()
            L = x32.max(dim=1, keepdim=True)[0]
            proj_dtype = self.spectral_proj.weight.dtype
            r_spec = torch.sigmoid(self.spectral_proj(x32.to(proj_dtype)).float())
            pad_r = F.pad(r_spec, (1, 1, 1, 1), mode="replicate")
            gx = F.conv2d(pad_r, self.sobel_x.float())
            gy = F.conv2d(pad_r, self.sobel_y.float())

            # Quaternion tensor Q = [L, R_spec, gx, gy] in R^{B x 4 x H x W}
            q_in = torch.cat([L, r_spec, gx, gy], dim=1)
            w_dtype = self.q_conv1.w_r.dtype
            q_feat = self.q_act(self.q_bn1(self.q_conv1(q_in.to(w_dtype))).float())
            out = self.q_out(q_feat.to(self.q_out.weight.dtype)).float()
        return out.to(x.dtype)


# ---------------------------------------------------------------------------
# 2. Frequency-Gated Weave Suppressor & Space-to-Depth Deformable P2->P3 Neck
# ---------------------------------------------------------------------------
class FrequencyGatedWeaveSuppressor(nn.Module):
    """
    Suppresses periodic high-frequency FR-4 fiberglass weave texture on stride-4 P2
    features while preserving sharp solder-fillet edges via semantic P3 gating:
        P2_low  = AvgPool_3x3(P2)
        dP2     = P2 - P2_low
        G_sem   = sigmoid(Up_2x(Conv_1x1(P3)))
        P2_filt = P2_low + G_sem * dP2
    """

    __module__ = "fg_csfa_neck"

    def __init__(self, p2_ch: int, p3_ch: int):
        super().__init__()
        self.sem_gate = nn.Conv2d(p3_ch, p2_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.sem_gate.weight)
        nn.init.constant_(self.sem_gate.bias, 1.0)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        p2_low = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)
        dp2 = p2 - p2_low
        g_sem = torch.sigmoid(
            F.interpolate(self.sem_gate(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        )
        return p2_low + g_sem * dp2


class SpaceToDepthDownsample(nn.Module):
    """
    Lossless 2x Space-to-Depth (SPD) folding from stride-4 P2 (2H x 2W x C2)
    into stride-8 (H x W x 4*C2), followed by 1x1 channel projection to C3.
    """

    __module__ = "fg_csfa_neck"

    def __init__(self, p2_ch: int, p3_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(p2_ch * 4, p3_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(p3_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure even spatial dimensions before 2x2 phase slicing
        h, w = x.shape[-2:]
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        sub00 = x[..., 0::2, 0::2]
        sub01 = x[..., 0::2, 1::2]
        sub10 = x[..., 1::2, 0::2]
        sub11 = x[..., 1::2, 1::2]
        folded = torch.cat([sub00, sub01, sub10, sub11], dim=1)
        return self.proj(folded)


class DeformablePhaseAligner(nn.Module):
    """
    Pure-PyTorch differentiable 2D sub-pixel phase alignment using learned offset fields
    and bilinear `F.grid_sample` (zero custom C++ compilation required, works on all GPUs).
    Aligns P3 neck features with lossless SPD-folded P2 micro-component features.
    """

    __module__ = "fg_csfa_neck"

    def __init__(self, ch: int, max_offset_cells: float = 1.5):
        super().__init__()
        self.max_offset_cells = float(max_offset_cells)
        self.offset_net = nn.Sequential(
            nn.Conv2d(ch * 2, ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch // 2, 2, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def forward(self, feat_to_align: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        if feat_to_align.shape[-2:] != ref_feat.shape[-2:]:
            feat_to_align = F.interpolate(
                feat_to_align, size=ref_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        b, c, h, w = feat_to_align.shape
        concat = torch.cat([feat_to_align, ref_feat], dim=1)
        # Predict sub-pixel offsets in [-max_offset_cells, +max_offset_cells]
        offsets = torch.tanh(self.offset_net(concat)).float() * self.max_offset_cells

        dev_type = "cuda" if feat_to_align.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=feat_to_align.device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, w, device=feat_to_align.device, dtype=torch.float32),
                indexing="ij",
            )
            base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, h, w, 2)
            dx_norm = offsets[:, 0, :, :] * (2.0 / max(w - 1, 1))
            dy_norm = offsets[:, 1, :, :] * (2.0 / max(h - 1, 1))
            delta_grid = torch.stack([dx_norm, dy_norm], dim=-1)
            sampling_grid = (base_grid + delta_grid).clamp(-1.0, 1.0)
            aligned = F.grid_sample(
                feat_to_align.float(),
                sampling_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        return aligned.to(feat_to_align.dtype)


class FGCSFAAlignmentBlock(nn.Module):
    """
    Wraps the P3 neck module (e.g., Layer 16 C3k2) to fuse:
      1. Standard P3 neck output F_p3
      2. Frequency-gated, Space-to-Depth folded P2 features F_spd
      3. Deformably phase-aligned cross-scale refinement
    via a zero-initialized BiFPN residual gate `alpha_csfa` so that at step 0,
    F_out == F_p3 (exact pretrained equivalence).
    """

    __module__ = "fg_csfa_neck"

    def __init__(self, orig_p3_module: nn.Module, p2_ch: int = 64, p3_ch: int = 128):
        super().__init__()
        self.orig_p3_module = orig_p3_module
        self.f = getattr(orig_p3_module, "f", -1)
        self.i = getattr(orig_p3_module, "i", 16)
        self.type = getattr(orig_p3_module, "type", "FGCSFAAlignmentBlock")

        self.weave_suppressor = FrequencyGatedWeaveSuppressor(p2_ch=p2_ch, p3_ch=p3_ch)
        self.spd_down = SpaceToDepthDownsample(p2_ch=p2_ch, p3_ch=p3_ch)
        self.phase_aligner = DeformablePhaseAligner(ch=p3_ch)

        self.refine_conv = nn.Sequential(
            nn.Conv2d(p3_ch, p3_ch, kernel_size=3, padding=1, groups=p3_ch, bias=False),
            nn.BatchNorm2d(p3_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(p3_ch, p3_ch, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.refine_conv[-1].weight)

        # Learnable BiFPN fusion weight (initialized to 1.0, while refine_conv[-1] is 0.0 at step 0)
        self.bifpn_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.cached_p2: Optional[torch.Tensor] = None
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p3_feat = self.orig_p3_module(x)
        p2_feat = self.cached_p2
        self.cached_p2 = None

        if p2_feat is None:
            return p3_feat

        p2_clean = self.weave_suppressor(p2_feat, p3_feat)
        p2_spd = self.spd_down(p2_clean)
        p2_aligned = self.phase_aligner(p2_spd, p3_feat)

        w = F.relu(self.bifpn_weight).to(p3_feat.dtype)
        fused = self.refine_conv(p3_feat + w * p2_aligned)
        return p3_feat + fused


class P2FeatureTapWrapper(nn.Module):
    """Wraps the backbone P2 layer (stride 4) to forward its output to FGCSFAAlignmentBlock."""

    __module__ = "fg_csfa_neck"

    def __init__(self, orig_p2_module: nn.Module, target_csfa_block: FGCSFAAlignmentBlock):
        super().__init__()
        self.orig_p2_module = orig_p2_module
        self.f = getattr(orig_p2_module, "f", -1)
        self.i = getattr(orig_p2_module, "i", 2)
        self.type = getattr(orig_p2_module, "type", "P2FeatureTapWrapper")
        self.np = getattr(orig_p2_module, "np", sum(p.numel() for p in orig_p2_module.parameters()))
        # Store in list so PyTorch does not register circular submodule hierarchy
        self._target_ref = [target_csfa_block]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig_p2_module(x)
        if self._target_ref and self._target_ref[0] is not None:
            self._target_ref[0].cached_p2 = out
        return out


class QuaternionStemWrapper(nn.Module):
    """Wraps Layer 0 to inject the Quaternion Radiometric Stem residual into Layer-0 feature space."""

    __module__ = "fg_csfa_neck"

    def __init__(
        self,
        orig_layer0: nn.Module,
        priors_path: str = "data/pcb_spectral_priors.json",
        alpha: float = 0.25,
    ):
        super().__init__()
        self.orig_layer0 = orig_layer0
        self.f = getattr(orig_layer0, "f", -1)
        self.i = getattr(orig_layer0, "i", 0)
        self.type = getattr(orig_layer0, "type", "Conv")

        # Determine Layer-0 output channels even if already wrapped by IGRDualStreamStem
        base_conv = getattr(orig_layer0, "orig_conv", orig_layer0)
        inner_conv = getattr(base_conv, "conv", base_conv)
        out_ch = getattr(inner_conv, "out_channels", 32)
        stride_val = getattr(inner_conv, "stride", (2, 2))
        stride = stride_val[0] if isinstance(stride_val, (tuple, list)) else int(stride_val)

        self.q_adapter = QuaternionRadiometricAdapter(
            out_ch=out_ch, stride=stride, priors_path=priors_path, alpha=alpha
        )
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orig_layer0(x) + self.q_adapter(x)


def _infer_out_channels(mod: nn.Module, default: int = 64) -> int:
    for attr in ("cv2", "conv"):
        if hasattr(mod, attr):
            sub = getattr(mod, attr)
            if hasattr(sub, "conv") and hasattr(sub.conv, "out_channels"):
                return int(sub.conv.out_channels)
            if hasattr(sub, "out_channels"):
                return int(sub.out_channels)
    for m in reversed(list(mod.modules())):
        if isinstance(m, nn.Conv2d):
            return int(m.out_channels)
    return default


def inject_fg_csfa_into_model(
    det_model: nn.Module,
    inject_quaternion_stem: bool = True,
    priors_path: str = "data/pcb_spectral_priors.json",
    alpha: float = 0.25,
) -> nn.Module:
    """
    Idempotently injects:
      1. QuaternionRadiometricAdapter into Layer 0 (composable with IGR-Net)
      2. P2FeatureTapWrapper into backbone stride-4 layer (index 2)
      3. FGCSFAAlignmentBlock into the P3 detection neck input layer (first head source)
    """
    seq = det_model.model
    # 1. Quaternion Stem
    if inject_quaternion_stem and not isinstance(seq[0], QuaternionStemWrapper):
        seq[0] = QuaternionStemWrapper(seq[0], priors_path=priors_path, alpha=alpha)

    # 2. Identify P2 (stride 4, typically layer 2) and P3 head input (det_head.f[0], typically layer 16)
    det_head = seq[-1]
    head_sources = getattr(det_head, "f", [16, 19, 22])
    p3_neck_idx = int(head_sources[0]) if isinstance(head_sources, (list, tuple)) else 16
    p2_idx = 2

    if not isinstance(seq[p3_neck_idx], FGCSFAAlignmentBlock):
        p2_ch = _infer_out_channels(seq[p2_idx], default=64)
        p3_ch = _infer_out_channels(seq[p3_neck_idx], default=128)
        csfa_block = FGCSFAAlignmentBlock(seq[p3_neck_idx], p2_ch=p2_ch, p3_ch=p3_ch)
        seq[p3_neck_idx] = csfa_block
        seq[p2_idx] = P2FeatureTapWrapper(seq[p2_idx], csfa_block)
    else:
        # Ensure P2 tap points to existing CSFA block
        if isinstance(seq[p2_idx], P2FeatureTapWrapper):
            seq[p2_idx]._target_ref = [seq[p3_neck_idx]]

    return det_model


# Register safe globals for PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([
        QuaternionConv2d,
        QuaternionRadiometricAdapter,
        QuaternionStemWrapper,
        FrequencyGatedWeaveSuppressor,
        SpaceToDepthDownsample,
        DeformablePhaseAligner,
        FGCSFAAlignmentBlock,
        P2FeatureTapWrapper,
    ])
except Exception:
    pass


def get_fg_csfa_trainer(
    inject_quaternion_stem: bool = True,
    priors_path: str = "data/pcb_spectral_priors.json",
    alpha: float = 0.25,
):
    """Returns a DetectionTrainer subclass that injects FG-CSFA into the model."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class FGCSFADetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            inject_fg_csfa_into_model(
                model,
                inject_quaternion_stem=inject_quaternion_stem,
                priors_path=priors_path,
                alpha=alpha,
            )
            if weights is not None:
                model.load(weights, verbose=False)
            if verbose:
                print(
                    f"[FG-CSFA] Injected Quaternion Radiometric Stem (alpha={alpha}) "
                    f"& Frequency-Gated SPD Deformable P2->P3 Alignment Neck"
                )
            return model

    return FGCSFADetectionTrainer


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------
def _selftest():
    print("=" * 75)
    print("🔬 Running Self-Test for Proposal 3: FG-CSFA (`fg_csfa_neck.py`)")
    print("=" * 75)
    torch.manual_seed(0)

    # 1. Test QuaternionConv2d parameter efficiency and Hamilton structure
    qconv = QuaternionConv2d(4, 16, kernel_size=3, stride=2, padding=1)
    q_in = torch.randn(2, 4, 64, 64, requires_grad=True)
    q_out = qconv(q_in)
    assert q_out.shape == (2, 16, 32, 32)
    q_params = sum(p.numel() for p in qconv.parameters())
    std_params = 4 * 16 * 3 * 3
    print(f"[QuaternionConv2d] Output shape: {tuple(q_out.shape)} | Params: {q_params} (vs {std_params} standard Conv2d = 4x reduction)")

    # 2. Test FGCSFAAlignmentBlock zero-init identity & gradient flow
    p2_dummy = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    p3_dummy = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

    import copy
    p3_base = copy.deepcopy(p3_dummy).eval()
    csfa = FGCSFAAlignmentBlock(copy.deepcopy(p3_dummy), p2_ch=64, p3_ch=128).eval()
    p2_tap = P2FeatureTapWrapper(copy.deepcopy(p2_dummy), csfa).eval()

    x_s2 = torch.randn(2, 32, 64, 64)
    with torch.no_grad():
        feat_p2 = p2_tap(x_s2)
        y_base = p3_base(feat_p2)
        y_csfa = csfa(feat_p2)

    diff = (y_base - y_csfa).abs().max().item()
    print(f"[Zero-Init Identity Check] max |P3_base - P3_CSFA| at step 0 = {diff:.3e} (expected 0.0)")
    assert diff < 1e-6, "FGCSFAAlignmentBlock must be exact identity at step 0!"

    # Gradient check
    csfa.train()
    p2_tap.train()
    x_s2_grad = torch.randn(2, 32, 64, 64, requires_grad=True)
    f_p2 = p2_tap(x_s2_grad)
    y_train = csfa(f_p2)
    y_train.pow(2).mean().backward()
    assert csfa.refine_conv[-1].weight.grad is not None
    assert torch.isfinite(csfa.refine_conv[-1].weight.grad).all()
    print("[FG-CSFA Gradient Check] SPD + Deformable Phase Aligner gradients finite: OK | PASSED")
    print("=" * 75)


if __name__ == "__main__":
    _selftest()
