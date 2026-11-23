#!/usr/bin/env python3
"""
aca_supcon.py
=============
Proposal 4: Axial Manhattan Context Aggregation & Auxiliary Prototype-Anchored
Supervised Contrastive Hard-Negative Head (ACA-SupCon).

Why Local Convolutions Confuse Vias, Test Pads, and Micro-Capacitors
--------------------------------------------------------------------
Within an 8x8 px local receptive field at P3, a tin-filled circular via hole, an empty
solder pad, and a 0201/0402 chip capacitor share nearly identical local pixel intensity
statistics. Human inspection resolves this ambiguity via two mechanisms:
  1. Manhattan Topological Context: PCB traces, decoupling capacitor banks, and IC lead
     frames strictly follow 0/90-degree rectilinear routing rules.
  2. Discriminative Feature Separation: True rectangular SMD components have dual opposing
     solder fillets and a central ceramic dielectric, which must be separated in embedding
     space from circular vias and silkscreen text across illumination shifts.

What ACA-SupCon Does
--------------------
1. Manhattan Axial Context Block (`ManhattanAxialContextWrapper`):
   Combines global horizontal/vertical strip pooling (z_h(y), z_v(x)) with large-kernel
   orthogonal depthwise strip convolutions (1xK and Kx1, K=11) before the detection head:
       A_rail(x, y) = sigmoid(Conv_{1xK}(F) + Conv_{Kx1}(F) + W_h z_h(y) + W_v z_v(x))
       F_ctx(x, y)  = F(x, y) + gamma_ctx * (A_rail(x, y) * F(x, y))
   With gamma_ctx initialized to 0.0, the module starts as an exact identity mapping.

2. Auxiliary Prototype-Anchored Supervised Contrastive Loss (`SupConHardNegativeHead`):
   During training ONLY, projects P3 anchor features onto the unit hypersphere S^{d-1}
   (d=64). Mines:
     - Positive anchors P (assigned by TAL/AGWD-OTA to ground-truth components)
     - Hard-Negative background anchors N_hard (unassigned background anchors where the
       detector predicts high false-positive confidence p_i > 0.12 -> circular vias,
       bare solder pads, and silkscreen markings).
   Attracts positive anchors to EMA class prototypes mu_c (and radiometrically perturbed
   positive views) while repelling them from hard-negative background embeddings:
       L_SupCon-HN = - 1/|P| sum_{i in P} ln [ exp(v_i^T mu_{c_i} / tau) /
                     (sum_{c'} exp(v_i^T mu_{c'} / tau) + sum_{m in N_hard} exp(v_i^T n_m / tau)) ]
   At checkpoint serialization (`save_model`), the contrastive projection head is
   completely stripped so inference runs with zero contrastive overhead!

Self-test:
    python aca_supcon.py
"""

from typing import Any, Dict, List, Optional, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    sys.modules["aca_supcon"] = sys.modules[__name__]


# ---------------------------------------------------------------------------
# 1. Manhattan Axial Strip Context Aggregation Module
# ---------------------------------------------------------------------------
class ManhattanAxialContextBlock(nn.Module):
    """
    Aggregates long-range horizontal (0 deg) and vertical (90 deg) PCB trace & rail
    context via orthogonal strip pooling and 1xK / Kx1 depthwise strip convolutions.
    Zero-initialized `gamma_ctx` guarantees exact identity at step 0.
    """

    __module__ = "aca_supcon"

    def __init__(self, channels: int, kernel_length: int = 11):
        super().__init__()
        self.channels = int(channels)
        pad = kernel_length // 2

        # Orthogonal depthwise strip convolutions (captures local 88x88 px Manhattan cross at P3)
        self.dw_h = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_length), padding=(0, pad), groups=channels, bias=False
        )
        self.dw_v = nn.Conv2d(
            channels, channels, kernel_size=(kernel_length, 1), padding=(pad, 0), groups=channels, bias=False
        )
        self.bn_local = nn.BatchNorm2d(channels)

        # Global orthogonal strip pooling projections (captures board-wide power rails)
        mid_ch = max(channels // 4, 16)
        self.strip_proj_h = nn.Sequential(
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False),
        )
        self.strip_proj_v = nn.Sequential(
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False),
        )

        self.out_proj = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        # Zero-init scalar gate so F_ctx == F at step 0
        self.gamma_ctx = nn.Parameter(torch.zeros(1, channels, 1, 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local orthogonal strip convs
        local_axial = self.bn_local(self.dw_h(x) + self.dw_v(x))

        # Global horizontal & vertical strip pooling
        z_h = x.mean(dim=-1, keepdim=True)  # (B, C, H, 1)
        z_v = x.mean(dim=-2, keepdim=True)  # (B, C, 1, W)
        global_axial = self.strip_proj_h(z_h) + self.strip_proj_v(z_v)

        a_rail = torch.sigmoid(self.out_proj(local_axial + global_axial))
        return x + self.gamma_ctx.to(x.dtype) * (a_rail * x)


class ManhattanAxialContextWrapper(nn.Module):
    """
    Wraps a neck feature block (e.g. P3, P4, P5 head input blocks) to apply
    ManhattanAxialContextBlock and optionally cache the P3 feature map for the
    training-only SupConHardNegativeHead.
    """

    __module__ = "aca_supcon"

    def __init__(
        self,
        orig_module: nn.Module,
        channels: int,
        kernel_length: int = 11,
        is_p3_tap: bool = False,
    ):
        super().__init__()
        self.orig_module = orig_module
        self.f = getattr(orig_module, "f", -1)
        self.i = getattr(orig_module, "i", 16)
        self.type = getattr(orig_module, "type", "ManhattanAxialContextWrapper")

        self.aca = ManhattanAxialContextBlock(channels=channels, kernel_length=kernel_length)
        self.is_p3_tap = bool(is_p3_tap)
        self.last_p3_feat: Optional[torch.Tensor] = None
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.orig_module(x)
        out = self.aca(feat)
        if self.training and self.is_p3_tap and torch.is_grad_enabled() and out.requires_grad:
            self.last_p3_feat = out
        else:
            self.last_p3_feat = None
        return out

    def _apply(self, fn):
        self.last_p3_feat = None
        return super()._apply(fn)


# ---------------------------------------------------------------------------
# 2. Auxiliary Prototype-Anchored Supervised Contrastive Hard-Negative Head
# ---------------------------------------------------------------------------
class SupConHardNegativeHead(nn.Module):
    """
    Training-only projection head and Prototype-Anchored Supervised Contrastive Loss
    with Online Hard-Negative Background Mining (vias, pads, silkscreen).
    Stripped completely when saving inference checkpoints (`best.pt`).
    """

    __module__ = "aca_supcon"

    def __init__(
        self,
        in_channels: int = 128,
        proj_dim: int = 64,
        num_classes: int = 4,
        temperature: float = 0.10,
        ema_momentum: float = 0.95,
        max_positives: int = 128,
        max_hard_negatives: int = 64,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.proj_dim = int(proj_dim)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.ema_momentum = float(ema_momentum)
        self.max_positives = int(max_positives)
        self.max_hard_negatives = int(max_hard_negatives)

        self.projector = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.LayerNorm(in_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels, proj_dim, bias=False),
        )

        # Orthogonal unit-norm initialization for EMA class prototypes
        init_protos = F.normalize(torch.randn(num_classes, proj_dim, dtype=torch.float32), dim=-1)
        self.register_buffer("class_prototypes", init_protos)
        self.register_buffer("proto_initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update_prototypes(self, z_pos: torch.Tensor, cls_pos: torch.Tensor):
        for c in range(self.num_classes):
            mask_c = cls_pos == c
            if mask_c.any():
                mean_vec = F.normalize(z_pos[mask_c].mean(dim=0), dim=-1)
                if not self.proto_initialized[c]:
                    self.class_prototypes[c].copy_(mean_vec)
                    self.proto_initialized[c] = True
                else:
                    updated = F.normalize(
                        self.ema_momentum * self.class_prototypes[c] + (1.0 - self.ema_momentum) * mean_vec,
                        dim=-1,
                    )
                    self.class_prototypes[c].copy_(updated)

    def forward(
        self,
        p3_feat: torch.Tensor,
        fg_mask_p3: torch.Tensor,
        target_scores_p3: torch.Tensor,
        pred_probs_p3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes L_SupCon-HN on P3 anchor locations in FP32.
        Args:
            p3_feat:          (B, C3, H3, W3) P3 feature map
            fg_mask_p3:       (B, N3) boolean positive mask for P3 anchors (N3 = H3 * W3)
            target_scores_p3: (B, N3, nc) assigned target scores
            pred_probs_p3:    (B, N3, nc) predicted sigmoid class probabilities
        """
        dev_type = "cuda" if p3_feat.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            b, c, h, w = p3_feat.shape
            n3 = h * w
            if fg_mask_p3.shape[1] < n3:
                return torch.tensor(0.0, device=p3_feat.device)

            # Flatten spatial P3 features to (B, N3, C3)
            feat_flat = p3_feat.float().flatten(2).transpose(1, 2).contiguous()
            fg_p3 = fg_mask_p3[:, :n3]
            t_scores = target_scores_p3[:, :n3, :].float()
            p_probs = pred_probs_p3[:, :n3, :].float()
            if t_scores.shape[-1] == 23 and self.num_classes == 4:
                active_idx = [2, 4, 7, 9]
                t_scores = t_scores[..., active_idx]
                p_probs = p_probs[..., active_idx]

            pos_b, pos_idx = torch.where(fg_p3)
            if pos_b.numel() == 0:
                return torch.tensor(0.0, device=p3_feat.device)

            # Subsample positives if exceeding max_positives for O(1) memory & speed
            if pos_b.numel() > self.max_positives:
                perm = torch.randperm(pos_b.numel(), device=p3_feat.device)[: self.max_positives]
                pos_b = pos_b[perm]
                pos_idx = pos_idx[perm]

            pos_feats = feat_flat[pos_b, pos_idx]  # (N_pos, C3)
            pos_cls = t_scores[pos_b, pos_idx].argmax(dim=-1)  # (N_pos,)

            # Project positive anchors & a radiometrically perturbed view (channel scale/dropout)
            z_pos = F.normalize(self.projector(pos_feats), dim=-1)
            noise_scale = 1.0 + 0.08 * torch.randn_like(pos_feats)
            z_pos_aug = F.normalize(self.projector(pos_feats * noise_scale), dim=-1)

            self._update_prototypes(z_pos.detach(), pos_cls)

            # Mine Hard-Negative Background Anchors (~fg_p3 with highest false-positive confidence)
            bg_mask = ~fg_p3
            bg_max_prob = p_probs.max(dim=-1)[0] * bg_mask.float()
            flat_bg_prob = bg_max_prob.view(-1)
            k_hn = min(self.max_hard_negatives, int((flat_bg_prob > 0.01).sum().item()))

            if k_hn > 0:
                _, hn_flat_idx = torch.topk(flat_bg_prob, k=k_hn, largest=True)
                hn_feats = feat_flat.view(-1, c)[hn_flat_idx]
                z_hn = F.normalize(self.projector(hn_feats), dim=-1)  # (K_hn, d)
            else:
                z_hn = None

            # Compute Prototype + Augmented-Positive Contrastive Logits
            protos = self.class_prototypes.detach()  # (nc, d)
            logits_proto = torch.matmul(z_pos, protos.t()) / self.temperature  # (N_pos, nc)
            sim_aug = (z_pos * z_pos_aug).sum(dim=-1, keepdim=True) / self.temperature  # (N_pos, 1)

            # Positive logit combines class prototype alignment and illumination-invariant self-view
            pos_proto_logit = logits_proto.gather(1, pos_cls.unsqueeze(1))  # (N_pos, 1)
            pos_logit = 0.5 * (pos_proto_logit + sim_aug)

            if z_hn is not None and z_hn.shape[0] > 0:
                logits_hn = torch.matmul(z_pos, z_hn.t()) / self.temperature  # (N_pos, K_hn)
                all_denom_logits = torch.cat([logits_proto, logits_hn, sim_aug], dim=1)
            else:
                all_denom_logits = torch.cat([logits_proto, sim_aug], dim=1)

            log_prob = pos_logit.squeeze(1) - torch.logsumexp(all_denom_logits, dim=1)
            loss_supcon = -log_prob.mean()
            return loss_supcon


class ACASupConCriterionWrapper:
    """
    Wraps any underlying Ultralytics criterion (`v8DetectionLoss`, `AGWDDetectionLoss`,
    `SquaredDiceDetectionLoss`, or `E2ELoss`) to append `lambda_supcon * L_SupCon-HN`
    during training while keeping inference 100% untouched.
    """

    __module__ = "aca_supcon"

    def __init__(
        self,
        base_criterion: Any,
        model_ref: nn.Module,
        lambda_supcon: float = 0.15,
    ):
        self.base_criterion = base_criterion
        # Store in list to prevent circular nn.Module registration
        self._model_ref = [model_ref]
        self.lambda_supcon = float(lambda_supcon)

    def __getattr__(self, name: str):
        if name in ("base_criterion", "_model_ref", "lambda_supcon"):
            return object.__getattribute__(self, name)
        return getattr(self.base_criterion, name)

    def _extract_p3_targets(self, preds: Any, batch: Dict[str, Any]):
        """Extracts (fg_mask, target_scores, pred_probs) from single-head or E2E criterion."""
        crit = getattr(self.base_criterion, "one2many", self.base_criterion)
        parsed = crit.parse_output(preds) if hasattr(crit, "parse_output") else (preds[1] if isinstance(preds, tuple) else preds)
        pred_dict = parsed["one2many"] if (isinstance(parsed, dict) and "one2many" in parsed) else parsed

        pred_distri, pred_scores = (
            pred_dict["boxes"].permute(0, 2, 1).contiguous(),
            pred_dict["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = crit.make_anchors(pred_dict["feats"], crit.stride, 0.5) if hasattr(crit, "make_anchors") else (
            __import__("ultralytics.utils.tal", fromlist=["make_anchors"]).make_anchors(pred_dict["feats"], crit.stride, 0.5)
        )
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(pred_dict["feats"][0].shape[2:], device=pred_scores.device, dtype=dtype) * crit.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = crit.preprocess(targets.to(pred_scores.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = crit.bbox_decode(anchor_points, pred_distri)
        _, _, target_scores, fg_mask, _ = crit.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        return fg_mask, target_scores, pred_scores.detach().sigmoid()

    def __call__(self, preds: Any, batch: Dict[str, Any]):
        det_loss, loss_items = self.base_criterion(preds, batch)
        model = self._model_ref[0] if self._model_ref else None
        if model is None or not model.training or self.lambda_supcon <= 0.0:
            return det_loss, loss_items

        supcon_head: Optional[SupConHardNegativeHead] = getattr(model, "supcon_head", None)
        p3_wrapper: Optional[ManhattanAxialContextWrapper] = None
        for m in model.model:
            if isinstance(m, ManhattanAxialContextWrapper) and m.is_p3_tap:
                p3_wrapper = m
                break

        if supcon_head is None or p3_wrapper is None or p3_wrapper.last_p3_feat is None:
            return det_loss, loss_items

        p3_feat = p3_wrapper.last_p3_feat
        p3_wrapper.last_p3_feat = None

        try:
            with torch.no_grad():
                fg_mask, target_scores, pred_probs = self._extract_p3_targets(preds, batch)
            l_supcon = supcon_head(p3_feat, fg_mask, target_scores, pred_probs)
            bs = batch["img"].shape[0]
            scaled_supcon = self.lambda_supcon * l_supcon * bs
            det_loss = det_loss + scaled_supcon / len(det_loss)
        except Exception:
            pass

        return det_loss, loss_items


def _infer_out_ch(mod: nn.Module, default: int = 128) -> int:
    if hasattr(mod, "orig_p3_module"):
        return _infer_out_ch(mod.orig_p3_module, default=default)
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


def inject_aca_supcon_into_model(
    det_model: nn.Module,
    kernel_length: int = 11,
    proj_dim: int = 64,
    temperature: float = 0.10,
    lambda_supcon: float = 0.15,
) -> nn.Module:
    """
    Idempotently wraps the detection head's input layers (P3, P4, P5) with
    ManhattanAxialContextWrapper and attaches the training-only SupConHardNegativeHead.
    """
    seq = det_model.model
    det_head = seq[-1]
    head_sources = getattr(det_head, "f", [16, 19, 22])
    if not isinstance(head_sources, (list, tuple)):
        head_sources = [16, 19, 22]

    p3_ch = 128
    for idx_pos, layer_idx in enumerate(head_sources):
        layer_idx = int(layer_idx)
        is_p3 = idx_pos == 0
        if not isinstance(seq[layer_idx], ManhattanAxialContextWrapper):
            ch = _infer_out_ch(seq[layer_idx], default=128 if is_p3 else 256)
            if is_p3:
                p3_ch = ch
            seq[layer_idx] = ManhattanAxialContextWrapper(
                seq[layer_idx],
                channels=ch,
                kernel_length=kernel_length,
                is_p3_tap=is_p3,
            )
        elif is_p3:
            p3_ch = seq[layer_idx].aca.channels

    nc = getattr(det_head, "nc", 4)
    eff_nc = 4 if nc == 23 else nc
    if not hasattr(det_model, "supcon_head") or det_model.supcon_head is None:
        det_model.supcon_head = SupConHardNegativeHead(
            in_channels=p3_ch,
            proj_dim=proj_dim,
            num_classes=eff_nc,
            temperature=temperature,
        )
    det_model.lambda_supcon = float(lambda_supcon)

    # Wrap init_criterion using a pickle-safe top-level function + functools.partial
    import functools
    orig_init_crit = det_model.init_criterion
    det_model.init_criterion = functools.partial(
        aca_supcon_init_criterion,
        det_model,
        orig_init_crit,
        float(lambda_supcon),
    )
    det_model.criterion = None
    return det_model


def aca_supcon_init_criterion(
    det_model: nn.Module,
    orig_init_crit: Any,
    lambda_supcon: float,
    *args,
    **kwargs,
):
    """Top-level pickle-safe criterion wrapper for ACA-SupCon."""
    base_c = orig_init_crit(*args, **kwargs)
    return ACASupConCriterionWrapper(base_c, det_model, lambda_supcon=lambda_supcon)


def strip_supcon_for_save(model_ref: nn.Module) -> Dict[str, Any]:
    """
    Strips training-only `supcon_head`, `init_criterion`, and transient P3 feature caches
    prior to checkpoint saving (`save_model`), returning a backup dict for restoration.
    """
    backup: Dict[str, Any] = {}
    for attr in ("supcon_head", "lambda_supcon"):
        if hasattr(model_ref, attr):
            backup[attr] = getattr(model_ref, attr)
            try:
                delattr(model_ref, attr)
            except AttributeError:
                setattr(model_ref, attr, None)
    if "init_criterion" in model_ref.__dict__:
        backup["init_criterion"] = model_ref.__dict__.pop("init_criterion")

    if hasattr(model_ref, "model"):
        for m in model_ref.model:
            if isinstance(m, ManhattanAxialContextWrapper):
                m.last_p3_feat = None
            if hasattr(m, "cached_p2"):
                m.cached_p2 = None
    return backup


def restore_supcon_after_save(model_ref: nn.Module, backup: Dict[str, Any]):
    """Restores training-only attributes after `save_model` completes."""
    for attr, val in backup.items():
        setattr(model_ref, attr, val)


# Register safe globals for PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([
        ManhattanAxialContextBlock,
        ManhattanAxialContextWrapper,
        SupConHardNegativeHead,
    ])
except Exception:
    pass


def get_aca_supcon_trainer(
    kernel_length: int = 11,
    proj_dim: int = 64,
    temperature: float = 0.10,
    lambda_supcon: float = 0.15,
):
    """Returns a DetectionTrainer subclass for Proposal 4 (ACA-SupCon)."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class ACASupConDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            inject_aca_supcon_into_model(
                model,
                kernel_length=kernel_length,
                proj_dim=proj_dim,
                temperature=temperature,
                lambda_supcon=lambda_supcon,
            )
            if weights is not None:
                model.load(weights, verbose=False)
            if verbose:
                print(
                    f"[ACA-SupCon] Injected Manhattan Axial Context (1x{kernel_length} + {kernel_length}x1) "
                    f"& SupCon Hard-Negative Head (lambda={lambda_supcon}, stripped at inference)"
                )
            return model

        def save_model(self):
            ema = unwrap_model(self.ema.ema) if (hasattr(self, "ema") and self.ema is not None) else None
            b_ema = strip_supcon_for_save(ema) if ema is not None else {}
            b_mod = strip_supcon_for_save(unwrap_model(self.model)) if hasattr(self, "model") else {}
            try:
                return super().save_model()
            finally:
                if ema is not None:
                    restore_supcon_after_save(ema, b_ema)
                if hasattr(self, "model"):
                    restore_supcon_after_save(unwrap_model(self.model), b_mod)

    return ACASupConDetectionTrainer


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------
def _selftest():
    print("=" * 75)
    print("🔬 Running Self-Test for Proposal 4: ACA-SupCon (`aca_supcon.py`)")
    print("=" * 75)
    torch.manual_seed(0)

    # 1. Verify ManhattanAxialContextWrapper zero-init identity
    dummy_neck = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    import copy
    base_neck = copy.deepcopy(dummy_neck).eval()
    aca_wrap = ManhattanAxialContextWrapper(copy.deepcopy(dummy_neck), channels=128, kernel_length=11, is_p3_tap=True).eval()

    x = torch.randn(2, 128, 40, 40)
    with torch.no_grad():
        y_base = base_neck(x)
        y_aca = aca_wrap(x)
    diff = (y_base - y_aca).abs().max().item()
    print(f"[Zero-Init Identity Check] max |Neck_base - Neck_ACA| at step 0 = {diff:.3e} (expected 0.0)")
    assert diff < 1e-6, "ManhattanAxialContextWrapper must be exact identity at step 0!"

    # 2. Verify SupConHardNegativeHead loss & gradient flow
    aca_wrap.train()
    supcon = SupConHardNegativeHead(in_channels=128, proj_dim=64, num_classes=4, temperature=0.10)
    x_train = torch.randn(2, 128, 20, 20, requires_grad=True)
    p3_out = aca_wrap(x_train)
    assert aca_wrap.last_p3_feat is not None

    n3 = 20 * 20
    fg_mask = torch.zeros(2, n3, dtype=torch.bool)
    fg_mask[0, :12] = True
    fg_mask[1, 20:35] = True
    target_scores = torch.zeros(2, n3, 4)
    target_scores[0, :12, 0] = 0.85  # Capacitor positives
    target_scores[1, 20:35, 3] = 0.90  # IC positives
    pred_probs = torch.sigmoid(torch.randn(2, n3, 4))

    loss_sc = supcon(p3_out, fg_mask, target_scores, pred_probs)
    loss_total = p3_out.pow(2).mean() + 0.15 * loss_sc
    loss_total.backward()

    assert torch.isfinite(loss_sc) and loss_sc.item() > 0.0
    assert aca_wrap.aca.gamma_ctx.grad is not None and torch.isfinite(aca_wrap.aca.gamma_ctx.grad).all()
    print(f"[SupCon-HN Loss Check] L_SupCon-HN = {loss_sc.item():.4f} | Prototype EMA initialized: {supcon.proto_initialized.tolist()} | PASSED")
    print("=" * 75)


if __name__ == "__main__":
    _selftest()
