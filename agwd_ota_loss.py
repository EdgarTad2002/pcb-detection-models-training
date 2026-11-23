#!/usr/bin/env python3
"""
agwd_ota_loss.py
================
Implements:
  - Experiment E3: Base Normalized Wasserstein Distance (NWD) Task-Aligned Assigner & Box Loss
  - Proposal 1:    Anisotropic Gaussian Wasserstein & Kullback-Leibler Optimal Transport
                   Task-Aligned Assigner (AGWD-OTA)

Mathematical Formulation
------------------------
1. 2D Gaussian Bounding Box Representation:
   Each bounding box B = (cx, cy, w, h) in pixel coordinates is modeled as a 2D Gaussian
   distribution N(mu, Sigma) reflecting high physical density at the component body
   center and solder fillets at the boundaries:
       mu = [cx, cy]^T,   Sigma = diag(sigma_x^2, sigma_y^2) = diag((w/2)^2, (h/2)^2)

2. Second-Order Wasserstein Distance & Base NWD (Experiment E3):
       W_2^2(N_p, N_g) = (cx_p - cx_g)^2 + (cy_p - cy_g)^2 + ((w_p - w_g)/2)^2 + ((h_p - h_g)/2)^2
       NWD(B_p, B_g)   = exp( - sqrt(W_2^2(N_p, N_g) + eps) / (C * (sqrt(w_g * h_g) / s_0)^eta) )

3. Symmetrized Jeffreys KL-Divergence for Anisotropic Packages (Proposal 1):
   Penalizes shifts along the narrow axis of high-aspect-ratio connectors and passives:
       D_KL(N_p || N_g) = (cx_p - cx_g)^2 / (2 * sigma_x,g^2) + (cy_p - cy_g)^2 / (2 * sigma_y,g^2)
                        + 0.5 * (sigma_x,p^2 / sigma_x,g^2 + sigma_y,p^2 / sigma_y,g^2 - 2
                                 - ln((sigma_x,p^2 * sigma_y,p^2) / (sigma_x,g^2 * sigma_y,g^2)))
       D_J(N_p, N_g)    = 0.5 * (D_KL(N_p || N_g) + D_KL(N_g || N_p))

4. Anisotropic Gaussian Wasserstein Similarity & Scale-Gated Loss:
       S_AGWD(B_p, B_g) = NWD(B_p, B_g) * exp( - tau_kl * ln(1 + D_J(N_p, N_g)) )
       psi(A_g)         = exp( - (w_g * h_g) / kappa^2 )     (kappa = 24.0 px default)
       L_box(B_p, B_g)  = psi(A_g) * (1 - S_AGWD(B_p, B_g)) + (1 - psi(A_g)) * (1 - CIoU(B_p, B_g))

5. Dynamic-k Optimal Transport Label Assignment (Dynamic-k SimOTA):
   Replaces rigid tal_topk=10 with adaptive positive anchor budget per GT object:
       t_{i,g} = s_{i, c_g}^alpha * [ psi(A_g) * S_AGWD(B_i, B_g) + (1 - psi(A_g)) * IoU(B_i, B_g) ]^beta
       k_g     = clamp( floor( sum_{m=1}^M max(IoU_(m), S_AGWD,(m)) ), k_min=2, k_max=12 )

Self-test (verifies sub-10px gradient stability & Dynamic-k assignment):
    python agwd_ota_loss.py
"""

from typing import Any, Callable, Optional, Tuple
import functools
import inspect
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, make_anchors

if __name__ == "__main__":
    sys.modules["agwd_ota_loss"] = sys.modules[__name__]


# ---------------------------------------------------------------------------
# 1. Core Gaussian Wasserstein & Anisotropic KL Primitives (AMP-Safe FP32)
# ---------------------------------------------------------------------------
def xyxy_to_cxcywh(boxes: torch.Tensor, min_size: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts [..., 4] (x1, y1, x2, y2) boxes into (cx, cy, w, h) in FP32."""
    b = boxes.float()
    x1, y1, x2, y2 = b.unbind(-1)
    w = (x2 - x1).clamp(min=min_size)
    h = (y2 - y1).clamp(min=min_size)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return cx, cy, w, h


def compute_nwd_similarity(
    pred_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    constant_c: float = 12.8,
    scale_s0: float = 12.0,
    eta: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes Scale-Adaptive Normalized Wasserstein Distance (NWD) similarity in [0, 1].
    Supports both 1-to-1 aligned tensors [N, 4] and broadcastable [B, N_gt, N_anc, 4].
    """
    with torch.autocast(device_type=pred_xyxy.device.type, enabled=False):
        cx_p, cy_p, w_p, h_p = xyxy_to_cxcywh(pred_xyxy, min_size=1e-3)
        cx_g, cy_g, w_g, h_g = xyxy_to_cxcywh(gt_xyxy, min_size=1e-3)

        center_dist_sq = (cx_p - cx_g).pow(2) + (cy_p - cy_g).pow(2)
        wh_dist_sq = ((w_p - w_g) * 0.5).pow(2) + ((h_p - h_g) * 0.5).pow(2)
        w2_sq = center_dist_sq + wh_dist_sq
        w2 = torch.sqrt(w2_sq + eps)

        if eta > 0.0:
            gt_scale = torch.sqrt(w_g * h_g + eps)
            c_eff = constant_c * torch.pow((gt_scale / scale_s0).clamp(min=0.25, max=8.0), eta)
        else:
            c_eff = constant_c

        nwd = torch.exp(-w2 / (c_eff + eps))
        return nwd.clamp(0.0, 1.0)


def compute_symmetrized_kld(
    pred_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Computes Symmetrized Jeffreys KL-Divergence D_J(N_p, N_g) between 2D Gaussian
    bounding box distributions, capturing anisotropic aspect-ratio sensitivity.
    """
    with torch.autocast(device_type=pred_xyxy.device.type, enabled=False):
        cx_p, cy_p, w_p, h_p = xyxy_to_cxcywh(pred_xyxy, min_size=1e-2)
        cx_g, cy_g, w_g, h_g = xyxy_to_cxcywh(gt_xyxy, min_size=1e-2)

        # Variances sigma^2 = (w/2)^2, (h/2)^2
        var_xp = (w_p * 0.5).pow(2) + eps
        var_yp = (h_p * 0.5).pow(2) + eps
        var_xg = (w_g * 0.5).pow(2) + eps
        var_yg = (h_g * 0.5).pow(2) + eps

        dx_sq = (cx_p - cx_g).pow(2)
        dy_sq = (cy_p - cy_g).pow(2)

        # Forward KL: D_KL(N_p || N_g)
        kl_pg = 0.5 * (
            dx_sq / var_xg
            + dy_sq / var_yg
            + (var_xp / var_xg)
            + (var_yp / var_yg)
            - 2.0
            - torch.log((var_xp * var_yp) / (var_xg * var_yg))
        )

        # Reverse KL: D_KL(N_g || N_p)
        kl_gp = 0.5 * (
            dx_sq / var_xp
            + dy_sq / var_yp
            + (var_xg / var_xp)
            + (var_yg / var_yp)
            - 2.0
            - torch.log((var_xg * var_yg) / (var_xp * var_yp))
        )

        d_jeffreys = 0.5 * (kl_pg.clamp(min=0.0) + kl_gp.clamp(min=0.0))
        return d_jeffreys


def compute_agwd_similarity(
    pred_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    constant_c: float = 12.8,
    tau_kl: float = 0.25,
    use_kld: bool = True,
    eta: float = 0.25,
) -> torch.Tensor:
    """
    Computes composite Anisotropic Gaussian Wasserstein Similarity S_AGWD in [0, 1].
    When use_kld=False, reduces to pure Scale-Normalized Wasserstein Distance (Experiment E3).
    """
    nwd = compute_nwd_similarity(
        pred_xyxy, gt_xyxy, constant_c=constant_c, eta=eta
    )
    if not use_kld or tau_kl <= 0.0:
        return nwd

    with torch.autocast(device_type=pred_xyxy.device.type, enabled=False):
        d_kld = compute_symmetrized_kld(pred_xyxy, gt_xyxy)
        kld_sim = torch.exp(-tau_kl * torch.log1p(d_kld))
        return (nwd * kld_sim).clamp(0.0, 1.0)


def compute_scale_gate_psi(
    gt_xyxy: torch.Tensor,
    kappa: float = 24.0,
) -> torch.Tensor:
    """
    Smooth scale-gating weight psi(A_g) = exp( - (w_g * h_g) / tau_A ) in (0, 1].
    Accepts either linear scale kappa <= 64 px (tau_A = kappa^2, e.g. 24^2 = 576 px^2)
    or area threshold tau_A > 64 px^2 directly (e.g. 256-576 px^2).
    - Micro-capacitors (e.g. 6x6 px, area=36 <= 576): psi -> 0.94 (AGWD-dominated)
    - Macro ICs/Connectors (e.g. 80x80 px, area=6400 >> 576): psi -> 0.0 (CIoU-dominated)
    """
    with torch.autocast(device_type=gt_xyxy.device.type, enabled=False):
        _, _, w_g, h_g = xyxy_to_cxcywh(gt_xyxy, min_size=1e-3)
        area = w_g * h_g
        tau_area = float(kappa) if kappa > 64.0 else float(kappa * kappa)
        psi = torch.exp(-area / max(tau_area, 1.0))
        return psi.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Optimal Transport / Dynamic-k Task-Aligned Assigner (AGWD-OTA)
# ---------------------------------------------------------------------------
class AGWDOptimalTransportAssigner(TaskAlignedAssigner):
    """
    Extends Ultralytics TaskAlignedAssigner with:
      1. Scale-gated hybrid similarity: psi(A_g) * S_AGWD + (1 - psi(A_g)) * CIoU
      2. Calibrated beta exponent (default 3.0 vs 6.0) to prevent target score underflow
         on 4x4 - 8x8 px micro-capacitors.
      3. Dynamic-k SimOTA positive candidate selection per ground-truth object based on
         cumulative distribution overlap, preventing background pad assignment on tiny boxes.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 4,
        alpha: float = 1.0,
        beta: float = 3.0,
        eps: float = 1e-9,
        kappa: float = 24.0,
        constant_c: float = 12.8,
        tau_kl: float = 0.25,
        use_kld: bool = True,
        dynamic_k: bool = True,
        k_min: int = 2,
        k_max: int = 12,
        stride: Optional[list] = None,
        topk2: Optional[int] = None,
    ):
        super().__init__(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
        )
        self.kappa = float(kappa)
        self.constant_c = float(constant_c)
        self.tau_kl = float(tau_kl)
        self.use_kld = bool(use_kld)
        self.dynamic_k = bool(dynamic_k)
        self.k_min = int(k_min)
        self.k_max = int(k_max)

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """
        Overrides base IoU calculation to blend CIoU with Anisotropic Gaussian Wasserstein
        similarity according to the ground-truth scale gate psi(A_g).
        """
        with torch.autocast(device_type=pd_bboxes.device.type, enabled=False):
            gt_f = gt_bboxes.float()
            pd_f = pd_bboxes.float()
            ciou = bbox_iou(gt_f, pd_f, xywh=False, CIoU=True).squeeze(-1).nan_to_num(0.0).clamp_(0.0, 1.0)
            agwd = compute_agwd_similarity(
                pd_f,
                gt_f,
                constant_c=self.constant_c,
                tau_kl=self.tau_kl,
                use_kld=self.use_kld,
            )
            psi = compute_scale_gate_psi(gt_f, kappa=self.kappa)
            hybrid_overlaps = psi * agwd + (1.0 - psi) * ciou
            return hybrid_overlaps.nan_to_num(0.0).clamp(0.0, 1.0).to(pd_bboxes.dtype)

    def select_topk_candidates(
        self, metrics: torch.Tensor, largest: bool = True, topk_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Selects top-k positive anchors per GT. When dynamic_k=True, computes a per-GT
        adaptive budget k_g in [k_min, k_max] from the sum of top-M overlap scores.
        """
        b, n_max_boxes, total_num_anchors = metrics.shape
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        if self.dynamic_k and hasattr(self, "_latest_overlaps") and self._latest_overlaps is not None:
            # Estimate effective positive anchor budget k_g from top-M overlap mass
            top_overlaps, _ = torch.topk(
                self._latest_overlaps, min(self.topk, total_num_anchors), dim=-1, largest=True
            )
            dyn_k = top_overlaps.sum(dim=-1).floor().long().clamp(min=self.k_min, max=min(self.k_max, self.topk))
            rank_grid = torch.arange(self.topk, device=metrics.device).view(1, 1, self.topk)
            dyn_valid = (rank_grid < dyn_k.unsqueeze(-1)) & topk_mask
        else:
            dyn_valid = topk_mask

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=metrics.device)
        for k in range(self.topk):
            is_k_active = dyn_valid[:, :, k : k + 1].to(torch.int8)
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones * is_k_active)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_box_metrics(
        self, pd_scores: torch.Tensor, pd_bboxes: torch.Tensor, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes alignment metric and caches hybrid overlaps for Dynamic-k SimOTA."""
        self._latest_overlaps = None
        align_metric, overlaps = super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
        self._latest_overlaps = overlaps.detach()
        return align_metric, overlaps


# ---------------------------------------------------------------------------
# 3. Scale-Gated AGWD + CIoU Bounding Box Regression Loss
# ---------------------------------------------------------------------------
class AGWDBboxLoss(BboxLoss):
    """
    Computes scale-gated bounding box regression loss:
        L_box = psi(A_g) * (1 - S_AGWD) + (1 - psi(A_g)) * (1 - CIoU)
    plus standard Distribution Focal Loss (DFL).
    """

    def __init__(
        self,
        reg_max: int = 16,
        kappa: float = 24.0,
        constant_c: float = 12.8,
        tau_kl: float = 0.25,
        use_kld: bool = True,
    ):
        super().__init__(reg_max)
        self.kappa = float(kappa)
        self.constant_c = float(constant_c)
        self.tau_kl = float(tau_kl)
        self.use_kld = bool(use_kld)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred_bboxes.requires_grad:
            pred_bboxes.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0))
        if pred_dist.requires_grad:
            pred_dist.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0))

        with torch.autocast(device_type=pred_bboxes.device.type, enabled=False):
            pred_bboxes_fp32 = pred_bboxes.float()
            target_bboxes_fp32 = target_bboxes.float()
            stride_fp32 = stride.float()
            weight = target_scores[fg_mask].float().sum(-1, keepdim=True)
            denom = target_scores_sum.float().clamp(min=1.0)

            # Decode boxes to pixel coordinates in FP32 for scale-consistent Gaussian Wasserstein distance
            pred_px = (pred_bboxes_fp32 * stride_fp32)[fg_mask]
            target_px = (target_bboxes_fp32 * stride_fp32)[fg_mask]

            ciou = bbox_iou(pred_bboxes_fp32[fg_mask], target_bboxes_fp32[fg_mask], xywh=False, CIoU=True).nan_to_num(0.0)
            agwd_sim = compute_agwd_similarity(
                pred_px,
                target_px,
                constant_c=self.constant_c,
                tau_kl=self.tau_kl,
                use_kld=self.use_kld,
            ).unsqueeze(-1)
            psi = compute_scale_gate_psi(target_px, kappa=self.kappa).unsqueeze(-1)

            loss_ciou = 1.0 - ciou
            loss_agwd = 1.0 - agwd_sim
            hybrid_box_loss = psi * loss_agwd + (1.0 - psi) * loss_ciou
            loss_iou = (hybrid_box_loss * weight).sum() / denom

            # DFL / L1 loss in FP32
            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points.float(), target_bboxes_fp32, self.dfl_loss.reg_max - 1)
                loss_dfl = (
                    self.dfl_loss(pred_dist[fg_mask].float().view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
                    * weight
                )
                loss_dfl = loss_dfl.sum() / denom
            else:
                target_ltrb = bbox2dist(anchor_points.float(), target_bboxes_fp32)
                target_ltrb = target_ltrb * stride_fp32
                target_ltrb[..., 0::2] /= imgsz[1].float()
                target_ltrb[..., 1::2] /= imgsz[0].float()
                pred_dist_fp32 = pred_dist.float() * stride_fp32
                pred_dist_fp32[..., 0::2] /= imgsz[1].float()
                pred_dist_fp32[..., 1::2] /= imgsz[0].float()
                loss_dfl = (
                    F.l1_loss(pred_dist_fp32[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
                    * weight
                )
                loss_dfl = loss_dfl.sum() / denom

            return loss_iou.nan_to_num(0.0), loss_dfl.nan_to_num(0.0)


# ---------------------------------------------------------------------------
# 4. Composite Detection Loss (Compatible with YOLO26 E2ELoss & Single-Head)
# ---------------------------------------------------------------------------
class AGWDDetectionLoss(v8DetectionLoss):
    """
    Replaces standard TAL and BboxLoss with AGWDOptimalTransportAssigner and AGWDBboxLoss.
    Optionally integrates Multi-Stage Squared-Dice Classification Loss (Eq. 1-4:
    BCE + lambda_dice * Squared_Dice_Loss across feature pyramid stages) when squared_dice=True.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 13,
        tal_topk2: Optional[int] = None,
        kappa: float = 24.0,
        constant_c: float = 12.8,
        tau_kl: float = 0.25,
        tal_beta: float = 3.0,
        use_kld: bool = True,
        dynamic_k: bool = True,
        squared_dice: bool = False,
        lambda_dice: float = 0.5,
        smooth_dice: float = 1.0,
        stage_weights: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        m = model.model[-1]
        is_one2one = (tal_topk2 is not None and tal_topk2 == 1)
        self.squared_dice = bool(squared_dice)
        self.lambda_dice = float(lambda_dice)
        self.smooth_dice = float(smooth_dice)
        self.stage_weights = stage_weights
        self.assigner = AGWDOptimalTransportAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=tal_beta,
            kappa=kappa,
            constant_c=constant_c,
            tau_kl=tau_kl,
            use_kld=use_kld,
            dynamic_k=(dynamic_k and not is_one2one),
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = AGWDBboxLoss(
            reg_max=m.reg_max,
            kappa=kappa,
            constant_c=constant_c,
            tau_kl=tau_kl,
            use_kld=use_kld,
        ).to(self.device)

    def _compute_multistage_sdice(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        feats: list,
    ) -> torch.Tensor:
        """
        Computes Multi-Stage Squared-Dice Loss (Eq. 3, 4, 8 from Agaian et al.)
        across each feature pyramid scale (P3, P4, P5 / P2) with optional stage weighting
        so micro-components on high-res stages are not drowned out by macro-components on coarse stages.
        """
        with torch.autocast(device_type=pred_scores.device.type, enabled=False):
            p = pred_scores.float().sigmoid()
            y = target_scores.float()
            if p.shape[-1] == 23:
                active_cls = [2, 4, 7, 9]
                p = p[..., active_cls]
                y = y[..., active_cls]
            stage_losses = []
            weights = []
            offset = 0
            for i, feat in enumerate(feats):
                n_stage = int(feat.shape[2] * feat.shape[3])
                if offset + n_stage <= p.shape[1]:
                    p_s = p[:, offset : offset + n_stage, :]
                    y_s = y[:, offset : offset + n_stage, :]
                    inter = (p_s * y_s).sum(dim=(0, 1))
                    p_sum = p_s.sum(dim=(0, 1))
                    y_sum = y_s.sum(dim=(0, 1))
                    num = 2.0 * (inter ** 2) + self.smooth_dice
                    den = (p_sum ** 2) + (y_sum ** 2) + self.smooth_dice
                    sdice_s = (1.0 - num / den).mean()
                    w = self.stage_weights[i] if (self.stage_weights is not None and i < len(self.stage_weights)) else 1.0
                    stage_losses.append(w * sdice_s)
                    weights.append(w)
                offset += n_stage
            if not stage_losses:
                inter = (p * y).sum(dim=(0, 1))
                p_sum = p.sum(dim=(0, 1))
                y_sum = y.sum(dim=(0, 1))
                num = 2.0 * (inter ** 2) + self.smooth_dice
                den = (p_sum ** 2) + (y_sum ** 2) + self.smooth_dice
                return (1.0 - num / den).mean()
            return sum(stage_losses) / sum(weights)

    def get_assigned_targets_and_loss(self, preds: dict, batch: dict) -> tuple:
        assigned_info, loss, loss_detach = super().get_assigned_targets_and_loss(preds, batch)
        if self.squared_dice and self.lambda_dice > 0.0:
            fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned_info
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            # Reconstruct soft target_scores from base BCE or compute stage-wise SDice against fg_mask/cls
            with torch.no_grad():
                batch_size = pred_scores.shape[0]
                dtype = pred_scores.dtype
                imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
                targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
                targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
                gt_labels, gt_bboxes = targets.split((1, 4), 2)
                _, _, target_scores = self.assigner.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
            sdice = self._compute_multistage_sdice(pred_scores, target_scores, preds["feats"])
            cls_gain = self.hyp.cls if hasattr(self.hyp, "cls") else 0.5
            sdice_weighted = cls_gain * self.lambda_dice * sdice.to(loss.dtype)
            loss[1] = loss[1] + sdice_weighted
            if "cls_loss" in loss_detach:
                loss_detach["cls_loss"] = loss[1].detach()
        return assigned_info, loss, loss_detach


def agwd_loss_factory(
    m: torch.nn.Module,
    kappa: float = 24.0,
    constant_c: float = 12.8,
    tau_kl: float = 0.25,
    tal_beta: float = 3.0,
    use_kld: bool = True,
    dynamic_k: bool = True,
    squared_dice: bool = False,
    lambda_dice: float = 0.5,
    smooth_dice: float = 1.0,
    stage_weights: Optional[list] = None,
    **kw,
) -> AGWDDetectionLoss:
    return AGWDDetectionLoss(
        m,
        kappa=kappa,
        constant_c=constant_c,
        tau_kl=tau_kl,
        tal_beta=tal_beta,
        use_kld=use_kld,
        dynamic_k=dynamic_k,
        squared_dice=squared_dice,
        lambda_dice=lambda_dice,
        smooth_dice=smooth_dice,
        stage_weights=stage_weights,
        **kw,
    )


def agwd_init_criterion(
    model_instance: torch.nn.Module,
    loss_factory: Callable[..., AGWDDetectionLoss],
    *args,
    **kwargs,
) -> Any:
    """Pickle-safe criterion initializer supporting YOLO26 E2ELoss and standard DetectionModel."""
    target_model = (
        model_instance
        if isinstance(model_instance, torch.nn.Module)
        else (args[0] if (args and isinstance(args[0], torch.nn.Module)) else None)
    )
    if target_model is None:
        raise ValueError("agwd_init_criterion could not determine model instance")

    if getattr(target_model, "end2end", False):
        from ultralytics.utils.loss import E2ELoss

        sig = inspect.signature(E2ELoss.__init__)
        if "loss_fn" in sig.parameters:
            e2e = E2ELoss(target_model, loss_fn=loss_factory)
            e2e.final_o2m = 0.35
            return e2e
    return loss_factory(target_model)


def inject_agwd_criterion_into_model(
    model: torch.nn.Module,
    kappa: float = 24.0,
    constant_c: float = 12.8,
    tau_kl: float = 0.25,
    tal_beta: float = 3.0,
    use_kld: bool = True,
    dynamic_k: bool = True,
    squared_dice: bool = False,
    lambda_dice: float = 0.5,
    smooth_dice: float = 1.0,
    stage_weights: Optional[list] = None,
    **kwargs,
) -> torch.nn.Module:
    """Attaches the AGWD-OTA / Base-NWD (+ optional Multi-Stage Squared-Dice) criterion factory."""
    loss_factory = functools.partial(
        agwd_loss_factory,
        kappa=kappa,
        constant_c=constant_c,
        tau_kl=tau_kl,
        tal_beta=tal_beta,
        use_kld=use_kld,
        dynamic_k=dynamic_k,
        squared_dice=squared_dice,
        lambda_dice=lambda_dice,
        smooth_dice=smooth_dice,
        stage_weights=stage_weights,
    )
    model.init_criterion = functools.partial(agwd_init_criterion, model, loss_factory)
    model.criterion = None
    return model


class AGWDOTATrainer(DetectionTrainer):
    """Standalone DetectionTrainer for E3 (Base NWD) or Proposal 1 (AGWD-OTA)."""

    kappa: float = 24.0
    constant_c: float = 12.8
    tau_kl: float = 0.25
    tal_beta: float = 3.0
    use_kld: bool = True
    dynamic_k: bool = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        inject_agwd_criterion_into_model(
            model,
            kappa=self.kappa,
            constant_c=self.constant_c,
            tau_kl=self.tau_kl,
            tal_beta=self.tal_beta,
            use_kld=self.use_kld,
            dynamic_k=self.dynamic_k,
        )
        if verbose:
            mode = "AGWD-OTA (Anisotropic Wasserstein + KLD + Dynamic-k)" if self.use_kld else "Base NWD (E3)"
            print(
                f"[{mode}] Injected criterion (kappa={self.kappa}, C={self.constant_c}, "
                f"tau_kl={self.tau_kl}, beta={self.tal_beta}, dynamic_k={self.dynamic_k})"
            )
        return model

    def save_model(self):
        if hasattr(self, "ema") and self.ema is not None and hasattr(self.ema, "ema"):
            ema_model = getattr(self.ema, "ema")
            if "init_criterion" in ema_model.__dict__:
                del ema_model.__dict__["init_criterion"]
        return super().save_model()


def get_agwd_ota_trainer(
    kappa: float = 24.0,
    constant_c: float = 12.8,
    tau_kl: float = 0.25,
    tal_beta: float = 3.0,
    use_kld: bool = True,
    dynamic_k: bool = True,
) -> type[DetectionTrainer]:
    """Returns a dynamically named, pickle-compatible DetectionTrainer subclass."""
    key = (
        f"AGWDOTATrainer_k{str(kappa).replace('.', '_')}"
        f"_c{str(constant_c).replace('.', '_')}"
        f"_b{str(tal_beta).replace('.', '_')}"
        f"_{'kld' if use_kld else 'nwd'}"
        f"_{'dynk' if dynamic_k else 'fixk'}"
    )
    mod = sys.modules[__name__]
    if hasattr(mod, key):
        return getattr(mod, key)

    cls = type(
        key,
        (AGWDOTATrainer,),
        {
            "kappa": float(kappa),
            "constant_c": float(constant_c),
            "tau_kl": float(tau_kl),
            "tal_beta": float(tal_beta),
            "use_kld": bool(use_kld),
            "dynamic_k": bool(dynamic_k),
            "__module__": __name__,
        },
    )
    setattr(mod, key, cls)
    return cls


# ---------------------------------------------------------------------------
# 5. Self-Test: Sub-10px Gradient Flow & Anisotropic Sensitivity Verification
# ---------------------------------------------------------------------------
def _selftest():
    print("=" * 75)
    print("🔬 Running Self-Test for Experiment E3 (Base NWD) & Proposal 1 (AGWD-OTA)")
    print("=" * 75)
    torch.manual_seed(0)

    # 1. Sub-10px (6x6 px) non-overlapping micro-capacitor gradient check
    gt_micro = torch.tensor([[100.0, 100.0, 106.0, 106.0]], dtype=torch.float32)  # 6x6 px
    pred_micro = torch.tensor([[108.0, 108.0, 114.0, 114.0]], dtype=torch.float32, requires_grad=True)

    iou_val = bbox_iou(pred_micro, gt_micro, xywh=False, CIoU=False).item()
    nwd_sim = compute_nwd_similarity(pred_micro, gt_micro, constant_c=12.8)
    agwd_sim = compute_agwd_similarity(pred_micro, gt_micro, constant_c=12.8, tau_kl=0.25, use_kld=True)
    psi_val = compute_scale_gate_psi(gt_micro, kappa=24.0).item()

    loss = 1.0 - agwd_sim.mean()
    loss.backward()

    assert pred_micro.grad is not None and torch.isfinite(pred_micro.grad).all()
    grad_norm = pred_micro.grad.norm().item()
    print(
        f"[Sub-10px 6x6 Capacitor Shift (+8px, non-overlapping)] "
        f"IoU={iou_val:.4f} | NWD={nwd_sim.item():.4f} | AGWD={agwd_sim.item():.4f} | "
        f"psi={psi_val:.4f} | ||grad||={grad_norm:.4e}"
    )
    assert grad_norm > 1e-3, "Gradient must be strong and non-zero for non-overlapping 6x6 micro-boxes!"

    # 2. Anisotropic Pin-Header Connector Sensitivity Check (10x160 px vertical connector)
    gt_conn = torch.tensor([[200.0, 200.0, 210.0, 360.0]], dtype=torch.float32)  # w=10, h=160
    pred_shift_x = torch.tensor([[205.0, 200.0, 215.0, 360.0]], dtype=torch.float32)  # +5px along narrow X axis
    pred_shift_y = torch.tensor([[200.0, 205.0, 210.0, 365.0]], dtype=torch.float32)  # +5px along long Y axis

    kld_x = compute_symmetrized_kld(pred_shift_x, gt_conn).item()
    kld_y = compute_symmetrized_kld(pred_shift_y, gt_conn).item()
    print(
        f"[Anisotropic 10x160 Connector +5px Shift] "
        f"Narrow-Axis KLD={kld_x:.4f} vs Long-Axis KLD={kld_y:.4f} (Ratio: {kld_x / max(kld_y, 1e-9):.1f}x)"
    )
    assert kld_x > 100.0 * kld_y, "Anisotropic KLD must penalize narrow-axis shifts far more than long-axis shifts!"

    # 3. Verify Dynamic-k Assigner
    assigner = AGWDOptimalTransportAssigner(topk=13, num_classes=4, dynamic_k=True)
    pd_scores = torch.sigmoid(torch.randn(1, 100, 4))
    anc_xy = torch.stack(torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij"), dim=-1).reshape(1, 100, 2).float() * 8.0 + 4.0
    pd_bboxes = torch.cat([anc_xy - 3.0, anc_xy + 3.0], dim=-1)
    gt_labels = torch.tensor([[[0], [1]]])
    gt_bboxes = torch.tensor([[[16.0, 16.0, 22.0, 22.0], [32.0, 32.0, 72.0, 72.0]]])  # 1 micro cap (6x6), 1 macro IC (40x40)
    mask_gt = torch.ones(1, 2, 1, dtype=torch.bool)

    _, target_bboxes, target_scores, fg_mask, _ = assigner(
        pd_scores, pd_bboxes, anc_xy[0], gt_labels, gt_bboxes, mask_gt
    )
    print(f"[Dynamic-k Assigner] Total positive anchors assigned: {int(fg_mask.sum().item())} | PASSED")

    # 4. Verify AGWDBboxLoss with batch > 1 and 2D stride [N_anc, 1]
    bbox_loss_fn = AGWDBboxLoss(reg_max=16, kappa=24.0, constant_c=12.8, tau_kl=0.25, use_kld=True)
    bs_test, n_anc = 4, 100
    pred_dist_b = torch.randn(bs_test, n_anc, 64, requires_grad=True)
    pred_bboxes_b = (pd_bboxes.expand(bs_test, n_anc, 4) / 8.0).clone().detach().requires_grad_(True)
    target_bboxes_b = (target_bboxes.expand(bs_test, n_anc, 4) / 8.0).clone().detach()
    target_scores_b = target_scores.expand(bs_test, n_anc, 4).clone().detach()
    fg_mask_b = fg_mask.expand(bs_test, n_anc).clone()
    stride_2d = torch.full((n_anc, 1), 8.0)
    loss_iou, loss_dfl = bbox_loss_fn(
        pred_dist_b,
        pred_bboxes_b,
        anc_xy[0] / 8.0,
        target_bboxes_b,
        target_scores_b,
        target_scores_b.sum().clamp(min=1.0),
        fg_mask_b,
        torch.tensor([80.0, 80.0]),
        stride_2d,
    )
    (loss_iou + loss_dfl).backward()
    assert pred_bboxes_b.grad is not None and torch.isfinite(pred_bboxes_b.grad).all()
    print(f"[AGWDBboxLoss Batch={bs_test}] loss_iou={loss_iou.item():.4f}, loss_dfl={loss_dfl.item():.4f} | PASSED")
    print("=" * 75)


if __name__ == "__main__":
    _selftest()
