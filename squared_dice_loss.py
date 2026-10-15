"""
squared_dice_loss.py
====================
Implementation of Binary Cross-Entropy (BCE) + Squared-Dice Loss (SDice)
classification loss for dense object detection on PCBs.

Reference Formulation:
    BCE = - 1/N * sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )              [Eq. 1]
    Dice_Loss = 1 - (2 * sum(p_i * y_i) + smooth) / (sum(p_i) + sum(y_i) + smooth)  [Eq. 2]
    Squared_Dice_Loss = 1 - (2 * (sum(p_i * y_i))^2 + smooth) / ((sum(p_i))^2 + (sum(y_i))^2 + smooth)  [Eq. 3]
    Base_Loss = BCE + lambda_dice * Squared_Dice_Loss                           [Eq. 4]

Note on Eq. 4: In the reference paper, Base_Loss is defined with unit weight (weight 1)
on the Dice-type term; we introduce lambda_dice as a tunable weighting coefficient.
Bounding box regression uses standard CIoU by default, with Squared-IoU available
only as an opt-in ablation.
"""

from typing import Any, Callable
import functools
import inspect
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import ultralytics
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors, bbox2dist

print(f"[SquaredDiceTrainer] ultralytics.__version__ = {ultralytics.__version__}")


class SquaredDiceLoss(nn.Module):
    """
    Computes Squared-Dice Loss (Eq. 3):
        L_SDice = 1 - (2 * (sum(p * y))^2 + smooth) / ((sum(p))^2 + (sum(y))^2 + smooth)
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_prob: Sigmoid probabilities of shape (batch, num_anchors, num_classes)
            target: Target assignment scores of shape (batch, num_anchors, num_classes)
        """
        p = pred_prob.float()
        y = target.float()

        # Compute across batch and spatial anchors for each class channel
        dims = (0, 1)
        intersection = (p * y).sum(dim=dims)
        p_sum = p.sum(dim=dims)
        y_sum = y.sum(dim=dims)

        numerator = 2.0 * (intersection ** 2) + self.smooth
        denominator = (p_sum ** 2) + (y_sum ** 2) + self.smooth
        sdice_per_class = 1.0 - (numerator / denominator)
        return sdice_per_class.mean()


class SquaredDiceBboxLoss(BboxLoss):
    """
    Bounding box loss with default plain CIoU (1 - iou).
    Optionally supports Squared-IoU (1 - iou^2 with clamp(max=1.0) only) as an ablation.
    """
    def __init__(self, reg_max: int = 16, squared: bool = False):
        super().__init__(reg_max)
        self.squared = squared

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        if self.squared:
            # Squared-IoU ablation: 1 - iou^2 with clamp(max=1.0) only (no lower clamp)
            iou_clamped = iou.clamp(max=1.0)
            loss_iou = ((1.0 - (iou_clamped ** 2)) * weight).sum() / target_scores_sum
        else:
            # Default faithful behavior: plain CIoU
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class SquaredDiceDetectionLoss(v8DetectionLoss):
    """
    Composite Detection Loss incorporating:
      1. BCE Classification Loss + lambda_dice * Squared-Dice Classification Loss
      2. Bounding Box Loss (plain CIoU by default; Squared-IoU as an opt-in ablation)
      3. Distribution Focal Loss (DFL) or L1 regression for reg_max=1
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        lambda_dice: float = 0.5,
        smooth: float = 1.0,
        squared_box: bool = False,
        **kwargs,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.lambda_dice = lambda_dice
        self.sdice_loss = SquaredDiceLoss(smooth=smooth)
        m = model.model[-1]
        self.bbox_loss = SquaredDiceBboxLoss(reg_max=m.reg_max, squared=squared_box).to(self.device)
        self.last_bce: float = 0.0
        self.last_sdice: float = 0.0
        self._step_count: int = 0

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)  # [box, cls, dfl]
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Ground truth preprocessing
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Task-Aligned Assignment
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 1. Base BCE classification loss
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss_bce = bce_loss.sum() / target_scores_sum

        # 2. Squared-Dice Loss component (Eq. 3 & Eq. 4)
        p_prob = pred_scores.sigmoid()
        sdice = self.sdice_loss(p_prob, target_scores.to(dtype))

        # Store unscaled losses as detached floats and log every ~100 steps
        self.last_bce = float(loss_bce.detach().item())
        self.last_sdice = float(sdice.detach().item())
        self._step_count += 1
        if self._step_count % 100 == 1:
            print(
                f"[SquaredDiceLoss] step={self._step_count} "
                f"bce={self.last_bce:.4f} "
                f"sdice={self.last_sdice:.4f}"
            )

        # Composite Classification: BCE + lambda_dice * SDice
        loss[1] = loss_bce + self.lambda_dice * sdice

        # 3. Bounding Box Loss (plain CIoU or Squared-IoU)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        box_gain = self.hyp.box if hasattr(self.hyp, "box") else (self.hyp.get("box", 7.5) if isinstance(self.hyp, dict) else 7.5)
        cls_gain = self.hyp.cls if hasattr(self.hyp, "cls") else (self.hyp.get("cls", 0.5) if isinstance(self.hyp, dict) else 0.5)
        dfl_gain = self.hyp.dfl if hasattr(self.hyp, "dfl") else (self.hyp.get("dfl", 1.5) if isinstance(self.hyp, dict) else 1.5)
        loss[0] *= box_gain
        loss[1] *= cls_gain
        loss[2] *= dfl_gain

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


def squared_dice_loss_factory(
    m: torch.nn.Module,
    lambda_dice: float = 0.5,
    smooth: float = 1.0,
    squared_box: bool = False,
    **kw,
) -> "SquaredDiceDetectionLoss":
    """Factory creating SquaredDiceDetectionLoss with given hyperparameters."""
    return SquaredDiceDetectionLoss(
        m,
        lambda_dice=lambda_dice,
        smooth=smooth,
        squared_box=squared_box,
        **kw,
    )


def squared_dice_init_criterion(
    model_instance: torch.nn.Module,
    loss_factory: Callable[..., "SquaredDiceDetectionLoss"],
    *args,
    **kwargs,
) -> Any:
    """
    Top-level criterion initialization function (fully pickle-compatible).
    Dispatches to E2ELoss for dual-head models (YOLO26) or single-head loss.
    """
    target_model = model_instance if isinstance(model_instance, torch.nn.Module) else (
        args[0] if (args and isinstance(args[0], torch.nn.Module)) else None
    )
    if target_model is None:
        raise ValueError("squared_dice_init_criterion could not determine model instance")

    if getattr(target_model, "end2end", False):
        try:
            from ultralytics.utils.loss import E2ELoss
        except ImportError as e:
            raise ImportError(
                f"Model has end2end=True, but E2ELoss could not be imported from ultralytics.utils.loss: {e}"
            ) from e

        sig = inspect.signature(E2ELoss.__init__)
        if "loss_fn" not in sig.parameters:
            raise TypeError(
                f"E2ELoss in Ultralytics {ultralytics.__version__} does not accept a 'loss_fn' argument. "
                f"Supported parameters: {list(sig.parameters.keys())}"
            )
        crit = E2ELoss(target_model, loss_fn=loss_factory)
        print(f"[SquaredDiceTrainer] Criterion: {type(crit).__name__}")
        print(f"[SquaredDiceTrainer]   one2one criterion: {type(getattr(crit, 'one2one', None)).__name__}")
        print(f"[SquaredDiceTrainer]   one2many criterion: {type(getattr(crit, 'one2many', None)).__name__}")
        return crit

    crit = loss_factory(target_model)
    print(f"[SquaredDiceTrainer] Criterion: {type(crit).__name__}")
    return crit


class SquaredDiceTrainer(DetectionTrainer):
    """
    DetectionTrainer configured to train with BCE + Squared-Dice Loss.
    Handles dual-head end2end architectures (YOLO26) and standard single-head detectors.
    """
    lambda_dice: float = 0.5
    smooth: float = 1.0
    squared_box: bool = False

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
        lambda_dice: float | None = None,
        smooth: float | None = None,
        squared_box: bool | None = None,
    ):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if lambda_dice is not None:
            self.lambda_dice = lambda_dice
        if smooth is not None:
            self.smooth = smooth
        if squared_box is not None:
            self.squared_box = squared_box

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        loss_factory = functools.partial(
            squared_dice_loss_factory,
            lambda_dice=self.lambda_dice,
            smooth=self.smooth,
            squared_box=self.squared_box,
        )
        model.init_criterion = functools.partial(
            squared_dice_init_criterion,
            model,
            loss_factory,
        )
        return model

    def save_model(self):
        """
        Safely serialize EMA checkpoint by removing any transient instance attributes.
        """
        if hasattr(self, "ema") and self.ema is not None and hasattr(self.ema, "ema"):
            ema_model = getattr(self.ema, "ema")
            if "init_criterion" in ema_model.__dict__:
                del ema_model.__dict__["init_criterion"]
        return super().save_model()


def get_squared_dice_trainer(
    lambda_dice: float = 0.5,
    smooth: float = 1.0,
    squared_box: bool = False,
) -> type[DetectionTrainer]:
    """
    Public factory returning a picklable DetectionTrainer class configured with the
    specified Squared-Dice hyperparameters without relying on module-level globals.
    """
    key = (
        f"SquaredDiceTrainer_l{str(lambda_dice).replace('.', '_')}"
        f"_s{str(smooth).replace('.', '_')}"
        f"_{'sq' if squared_box else 'ciou'}"
    )
    mod = sys.modules[__name__]
    if hasattr(mod, key):
        return getattr(mod, key)

    cls = type(
        key,
        (SquaredDiceTrainer,),
        {
            "lambda_dice": lambda_dice,
            "smooth": smooth,
            "squared_box": squared_box,
            "__module__": __name__,
        },
    )
    setattr(mod, key, cls)
    return cls
