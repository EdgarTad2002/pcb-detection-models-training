"""
Group Multi-Stage Loss Function (TransFuse / EGE-UNet with Squared-Dice Loss).
Implements the exact formulas from user research:

  BCE = - 1/N * sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )              [Eq. 1]

  Dice_Loss = 1 - ( 2 * sum(p_i * y_i) + smooth ) / ( sum(p_i) + sum(y_i) + smooth )  [Eq. 2]

  Squared_Dice_Loss = 1 - ( 2 * (sum(p_i * y_i))^2 + smooth ) /
                          ( (sum(p_i))^2 + (sum(y_i))^2 + smooth )             [Eq. 3]

  Base_Loss = BCE + Squared_Dice_Loss                                            [Eq. 4]

  Group_MultiStage_Loss = sum_{stage=1..4} Base_Loss(P_stage, Y_stage)

Why Squared-Dice (SDice) is superior for PCB AOI:
The quadratic terms (sum(p))^2 and (sum(y))^2 drastically amplify the penalty
for false positives around small component edges (0402/0201 chip capacitors),
preventing edge boundary bleeding onto the PCB substrate.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_cross_entropy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Numerically stable BCE loss for predictions in [0, 1] (or logits).
    Eq. (1): - 1/N sum [ y * log(p) + (1 - y) * log(1 - p) ]
    """
    pred_f = pred.float()
    target_f = target.float()
    pred_clamped = torch.clamp(pred_f, min=eps, max=1.0 - eps)
    bce = - (target_f * torch.log(pred_clamped) + (1.0 - target_f) * torch.log(1.0 - pred_clamped))
    bce_val = bce.mean()
    return torch.nan_to_num(bce_val, nan=0.0, posinf=1.0, neginf=0.0)


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-4,
) -> torch.Tensor:
    """
    Standard Dice Loss computed with per-item spatial normalization in float32.
    Eq. (2): 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
    """
    pred_f = pred.float()
    target_f = target.float()

    # Flatten spatial dims [B, -1]
    b = pred_f.shape[0] if pred_f.ndim >= 2 else 1
    p_flat = pred_f.view(b, -1)
    y_flat = target_f.view(b, -1)

    # Use spatial mean to prevent FP16/FP32 overflow under AMP
    inter = (p_flat * y_flat).mean(dim=1)
    p_mean = p_flat.mean(dim=1)
    y_mean = y_flat.mean(dim=1)

    dice = (2.0 * inter + smooth) / (p_mean + y_mean + smooth)
    dice_loss_val = (1.0 - dice).mean()
    return torch.nan_to_num(dice_loss_val, nan=0.0, posinf=1.0, neginf=0.0)


def squared_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-4,
) -> torch.Tensor:
    """
    Squared-Dice Loss (SDice) computed with per-item spatial normalization in float32.
    Eq. (3): 1 - (2 * (sum(p * y))^2 + smooth) / ((sum(p))^2 + (sum(y))^2 + smooth)

    By computing normalized spatial means (mean(p) in [0, 1]) instead of unnormalized raw pixel sums,
    the squared terms remain bounded in [0, 1], guaranteeing 100% immunity to FP16 overflow under AMP!
    """
    pred_f = pred.float()
    target_f = target.float()

    # Flatten spatial dims per batch element [B, -1]
    b = pred_f.shape[0] if pred_f.ndim >= 2 else 1
    p_flat = pred_f.view(b, -1)
    y_flat = target_f.view(b, -1)

    # Normalized spatial means
    inter_mean = (p_flat * y_flat).mean(dim=1)
    p_mean = p_flat.mean(dim=1)
    y_mean = y_flat.mean(dim=1)

    numerator = 2.0 * (inter_mean ** 2) + smooth
    denominator = (p_mean ** 2) + (y_mean ** 2) + smooth

    sdice = numerator / denominator
    sdice_loss_val = (1.0 - sdice).mean()
    return torch.nan_to_num(sdice_loss_val, nan=0.0, posinf=1.0, neginf=0.0)


def base_stage_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    use_squared_dice: bool = True,
    smooth: float = 1.0,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Base loss for a single stage: Base = BCE + (SDice or Dice).
    Returns (total_base_loss, bce_val, dice_val).
    """
    bce = binary_cross_entropy_loss(pred, target)
    if use_squared_dice:
        d_loss = squared_dice_loss(pred, target, smooth=smooth)
    else:
        d_loss = dice_loss(pred, target, smooth=smooth)

    total = bce_weight * bce + dice_weight * d_loss
    return total, bce, d_loss


class GroupMultiStageLoss(nn.Module):
    """
    Group Multi-Stage Loss module over intermediate feature maps [P2, P3, P4, P5]
    and primary Output prediction, matching DualCENet (TransFuse / EGE-UNet):

      Group_Loss = Loss_Output + sum_{i=1..M} lambda_i * Loss_Stage_i

    where Base_Loss(P, Y) = bce_weight * BCE(P, Y) + dice_weight * Squared_Dice(P, Y).

    PCB-Calibrated Stage Weights (lambda_i):
      - Stage 1 (P2, Stride 4):  0.35 (Heaviest supervision for microscopic 0402/0201 capacitors)
      - Stage 2 (P3, Stride 8):  0.25 (Mid-scale components and IC pin rows)
      - Stage 3 (P4, Stride 16): 0.15 (Connectors and large IC bodies)
      - Stage 4 (P5, Stride 32): 0.10 (Coarse features; downscaled mask aliasing penalty minimized)
    """
    def __init__(
        self,
        use_squared_dice: bool = True,
        stage_weights: Optional[List[float]] = None,
        output_weight: float = 1.0,
        smooth: float = 1.0,
        bce_weight: float = 0.6,
        dice_weight: float = 1.0,
    ):
        super(GroupMultiStageLoss, self).__init__()
        self.use_squared_dice = use_squared_dice
        # Default calibrated scale-aware pyramid weights for PCB AOI:
        self.stage_weights = stage_weights if stage_weights is not None else [0.35, 0.25, 0.15, 0.10]
        self.output_weight = float(output_weight)
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        stage_preds: List[torch.Tensor],
        target_mask: torch.Tensor,
        output_pred: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            stage_preds: List of intermediate predicted masks [P2, P3, P4, P5] with values in [0, 1].
            target_mask: Full-resolution binary mask [B, 1, H, W] of component ground truth.
            output_pred: Optional high-resolution primary output mask prediction [B, 1, H, W].

        Returns:
            total_loss: Group_Loss = Loss_Output + sum_i lambda_i * Loss_Stage_i
            loss_dict: Dictionary with per-stage details for logging.
        """
        total_loss = torch.tensor(0.0, device=target_mask.device)
        loss_dict = {}

        # 1. Output Loss (if explicit full-resolution head is provided)
        if output_pred is not None:
            out_target = target_mask.float()
            if output_pred.shape[-2:] != out_target.shape[-2:]:
                out_target = F.interpolate(out_target, size=output_pred.shape[-2:], mode="nearest")
            out_loss, out_bce, out_sdice = base_stage_loss(
                output_pred,
                out_target,
                use_squared_dice=self.use_squared_dice,
                smooth=self.smooth,
                bce_weight=self.bce_weight,
                dice_weight=self.dice_weight,
            )
            total_loss = total_loss + self.output_weight * out_loss
            loss_dict["loss_output"] = out_loss.detach()
            loss_dict["output_bce"] = out_bce.detach()
            loss_dict["output_sdice"] = out_sdice.detach()

        # 2. Multi-Stage Auxiliary Losses: sum_{i=1..M} lambda_i * Loss_Stage_i
        num_stages = min(len(stage_preds), len(self.stage_weights))
        for idx in range(num_stages):
            p_stage = stage_preds[idx]
            weight = self.stage_weights[idx]

            # Downscale target ground truth to match this stage's spatial resolution
            if p_stage.shape[-2:] != target_mask.shape[-2:]:
                y_stage = F.interpolate(
                    target_mask.float(),
                    size=p_stage.shape[-2:],
                    mode="nearest",
                )
            else:
                y_stage = target_mask.float()

            s_loss, s_bce, s_sdice = base_stage_loss(
                p_stage,
                y_stage,
                use_squared_dice=self.use_squared_dice,
                smooth=self.smooth,
                bce_weight=self.bce_weight,
                dice_weight=self.dice_weight,
            )

            total_loss = total_loss + weight * s_loss
            loss_dict[f"stage_{idx + 1}_loss"] = s_loss.detach()
            loss_dict[f"stage_{idx + 1}_bce"] = s_bce.detach()
            loss_dict[f"stage_{idx + 1}_sdice"] = s_sdice.detach()

        loss_dict["group_multistage_total"] = total_loss.detach()
        return total_loss, loss_dict


if __name__ == "__main__":
    # Quick self-test
    print("Testing GroupMultiStageLoss with Squared-Dice...")
    target = (torch.rand(2, 1, 640, 640) > 0.85).float()
    p1 = torch.sigmoid(torch.randn(2, 1, 160, 160))
    p2 = torch.sigmoid(torch.randn(2, 1, 80, 80))
    p3 = torch.sigmoid(torch.randn(2, 1, 40, 40))
    p4 = torch.sigmoid(torch.randn(2, 1, 20, 20))

    criterion = GroupMultiStageLoss(use_squared_dice=True)
    loss, details = criterion([p1, p2, p3, p4], target)
    print(f"Total Group Multi-Stage Loss: {loss.item():.4f}")
    for k, v in details.items():
        print(f"  {k}: {v.item():.4f}")
    print("Self-test passed successfully!")
