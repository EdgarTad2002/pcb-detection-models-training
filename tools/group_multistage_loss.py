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
    pred_clamped = torch.clamp(pred, min=eps, max=1.0 - eps)
    bce = - (target * torch.log(pred_clamped) + (1.0 - target) * torch.log(1.0 - pred_clamped))
    return bce.mean()


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Standard Dice Loss.
    Eq. (2): 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    cardinality = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice


def squared_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Squared-Dice Loss (SDice).
    Eq. (3): 1 - (2 * (sum(p * y))^2 + smooth) / ((sum(p))^2 + (sum(y))^2 + smooth)
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    sum_pred = pred_flat.sum()
    sum_target = target_flat.sum()

    numerator = 2.0 * (intersection ** 2) + smooth
    denominator = (sum_pred ** 2) + (sum_target ** 2) + smooth

    sdice = numerator / denominator
    return 1.0 - sdice


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
    Group Multi-Stage Loss module over intermediate feature maps [P1, P2, P3, P4].
    Computes Base_Loss at each stage against downscaled target ground truth masks.
    """
    def __init__(
        self,
        use_squared_dice: bool = True,
        stage_weights: Optional[List[float]] = None,
        smooth: float = 1.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super(GroupMultiStageLoss, self).__init__()
        self.use_squared_dice = use_squared_dice
        self.stage_weights = stage_weights if stage_weights else [1.0, 1.0, 1.0, 1.0]
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        stage_preds: List[torch.Tensor],
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            stage_preds: List of 4 predicted masks [P1, P2, P3, P4] with values in [0, 1].
            target_mask: Full-resolution binary mask [B, 1, H, W] of component ground truth.

        Returns:
            total_loss: Weighted sum of stage losses.
            loss_dict: Dictionary with per-stage details for logging.
        """
        total_loss = torch.tensor(0.0, device=target_mask.device)
        loss_dict = {}

        for idx, (p_stage, weight) in enumerate(zip(stage_preds, self.stage_weights), start=1):
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
            loss_dict[f"stage_{idx}_loss"] = s_loss.detach()
            loss_dict[f"stage_{idx}_bce"] = s_bce.detach()
            loss_dict[f"stage_{idx}_sdice"] = s_sdice.detach()

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
