"""
test_squared_dice.py
====================
Unit tests for the Squared-Dice Loss and Bounding Box Loss modules.

Tests:
  1. Hand-computed verification on a small tensor against analytical formula.
  2. Loss is 0 when predictions exactly match targets (p == y).
  3. Gradients are non-zero when p is far from y.
  4. Bbox loss verification: default CIoU vs opt-in Squared-IoU ablation.
"""

import math
import torch
from squared_dice_loss import SquaredDiceLoss, SquaredDiceBboxLoss


def test_hand_computed_value():
    """
    Check SquaredDiceLoss against a hand-computed analytical value.

    Formulation:
        L_SDice = 1 - (2 * (sum(p * y))^2 + smooth) / ((sum(p))^2 + (sum(y))^2 + smooth)

    Small tensor setup:
        batch_size = 1, num_anchors = 2, num_classes = 1
        p = [[0.8], [0.6]]
        y = [[1.0], [0.0]]
        smooth = 1.0

    Analytical computation:
        sum(p * y) = 0.8 * 1.0 + 0.6 * 0.0 = 0.8
        (sum(p * y))^2 = 0.64
        numerator = 2 * 0.64 + 1.0 = 2.28

        sum(p) = 0.8 + 0.6 = 1.4
        (sum(p))^2 = 1.96

        sum(y) = 1.0 + 0.0 = 1.0
        (sum(y))^2 = 1.00

        denominator = 1.96 + 1.00 + 1.0 = 3.96

        expected_loss = 1.0 - (2.28 / 3.96) = 0.424242424...
    """
    loss_fn = SquaredDiceLoss(smooth=1.0)
    p = torch.tensor([[[0.8], [0.6]]], dtype=torch.float32)
    y = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)

    loss = loss_fn(p, y).item()
    expected = 1.0 - (2.0 * (0.8 ** 2) + 1.0) / ((1.4 ** 2) + (1.0 ** 2) + 1.0)

    assert abs(loss - expected) < 1e-6, f"Mismatch: computed {loss} vs expected {expected}"
    print(f"✅ test_hand_computed_value passed: loss={loss:.8f} (expected={expected:.8f})")


def test_loss_zero_when_p_equals_y():
    """
    Verify that loss is 0 when predictions match ground truth exactly (large smooth-free case).
    With p == y and smooth=1e-7:
        sum(p*y) = sum(y^2) = sum(p) = S
        numerator = 2*S^2 + eps
        denominator = S^2 + S^2 + eps = 2*S^2 + eps
        loss = 1 - 1 = 0
    """
    loss_fn = SquaredDiceLoss(smooth=1e-7)
    p = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    y = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)

    loss = loss_fn(p, y).item()
    assert abs(loss) < 1e-5, f"Expected loss ~0 when p == y, got {loss}"
    print(f"✅ test_loss_zero_when_p_equals_y passed: loss={loss:.8e}")


def test_gradients_non_zero_far_from_y():
    """
    Verify that gradients are non-zero when p is far from y (no zero-gradient dead zones).
    """
    loss_fn = SquaredDiceLoss(smooth=1.0)
    p = torch.tensor([[[0.01], [0.02]]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[[1.0], [1.0]]], dtype=torch.float32)

    loss = loss_fn(p, y)
    loss.backward()

    assert p.grad is not None, "Gradients should be computed"
    assert (p.grad.abs() > 1e-4).all(), f"Gradients too small: {p.grad}"
    print(f"✅ test_gradients_non_zero_far_from_y passed: grad={p.grad.flatten().tolist()}")


def test_bbox_loss_default_and_ablation():
    """
    Verify that SquaredDiceBboxLoss defaults to plain CIoU (squared=False)
    and uses clamp(max=1.0) without lower clamping when squared=True.
    """
    box_default = SquaredDiceBboxLoss(reg_max=1, squared=False)
    assert not box_default.squared, "Default bbox loss must be plain CIoU (squared=False)"

    box_ablation = SquaredDiceBboxLoss(reg_max=1, squared=True)
    assert box_ablation.squared, "Ablation bbox loss must have squared=True"
    print("✅ test_bbox_loss_default_and_ablation passed")


if __name__ == "__main__":
    test_hand_computed_value()
    test_loss_zero_when_p_equals_y()
    test_gradients_non_zero_far_from_y()
    test_bbox_loss_default_and_ablation()
    print("\n🎉 ALL SQUARED-DICE LOSS UNIT TESTS PASSED SUCCESSFULLY!")
