#!/usr/bin/env python3
"""
tests/test_retinex_operators.py

Unit tests for weighted variational Retinex discrete operators,
Fourier transfer functions, adjoint properties, batching consistency,
and convergence criteria.
"""

import numpy as np
import pytest
import torch
import torch.fft

from tools.weighted_variational_retinex import (
    compute_fft_kernel_derivatives,
    decompose_weighted_variational,
    gradient_h,
    gradient_v,
    process_rgb_batch,
    process_rgb_image,
)


def test_spatial_dimension_assertion():
    """Verify that h < 2 or w < 2 raises an explicit AssertionError."""
    with pytest.raises(AssertionError):
        compute_fft_kernel_derivatives(1, 10, device="cpu", cache=False)

    with pytest.raises(AssertionError):
        compute_fft_kernel_derivatives(10, 1, device="cpu", cache=False)

    with pytest.raises(AssertionError):
        compute_fft_kernel_derivatives(0, 0, device="cpu", cache=False)


def test_forward_differences_fft_match():
    """
    Verify on a random float32 (H, W) tensor that:
      ifft2(F_kh * fft2(x)).real == gradient_h(x) (atol 1e-5)
      ifft2(F_kv * fft2(x)).real == gradient_v(x) (atol 1e-5)
    """
    torch.manual_seed(42)
    h, w = 64, 48
    x = torch.randn((h, w), dtype=torch.float32)

    F_kh, F_kh_conj, F_kv, F_kv_conj, _ = compute_fft_kernel_derivatives(h, w, device="cpu", cache=False)

    # Horizontal forward difference
    fft_x = torch.fft.fft2(x)
    grad_h_fft = torch.fft.ifft2(F_kh * fft_x).real
    grad_h_spatial = gradient_h(x)

    assert torch.allclose(grad_h_fft, grad_h_spatial, atol=1e-5), (
        f"Horizontal gradient mismatch! Max diff: {torch.max(torch.abs(grad_h_fft - grad_h_spatial)).item()}"
    )

    # Vertical forward difference
    grad_v_fft = torch.fft.ifft2(F_kv * fft_x).real
    grad_v_spatial = gradient_v(x)

    assert torch.allclose(grad_v_fft, grad_v_spatial, atol=1e-5), (
        f"Vertical gradient mismatch! Max diff: {torch.max(torch.abs(grad_v_fft - grad_v_spatial)).item()}"
    )


def test_adjoint_operator_inner_product():
    """
    Adjoint check:
      <gradient_h(x), y> == <x, ifft2(F_kh_conj * fft2(y)).real> (rtol 1e-4)
      <gradient_v(x), y> == <x, ifft2(F_kv_conj * fft2(y)).real> (rtol 1e-4)
    """
    torch.manual_seed(123)
    h, w = 50, 70
    x = torch.randn((h, w), dtype=torch.float32)
    y = torch.randn((h, w), dtype=torch.float32)

    F_kh, F_kh_conj, F_kv, F_kv_conj, _ = compute_fft_kernel_derivatives(h, w, device="cpu", cache=False)

    # Horizontal adjoint test
    grad_h_x = gradient_h(x)
    adj_h_y = torch.fft.ifft2(F_kh_conj * torch.fft.fft2(y)).real

    inner_prod_h_1 = torch.sum(grad_h_x * y)
    inner_prod_h_2 = torch.sum(x * adj_h_y)

    assert torch.allclose(inner_prod_h_1, inner_prod_h_2, rtol=1e-4, atol=1e-5), (
        f"Horizontal adjoint failed! <Ax, y>={inner_prod_h_1.item()} != <x, A*y>={inner_prod_h_2.item()}"
    )

    # Vertical adjoint test
    grad_v_x = gradient_v(x)
    adj_v_y = torch.fft.ifft2(F_kv_conj * torch.fft.fft2(y)).real

    inner_prod_v_1 = torch.sum(grad_v_x * y)
    inner_prod_v_2 = torch.sum(x * adj_v_y)

    assert torch.allclose(inner_prod_v_1, inner_prod_v_2, rtol=1e-4, atol=1e-5), (
        f"Vertical adjoint failed! <Ax, y>={inner_prod_v_1.item()} != <x, A*y>={inner_prod_v_2.item()}"
    )


def test_batch_consistency():
    """
    Verify that batched execution (B, H, W) gives identical results
    to unbatched slice-by-slice execution due to per-image scalar means.
    """
    torch.manual_seed(999)
    b, h, w = 3, 32, 32
    batch_img = torch.rand((b, h, w), dtype=torch.float32).clamp(min=0.05, max=1.0)

    # Batched run
    R_batch, L_batch, S_enh_batch = decompose_weighted_variational(
        batch_img,
        c1=0.01,
        c2=0.1,
        lambd=1.0,
        gamma=2.2,
        max_iter=5,
    )

    # Sequential individual runs
    for i in range(b):
        single_img = batch_img[i : i + 1]
        R_single, L_single, S_enh_single = decompose_weighted_variational(
            single_img,
            c1=0.01,
            c2=0.1,
            lambd=1.0,
            gamma=2.2,
            max_iter=5,
        )
        assert torch.allclose(R_batch[i : i + 1], R_single, atol=1e-6)
        assert torch.allclose(L_batch[i : i + 1], L_single, atol=1e-6)
        assert torch.allclose(S_enh_batch[i : i + 1], S_enh_single, atol=1e-6)


def test_process_rgb_batch_matches_single():
    """Verify that process_rgb_batch matches sequential process_rgb_image calls."""
    np.random.seed(42)
    img1 = np.random.randint(20, 240, (32, 32, 3), dtype=np.uint8)
    img2 = np.random.randint(20, 240, (32, 32, 3), dtype=np.uint8)

    # Batched
    batch_enh = process_rgb_batch([img1, img2], max_iter=5)

    # Single
    import cv2
    _, _, single1 = process_rgb_image(img1, max_iter=5)
    _, _, single2 = process_rgb_image(img2, max_iter=5)
    single1_bgr = cv2.cvtColor(single1, cv2.COLOR_RGB2BGR)
    single2_bgr = cv2.cvtColor(single2, cv2.COLOR_RGB2BGR)

    assert np.allclose(batch_enh[0], single1_bgr, atol=1)
    assert np.allclose(batch_enh[1], single2_bgr, atol=1)


def test_convergence_stopping_rule_and_return_info():
    """
    Verify that decompose_weighted_variational respects convergence thresholds,
    stops when eps_r and eps_l are below tolerance, and returns info dict.
    """
    torch.manual_seed(101)
    img = torch.rand((1, 32, 32), dtype=torch.float32).clamp(min=0.1, max=0.9)

    # Test with lenient tolerance (should stop before max_iter=20)
    R, L, S_enh, info = decompose_weighted_variational(
        img,
        max_iter=20,
        eps1=0.1,
        eps2=0.1,
        return_info=True,
    )

    assert "iterations" in info
    assert "eps_r" in info
    assert "eps_l" in info
    assert "converged" in info
    assert info["iterations"] <= 20
    assert info["converged"] is True


def test_process_rgb_image_parameter_plumbing():
    """
    Verify that process_rgb_image accepts and forwards c1, c2, lambd, gamma, max_iter, eps1, eps2, return_info.
    """
    np_img = np.random.randint(0, 256, (40, 50, 3), dtype=np.uint8)

    refl, illum, enh, info = process_rgb_image(
        np_img,
        device="cpu",
        c1=0.02,
        c2=0.2,
        lambd=1.5,
        gamma=2.0,
        max_iter=4,
        eps1=1e-3,
        eps2=1e-3,
        return_info=True,
    )

    assert refl.shape == np_img.shape
    assert illum.shape == (40, 50)
    assert enh.shape == np_img.shape
    assert isinstance(info, dict)
    assert info["iterations"] <= 4
