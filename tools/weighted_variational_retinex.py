#!/usr/bin/env python3
"""
tools/weighted_variational_retinex.py

Faithful implementation of:
"A weighted variational model for simultaneous reflectance and illumination estimation"
(CVPR 2016, Xueyang Fu, Delu Zeng, Yue Huang, Xiao-Ping Zhang, Xinghao Ding)
Vectorized in PyTorch using 2D FFT and ADMM optimization.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.fft

# Module-level configuration flag for scalar approximation of spatially varying weights
# (Fu et al., CVPR 2016, Eq. 6 & 8)
use_mean_weights: bool = True

# In-memory cache for FFT derivative operators keyed by (H, W, device_str)
_KERNEL_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def shrink(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft-thresholding / shrinkage operator."""
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def compute_fft_kernel_derivatives(
    h: int,
    w: int,
    device: Union[torch.device, str] = "cpu",
    cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes optical transfer functions of horizontal and vertical forward difference operators
    in the Fourier domain (discrete gradient frequency response and Laplacian denominator).

    Forward difference operator:
        gradient_h(x)[i, j] = x[i, j+1] - x[i, j]  (circularly wrapped)
        gradient_v(x)[i, j] = x[i+1, j] - x[i, j]  (circularly wrapped)

    In circular convolution (y = x * k):
        (x * k)[i, j] = sum_{m, n} x[i - m, j - n] * k[m, n]
    For forward horizontal difference x[i, j+1] - x[i, j]:
        kh[0, 0] = -1.0  ->  -x[i, j]
        kh[0, -1] = 1.0  ->  +x[i, j - (-1)] = +x[i, j+1]
    For forward vertical difference x[i+1, j] - x[i, j]:
        kv[0, 0] = -1.0  ->  -x[i, j]
        kv[-1, 0] = 1.0  ->  +x[i - (-1), j] = +x[i+1, j]
    """
    assert h >= 2 and w >= 2, f"Spatial dimensions must be at least 2x2, got {h}x{w}"

    if not isinstance(device, torch.device):
        device = torch.device(device)
    dev_key = f"{device.type}:{device.index if device.index is not None else 0}"
    cache_key = (h, w, dev_key)

    if cache and cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    # Horizontal difference kernel: kh[0, 0] = -1.0, kh[0, -1] = 1.0
    kh = torch.zeros((h, w), dtype=torch.float32, device=device)
    kh[0, 0] = -1.0
    kh[0, -1] = 1.0
    F_kh = torch.fft.fft2(kh)
    F_kh_conj = torch.conj(F_kh)

    # Vertical difference kernel: kv[0, 0] = -1.0, kv[-1, 0] = 1.0
    kv = torch.zeros((h, w), dtype=torch.float32, device=device)
    kv[0, 0] = -1.0
    kv[-1, 0] = 1.0
    F_kv = torch.fft.fft2(kv)
    F_kv_conj = torch.conj(F_kv)

    # Laplacian denominator: |F_kh|^2 + |F_kv|^2
    laplacian_denom = (F_kh_conj * F_kh + F_kv_conj * F_kv).real

    operators = (F_kh, F_kh_conj, F_kv, F_kv_conj, laplacian_denom)
    if cache:
        _KERNEL_CACHE[cache_key] = operators
    return operators


def gradient_h(x: torch.Tensor) -> torch.Tensor:
    """Periodic forward horizontal gradient: x[..., j+1] - x[..., j]."""
    return torch.roll(x, -1, dims=-1) - x


def gradient_v(x: torch.Tensor) -> torch.Tensor:
    """Periodic forward vertical gradient: x[..., i+1, :] - x[..., i, :]."""
    return torch.roll(x, -1, dims=-2) - x


def decompose_weighted_variational(
    img_tensor: torch.Tensor,
    c1: float = 0.01,
    c2: float = 0.1,
    lambd: float = 1.0,
    gamma: float = 2.2,
    max_iter: int = 15,
    eps1: float = 1e-3,
    eps2: float = 1e-3,
    use_mean_weights: Optional[bool] = None,
    return_info: bool = False,
):
    """
    Decomposes an image into Reflectance (R) and Illumination (L) following Fu et al. (CVPR 2016).
    Supports 2D (H, W), 3D single (1, H, W), or batched (B, H, W) tensors normalized in (0, 1].

    Parameters:
        img_tensor: (H, W), (1, H, W), or (B, H, W) tensor in (0, 1]
        c1: penalty parameter for reflectance ADMM subproblem (paper default: 0.01)
        c2: penalty parameter for illumination subproblem (paper default: 0.1)
        lambd: regularization weight lambda (paper default: 1.0)
        gamma: illumination gamma correction factor for S_enhanced (paper default: 2.2)
        max_iter: maximum ADMM iterations (paper default: 15)
        eps1: relative convergence threshold for r: ||r^k - r^{k-1}|| / ||r^{k-1}|| (paper: 1e-3)
        eps2: relative convergence threshold for l: ||l^k - l^{k-1}|| / ||l^{k-1}|| (paper: 1e-3)
        use_mean_weights: whether to use scalar spatial mean weights in the FFT denominators.
                          If None, falls back to module-level `use_mean_weights` (default: True).
        return_info: if True, returns (R, L, S_enhanced, info_dict)

    Returns:
        R_final: Reflectance tensor, same spatial/batch shape as input, in [0, 1]
        L_final: Illumination tensor, same spatial/batch shape as input, in (0, inf)
        S_enhanced: Enhanced image R * (L / max(L))^(1/gamma), in [0, 1]
        info: (optional) dict with iterations, eps_r, eps_l, converged
    """
    if use_mean_weights is None:
        use_mean_weights_effective = globals()["use_mean_weights"]
    else:
        use_mean_weights_effective = use_mean_weights

    # Handle shape flexibility: (H, W) -> (1, H, W)
    if img_tensor.dim() == 2:
        s_input = img_tensor.unsqueeze(0)
        is_2d = True
    elif img_tensor.dim() == 3:
        s_input = img_tensor
        is_2d = False
    else:
        raise ValueError(f"Expected 2D (H, W) or 3D (B, H, W) tensor, got shape {img_tensor.shape}")

    device = s_input.device
    b, h, w = s_input.shape

    # Epsilon to prevent log(0)
    eps = 1e-4
    S = torch.clamp(s_input, min=eps, max=1.0)
    s = torch.log(S)

    # Precompute / fetch cached FFT derivative operators
    F_kh, F_kh_conj, F_kv, F_kv_conj, denom_diff = compute_fft_kernel_derivatives(h, w, device)

    # Initial state: r^0 = 0, l^0 = s
    r = torch.zeros_like(s)
    l = s.clone()
    R = torch.exp(r)
    L = torch.exp(l)

    b_h = torch.zeros_like(s)
    b_v = torch.zeros_like(s)
    d_h = torch.zeros_like(s)
    d_v = torch.zeros_like(s)

    iters_completed = 0
    final_eps_r = 0.0
    final_eps_l = 0.0
    converged = False

    for k in range(max_iter):
        iters_completed = k + 1
        r_prev = r.clone()
        l_prev = l.clone()

        # ----------------------------------------------------
        # (P1) Update d (Shrinkage operator)
        # ----------------------------------------------------
        grad_r_h = gradient_h(r)
        grad_r_v = gradient_v(r)

        tau = 1.0 / (2.0 * lambd)
        d_h = shrink(R * grad_r_h + b_h, tau)
        d_v = shrink(R * grad_r_v + b_v, tau)

        # ----------------------------------------------------
        # (P2) Update r (Closed-form FFT Least-Squares)
        # ----------------------------------------------------
        # Scalar approximation documentation:
        # In Fu et al. (CVPR 2016) Eq. 6 & 8, the objective contains spatially varying
        # weight matrices W^r = R^{k-1} and W^l = L^{k-1} inside the gradient penalty.
        # Rigorously, a spatially varying weight inside div(W^2 grad(r)) does not diagonalize
        # in the Fourier basis (pointwise multiplication in space is a convolution in frequency).
        # To retain an exact, closed-form O(N log N) FFT inversion per iteration,
        # Fu et al.'s formulation is interpreted by approximating the spatially varying
        # weight operator with its per-image spatial scalar mean (mean_R and mean_L).
        # When use_mean_weights=True, per-image means mean_R = mean(R) and mean_L = mean(L)
        # are used. If use_mean_weights=False, unit weights (1.0) are used in denominators.
        term_d_b_h = d_h - b_h
        term_d_b_v = d_v - b_v
        Phi = F_kh_conj * torch.fft.fft2(term_d_b_h) + F_kv_conj * torch.fft.fft2(term_d_b_v)

        numerator_r = torch.fft.fft2(s - l) + c1 * lambd * Phi
        if use_mean_weights_effective:
            # Per-image scalar mean of shape (B, 1, 1) to support batched evaluation
            mean_R = torch.mean(R, dim=(-2, -1), keepdim=True).clamp(min=1e-3)
            denom_r = 1.0 + c1 * lambd * mean_R * denom_diff
        else:
            denom_r = 1.0 + c1 * lambd * denom_diff

        r = torch.fft.ifft2(numerator_r / denom_r).real
        r = torch.clamp(r, max=0.0)  # Constraint r <= 0 (since R <= 1)
        R = torch.exp(r)

        # Multiplier update
        b_h = b_h + R * gradient_h(r) - d_h
        b_v = b_v + R * gradient_v(r) - d_v

        # ----------------------------------------------------
        # (P3) Update l (Closed-form FFT Illumination Smoothing)
        # ----------------------------------------------------
        numerator_l = torch.fft.fft2(s - r)
        if use_mean_weights_effective:
            mean_L = torch.mean(L, dim=(-2, -1), keepdim=True).clamp(min=1e-3)
            denom_l = 1.0 + c2 * mean_L * denom_diff
        else:
            denom_l = 1.0 + c2 * denom_diff

        l = torch.fft.ifft2(numerator_l / denom_l).real
        l = torch.clamp(l, min=s)  # Constraint s <= l (since S <= L)
        L = torch.exp(l)

        # ----------------------------------------------------
        # Convergence check (Fu et al., Section 4):
        # eps_r = ||r^k - r^{k-1}||_2 / ||r^{k-1}||_2 <= eps1
        # eps_l = ||l^k - l^{k-1}||_2 / ||l^{k-1}||_2 <= eps2
        # Guard denominator against division by zero (e.g. at k=0 when r^0=0)
        # ----------------------------------------------------
        diff_r_norm = torch.linalg.norm((r - r_prev).reshape(b, -1), dim=1)
        prev_r_norm = torch.linalg.norm(r_prev.reshape(b, -1), dim=1)
        eps_r_batch = diff_r_norm / (prev_r_norm + 1e-12)

        diff_l_norm = torch.linalg.norm((l - l_prev).reshape(b, -1), dim=1)
        prev_l_norm = torch.linalg.norm(l_prev.reshape(b, -1), dim=1)
        eps_l_batch = diff_l_norm / (prev_l_norm + 1e-12)

        max_eps_r = float(eps_r_batch.max().item())
        max_eps_l = float(eps_l_batch.max().item())
        final_eps_r = max_eps_r
        final_eps_l = max_eps_l

        if max_eps_r <= eps1 and max_eps_l <= eps2:
            converged = True
            break

    # Final estimated components
    R_final = torch.clamp(torch.exp(r), 0.0, 1.0)
    L_final = torch.clamp(torch.exp(l), min=eps)

    # Gamma correction on Illumination (Fu et al., Eq. 9 and 11, W=1):
    # S_enhanced = R * L_norm^(1/gamma), with L normalized to (0, 1]
    L_max = torch.amax(L_final, dim=(-2, -1), keepdim=True).clamp(min=eps)
    L_norm = torch.clamp(L_final / L_max, min=eps, max=1.0)
    L_prime = torch.pow(L_norm, 1.0 / gamma)
    S_enhanced = torch.clamp(R_final * L_prime, 0.0, 1.0)

    if is_2d:
        R_final = R_final.squeeze(0)
        L_final = L_final.squeeze(0)
        S_enhanced = S_enhanced.squeeze(0)

    if return_info:
        info = {
            "iterations": iters_completed,
            "eps_r": final_eps_r,
            "eps_l": final_eps_l,
            "converged": converged,
        }
        return R_final, L_final, S_enhanced, info

    return R_final, L_final, S_enhanced


def process_rgb_image(
    img_bgr: np.ndarray,
    device: Union[torch.device, str] = "cpu",
    c1: float = 0.01,
    c2: float = 0.1,
    lambd: float = 1.0,
    gamma: float = 2.2,
    max_iter: int = 15,
    eps1: float = 1e-3,
    eps2: float = 1e-3,
    use_mean_weights: Optional[bool] = None,
    return_info: bool = False,
):
    """
    Processes an RGB image in HSV V-channel using Fu et al. (CVPR 2016) weighted variational Retinex.
    Preserves color balance and hue while equalizing illumination and boosting shadowed details.

    Parameters:
        img_bgr: Input BGR image (uint8 numpy array)
        device: PyTorch device ('cpu' or 'cuda')
        c1, c2, lambd, gamma, max_iter, eps1, eps2: Decomposition parameters forwarded to decompose_weighted_variational
        use_mean_weights: Optional override for scalar approximation of denominator weights
        return_info: If True, returns a 4-tuple (rgb_reflectance, L_v_np, rgb_enhanced, info)

    Returns:
        rgb_reflectance: Pure Reflectance RGB image (intrinsic surface albedo)
        L_v_np: Estimated illumination V-channel (uint8, 0-255)
        rgb_enhanced: Illumination-normalized enhanced RGB image
        info: (optional) Decomposition convergence dictionary
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Process V-channel in HSV domain (preserves color balance, fast)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_norm = img_hsv[:, :, 2] / 255.0

    v_tensor = torch.from_numpy(v_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        decomp_result = decompose_weighted_variational(
            v_tensor,
            c1=c1,
            c2=c2,
            lambd=lambd,
            gamma=gamma,
            max_iter=max_iter,
            eps1=eps1,
            eps2=eps2,
            use_mean_weights=use_mean_weights,
            return_info=return_info,
        )

    if return_info:
        R_v, L_v, S_enh_v, info = decomp_result
    else:
        R_v, L_v, S_enh_v = decomp_result
        info = None

    R_v_np = (R_v.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    L_v_np = (L_v.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    S_enh_v_np = (S_enh_v.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Reconstruct enhanced RGB image
    hsv_enh = img_hsv.copy()
    hsv_enh[:, :, 2] = S_enh_v_np
    hsv_enh = np.clip(hsv_enh, 0, 255).astype(np.uint8)
    rgb_enhanced = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2RGB)

    # Reconstruct pure Reflectance RGB image (intrinsic surface albedo)
    hsv_refl = img_hsv.copy()
    hsv_refl[:, :, 2] = R_v_np
    hsv_refl = np.clip(hsv_refl, 0, 255).astype(np.uint8)
    rgb_reflectance = cv2.cvtColor(hsv_refl, cv2.COLOR_HSV2RGB)

    if return_info:
        return rgb_reflectance, L_v_np, rgb_enhanced, info

    return rgb_reflectance, L_v_np, rgb_enhanced


def process_rgb_batch(
    images_bgr: List[np.ndarray],
    device: Union[torch.device, str] = "cpu",
    c1: float = 0.01,
    c2: float = 0.1,
    lambd: float = 1.0,
    gamma: float = 2.2,
    max_iter: int = 15,
    eps1: float = 1e-3,
    eps2: float = 1e-3,
    use_mean_weights: Optional[bool] = None,
) -> List[np.ndarray]:
    """
    Batched processing of equal-sized RGB images in HSV V-channel.
    Returns list of enhanced BGR images.
    """
    if not images_bgr:
        return []

    # Check if all images have the same shape
    shapes = [img.shape for img in images_bgr]
    if len(set(shapes)) > 1:
        # Fall back to individual processing if sizes differ
        out_list = []
        for img in images_bgr:
            _, _, enh = process_rgb_image(
                img,
                device=device,
                c1=c1,
                c2=c2,
                lambd=lambd,
                gamma=gamma,
                max_iter=max_iter,
                eps1=eps1,
                eps2=eps2,
                use_mean_weights=use_mean_weights,
            )
            out_list.append(cv2.cvtColor(enh, cv2.COLOR_RGB2BGR))
        return out_list

    # Convert to HSV and extract V channels
    hsv_list = [cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV).astype(np.float32) for img in images_bgr]
    v_stack = np.stack([hsv[:, :, 2] / 255.0 for hsv in hsv_list], axis=0)  # (B, H, W)
    v_tensor = torch.from_numpy(v_stack).to(device)

    with torch.no_grad():
        _, _, S_enh = decompose_weighted_variational(
            v_tensor,
            c1=c1,
            c2=c2,
            lambd=lambd,
            gamma=gamma,
            max_iter=max_iter,
            eps1=eps1,
            eps2=eps2,
            use_mean_weights=use_mean_weights,
        )

    S_enh_np = (S_enh.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    enhanced_bgr_list = []
    for i, hsv in enumerate(hsv_list):
        hsv_copy = hsv.copy()
        hsv_copy[:, :, 2] = S_enh_np[i]
        hsv_copy = np.clip(hsv_copy, 0, 255).astype(np.uint8)
        rgb_enh = cv2.cvtColor(hsv_copy, cv2.COLOR_HSV2RGB)
        bgr_enh = cv2.cvtColor(rgb_enh, cv2.COLOR_RGB2BGR)
        enhanced_bgr_list.append(bgr_enh)

    return enhanced_bgr_list


if __name__ == "__main__":
    sample_path = Path("data_samples/native_images/Arty_Top_jpg.rf.7bc260a89099530b771500c983e1669e.jpg")
    if sample_path.exists():
        print(f"Reading: {sample_path}")
        img = cv2.imread(str(sample_path))
        t0 = time.time()
        refl, illum, enh, info = process_rgb_image(
            img,
            device="cuda" if torch.cuda.is_available() else "cpu",
            c1=0.01,
            c2=0.1,
            lambd=1.0,
            gamma=2.2,
            max_iter=15,
            eps1=1e-3,
            eps2=1e-3,
            return_info=True,
        )
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.3f}s: {info}")

        out_dir = Path("results/variational_retinex_demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "reflectance.jpg"), cv2.cvtColor(refl, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "illumination.jpg"), illum)
        cv2.imwrite(str(out_dir / "enhanced.jpg"), cv2.cvtColor(enh, cv2.COLOR_RGB2BGR))
        print(f"✅ Successfully decomposed and saved to: {out_dir}")
