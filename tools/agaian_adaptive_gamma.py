"""
Agaian Adaptive Gamma Correction (AGC) for PCB AOI Illumination Balancing.
Based on Eq. (23) of IEEE Access 2023 Paper (DOI: 10.1109/ACCESS.2023.3344534):
"Adaptive Single Low-Light Image Enhancement by Fractional Stretching in Logarithmic Domain"
(Trongtirakul, Agaian, and Wu, 2023):

    f(L*) = L*_max * ( L* / L*_max )^gamma

where:
    gamma = ln( c(L*) / ( p(L*)^(-1) + 2 ) ) / ln(2)

c(L*) : Image-dependent Cumulative Distribution Function (CDF).
p(L*) : Image-dependent Probability Density Function (PDF).
L*     : Luma / Illumination channel (in CIELAB or Retinex Illumination L).

Supports both NumPy (offline dataset generation) and PyTorch (online batch training).
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None


def apply_agaian_agc_numpy(
    luma: np.ndarray,
    gamma_min: float = 0.25,
    gamma_max: float = 2.2,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Agaian Adaptive Gamma Correction to a 2D luma/illumination array.

    Args:
        luma: 2D array (H, W) in range [0, 1] or [0, 255].
        gamma_min: Lower bound for gamma (maximum shadow boost).
        gamma_max: Upper bound for gamma (prevent highlight over-darkening).
        eps: Small constant for numerical stability.

    Returns:
        enhanced_luma: Enhanced 2D array matching the input dtype and scale.
        gamma_map: Spatial 2D map of computed gamma exponents.
    """
    is_uint8 = luma.dtype == np.uint8
    if is_uint8:
        l_float = luma.astype(np.float32) / 255.0
    else:
        l_float = np.clip(luma.astype(np.float32), 0.0, 1.0)

    l_max = float(np.max(l_float))
    if l_max < 1e-4:
        return (luma, np.ones_like(l_float))

    # 1. Compute 256-bin histogram for empirical PDF and CDF
    hist, bin_edges = np.histogram(l_float, bins=256, range=(0.0, 1.0))
    pdf = hist.astype(np.float32) / float(l_float.size)
    cdf = np.cumsum(pdf)

    # 2. Map pixel intensities to bin indices [0, 255]
    bin_indices = np.clip((l_float * 255.0).astype(np.int32), 0, 255)
    p_L = pdf[bin_indices]
    c_L = cdf[bin_indices]

    # 3. Agaian Formula (Eq. 23):
    # gamma = ln( c(L*) / ( p(L*)^(-1) + 2 ) ) / ln(2)
    p_inv = 1.0 / np.maximum(p_L, eps)
    denom = p_inv + 2.0
    ratio = np.maximum(c_L / denom, eps)

    # Scale gamma smoothly to range [gamma_min, gamma_max]
    raw_gamma = np.log(ratio) / np.log(2.0)
    # When ratio < 1 (shadows), raw_gamma is negative. We map it via sigmoid/clamping:
    # A standard AGC mapping centered around 1.0:
    gamma_map = np.clip(1.0 + 0.35 * raw_gamma, gamma_min, gamma_max)

    # 4. Exponentiation: f(L*) = L*_max * (L* / L*_max)^gamma
    norm_l = l_float / l_max
    enhanced = l_max * np.power(np.maximum(norm_l, eps), gamma_map)
    enhanced = np.clip(enhanced, 0.0, 1.0)

    if is_uint8:
        return (enhanced * 255.0).astype(np.uint8), gamma_map
    return enhanced, gamma_map


def enhance_image_cielab_agc(
    img_bgr: np.ndarray,
    gamma_min: float = 0.25,
    gamma_max: float = 2.2,
) -> np.ndarray:
    """
    Converts BGR to CIELAB, applies Agaian AGC exclusively to the L* (lightness) channel,
    and converts back to BGR. Keeps chrominance (a*, b*) 100% intact to eliminate color distortion.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    l_enhanced, _ = apply_agaian_agc_numpy(l_chan, gamma_min=gamma_min, gamma_max=gamma_max)
    lab[:, :, 0] = l_enhanced
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_agaian_agc_torch(
    luma_tensor: torch.Tensor,
    gamma_min: float = 0.25,
    gamma_max: float = 2.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable PyTorch implementation of Agaian AGC for batch tensors [B, 1, H, W] in [0, 1].
    """
    b, c, h, w = luma_tensor.shape
    num_pixels = float(h * w)

    enhanced_list = []
    for i in range(b):
        img_single = luma_tensor[i, 0]
        l_max = torch.max(img_single)
        if l_max < 1e-4:
            enhanced_list.append(img_single.unsqueeze(0))
            continue

        # Soft histogram approximation via 64 bins
        bins = torch.linspace(0.0, 1.0, steps=64, device=luma_tensor.device)
        bin_idx = torch.clamp((img_single * 63.0).long(), 0, 63)
        hist = torch.bincount(bin_idx.flatten(), minlength=64).float()
        pdf = hist / num_pixels
        cdf = torch.cumsum(pdf, dim=0)

        p_L = pdf[bin_idx]
        c_L = cdf[bin_idx]

        p_inv = 1.0 / (p_L + eps)
        denom = p_inv + 2.0
        ratio = torch.clamp(c_L / denom, min=eps)

        raw_gamma = torch.log(ratio) / math.log(2.0)
        gamma = torch.clamp(1.0 + 0.35 * raw_gamma, min=gamma_min, max=gamma_max)

        norm_l = img_single / l_max
        enhanced = l_max * torch.pow(torch.clamp(norm_l, min=eps), gamma)
        enhanced_list.append(torch.clamp(enhanced, 0.0, 1.0).unsqueeze(0))

    return torch.stack(enhanced_list, dim=0)


import math
