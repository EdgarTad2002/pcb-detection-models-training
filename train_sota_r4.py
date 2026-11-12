#!/usr/bin/env python3
"""
SOTA Trainer Wrapper for pcb-native-reviewed-r4-final:
Injects Identity-Preserving Multi-Scale Center-Surround Retinex +
Anisotropic Sargis-Scharr Edge Pre-Conditioning into Ultralytics Trainer & Validator
while preserving 100% of COCO-pretrained backbone weights and standard best.pt export.
"""
import sys
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

import train as base_train


class RetinexSargisEdgeTransform:
    """Fast GPU Multi-Scale Retinex + Orthogonal Scharr ('Sargis') Edge Residual Conditioner."""

    def __init__(self, ret_alpha: float = 0.10, edge_alpha: float = 0.08, eps: float = 1e-4):
        self.ret_alpha = float(ret_alpha)
        self.edge_alpha = float(edge_alpha)
        self.eps = float(eps)
        self._kernels = {}

    def _get_scharr_kernels(self, device: torch.device, dtype: torch.dtype):
        key = (device, dtype)
        if key not in self._kernels:
            kx = torch.tensor(
                [[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]],
                device=device,
                dtype=dtype,
            ).unsqueeze(0) / 16.0
            ky = kx.transpose(-1, -2).contiguous()
            self._kernels[key] = (kx, ky)
        return self._kernels[key]

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img is None or not torch.is_tensor(img) or img.ndim != 4:
            return img
        orig_dtype = img.dtype
        x = img.float()
        # Normalize if in [0, 255]
        scale = 255.0 if x.max() > 1.5 else 1.0
        if scale > 1.0:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)

        # 1. Multi-Scale Center-Surround Retinex Reflectance (scales 3x3 and 9x9)
        if self.ret_alpha > 0.0:
            illum_fine = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            illum_coarse = F.avg_pool2d(x, kernel_size=9, stride=1, padding=4)
            log_x = torch.log(x + self.eps)
            refl = 0.5 * (
                (log_x - torch.log(illum_fine + self.eps))
                + (log_x - torch.log(illum_coarse + self.eps))
            )
            refl = torch.tanh(refl * 2.5)  # zero-centered illumination-invariant detail in [-1, 1]
        else:
            refl = 0.0

        # 2. Anisotropic Horizontal/Vertical Sargis-Scharr Edge Energy on Luminance
        if self.edge_alpha > 0.0:
            lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            kx, ky = self._get_scharr_kernels(x.device, x.dtype)
            gx = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), kx)
            gy = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), ky)
            edge_mag = torch.sqrt(gx * gx + gy * gy + self.eps)
            # Zero-mean local contrast normalization of edge energy
            edge_mean = F.avg_pool2d(edge_mag, kernel_size=7, stride=1, padding=3)
            edge_res = torch.tanh((edge_mag - edge_mean) * 3.0)
        else:
            edge_res = 0.0

        out = (x + self.ret_alpha * refl + self.edge_alpha * edge_res).clamp(0.0, 1.0)
        if scale > 1.0:
            out = out * 255.0
        return out.to(orig_dtype)


def install_retinex_sargis_hook(ret_alpha: float = 0.10, edge_alpha: float = 0.08):
    transform = RetinexSargisEdgeTransform(ret_alpha=ret_alpha, edge_alpha=edge_alpha)

    orig_train_preprocess = DetectionTrainer.preprocess_batch
    orig_val_preprocess = DetectionValidator.preprocess

    def patched_train_preprocess(self, batch):
        batch = orig_train_preprocess(self, batch)
        if "img" in batch:
            batch["img"] = transform(batch["img"])
        return batch

    def patched_val_preprocess(self, batch):
        batch = orig_val_preprocess(self, batch)
        if "img" in batch:
            batch["img"] = transform(batch["img"])
        return batch

    DetectionTrainer.preprocess_batch = patched_train_preprocess
    DetectionValidator.preprocess = patched_val_preprocess
    print(
        f"[SOTA-R4] Installed Identity-Preserving Retinex (alpha={ret_alpha}) + "
        f"Sargis-Scharr Edge (alpha={edge_alpha}) Pre-Conditioner."
    )


if __name__ == "__main__":
    use_ret_edge = "--use-retinex-sargis-edge" in sys.argv
    if use_ret_edge:
        sys.argv.remove("--use-retinex-sargis-edge")
        install_retinex_sargis_hook(ret_alpha=0.10, edge_alpha=0.08)

    # Ensure yolo11m and yolo26x keys are registered in train.MODEL_DISPATCH_REGISTRY if present
    if hasattr(base_train, "MODEL_DISPATCH_REGISTRY"):
        reg = base_train.MODEL_DISPATCH_REGISTRY
        if "yolo26s" in reg:
            for k, w in [("yolo11m", "yolo11m.pt"), ("yolo26x", "yolo26x.pt")]:
                if k not in reg:
                    entry = dict(reg["yolo26s"])
                    entry["weights"] = w
                    reg[k] = entry

    base_train.main()
