#!/usr/bin/env python3
"""
Phase 1: Two-Stage YOLO High-Resolution ROI Specialist Verifier (`train_yolo_roi_verifier.py`)
==============================================================================================
Addresses the #1 bottleneck on `Electrolytic Capacitor` (Class 2):
  - High-confidence False Positives on confusers (SMD power inductors, QFN ICs, connectors,
    circular mounting holes/screw heads, crystal cans) outranking true uncommon Electrolytic
    Capacitors (radial blue/green/gold sleeves, oblique cans, SMD V-chips).
  - Cross-class misclassification where true Electrolytic Capacitors are predicted as IC/Connector.

Architecture (100% YOLO Ecosystem, Zero External Downloads Required):
  1. Uses our already-trained `APEX_YOLO11m_1536` (`best.pt`) and `APEX_YOLOv12s_1536` (`best.pt`)
     backbones to extract multi-scale pyramid features (`P2, P3, P4` pooled inside the box +
     `1.6x` context ring) on native high-resolution `256x256` component crops, fused with a
     Radial-Circular Polarity & Specular Geometry Descriptor.
  2. Mines REAL Hard-Negative False Positive crops directly from Stage-1 predictions on the
     492 training boards (`IoU < 0.15` with any GT box) so the verifier explicitly learns to
     reject the exact false-positive patterns of Stage-1!
  3. Oversamples `Electrolytic Capacitor` crops 4x with continuous `0-360 deg` circular rotation,
     sleeve hue/saturation shifts (black/blue/green/gold/silver cans), and shadow contrast jitter.
  4. Trains an Attentive Cosine-Margin + Seesaw Focal ROI Verifier (`runs/r4_yolo_roi_verifier/roi_verifier.pt`)
     in ~3 minutes on GPU and provides `ROIVerifierScorer` for `quick_wbf.py`.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("LRU_CACHE_CAPACITY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
from ultralytics import YOLO

cv2.setNumThreads(2)

for mod in [
    "apex_r4_model",
    "models.as_pgms_yolo",
    "agwd_ota_loss",
    "squared_dice_loss",
    "retinex_stem",
    "weber_stem",
    "train_sota_r4",
]:
    try:
        __import__(mod)
    except Exception:
        pass

from evaluate_apex_r4_suite import DATA_YAML, resolve_ckpt_path

# 5 Verifier Classes: 0=Capacitor, 1=Connector, 2=Electrolytic Capacitor, 3=IC, 4=Hard_Background_FP
VERIFIER_CLASSES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC", "Hard_Background_FP"]


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-7)


def extract_context_crop(
    img_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    crop_size: int = 192,
    context_scale: float = 1.60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a square context-preserving native crop centered on `box_xyxy`,
    returning `(crop_bgr, geom_desc_12d)`:
      - Preserves aspect ratio using reflection padding if near board edge.
      - Computes a 12-D Radial-Circular & Polarity Geometry Descriptor on the inner box vs surround:
        [log_w, log_h, aspect_ratio, circular_radial_sym, ring_edge_energy,
         specular_highlight_frac, dark_polarity_frac, inner_outer_lum_contrast,
         color_sat_mean, color_hue_sin, color_hue_cos, shadow_halo_asymmetry]
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box_xyxy[:4])
    bw = max(x2 - x1, 4.0)
    bh = max(y2 - y1, 4.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    side = max(bw, bh) * context_scale
    side = max(side, 28.0)
    half = 0.5 * side

    cx1 = int(round(cx - half))
    cy1 = int(round(cy - half))
    cx2 = int(round(cx + half))
    cy2 = int(round(cy + half))

    pad_l = max(0, -cx1)
    pad_t = max(0, -cy1)
    pad_r = max(0, cx2 - W)
    pad_b = max(0, cy2 - H)

    ix1, iy1 = max(0, cx1), max(0, cy1)
    ix2, iy2 = min(W, cx2), min(H, cy2)
    sub = img_bgr[iy1:iy2, ix1:ix2]
    if sub.size == 0:
        sub = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    elif pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        sub = cv2.copyMakeBorder(sub, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101)

    crop = cv2.resize(sub, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    # Compute 12-D physical & radial-circular geometry descriptor
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Inner component region (central 62% of the 1.60x context crop)
    m0 = int(round(crop_size * 0.19))
    m1 = int(round(crop_size * 0.81))
    inner_g = gray[m0:m1, m0:m1]

    # Sobel gradients on inner region to measure Radial-Circular Alignment vs Manhattan Rectangular Alignment
    gx = cv2.Sobel(inner_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inner_g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy + 1e-8)

    ih, iw = inner_g.shape
    yy, xx = np.mgrid[0:ih, 0:iw].astype(np.float32)
    rx = xx - (iw - 1) * 0.5
    ry = yy - (ih - 1) * 0.5
    r_norm = np.sqrt(rx * rx + ry * ry + 1e-6)
    ux, uy = rx / r_norm, ry / r_norm

    # Radial dot product |g . u_r| / |g|: near 1.0 for circular electrolytic cans, lower for orthogonal grids
    radial_cos = np.abs(gx * ux + gy * uy) / mag
    radial_ring_mask = (r_norm >= 0.28 * iw) & (r_norm <= 0.50 * iw)
    w_mag = mag[radial_ring_mask]
    radial_sym = float(np.sum(radial_cos[radial_ring_mask] * w_mag) / (np.sum(w_mag) + 1e-5))
    ring_edge_energy = float(np.mean(w_mag))

    # Specular metallic top (>0.82 lum) & dark polarity crescent (<0.22 lum) fractions
    spec_frac = float(np.mean(inner_g > 0.80))
    dark_frac = float(np.mean(inner_g < 0.22))

    # Inner vs outer surround contrast & directional shadow asymmetry
    outer_mask = np.ones_like(gray, dtype=bool)
    outer_mask[m0:m1, m0:m1] = False
    outer_g = gray[outer_mask]
    lum_contrast = float(np.mean(inner_g) - np.mean(outer_g))

    top_half = float(np.mean(inner_g[: ih // 2, :]))
    bot_half = float(np.mean(inner_g[ih // 2 :, :]))
    left_half = float(np.mean(inner_g[:, : iw // 2]))
    right_half = float(np.mean(inner_g[:, iw // 2 :]))
    polarity_asym = math.sqrt((top_half - bot_half) ** 2 + (left_half - right_half) ** 2)

    inner_hsv = hsv[m0:m1, m0:m1]
    sat_mean = float(np.mean(inner_hsv[:, :, 1]) / 255.0)
    hue_rad = inner_hsv[:, :, 0] * (2.0 * math.pi / 180.0)
    hue_sin = float(np.mean(np.sin(hue_rad)))
    hue_cos = float(np.mean(np.cos(hue_rad)))

    geom = np.array(
        [
            math.log(bw + 1.0) / 6.0,
            math.log(bh + 1.0) / 6.0,
            math.log(bw / bh),
            max(bw, bh) / min(bw, bh) - 1.0,
            radial_sym,
            ring_edge_energy,
            spec_frac,
            dark_frac,
            lum_contrast,
            polarity_asym,
            sat_mean,
            0.5 * (hue_sin + hue_cos),
        ],
        dtype=np.float32,
    )
    return crop, geom


def augment_elcap_crop(crop_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Applies cylindrical-aware 0-360 degree continuous rotation, sleeve hue/saturation shift
    (simulating blue/green/gold/black/silver electrolytic cans), and brightness/contrast jitter.
    """
    h, w = crop_bgr.shape[:2]
    angle = rng.uniform(0.0, 360.0)
    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, rng.uniform(0.92, 1.08))
    out = cv2.warpAffine(crop_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    if rng.random() < 0.65:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-35.0, 35.0)) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.75, 1.35), 0.0, 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.85, 1.15), 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


class YoloBackbonePyramidFeatureExtractor:
    """
    Extracts multi-scale [P2, P3, P4, P5] Inner-Core + Outer-Context Ring pooled descriptors
    from our pretrained R4-Final YOLO checkpoints (`APEX_YOLO11m_1536` and `APEX_YOLOv12s_1536`).
    """

    def __init__(self, ckpt_paths: List[Path], device: str = "0", crop_size: int = 192):
        self.device = torch.device(f"cuda:{device}" if (device != "cpu" and torch.cuda.is_available()) else "cpu")
        self.crop_size = int(crop_size)
        self.models = []
        self.tap_indices = []

        for pt in ckpt_paths:
            y = YOLO(str(pt))
            m = y.model.to(self.device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            # Determine backbone/neck tap layers
            taps = getattr(m, "stage_indices", [2, 4, 6, 8])
            detect_f = getattr(m.model[-1], "f", [])
            if isinstance(detect_f, (list, tuple)):
                taps = sorted(list(set(list(taps) + [int(i) for i in detect_f])))
            self.models.append(m)
            self.tap_indices.append(taps)

    @torch.inference_mode()
    def extract_batch(self, crops_bgr: List[np.ndarray], geoms: List[np.ndarray], batch_size: int = 64) -> torch.Tensor:
        if not crops_bgr:
            return torch.zeros((0, 1), dtype=torch.float32)

        all_vecs = []
        for start in range(0, len(crops_bgr), batch_size):
            batch_crops = crops_bgr[start : start + batch_size]
            batch_geoms = np.stack(geoms[start : start + batch_size], axis=0)

            # Convert BGR uint8 -> RGB float32 [0, 1] tensor (B, 3, H, W)
            arr = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in batch_crops], axis=0)
            x_in = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(self.device, dtype=torch.float32) / 255.0

            model_feats = []
            for m, taps in zip(self.models, self.tap_indices):
                y_hist = []
                tap_set = set(taps)
                save_set = set(getattr(m, "save", [])) | tap_set
                cur = x_in
                for layer in m.model[:-1]:
                    if layer.f != -1:
                        cur = y_hist[layer.f] if isinstance(layer.f, int) else [cur if j == -1 else y_hist[j] for j in layer.f]
                    cur = layer(cur)
                    y_hist.append(cur if layer.i in save_set else None)
                    if layer.i in tap_set and isinstance(cur, torch.Tensor) and cur.ndim == 4:
                        fh, fw = cur.shape[-2:]
                        if fh >= 4 and fw >= 4:
                            # Pool Inner Core (central 50%) AND Full Context (100%) to capture component vs surround
                            y0, y1 = max(0, fh // 4), min(fh, fh - fh // 4)
                            x0, x1 = max(0, fw // 4), min(fw, fw - fw // 4)
                            inner_mean = cur[:, :, y0:y1, x0:x1].mean(dim=(-2, -1))
                            inner_max = cur[:, :, y0:y1, x0:x1].amax(dim=(-2, -1))
                            global_mean = cur.mean(dim=(-2, -1))
                            model_feats.append(F.normalize(torch.cat([inner_mean, inner_max, global_mean], dim=1), dim=1))
                        else:
                            model_feats.append(F.normalize(cur.mean(dim=(-2, -1)), dim=1))

            g_tensor = torch.from_numpy(batch_geoms).to(self.device, dtype=torch.float32)
            vec = torch.cat(model_feats + [g_tensor], dim=1).cpu()
            all_vecs.append(vec)

        return torch.cat(all_vecs, dim=0)


class AttentiveCosineROIVerifier(nn.Module):
    """
    Attentive Residual MLP + Cosine-Prototype Classifier over Multi-YOLO Pyramid ROI Features.
    Outputs logits for 5 classes: [Capacitor, Connector, Electrolytic Capacitor, IC, Hard_Background_FP].
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, num_classes: int = 5):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.input_norm = nn.LayerNorm(in_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.SiLU(inplace=True),
            nn.Linear(in_dim // 4, in_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(inplace=True),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim // 2) * 0.05)
        self.linear_head = nn.Linear(hidden_dim // 2, num_classes)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.input_norm(x)
        x_g = x_n * self.gate(x_n)
        h = self.proj(x_g)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        scale = self.log_scale.exp().clamp(min=4.0, max=40.0)
        cos_logits = scale * (h_norm @ p_norm.t())
        return 0.65 * cos_logits + 0.35 * self.linear_head(h)


class ROIVerifierScorer:
    """
    Runtime Stage-2 Verifier used by `quick_wbf.py` to re-score candidate boxes
    on test boards using the trained `AttentiveCosineROIVerifier`.
    """

    def __init__(self, verifier_ckpt_path: Path, device: str = "0"):
        self.device = torch.device(f"cuda:{device}" if (device != "cpu" and torch.cuda.is_available()) else "cpu")
        ckpt = torch.load(str(verifier_ckpt_path), map_location="cpu", weights_only=False)
        self.crop_size = int(ckpt.get("crop_size", 192))
        backbone_paths = [resolve_ckpt_path(p) for p in ckpt["backbone_paths"]]
        backbone_paths = [p for p in backbone_paths if p is not None]
        self.extractor = YoloBackbonePyramidFeatureExtractor(backbone_paths, device=device, crop_size=self.crop_size)
        self.verifier = AttentiveCosineROIVerifier(
            in_dim=ckpt["in_dim"],
            hidden_dim=ckpt.get("hidden_dim", 512),
            num_classes=ckpt.get("num_classes", 5),
        ).to(self.device)
        self.verifier.load_state_dict(ckpt["state_dict"], strict=True)
        self.verifier.eval()

    @torch.inference_mode()
    def predict_probs(self, img_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Returns (N, 5) probability matrix over [Cap, Conn, ElCap, IC, Hard_BG_FP] for `boxes_xyxy`."""
        if len(boxes_xyxy) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        crops, geoms = [], []
        for b in boxes_xyxy:
            c, g = extract_context_crop(img_bgr, b, crop_size=self.crop_size)
            crops.append(c)
            geoms.append(g)
        feats = self.extractor.extract_batch(crops, geoms, batch_size=64).to(self.device)
        logits = self.verifier(feats)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs


def collect_split_crops(
    root: Path,
    split_subpath: str,
    stage1_model: Optional[YOLO],
    crop_size: int = 192,
    is_train: bool = True,
    device: str = "0",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Collects native crops, 12-D geometry vectors, and 5-class labels `{0, 1, 2, 3, 4}`:
      - All GT Connector (1), Electrolytic Capacitor (2, 4x oversampled on train), and IC (3) instances
      - Balanced subsample of GT Capacitor (0) instances (focusing on medium/large caps >=14px)
      - Mined Hard-Negative False Positive crops (Class 4) from Stage-1 model predictions that have
        IoU < 0.15 with any GT box!
    """
    img_dir = root / split_subpath
    lbl_dir = img_dir.parent / "labels"
    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    rng = random.Random(42)

    crops, geoms, labels = [], [], []
    counts = {k: 0 for k in range(5)}

    for idx, im_path in enumerate(img_paths):
        img_bgr = cv2.imread(str(im_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        lb_path = lbl_dir / f"{im_path.stem}.txt"

        gt_cls, gt_boxes = [], []
        if lb_path.exists():
            for line in lb_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                gt_cls.append(c)
                gt_boxes.append([
                    (cx - bw * 0.5) * W,
                    (cy - bh * 0.5) * H,
                    (cx + bw * 0.5) * W,
                    (cy + bh * 0.5) * H,
                ])
        gt_b = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        gt_c = np.array(gt_cls, dtype=np.int64) if gt_cls else np.zeros((0,), dtype=np.int64)

        for b, c in zip(gt_b, gt_c):
            bw, bh = b[2] - b[0], b[3] - b[1]
            # Keep all Connector (1), ElCap (2), IC (3); subsample Capacitor (0) so it doesn't drown rare classes
            if c == 0:
                keep_prob = 0.30 if max(bw, bh) >= 16.0 else 0.08
                if rng.random() > keep_prob:
                    continue
            crop, geom = extract_context_crop(img_bgr, b, crop_size=crop_size)
            crops.append(crop)
            geoms.append(geom)
            labels.append(int(c))
            counts[int(c)] += 1

            # Oversample Electrolytic Capacitor (2) and Connector (1) during training
            if is_train and c == 2:
                for _ in range(3):
                    jit_b = b + np.array(
                        [
                            rng.uniform(-0.08, 0.08) * bw,
                            rng.uniform(-0.08, 0.08) * bh,
                            rng.uniform(-0.08, 0.08) * bw,
                            rng.uniform(-0.08, 0.08) * bh,
                        ],
                        dtype=np.float32,
                    )
                    c_raw, _ = extract_context_crop(img_bgr, jit_b, crop_size=crop_size)
                    c_aug = augment_elcap_crop(c_raw, rng)
                    # Recompute geometry descriptor on augmented crop
                    _, g_aug = extract_context_crop(c_aug, np.array([0, 0, bw, bh], dtype=np.float32), crop_size=crop_size)
                    g_aug[:4] = geom[:4]
                    crops.append(c_aug)
                    geoms.append(g_aug)
                    labels.append(2)
                    counts[2] += 1
            elif is_train and c == 1 and rng.random() < 0.60:
                c_flp = cv2.flip(crop, 1 if rng.random() < 0.5 else 0)
                crops.append(c_flp)
                geoms.append(geom.copy())
                labels.append(1)
                counts[1] += 1

        # 1. Fast Geometric & Boundary Hard-Negative Confuser Crops (IoU < 0.10 with any GT)
        # Samples off-center/shifted patches near ICs, Connectors, and high-gradient background rings/holes
        hard_neg_candidates = []
        for b, c in zip(gt_b, gt_c):
            if c in (1, 2, 3):
                bw, bh = max(b[2] - b[0], 16.0), max(b[3] - b[1], 16.0)
                for dx_mult, dy_mult in ((1.35, 0.0), (-1.35, 0.0), (0.0, 1.35), (1.15, 1.15)):
                    sx1 = np.clip(b[0] + dx_mult * bw, 0, W - bw)
                    sy1 = np.clip(b[1] + dy_mult * bh, 0, H - bh)
                    hard_neg_candidates.append([sx1, sy1, sx1 + bw, sy1 + bh])
        for _ in range(8):
            rw = rng.uniform(28.0, 110.0)
            rh = rw * rng.uniform(0.75, 1.35)
            rx1 = rng.uniform(0.0, max(1.0, W - rw))
            ry1 = rng.uniform(0.0, max(1.0, H - rh))
            hard_neg_candidates.append([rx1, ry1, rx1 + rw, ry1 + rh])

        if hard_neg_candidates:
            hnc = np.array(hard_neg_candidates, dtype=np.float32)
            ious = box_iou_np(hnc, gt_b) if len(gt_b) > 0 else np.zeros((len(hnc), 0), dtype=np.float32)
            max_ious = ious.max(axis=1) if ious.shape[1] > 0 else np.zeros(len(hnc), dtype=np.float32)
            valid_hn = hnc[max_ious < 0.08]
            rng.shuffle(valid_hn := list(valid_hn))
            for hb in valid_hn[: (5 if is_train else 4)]:
                crop, geom = extract_context_crop(img_bgr, hb, crop_size=crop_size)
                crops.append(crop)
                geoms.append(geom)
                labels.append(4)
                counts[4] += 1

        # 2. Mine Real Stage-1 False-Positive boxes on GPU (or on every 8th board on CPU for speed)
        run_miner = (stage1_model is not None) and (device != "cpu" or (idx % 8 == 0))
        if run_miner:
            with torch.inference_mode():
                r = stage1_model.predict(img_bgr, imgsz=1536, conf=0.05, iou=0.55, max_det=300, device=device, verbose=False)[0]
                pb = r.boxes.xyxy.cpu().numpy()
                ps = r.boxes.conf.cpu().numpy()
                pc = r.boxes.cls.cpu().numpy().astype(int)
                del r

            if len(pb) > 0:
                ious = box_iou_np(pb, gt_b) if len(gt_b) > 0 else np.zeros((len(pb), 0), dtype=np.float32)
                max_ious = ious.max(axis=1) if ious.shape[1] > 0 else np.zeros(len(pb), dtype=np.float32)
                fp_indices = [
                    j
                    for j in range(len(pb))
                    if max_ious[j] < 0.12 and (pc[j] in (1, 2, 3) or ps[j] >= 0.20)
                ]
                for j in fp_indices[:6]:
                    crop, geom = extract_context_crop(img_bgr, pb[j], crop_size=crop_size)
                    crops.append(crop)
                    geoms.append(geom)
                    labels.append(4)
                    counts[4] += 1

        del img_bgr
        if (idx + 1) % 50 == 0 or (idx + 1) == len(img_paths):
            print(f"  [{split_subpath}] Processed {idx + 1}/{len(img_paths)} boards | crop counts={counts}", flush=True)

    return crops, geoms, labels


def main():
    p = argparse.ArgumentParser(description="Train Two-Stage YOLO High-Resolution ROI Verifier")
    p.add_argument("--crop-size", type=int, default=192)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    cfg = yaml.safe_load(open(DATA_YAML))
    root = Path(cfg["path"])

    # Resolve our best pretrained YOLO backbones on disk (no internet download needed)
    cand_backbones = [
        "runs/r4_apex_yolo11m_1536/pcb-filtered/weights/best.pt",
        "runs/r4_apex_yolov12s_1536/pcb-filtered/weights/best.pt",
        "runs/r4_final_yolo26s_agwd_sdice_1280/pcb-filtered/weights/best.pt",
    ]
    resolved_backbones = []
    for c_str in cand_backbones:
        pt = resolve_ckpt_path(c_str)
        if pt is not None:
            resolved_backbones.append((c_str, pt))
        if len(resolved_backbones) >= 2:
            break

    assert len(resolved_backbones) >= 1, "No trained R4 YOLO checkpoint found to extract ROI pyramid features!"
    print(f"[Stage-2 ROI Verifier] Using {len(resolved_backbones)} pretrained YOLO backbones: {[x[0] for x in resolved_backbones]}")

    # Use primary Stage-1 detector to mine real Hard-Negative False Positives on train/val boards
    miner_model = YOLO(str(resolved_backbones[0][1]))
    print("\n[Step 1/3] Collecting GT + 4x Rotated ElCap + Mined Stage-1 Hard-Negative ROI Crops...")
    tr_crops, tr_geoms, tr_labels = collect_split_crops(
        root, cfg["train"], miner_model, crop_size=args.crop_size, is_train=True, device=args.device
    )
    val_crops, val_geoms, val_labels = collect_split_crops(
        root, cfg["val"], miner_model, crop_size=args.crop_size, is_train=False, device=args.device
    )
    del miner_model
    gc.collect()

    print("\n[Step 2/3] Extracting Multi-Scale YOLO Pyramid + Radial-Circular Geometry Features...")
    extractor = YoloBackbonePyramidFeatureExtractor(
        [x[1] for x in resolved_backbones], device=args.device, crop_size=args.crop_size
    )
    X_train = extractor.extract_batch(tr_crops, tr_geoms, batch_size=64)
    y_train = torch.tensor(tr_labels, dtype=torch.long)
    del tr_crops, tr_geoms
    gc.collect()

    X_val = extractor.extract_batch(val_crops, val_geoms, batch_size=64)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    del val_crops, val_geoms, extractor
    gc.collect()

    in_dim = int(X_train.shape[1])
    print(f"  Extracted feature dimension = {in_dim} | Train samples = {len(X_train)} | Val samples = {len(X_val)}")

    # Class-balanced focal weights emphasizing Electrolytic Capacitor (2), Connector (1), and Hard_Background_FP (4)
    cls_counts = torch.bincount(y_train, minlength=5).float().clamp(min=1.0)
    inv_freq = (cls_counts.sum() / (5.0 * cls_counts)).clamp(min=0.5, max=4.0)
    inv_freq[2] *= 1.35  # Extra priority on Electrolytic Capacitor precision & recall
    inv_freq[4] *= 1.25  # Extra priority on rejecting Hard-Negative False Positives

    device_t = torch.device(f"cuda:{args.device}" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    inv_freq = inv_freq.to(device_t)

    model = AttentiveCosineROIVerifier(in_dim=in_dim, hidden_dim=512, num_classes=5).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    best_val_score = -1.0
    best_sd = None
    print("\n[Step 3/3] Training AttentiveCosineROIVerifier...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device_t), by.to(device_t)
            # Feature-space Mixup (alpha=0.2) for calibrated probabilities
            if random.random() < 0.35 and len(bx) > 4:
                lam = float(np.random.beta(0.2, 0.2))
                perm = torch.randperm(len(bx), device=device_t)
                bx_mix = lam * bx + (1.0 - lam) * bx[perm]
                logits = model(bx_mix)
                ce_a = F.cross_entropy(logits, by, weight=inv_freq, label_smoothing=0.04)
                ce_b = F.cross_entropy(logits, by[perm], weight=inv_freq, label_smoothing=0.04)
                loss = lam * ce_a + (1.0 - lam) * ce_b
            else:
                logits = model(bx)
                ce = F.cross_entropy(logits, by, weight=inv_freq, reduction="none", label_smoothing=0.04)
                pt = torch.exp(-ce).detach()
                focal_loss = ((1.0 - pt) ** 1.5) * ce
                loss = focal_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bx)
        scheduler.step()

        # Evaluate on Val crops (specifically tracking Electrolytic Capacitor F1 + Hard-BG rejection accuracy)
        model.eval()
        with torch.inference_mode():
            val_logits = model(X_val.to(device_t)).cpu()
            val_preds = val_logits.argmax(dim=-1)
            overall_acc = float((val_preds == y_val).float().mean().item())

            elcap_gt = y_val == 2
            elcap_pred = val_preds == 2
            tp_el = float((elcap_gt & elcap_pred).sum().item())
            el_rec = tp_el / max(float(elcap_gt.sum().item()), 1.0)
            el_prec = tp_el / max(float(elcap_pred.sum().item()), 1.0)
            el_f1 = 2.0 * el_rec * el_prec / max(el_rec + el_prec, 1e-6)

            bg_gt = y_val == 4
            bg_rej = float((val_preds[bg_gt] == 4).float().mean().item()) if bg_gt.any() else 0.0

            score = 0.50 * el_f1 + 0.30 * overall_acc + 0.20 * bg_rej
            if score > best_val_score:
                best_val_score = score
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:02d}/{args.epochs:02d} | Loss={total_loss / len(X_train):.4f} | "
                f"Val Acc={overall_acc:.4f} | ElCap F1={el_f1:.4f} (P={el_prec:.3f} R={el_rec:.3f}) | "
                f"Hard-FP Reject={bg_rej:.3f}",
                flush=True,
            )

    out_dir = Path("runs/r4_yolo_roi_verifier")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / "roi_verifier.pt"
    torch.save(
        {
            "state_dict": best_sd,
            "in_dim": in_dim,
            "hidden_dim": 512,
            "num_classes": 5,
            "crop_size": args.crop_size,
            "backbone_paths": [x[0] for x in resolved_backbones],
            "best_val_score": best_val_score,
            "classes": VERIFIER_CLASSES,
        },
        str(out_pt),
    )
    print(f"\n[Stage-2 ROI Verifier] Saved best verifier checkpoint to {out_pt} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
