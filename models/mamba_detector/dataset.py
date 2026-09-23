"""
PCB Dataset & DataLoader for Standalone Mamba Detector
=======================================================
Reads standard YOLO-formatted datasets (images + labels/*.txt).
Automatically detects and supports:
- Canonical 4-Class Taxonomy (datasets/pcb-unified-4class)
- Legacy 23-Class Taxonomy (auto-aliasing target classes: Cap, Conn, ElCap, IC)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

# Class mapping from legacy 23-class Roboflow to canonical 4 classes
LEGACY_MAPPING = {
    1: 0,   # Capacitor Jumper -> Capacitor
    2: 0,   # Capacitor -> Capacitor
    4: 1,   # Connector -> Connector
    7: 2,   # Electrolytic Capacitor -> Electrolytic Capacitor
    9: 3,   # IC -> IC
    22: 3,  # iC typo -> IC
}


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resizes and pads image to target shape with aspect ratio preservation."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2.0, dh / 2.0  # divide padding into 2 sides

    if shape[::-1] != new_unpad:
        if HAS_CV2:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(new_unpad, resample=Image.BILINEAR)
            img = np.array(pil_img)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if HAS_CV2:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        # np.pad for RGB or 2D image
        pad_width = ((top, bottom), (left, right), (0, 0)) if img.ndim == 3 else ((top, bottom), (left, right))
        img = np.pad(img, pad_width, mode="constant", constant_values=color[0])

    return img, r, (dw, dh)


class PCBDataset(Dataset):
    """
    PyTorch Dataset for PCB Component Detection.
    """

    def __init__(
        self,
        data_yaml_path: str,
        split: str = "train",
        imgsz: int = 640,
        augment: bool = True,
    ):
        super().__init__()
        self.split = split
        self.imgsz = imgsz
        self.augment = augment

        yaml_p = Path(data_yaml_path)
        assert yaml_p.exists(), f"Dataset config not found at: {yaml_p}"

        with open(yaml_p) as f:
            cfg = yaml.safe_load(f)

        dataset_root = yaml_p.parent
        if "path" in cfg and cfg["path"]:
            root_candidate = Path(cfg["path"])
            if root_candidate.exists():
                dataset_root = root_candidate

        split_key = "val" if split == "valid" else split
        split_path_str = cfg.get(split_key, f"{split}/images")
        img_dir = dataset_root / split_path_str

        # Look for sibling labels folder
        if not img_dir.exists():
            img_dir = dataset_root / split / "images"

        self.img_dir = img_dir
        self.lbl_dir = img_dir.parent / "labels"

        assert self.img_dir.exists(), f"Image directory does not exist: {self.img_dir}"

        # Collect image files
        self.img_files = sorted(
            list(self.img_dir.glob("*.jpg"))
            + list(self.img_dir.glob("*.png"))
            + list(self.img_dir.glob("*.jpeg"))
        )
        assert len(self.img_files) > 0, f"No images found in {self.img_dir}"

        # Determine if dataset is native 4-class or legacy 23-class
        self.nc = cfg.get("nc", len(cfg.get("names", [])))
        self.is_native_4class = (self.nc == 4)

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        img_p = self.img_files[idx]
        if HAS_CV2:
            bgr = cv2.imread(str(img_p))
            assert bgr is not None, f"Failed to load image: {img_p}"
            rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            with Image.open(str(img_p)) as im:
                rgb_orig = np.array(im.convert("RGB"))

        h0, w0 = rgb_orig.shape[:2]

        # 1. Letterbox resize
        rgb, ratio, (pad_w, pad_h) = letterbox(rgb_orig, (self.imgsz, self.imgsz))

        # 2. Load ground truth bounding boxes
        lbl_p = self.lbl_dir / (img_p.stem + ".txt")
        boxes = []
        if lbl_p.exists():
            with open(lbl_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        raw_cid = int(parts[0])
                        # Map class index
                        if self.is_native_4class:
                            if raw_cid >= 4:
                                continue
                            cid = raw_cid
                        else:
                            if raw_cid not in LEGACY_MAPPING:
                                continue
                            cid = LEGACY_MAPPING[raw_cid]

                        cx, cy, bw, bh = map(float, parts[1:5])
                        # Convert normalized (cx, cy, w, h) to original pixels
                        x1 = (cx - bw / 2.0) * w0
                        y1 = (cy - bh / 2.0) * h0
                        x2 = (cx + bw / 2.0) * w0
                        y2 = (cy + bh / 2.0) * h0

                        # Transform to letterbox pixel coordinates
                        x1 = x1 * ratio + pad_w
                        y1 = y1 * ratio + pad_h
                        x2 = x2 * ratio + pad_w
                        y2 = y2 * ratio + pad_h

                        # Clip to bounds
                        x1 = max(0.0, min(float(self.imgsz), x1))
                        y1 = max(0.0, min(float(self.imgsz), y1))
                        x2 = max(0.0, min(float(self.imgsz), x2))
                        y2 = max(0.0, min(float(self.imgsz), y2))

                        if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
                            boxes.append([float(cid), x1, y1, x2, y2])

        # 3. Horizontal flip augmentation
        if self.augment and (np.random.rand() > 0.5):
            rgb = np.ascontiguousarray(np.fliplr(rgb))
            if len(boxes) > 0:
                for b in boxes:
                    x1, x2 = b[1], b[3]
                    b[1] = self.imgsz - x2
                    b[3] = self.imgsz - x1

        # Tensor conversion (3, H, W) in [0.0, 1.0]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 5), dtype=torch.float32)

        meta = {
            "img_path": str(img_p),
            "stem": img_p.stem,
            "orig_shape": (h0, w0),
            "ratio": ratio,
            "padding": (pad_w, pad_h),
        }

        return img_tensor, boxes_tensor, meta


def pcb_collate_fn(batch):
    """Batches images and keeps variable-count bounding boxes as a list."""
    images, boxes, metas = zip(*batch)
    batched_images = torch.stack(images, dim=0)
    return batched_images, list(boxes), list(metas)
