#!/usr/bin/env python3
"""
Two-Stage PCB Component Detector (OBB-based Rectification & De-rotation + Component Detection)
=============================================================================================
Stage 1: Board-Level Macro Localization & Rectification
         Uses YOLOv11n-OBB (SanderGi/PCB-OBB, 2.6M params) to detect board boundaries,
         rotation angle, and 4 corner vertices. Unwarps/de-rotates perspective to a canonical
         0-degree orthogonal rectangular crop.
Stage 2: Component-Level Micro Detection
         Feeds canonical rectified board into our trained YOLO26 / Physics-Spectral detector
         to detect Capacitors, ICs, and Connectors with maximum precision.
Stage 3: Coordinate Reprojection (Homography Inversion)
         Projects component bounding boxes back to original camera coordinate frame.

Usage:
    python tools/two_stage_pcb_detector.py \
        --source data_samples/images/pcb141rec1_jpg.rf.b78efc04fe117f0bf985abac65a97519.jpg \
        --obb-weights weights/pcb_obb_yolov11n.pt \
        --comp-weights runs/yolov26s/pcb-filtered/weights/best.pt \
        --out-dir runs/two_stage_demo
"""

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_OBB_URL = "https://huggingface.co/SanderGi/PCB-OBB/resolve/main/best.pt?download=true"


def ensure_obb_weights(weights_path: Path) -> Path:
    """Download SanderGi YOLOv11n-OBB weights if not present locally."""
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if not weights_path.exists() or weights_path.stat().st_size < 1000000:
        print(f"📥 Downloading SanderGi YOLOv11n-OBB weights to {weights_path}...")
        urllib.request.urlretrieve(DEFAULT_OBB_URL, str(weights_path))
        print("✅ Download complete!")
    return weights_path


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points in clockwise order:
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff (x - y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff (x - y)
    return rect


def rectify_perspective(img: np.ndarray, corners: np.ndarray):
    """
    Applies a 4-point perspective transform to extract a canonical 0-degree
    orthogonal rectangular image of the PCB.
    """
    rect = order_points(corners)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(int(width_a), int(width_b))

    # Compute height of new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(int(height_a), int(height_b))

    # Avoid zero dimension
    max_w = max(max_w, 64)
    max_h = max(max_h, 64)

    # Align aspect ratio with input image to eliminate 90-degree axis ambiguity
    h_in, w_in = img.shape[:2]
    if (w_in >= h_in and max_w < max_h) or (w_in < h_in and max_w > max_h):
        rect = np.roll(rect, -1, axis=0)
        max_w, max_h = max_h, max_w

    dst = np.array(
        [
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ],
        dtype="float32",
    )

    M_warp = cv2.getPerspectiveTransform(rect, dst)
    unwarped = cv2.warpPerspective(img, M_warp, (max_w, max_h))
    return unwarped, M_warp, (max_w, max_h)


class TwoStagePCBDetector:
    def __init__(
        self,
        obb_weights: str,
        comp_weights: str,
        device: str = "auto",
        obb_conf: float = 0.25,
        comp_conf: float = 0.10,
        pad_ratio: float = 0.15,
    ):
        self.device = device
        self.obb_conf = obb_conf
        self.comp_conf = comp_conf
        self.pad_ratio = pad_ratio

        print(f"Loading Stage 1 OBB model: {obb_weights}")
        self.obb_model = YOLO(obb_weights)

        print(f"Loading Stage 2 Component model: {comp_weights}")
        self.comp_model = YOLO(comp_weights)

    def detect_and_rectify(self, img_bgr: np.ndarray):
        """
        Runs Stage 1: Detects PCB OBB, extracts corners, and unwarps image.
        Returns:
            rectified_img: np.ndarray (H, W, 3)
            M_warp: np.ndarray (3, 3) homography
            corners_orig: np.ndarray (4, 2) in original image coordinates
            angle_deg: float rotation angle
            conf: float detection confidence
        """
        h_orig, w_orig = img_bgr.shape[:2]

        # Add border padding to facilitate OBB detection if PCB takes up most of the frame
        if self.pad_ratio > 0:
            pad_h = int(h_orig * self.pad_ratio)
            pad_w = int(w_orig * self.pad_ratio)
            padded = cv2.copyMakeBorder(
                img_bgr, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[114, 114, 114]
            )
        else:
            pad_h, pad_w = 0, 0
            padded = img_bgr

        res = self.obb_model.predict(padded, conf=self.obb_conf, verbose=False)
        if len(res) == 0 or len(res[0].obb) == 0:
            # Fallback: Treat entire image as board if no OBB detected
            M_ident = np.eye(3, dtype=np.float32)
            corners = np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]], dtype=np.float32)
            return img_bgr, M_ident, corners, 0.0, 0.0

        # Select highest confidence PCB detection
        best_idx = int(torch.argmax(res[0].obb.conf))
        corners_padded = res[0].obb.xyxyxyxy[best_idx].cpu().numpy()
        angle_rad = float(res[0].obb.xywhr[best_idx, 4])
        angle_deg = angle_rad * 180.0 / np.pi
        conf = float(res[0].obb.conf[best_idx])

        # Shift corners back from padded space to original image space
        corners_orig = corners_padded.copy()
        corners_orig[:, 0] -= pad_w
        corners_orig[:, 1] -= pad_h

        # Perform 4-point perspective rectification on original image
        rectified_img, M_warp, _ = rectify_perspective(img_bgr, corners_orig)
        return rectified_img, M_warp, corners_orig, angle_deg, conf

    def detect_components(self, rectified_img: np.ndarray):
        """Runs Stage 2: Detects components on rectified board."""
        res = self.comp_model.predict(rectified_img, conf=self.comp_conf, verbose=False)
        return res[0]

    def reproject_boxes(self, boxes_xyxy: np.ndarray, M_warp: np.ndarray):
        """
        Projects 2D axis-aligned bounding boxes from rectified frame back to original
        frame as 4-corner polygons using inverse homography M_inv.
        """
        if len(boxes_xyxy) == 0:
            return np.zeros((0, 4, 2), dtype=np.float32)

        ret, M_inv = cv2.invert(M_warp)
        if not ret:
            M_inv = np.eye(3, dtype=np.float32)

        N = len(boxes_xyxy)
        # 4 corners for each box in rectified frame: (tl, tr, br, bl)
        polys = np.zeros((N, 4, 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            polys[i] = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # Apply perspective transform
        polys_reshaped = polys.reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(polys_reshaped, M_inv)
        return proj.reshape(N, 4, 2)

    def predict_full_pipeline(self, img_bgr: np.ndarray):
        """Runs complete 2-stage inference pipeline."""
        rect_img, M_warp, corners_orig, angle_deg, obb_conf = self.detect_and_rectify(img_bgr)
        comp_res = self.detect_components(rect_img)

        boxes_xyxy = comp_res.boxes.xyxy.cpu().numpy()
        scores = comp_res.boxes.conf.cpu().numpy()
        clses = comp_res.boxes.cls.cpu().numpy().astype(int)

        reprojected_polys = self.reproject_boxes(boxes_xyxy, M_warp)

        return {
            "rectified_image": rect_img,
            "M_warp": M_warp,
            "board_corners": corners_orig,
            "board_angle_deg": angle_deg,
            "board_conf": obb_conf,
            "comp_boxes_rectified": boxes_xyxy,
            "comp_polys_original": reprojected_polys,
            "comp_scores": scores,
            "comp_classes": clses,
            "comp_names": self.comp_model.names,
        }


def visualize_pipeline(img_orig: np.ndarray, result: dict, out_path: str):
    """Draws side-by-side visualization of the entire 3-stage process."""
    import matplotlib.pyplot as plt

    # 1. Original image with PCB boundary
    vis_orig = img_orig.copy()
    corners = result["board_corners"].astype(np.int32)
    cv2.polylines(vis_orig, [corners], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.putText(
        vis_orig,
        f"PCB (OBB): {result['board_conf']:.2f}, Angle: {result['board_angle_deg']:.1f} deg",
        (max(10, corners[:, 0].min()), max(30, corners[:, 1].min() - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 2. Rectified image with component detections
    vis_rect = result["rectified_image"].copy()
    class_colors = {
        "Capacitor": (255, 128, 0),             # Cyan/Blue
        "IC": (0, 0, 255),                      # Red
        "Connector": (255, 0, 255),              # Magenta
        "Electrolytic Capacitor": (0, 255, 255), # Yellow
    }

    for box, score, cls_id in zip(
        result["comp_boxes_rectified"], result["comp_scores"], result["comp_classes"]
    ):
        cls_name = result["comp_names"].get(cls_id, str(cls_id))
        color = class_colors.get(cls_name, (0, 255, 0))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_rect, (x1, y1), (x2, y2), color, 2)
        if score > 0.4:
            cv2.putText(
                vis_rect,
                f"{cls_name[:3]}:{score:.2f}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    # 3. Original image with reprojected components
    vis_reproj = img_orig.copy()
    for poly, score, cls_id in zip(
        result["comp_polys_original"], result["comp_scores"], result["comp_classes"]
    ):
        cls_name = result["comp_names"].get(cls_id, str(cls_id))
        color = class_colors.get(cls_name, (0, 255, 0))
        pts = poly.astype(np.int32)
        cv2.polylines(vis_reproj, [pts], isClosed=True, color=color, thickness=2)

    # Build 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"Stage 1: OBB Board Localization\nAngle: {result['board_angle_deg']:.1f}°, Conf: {result['board_conf']:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vis_rect, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Stage 2: Canonical Rectified 0° PCB\n{len(result['comp_boxes_rectified'])} Components Detected",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(vis_reproj, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        "Stage 3: Coordinate Reprojection\nInverse Homography back to Camera",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 Saved pipeline visualization to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Two-Stage OBB Rectification + Micro-Component Detector")
    p.add_argument("--source", type=str, required=True, help="Path to input image or directory")
    p.add_argument(
        "--obb-weights",
        type=str,
        default="weights/pcb_obb_yolov11n.pt",
        help="Path to YOLOv11n-OBB model weights",
    )
    p.add_argument(
        "--comp-weights",
        type=str,
        default="runs/yolov26s/pcb-filtered/weights/best.pt",
        help="Path to trained component detector weights",
    )
    p.add_argument("--obb-conf", type=float, default=0.25, help="Stage 1 confidence threshold")
    p.add_argument("--comp-conf", type=float, default=0.15, help="Stage 2 confidence threshold")
    p.add_argument("--out-dir", type=str, default="runs/two_stage_demo", help="Output directory")
    args = p.parse_args()

    obb_weights = ensure_obb_weights(Path(args.obb_weights))
    detector = TwoStagePCBDetector(
        obb_weights=str(obb_weights),
        comp_weights=args.comp_weights,
        obb_conf=args.obb_conf,
        comp_conf=args.comp_conf,
    )

    source_path = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_file():
        img = cv2.imread(str(source_path))
        if img is None:
            print(f"Error reading image: {source_path}")
            return
        result = detector.predict_full_pipeline(img)
        print(f"\nResults for {source_path.name}:")
        print(f"  Board Angle: {result['board_angle_deg']:.2f}° (Conf: {result['board_conf']:.3f})")
        print(f"  Components Found: {len(result['comp_boxes_rectified'])}")
        out_vis = out_dir / f"rectified_{source_path.stem}.png"
        visualize_pipeline(img, result, str(out_vis))
    else:
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        print(f"Processing {len(images)} images in {source_path}...")
        for img_path in images[:10]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            result = detector.predict_full_pipeline(img)
            out_vis = out_dir / f"rectified_{img_path.stem}.png"
            visualize_pipeline(img, result, str(out_vis))


if __name__ == "__main__":
    main()
