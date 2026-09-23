"""
Anchor-Free Detection Loss & Target Assignment
================================================
Implements Center-Based Target Assignment, Sigmoid Focal Loss,
Generalized IoU (GIoU) Loss, and Centerness Quality Loss.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes Generalized IoU between two sets of boxes [x1, y1, x2, y2].
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-7)

    # Enclosing box
    c_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    c_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    c_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    c_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    c_area = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)

    giou = iou - (c_area - union) / c_area.clamp(min=1e-7)
    return giou


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Sigmoid focal loss for class imbalance.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


class DetectionLoss(nn.Module):
    """
    Anchor-Free Detection Loss for VMamba Detector.
    """

    def __init__(
        self,
        num_classes: int = 4,
        strides: List[int] = [4, 8, 16, 32],
        reg_ranges: List[Tuple[float, float]] = [(-1, 64), (32, 128), (64, 256), (128, 100000)],
        center_radius: float = 1.5,
        loss_cls_weight: float = 1.0,
        loss_box_weight: float = 2.0,
        loss_ctr_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.reg_ranges = reg_ranges
        self.center_radius = center_radius
        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight
        self.loss_ctr_weight = loss_ctr_weight

    def generate_points(self, feat_shapes: List[Tuple[int, int]], device: torch.device) -> List[torch.Tensor]:
        """Generates (x, y) spatial grid locations on the original canvas."""
        points = []
        for (h, w), stride in zip(feat_shapes, self.strides):
            shifts_x = (torch.arange(0, w, device=device) + 0.5) * stride
            shifts_y = (torch.arange(0, h, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            points.append(torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1))
        return points

    def forward(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        centernesses: List[torch.Tensor],
        gt_boxes_list: List[torch.Tensor], # List of (N_i, 5) tensors: [cls_id, x1, y1, x2, y2]
    ) -> Dict[str, torch.Tensor]:
        device = cls_scores[0].device
        feat_shapes = [feat.shape[2:] for feat in cls_scores]
        points_per_lvl = self.generate_points(feat_shapes, device)

        batch_size = cls_scores[0].shape[0]

        # Flatten predictions across all pyramid levels
        all_cls = []
        all_bbox = []
        all_ctr = []
        all_points = []
        all_strides = []

        for lvl, (cls_lvl, bbox_lvl, ctr_lvl, pts_lvl, stride) in enumerate(
            zip(cls_scores, bbox_preds, centernesses, points_per_lvl, self.strides)
        ):
            B, C, H, W = cls_lvl.shape
            # (B, H, W, C) -> (B, H*W, C)
            all_cls.append(cls_lvl.permute(0, 2, 3, 1).reshape(B, -1, C))
            all_bbox.append(bbox_lvl.permute(0, 2, 3, 1).reshape(B, -1, 4))
            all_ctr.append(ctr_lvl.permute(0, 2, 3, 1).reshape(B, -1))
            all_points.append(pts_lvl) # (H*W, 2)
            all_strides.append(torch.full((H * W,), stride, device=device))

        flat_cls = torch.cat(all_cls, dim=1) # (B, Total_pts, num_classes)
        flat_bbox = torch.cat(all_bbox, dim=1) # (B, Total_pts, 4) [l, t, r, b]
        flat_ctr = torch.cat(all_ctr, dim=1) # (B, Total_pts)
        flat_points = torch.cat(all_points, dim=0) # (Total_pts, 2)
        flat_strides = torch.cat(all_strides, dim=0) # (Total_pts,)

        total_pts = flat_points.shape[0]

        # Target tensors
        target_cls = torch.zeros((batch_size, total_pts, self.num_classes), device=device)
        target_bbox = torch.zeros((batch_size, total_pts, 4), device=device)
        target_ctr = torch.zeros((batch_size, total_pts), device=device)
        pos_mask = torch.zeros((batch_size, total_pts), dtype=torch.bool, device=device)

        px, py = flat_points[:, 0], flat_points[:, 1]

        for b_idx, gt in enumerate(gt_boxes_list):
            if gt.numel() == 0:
                continue

            gt_cls = gt[:, 0].long()
            gt_x1, gt_y1, gt_x2, gt_y2 = gt[:, 1], gt[:, 2], gt[:, 3], gt[:, 4]

            # Distances from all points to all gt boxes: (Total_pts, Num_gt)
            l = px.unsqueeze(1) - gt_x1.unsqueeze(0)
            t = py.unsqueeze(1) - gt_y1.unsqueeze(0)
            r = gt_x2.unsqueeze(0) - px.unsqueeze(1)
            b = gt_y2.unsqueeze(0) - py.unsqueeze(1)

            is_in_box = (l > 0) & (t > 0) & (r > 0) & (b > 0)
            if not is_in_box.any():
                continue

            max_reg = torch.maximum(torch.maximum(l, r), torch.maximum(t, b)) # (Total_pts, Num_gt)
            areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1) # (Num_gt,)

            # For each point, pick the box with minimum area among matching boxes
            matched_areas = torch.where(is_in_box, areas.unsqueeze(0), torch.tensor(float("inf"), device=device))
            min_area, min_idx = matched_areas.min(dim=1)
            valid = min_area < float("inf")

            if valid.any():
                pos_mask[b_idx, valid] = True
                matched_gt_idx = min_idx[valid]
                matched_cls = gt_cls[matched_gt_idx]

                # One-hot class targets
                target_cls[b_idx, valid, matched_cls] = 1.0

                # Bbox distances [l, t, r, b]
                tgt_l = l[valid, matched_gt_idx]
                tgt_t = t[valid, matched_gt_idx]
                tgt_r = r[valid, matched_gt_idx]
                tgt_b = b[valid, matched_gt_idx]
                target_bbox[b_idx, valid] = torch.stack([tgt_l, tgt_t, tgt_r, tgt_b], dim=-1)

                # Centerness target: sqrt( (min(l,r)/max(l,r)) * (min(t,b)/max(t,b)) )
                lr_min = torch.minimum(tgt_l, tgt_r)
                lr_max = torch.maximum(tgt_l, tgt_r)
                tb_min = torch.minimum(tgt_t, tgt_b)
                tb_max = torch.maximum(tgt_t, tgt_b)
                ctr = torch.sqrt((lr_min / lr_max.clamp(min=1e-7)) * (tb_min / tb_max.clamp(min=1e-7)))
                target_ctr[b_idx, valid] = ctr

        # Number of positive targets
        num_pos = pos_mask.sum().clamp(min=1.0)

        # 1. Classification Loss (Focal Loss over all points)
        loss_cls = sigmoid_focal_loss(flat_cls, target_cls).sum() / num_pos

        # 2. Bbox Loss (GIoU Loss over positive points only)
        if pos_mask.any():
            pos_pred_ltrb = flat_bbox[pos_mask]
            pos_tgt_ltrb = target_bbox[pos_mask]
            pos_pts = flat_points.unsqueeze(0).repeat(batch_size, 1, 1)[pos_mask]

            # Convert (l, t, r, b) to (x1, y1, x2, y2)
            pred_boxes = torch.stack([
                pos_pts[:, 0] - pos_pred_ltrb[:, 0],
                pos_pts[:, 1] - pos_pred_ltrb[:, 1],
                pos_pts[:, 0] + pos_pred_ltrb[:, 2],
                pos_pts[:, 1] + pos_pred_ltrb[:, 3],
            ], dim=-1)

            tgt_boxes = torch.stack([
                pos_pts[:, 0] - pos_tgt_ltrb[:, 0],
                pos_pts[:, 1] - pos_tgt_ltrb[:, 1],
                pos_pts[:, 0] + pos_tgt_ltrb[:, 2],
                pos_pts[:, 1] + pos_tgt_ltrb[:, 3],
            ], dim=-1)

            giou = compute_giou(pred_boxes, tgt_boxes)
            loss_box = (1.0 - giou).mean()

            # 3. Centerness Loss
            pos_pred_ctr = flat_ctr[pos_mask]
            pos_tgt_ctr = target_ctr[pos_mask]
            loss_ctr = F.binary_cross_entropy_with_logits(pos_pred_ctr, pos_tgt_ctr)
        else:
            loss_box = flat_bbox.sum() * 0.0
            loss_ctr = flat_ctr.sum() * 0.0

        total_loss = (
            self.loss_cls_weight * loss_cls
            + self.loss_box_weight * loss_box
            + self.loss_ctr_weight * loss_ctr
        )

        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_ctr": loss_ctr,
            "num_pos": num_pos,
        }
