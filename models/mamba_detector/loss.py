"""
Anchor-Free Detection Loss & Target Assignment
================================================
Implements Center-Based Target Assignment, Sigmoid Focal Loss,
Generalized IoU (GIoU) Loss, and Centerness Quality Loss.
Enhanced with:
1. Guaranteed Positive Target Assignment for micro-components.
2. Pyramid scale allocation (P2 through P5).
3. Focal loss gamma=2.5, label smoothing, and loss reweighting (cls=1.5, box=5.0, ctr=1.0).
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
    gamma: float = 2.5,
) -> torch.Tensor:
    """
    Sigmoid focal loss for class imbalance with enhanced gamma=2.5.
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
    Enhanced Anchor-Free Detection Loss for VMamba Detector.
    Features:
    - Micro-component guaranteed positive assignment
    - Multi-scale pyramid level allocation (reg_ranges)
    - Multitask reweighting (cls=1.5, box=5.0, ctr=1.0)
    - Focal loss gamma=2.5 with label smoothing
    """

    def __init__(
        self,
        num_classes: int = 4,
        strides: List[int] = [4, 8, 16, 32],
        reg_ranges: List[Tuple[float, float]] = [(-1, 48), (24, 96), (64, 192), (128, 100000)],
        center_radius: float = 1.5,
        loss_cls_weight: float = 1.5,
        loss_box_weight: float = 5.0,
        loss_ctr_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.5,
        label_smoothing: float = 0.10,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.reg_ranges = reg_ranges
        self.center_radius = center_radius
        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight
        self.loss_ctr_weight = loss_ctr_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

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
        gt_boxes_list: List[torch.Tensor],  # List of (N_i, 5) tensors: [cls_id, x1, y1, x2, y2]
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
        all_reg_min = []
        all_reg_max = []

        for lvl, (cls_lvl, bbox_lvl, ctr_lvl, pts_lvl, stride) in enumerate(
            zip(cls_scores, bbox_preds, centernesses, points_per_lvl, self.strides)
        ):
            B, C, H, W = cls_lvl.shape
            num_pts = H * W
            all_cls.append(cls_lvl.permute(0, 2, 3, 1).reshape(B, -1, C))
            all_bbox.append(bbox_lvl.permute(0, 2, 3, 1).reshape(B, -1, 4))
            all_ctr.append(ctr_lvl.permute(0, 2, 3, 1).reshape(B, -1))
            all_points.append(pts_lvl)
            all_strides.append(torch.full((num_pts,), stride, device=device, dtype=torch.float32))
            rmin, rmax = self.reg_ranges[lvl]
            all_reg_min.append(torch.full((num_pts,), rmin, device=device, dtype=torch.float32))
            all_reg_max.append(torch.full((num_pts,), rmax, device=device, dtype=torch.float32))

        flat_cls = torch.cat(all_cls, dim=1)  # (B, Total_pts, num_classes)
        flat_bbox = torch.cat(all_bbox, dim=1)  # (B, Total_pts, 4) [l, t, r, b]
        flat_ctr = torch.cat(all_ctr, dim=1)  # (B, Total_pts)
        flat_points = torch.cat(all_points, dim=0)  # (Total_pts, 2)
        flat_strides = torch.cat(all_strides, dim=0)  # (Total_pts,)
        flat_reg_min = torch.cat(all_reg_min, dim=0)  # (Total_pts,)
        flat_reg_max = torch.cat(all_reg_max, dim=0)  # (Total_pts,)

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
            num_gt = gt.shape[0]

            cx = (gt_x1 + gt_x2) * 0.5
            cy = (gt_y1 + gt_y2) * 0.5
            w = (gt_x2 - gt_x1).clamp(min=1.0)
            h = (gt_y2 - gt_y1).clamp(min=1.0)
            max_obj_dim = torch.maximum(w, h)  # (Num_gt,)

            # Improvement 1: Dynamic expansion for micro-components
            # If a component is small relative to stride, expand effective box by at least 0.65*stride
            # around its center so that the nearest grid point is GUARANTEED to be inside!
            expansion = flat_strides.unsqueeze(1) * 0.65
            eff_x1 = torch.minimum(gt_x1.unsqueeze(0), cx.unsqueeze(0) - expansion)
            eff_x2 = torch.maximum(gt_x2.unsqueeze(0), cx.unsqueeze(0) + expansion)
            eff_y1 = torch.minimum(gt_y1.unsqueeze(0), cy.unsqueeze(0) - expansion)
            eff_y2 = torch.maximum(gt_y2.unsqueeze(0), cy.unsqueeze(0) + expansion)

            # Distances from points to effective box boundaries: (Total_pts, Num_gt)
            l = px.unsqueeze(1) - eff_x1
            t = py.unsqueeze(1) - eff_y1
            r = eff_x2 - px.unsqueeze(1)
            b = eff_y2 - py.unsqueeze(1)

            is_in_box = (l > 0) & (t > 0) & (r > 0) & (b > 0)

            # Center sampling: points must also be within center_radius * stride of box center
            c_dist_x = (px.unsqueeze(1) - cx.unsqueeze(0)).abs()
            c_dist_y = (py.unsqueeze(1) - cy.unsqueeze(0)).abs()
            radius_bound = flat_strides.unsqueeze(1) * self.center_radius
            is_in_center = (c_dist_x <= radius_bound) & (c_dist_y <= radius_bound)

            # Improvement 2: Multi-Scale Pyramid Level Allocation
            # A box is only assigned to pyramid levels matching its scale
            scale_valid = (max_obj_dim.unsqueeze(0) >= flat_reg_min.unsqueeze(1)) & (
                max_obj_dim.unsqueeze(0) <= flat_reg_max.unsqueeze(1)
            )

            # Final candidate mask per point and per ground truth
            candidate_mask = is_in_box & is_in_center & scale_valid

            # Fallback for any GT that didn't match: assign to nearest point on level 0 (P2, stride 4)
            unmatched_gts = (~candidate_mask.any(dim=0)).nonzero(as_tuple=True)[0]
            if len(unmatched_gts) > 0:
                p2_mask = (flat_strides == self.strides[0])
                p2_indices = p2_mask.nonzero(as_tuple=True)[0]
                p2_px, p2_py = px[p2_indices], py[p2_indices]
                for ug_idx in unmatched_gts:
                    ug_cx, ug_cy = cx[ug_idx], cy[ug_idx]
                    dists = (p2_px - ug_cx) ** 2 + (p2_py - ug_cy) ** 2
                    nearest_local = dists.argmin()
                    nearest_pt_idx = p2_indices[nearest_local]
                    candidate_mask[nearest_pt_idx, ug_idx] = True

            if not candidate_mask.any():
                continue

            areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)  # (Num_gt,)

            # For each point, assign to the candidate box with the smallest area (prioritize small components)
            matched_areas = torch.where(candidate_mask, areas.unsqueeze(0), torch.tensor(float("inf"), device=device))
            min_area, min_idx = matched_areas.min(dim=1)
            valid = min_area < float("inf")

            if valid.any():
                pos_mask[b_idx, valid] = True
                matched_gt_idx = min_idx[valid]
                matched_cls = gt_cls[matched_gt_idx]

                # Improvement 3: Label smoothing on class targets
                # Soften one-hot targets to prevent overconfidence
                smooth_val = self.label_smoothing / self.num_classes
                target_cls[b_idx, valid] = smooth_val
                target_cls[b_idx, valid, matched_cls] = 1.0 - self.label_smoothing + smooth_val

                # Bbox distances [l, t, r, b] from selected ground truth
                valid_pt_indices = valid.nonzero(as_tuple=True)[0]
                tgt_l = l[valid_pt_indices, matched_gt_idx]
                tgt_t = t[valid_pt_indices, matched_gt_idx]
                tgt_r = r[valid_pt_indices, matched_gt_idx]
                tgt_b = b[valid_pt_indices, matched_gt_idx]
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

        # 1. Classification Loss (Sigmoid Focal Loss with gamma=2.5)
        loss_cls = sigmoid_focal_loss(
            flat_cls, target_cls, alpha=self.focal_alpha, gamma=self.focal_gamma
        ).sum() / num_pos

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
