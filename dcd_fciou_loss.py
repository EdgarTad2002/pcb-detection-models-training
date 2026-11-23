"""Explicit paper-equation FC-IoU adaptation, separate from unmodified DCD-Net code.

Reference: DCD-Net, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3629085, equation 5.
This is a project adaptation, not a claim that the published repository enables it.
"""


def paper_fciou_loss(ciou_score, raw_iou, lower=0.05, upper=0.95):
    if not 0 <= lower < upper <= 1:
        raise ValueError("Require 0 <= lower < upper <= 1")
    focaler = ((raw_iou - lower) / (upper - lower)).clamp(0, 1)
    return 1 - ciou_score + raw_iou - focaler


def enable_paper_fciou(lower=0.05, upper=0.95):
    """Patch only the imported isolated author's BboxLoss; preserve its DFL term."""
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.metrics import bbox_iou
    if getattr(BboxLoss, "_pcb_paper_fciou", False):
        raise RuntimeError("Paper FC-IoU already enabled in this process")
    # Validate bounds before changing any process state.
    if not 0 <= lower < upper <= 1:
        raise ValueError("Require 0 <= lower < upper <= 1")
    original = BboxLoss.forward

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        loss_ciou, loss_dfl = original(self, pred_dist, pred_bboxes, anchor_points,
                                       target_bboxes, target_scores, target_scores_sum, fg_mask)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        focaler = ((iou - lower) / (upper - lower)).clamp(0, 1)
        correction = ((iou - focaler) * weight).sum() / target_scores_sum
        return loss_ciou + correction, loss_dfl

    BboxLoss.forward = forward
    BboxLoss._pcb_paper_fciou = True
