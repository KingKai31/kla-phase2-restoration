"""
ROI-differential preservation loss (Task 3, final Phase 2 loss-term
candidate #6) - grounded in Zhang et al. 2025 (ScienceDirect), "An
unsupervised denoising via differential feature learning under high-level
noise indistinguishable from defect edges": apply stronger fidelity
preservation specifically in high-local-structure regions, since a
uniform loss can't distinguish real fine structure from noise by itself.

NOT assumed to help - see scripts/roi_loss_ablation.py for the pre-
registered controlled comparison (decision rule fixed BEFORE running it)
and scripts/defect_preservation_stress_test.py for the actual defect-
survival evidence this term is meant to justify.

Real, disclosed risk: local variance is elevated by BOTH real fine
structure AND noise - the naive version below (a hard top-k% variance
mask) cannot fully distinguish them, so this term could amplify noise in
high-variance-but-actually-noisy regions rather than protecting real
structure. Watched for directly in the ablation, not assumed away.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIPreservationLoss(nn.Module):
    def __init__(self, patch_size: int = 7, roi_percentile: float = 90.0, roi_boost: float = 3.0):
        super().__init__()
        self.patch_size = patch_size
        self.roi_percentile = roi_percentile
        self.roi_boost = roi_boost

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = self.patch_size // 2
        local_mean = F.avg_pool2d(target, self.patch_size, stride=1, padding=pad)
        local_sq_mean = F.avg_pool2d(target ** 2, self.patch_size, stride=1, padding=pad)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)

        # per-image threshold (quantile computed per-sample in the batch, not
        # pooled across the whole batch) - a fixed global threshold would let
        # one unusually high-variance image in a batch dominate which pixels
        # count as "ROI" for every other image in that batch
        b = target.shape[0]
        flat_var = local_var.view(b, -1)
        threshold = torch.quantile(flat_var, self.roi_percentile / 100.0, dim=1).view(b, 1, 1, 1)
        roi_mask = (local_var >= threshold).float()

        weight_map = 1.0 + self.roi_boost * roi_mask
        pixel_error = torch.abs(pred - target)
        return (pixel_error * weight_map).mean()


class StageBCompositeLossWithROI(nn.Module):
    """Wraps the existing 5-term StageBCompositeLoss and adds the 6th ROI
    term as a separate, independently-weighted component - the base 5-term
    loss is NOT modified in place, matching this project's established
    pattern of keeping validated loss stacks intact and adding new ones
    alongside rather than editing them (see stageB_composite.py's own
    docstring for the same rationale relative to Stage A's loss)."""

    def __init__(self, base_loss, roi_weight: float = 0.1,
                 roi_patch_size: int = 7, roi_percentile: float = 90.0, roi_boost: float = 3.0):
        super().__init__()
        self.base_loss = base_loss
        self.roi_loss = ROIPreservationLoss(roi_patch_size, roi_percentile, roi_boost)
        self.w_roi = roi_weight

    def forward(self, raw_pred, target):
        total, parts = self.base_loss(raw_pred, target)
        pred_c = torch.clamp(raw_pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        roi_loss = self.roi_loss(pred_c, target_c)
        total = total + self.w_roi * roi_loss
        parts["roi_preservation"] = roi_loss.item()
        return total, parts
