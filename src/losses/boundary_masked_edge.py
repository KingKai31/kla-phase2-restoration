"""
Item 3 (improvement pass, time-boxed): the direct structural fix for the
Axis 5 diagnosis (reports/TECHNICAL_AUDIT.md sec.9) - the existing
SobelEdgeLoss (src/losses/stageB_composite.py) averages gradient-
magnitude error over ALL pixels, so real boundaries (<5% of pixels) are
diluted by 90%+ flat-region gradient noise. Doubling its weight (Axis 1b)
changed nothing because a diluted term scaled by 2 is still diluted.

BoundaryMaskedEdgeLoss computes the same Charbonnier-on-gradient-magnitude
loss, but only over pixels where the GT gradient magnitude is in the top
decile PER IMAGE - i.e. only at real boundaries. Uses reflect padding
(not the base SobelEdgeLoss's zero padding) so this term does not inherit
the border-brightness artifact characterized in
tests/test_src_modules.py::test_sobel_border_is_contaminated_by_zero_padding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryMaskedEdgeLoss(nn.Module):
    def __init__(self, percentile: float = 90.0, eps: float = 1e-3):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.percentile = percentile
        self.eps2 = eps ** 2

    def _gradient_magnitude(self, x):
        xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(xp, self.kx)
        gy = F.conv2d(xp, self.ky)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target):
        pred_edges = self._gradient_magnitude(pred)
        target_edges = self._gradient_magnitude(target)

        b = target_edges.shape[0]
        flat = target_edges.view(b, -1)
        thresh = torch.quantile(flat, self.percentile / 100.0, dim=1).view(b, 1, 1, 1)
        boundary_mask = (target_edges >= thresh).float()

        diff = pred_edges - target_edges
        charb = torch.sqrt(diff * diff + self.eps2)
        # per-image mean over the masked (boundary) pixels only, not a
        # blanket sum-over-batch mean (which would let one image with a
        # larger boundary fraction dominate)
        masked_sum = (charb * boundary_mask).sum(dim=(1, 2, 3))
        mask_count = boundary_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (masked_sum / mask_count).mean()
