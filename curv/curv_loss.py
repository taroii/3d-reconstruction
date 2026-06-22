"""
Curvature-weighted loss (Option 2) for the D2USt3R finetune, plus a gradient
control. Drop-in criteria that reweight the per-pixel pointmap loss by a local
geometric signal, upweighting complex regions (edges/corners). No new params,
no prediction.

  CurvWeightedConfLoss - weight by |mean curvature| (2nd-order surface bending)
  GradWeightedConfLoss - weight by |pointmap gradient| (1st-order); the CONTROL
                         that tests whether the gain is curvature-specific or
                         just generic edge emphasis.

gamma=0 reduces EXACTLY to ConfLoss -> the baseline arm shares one code path.

Installed into the DDUSt3R clone by curv/server/install_curv_loss.py, which
appends `from dust3r.curv_loss import *` to dust3r/losses.py so these resolve in
training.py's eval() context. Signals are computed on the SAME normalized GT
pointmap the loss supervises (Regr3D.get_all_pts3d), so they are scale-consistent.
"""
import torch
from dust3r.losses import ConfLoss


def mean_curvature_mag(P):
    """|mean curvature| per pixel from a pointmap P (B,H,W,3) via the 4-neighbour
    umbrella Laplacian (2nd order). Returns (B,H,W); 1-px border = 0."""
    up, down = P[:, :-2, 1:-1, :], P[:, 2:, 1:-1, :]
    left, right = P[:, 1:-1, :-2, :], P[:, 1:-1, 2:, :]
    center = P[:, 1:-1, 1:-1, :]
    lap = (up + down + left + right) * 0.25 - center
    out = P.new_zeros(P.shape[:3])
    out[:, 1:-1, 1:-1] = 0.5 * lap.norm(dim=-1)
    return out


def pointmap_grad_mag(P):
    """|spatial gradient| of a pointmap P (B,H,W,3) (1st order) -- the control
    signal. Returns (B,H,W)."""
    gy = P.new_zeros(P.shape[:3])
    gx = P.new_zeros(P.shape[:3])
    gy[:, 1:, :] = (P[:, 1:, :, :] - P[:, :-1, :, :]).norm(dim=-1)
    gx[:, :, 1:] = (P[:, :, 1:, :] - P[:, :, :-1, :]).norm(dim=-1)
    return torch.sqrt(gx * gx + gy * gy)


def _norm_weight(S, valid, gamma, clamp):
    """w = 1 + gamma * (S normalized by its per-image valid-mean), clamped. Per-
    image normalization makes gamma scale-free across signals; clamp bounds
    silhouette/cliff spikes so a few pixels can't dominate."""
    m = valid.float()
    mean = (S * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1) + 1e-8
    return 1.0 + gamma * (S / mean[:, None, None]).clamp(max=clamp)


class _WeightedConfLoss(ConfLoss):
    """ConfLoss whose per-pixel pointmap loss is multiplied by a geometric weight
    before confidence weighting. Subclasses define _signal(P). gamma=0 == ConfLoss."""

    def __init__(self, pixel_loss, alpha=1, gamma=1.0, clamp=10.0):
        super().__init__(pixel_loss, alpha)
        self.gamma = gamma
        self.clamp = clamp

    def _signal(self, P):
        raise NotImplementedError

    def get_name(self):
        return f'{type(self).__name__}(g={self.gamma}, {self.pixel_loss})'

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        ((loss1, msk1), (loss2, msk2)), details = \
            self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        gt_pts1, gt_pts2, _, _, valid1, valid2, _ = \
            self.pixel_loss.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        w1 = _norm_weight(self._signal(gt_pts1), valid1, self.gamma, self.clamp)
        w2 = _norm_weight(self._signal(gt_pts2), valid2, self.gamma, self.clamp)
        loss1 = loss1 * w1[msk1]                  # msk1 == valid1, same pixel order
        loss2 = loss2 * w2[msk2]

        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        return conf_loss1 + conf_loss2, dict(
            conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2),
            gamma=self.gamma, **details)


class CurvWeightedConfLoss(_WeightedConfLoss):
    """Weight by |mean curvature| (2nd-order). The method."""
    def _signal(self, P):
        return mean_curvature_mag(P)


class GradWeightedConfLoss(_WeightedConfLoss):
    """Weight by |pointmap gradient| (1st-order). The control."""
    def _signal(self, P):
        return pointmap_grad_mag(P)
