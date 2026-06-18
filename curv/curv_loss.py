"""
Option 2 (curvature-weighted loss) for the D2USt3R finetune. A drop-in criterion
that reweights the per-pixel pointmap loss by local GT curvature, upweighting
geometrically complex regions (edges/corners). No new params, no prediction.

Installed into the DDUSt3R clone by curv/server/install_curv_loss.py, which copies
this file to DDUSt3R/dust3r/curv_loss.py and appends
`from dust3r.curv_loss import *` to dust3r/losses.py, so the criterion string
`CurvWeightedConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, gamma=1.0)`
resolves in training.py's eval() context.

Curvature is computed on the SAME normalized GT pointmap the loss supervises
(gt_pts from Regr3D.get_all_pts3d), so it is scale-consistent. We use the
magnitude of discrete mean curvature (umbrella Laplacian) as the weight signal.

gamma=0 reduces EXACTLY to ConfLoss -> Arm A and Arm B share one code path.
"""
import torch
from dust3r.losses import ConfLoss


def mean_curvature_mag(P):
    """|mean curvature| per pixel from a pointmap P (B,H,W,3) via the 4-neighbour
    umbrella Laplacian. Returns (B,H,W); the 1-px border is left at 0."""
    up, down = P[:, :-2, 1:-1, :], P[:, 2:, 1:-1, :]
    left, right = P[:, 1:-1, :-2, :], P[:, 1:-1, 2:, :]
    center = P[:, 1:-1, 1:-1, :]
    lap = (up + down + left + right) * 0.25 - center
    Hc = 0.5 * lap.norm(dim=-1)
    out = P.new_zeros(P.shape[:3])
    out[:, 1:-1, 1:-1] = Hc
    return out


def curvature_weight_map(P, valid, gamma, clamp=10.0):
    """w = 1 + gamma * (|K| normalized by its per-image valid-mean), clamped.
    Per-image normalization makes gamma scale-free; the clamp bounds silhouette /
    depth-cliff spikes so a handful of pixels can't dominate the loss."""
    K = mean_curvature_mag(P)
    m = valid.float()
    meanK = (K * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1) + 1e-8
    kb = K / meanK[:, None, None]
    return 1.0 + gamma * kb.clamp(max=clamp)


class CurvWeightedConfLoss(ConfLoss):
    """ConfLoss whose per-pixel pointmap loss is multiplied by a curvature weight
    before confidence weighting. gamma=0 == plain ConfLoss (Arm A baseline)."""

    def __init__(self, pixel_loss, alpha=1, gamma=1.0, clamp=10.0):
        super().__init__(pixel_loss, alpha)
        self.gamma = gamma
        self.clamp = clamp

    def get_name(self):
        return f'CurvWeightedConfLoss(g={self.gamma}, {self.pixel_loss})'

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        ((loss1, msk1), (loss2, msk2)), details = \
            self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        # recover the SAME normalized GT pointmaps + masks the loss used
        gt_pts1, gt_pts2, _, _, valid1, valid2, _ = \
            self.pixel_loss.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        w1 = curvature_weight_map(gt_pts1, valid1, self.gamma, self.clamp)
        w2 = curvature_weight_map(gt_pts2, valid2, self.gamma, self.clamp)
        loss1 = loss1 * w1[msk1]                 # msk1 == valid1, same pixel order
        loss2 = loss2 * w2[msk2]

        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        return conf_loss1 + conf_loss2, dict(
            conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2),
            curv_gamma=self.gamma, **details)
