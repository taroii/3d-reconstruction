"""
Curvature head (Option 1 from the brief): a DPT head predicting per-pixel
curvature Chat in W x H x 1, supervised against GT curvature.

This is the standalone, frozen-encoder learnability prototype -- the direct
analogue of the N_phi normal head, built on the FROZEN D2USt3R encoder. It
answers "can a head on these features recover curvature at all?" before we wire
the term into D2USt3R's full pair-training loop (L_static + L_dyn + lambda*L_curv;
see PLAN.md Phase 3, which lives in the DDUSt3R training code).

We hook 4 evenly spaced encoder blocks (canonical DPT-on-ViT) and reuse croco's
DPTOutputAdapter via dust3r's PixelwiseTaskWithDPT. Output: 2 channels ->
curvature (1) + confidence omega in [0,1] (1). Camera-frame, pose-free.
"""

import os
import sys
import torch
import torch.nn as nn

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

from dust3r.heads.dpt_head import PixelwiseTaskWithDPT  # noqa: E402


class CurvatureHead(nn.Module):
    """Frozen encoder + trainable DPT head -> (curvature, omega)."""

    def __init__(self, backbone, hooks=None, feature_dim=256, last_dim=128):
        super().__init__()
        self.enc_embed_dim = backbone.enc_embed_dim
        self.patch_embed = backbone.patch_embed
        self.enc_blocks = backbone.enc_blocks
        self.enc_norm = backbone.enc_norm
        enc_depth = len(self.enc_blocks)
        # 4 evenly spaced encoder blocks (0-indexed block outputs)
        self.hooks = hooks or [enc_depth // 4 - 1, enc_depth // 2 - 1,
                               3 * enc_depth // 4 - 1, enc_depth - 1]
        ed = self.enc_embed_dim
        self.dpt = PixelwiseTaskWithDPT(
            num_channels=2,                       # 1 curvature + 1 conf
            feature_dim=feature_dim, last_dim=last_dim,
            hooks_idx=[0, 1, 2, 3],               # index into our 4 collected layers
            dim_tokens=[ed, ed, ed, ed],
            postprocess=None, head_type="regression")
        # freeze the encoder; only the DPT head trains
        for p in self.patch_embed.parameters():
            p.requires_grad_(False)
        for p in self.enc_blocks.parameters():
            p.requires_grad_(False)
        for p in self.enc_norm.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, img, true_shape):
        """Run frozen encoder, collect features at the 4 hook layers."""
        x, pos = self.patch_embed(img, true_shape=true_shape)
        feats = []
        hookset = set(self.hooks)
        for i, blk in enumerate(self.enc_blocks):
            x = blk(x, pos)
            if i in hookset:
                feats.append(x)
        feats[-1] = self.enc_norm(feats[-1])      # norm the final hook (as stock head does)
        return feats

    def forward(self, img, true_shape):
        feats = self.encode(img, true_shape)
        H, W = int(true_shape[0, 0]), int(true_shape[0, 1])
        out = self.dpt(feats, (H, W))             # (B, 2, H, W)
        curv = out[:, 0]                          # raw (compressed) curvature
        omega = torch.sigmoid(out[:, 1])          # confidence in [0,1]
        return curv, omega

    def trainable_parameters(self):
        return [p for p in self.dpt.parameters() if p.requires_grad]


def curvature_conf_loss(curv_pred, omega, curv_gt, valid, alpha=0.2, delta=1.0):
    """Confidence-weighted robust (Huber) curvature loss, DUSt3R-style conf:
       mean_valid[ omega * huber(Chat - K) - alpha * log(omega) ].
    Huber (not MSE) because curvature is heavy-tailed. Returns (loss, L1_MAE)
    for monitoring; both computed on whatever target space (compressed or raw)
    the caller supervises in."""
    # curv_pred (B,H,W), curv_gt (B,H,W), valid (B,H,W)
    diff = curv_pred - curv_gt
    ad = diff.abs()
    huber = torch.where(ad <= delta, 0.5 * diff * diff, delta * (ad - 0.5 * delta))
    v = valid.float()
    denom = v.sum().clamp(min=1.0)
    loss = ((omega * huber - alpha * torch.log(omega.clamp(min=1e-6))) * v).sum() / denom
    with torch.no_grad():
        mae = (ad * v).sum() / denom
    return loss, mae
