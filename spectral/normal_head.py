"""
N_phi: single-image appearance->normal prior (PLAN step 1a). Full DPT-style head
on the FROZEN DUSt3R encoder (encoder-only = single-image, as the template
requires; the stock dust3r head instead mixes decoder/cross-attention layers).

We hook 4 evenly spaced encoder blocks (canonical DPT-on-ViT) and reuse croco's
DPTOutputAdapter via dust3r's PixelwiseTaskWithDPT. Output: 4 channels ->
unit normal (3) + confidence omega in [0,1] (1). Camera-frame, pose-free.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

from dust3r.heads.dpt_head import PixelwiseTaskWithDPT  # noqa: E402


class NormalHead(nn.Module):
    """Frozen encoder + trainable DPT head -> (normal, omega)."""

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
            num_channels=4,                       # 3 normal + 1 conf
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
        out = self.dpt(feats, (H, W))             # (B, 4, H, W)
        normal = F.normalize(out[:, :3], dim=1)   # unit camera-frame normal
        omega = torch.sigmoid(out[:, 3])          # confidence in [0,1]
        return normal, omega

    def trainable_parameters(self):
        return [p for p in self.dpt.parameters() if p.requires_grad]


def angular_conf_loss(normal_pred, omega, normal_gt, valid, alpha=0.2):
    """Confidence-weighted angular loss (DUSt3R-style conf):
       mean_valid[ omega * (1 - <n_pred, n_gt>) - alpha * log(omega) ].
    Returns (loss, mean_angular_error_degrees) for monitoring."""
    # normal_pred (B,3,H,W), normal_gt (B,H,W,3), valid (B,H,W)
    npred = normal_pred.permute(0, 2, 3, 1)                     # (B,H,W,3)
    cos = (npred * normal_gt).sum(-1).clamp(-1, 1)              # (B,H,W)
    ang = 1.0 - cos
    v = valid.float()
    denom = v.sum().clamp(min=1.0)
    loss = ((omega * ang - alpha * torch.log(omega.clamp(min=1e-6))) * v).sum() / denom
    with torch.no_grad():
        deg = torch.rad2deg(torch.arccos(cos.clamp(-1, 1)))
        mae = (deg * v).sum() / denom
    return loss, mae
