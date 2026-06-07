"""
BandPCOptimizer (PLAN step 1b): PointCloudOptimizer + the high-band normal prior
L_geo and the DC scale anchor L_dc. Non-invasive: we let the parent compute the
full L_align (corr + flow + temporal + depth-prior) and ADD our terms.

L_geo is pose-free by construction -- normals come from the camera-frame pointmap
_fast_depthmap_to_pts3d(depth, grid, focal, pp), i.e. the value BEFORE
geotrf(im_poses, .) -- so it sharpens within-frame shape without touching pose.

UNTESTED ON GPU until N_phi is trained; the math mirrors normals.py exactly.
"""
import torch
import torch.nn.functional as F

from dust3r.cloud_opt.optimizer import PointCloudOptimizer, _fast_depthmap_to_pts3d


def normals_from_points_t(P):
    """Differentiable unit normals from a camera-frame point map (B,H,W,3),
    oriented toward the camera (n . P < 0). Matches normals.normals_from_points."""
    gy = torch.gradient(P, dim=1)[0]
    gx = torch.gradient(P, dim=2)[0]
    n = torch.cross(gx, gy, dim=-1)
    n = F.normalize(n, dim=-1, eps=1e-12)
    flip = (n * P).sum(-1, keepdim=True) > 0
    return torch.where(flip, -n, n)


class BandPCOptimizer(PointCloudOptimizer):
    def __init__(self, *args, w_geo=0.0, w_dc=0.0, eta=0.0, s0=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_geo = w_geo
        self.w_dc = w_dc
        self.eta = eta                 # curvature term (deferred; needs sparse L chi)
        self.s0 = s0                   # DC anchor target (metric scale)
        self.Nhat = None               # (N,H,W,3) predicted normals (fixed)
        self.omega = None              # (N,H,W) predicted confidence (fixed)

    @torch.no_grad()
    def set_normal_prior(self, Nhat, omega):
        """Stationary attractor: fixed appearance->normal predictions on the grid."""
        self.register_buffer("Nhat", Nhat.to(self.device))
        self.register_buffer("omega", omega.to(self.device))

    def _camera_frame_points(self):
        """Pose-free per-image camera-frame pointmaps (list of (H,W,3))."""
        depth = self.get_depthmaps(raw=True)               # (N, max_area)
        rel = _fast_depthmap_to_pts3d(depth, self._grid, self.get_focals(),
                                      pp=self.get_principal_points())
        return [rel[i, :h * w].view(h, w, 3) for i, (h, w) in enumerate(self.imshapes)]

    def geo_loss(self):
        P = torch.stack(self._camera_frame_points())       # (N,H,W,3)  (Sintel: same shape)
        N = normals_from_points_t(P)                       # (N,H,W,3)
        cos = (N * self.Nhat).sum(-1).clamp(-1, 1)         # (N,H,W)
        gate = self.omega
        if self.dynamic_masks is not None:
            dyn = torch.stack([m.float().to(self.device) for m in self.dynamic_masks])
            gate = gate * (1.0 - dyn)
        return (gate * (1.0 - cos)).sum() / gate.sum().clamp(min=1.0)

    def dc_loss(self):
        depths = torch.cat([d.reshape(-1) for d in self.get_depthmaps(raw=False)])
        return (torch.log(depths.median()) - torch.log(torch.tensor(
            self.s0, device=self.device))) ** 2

    def forward_batchify(self, epoch=9999):
        loss, flow_loss = super().forward_batchify(epoch)
        if self.w_geo > 0 and self.Nhat is not None:
            loss = loss + self.w_geo * self.geo_loss()
        if self.w_dc > 0 and self.s0 is not None:
            loss = loss + self.w_dc * self.dc_loss()
        return loss, flow_loss

    @torch.no_grad()
    def residual_field(self):
        """Per-image per-pixel alignment residual (for the spectral diagnostic):
        ||proj_pts3d - aligned_pred|| pooled (max) over incident edges."""
        from dust3r.utils.geometry import geotrf
        pw_poses = self.get_pw_poses()
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj = self.get_pts3d(raw=True)                    # (N, max_area, 3)
        ai = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aj = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)
        res = [torch.zeros(self.max_area, device=self.device) for _ in range(self.n_imgs)]
        ri = (proj[self._ei] - ai).norm(dim=-1)            # (E, max_area)
        rj = (proj[self._ej] - aj).norm(dim=-1)
        for e, (i, j) in enumerate(self.edges):
            res[i] = torch.maximum(res[i], ri[e])
            res[j] = torch.maximum(res[j], rj[e])
        return [res[i][:h * w].view(h, w) for i, (h, w) in enumerate(self.imshapes)]
