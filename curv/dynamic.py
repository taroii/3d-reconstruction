"""
Dynamic-aware supervision pieces for the Faithful L_cc route (Task 2). Construct
-- from GT depth + relative pose + GT optical flow -- the masks and the
dynamic-aligned (SDAP) pointmap target that D2USt3R's unreleased dynamic loss
used, so we can reintroduce it and stack L_cc (cc_loss.py) on top.

torch-ONLY (no dust3r import) so the synthetic gates below run on the local PC:

    python dynamic.py        # gates: cam_flow, M_dyn, M_occ, SDAP target

Conventions (shared with cc_loss.flow_warp): pointmaps are (B,H,W,3); flow is
(B,2,H,W) with channel 0 = x (col), 1 = y (row); a flow maps a pixel in ITS view
to the (x,y) sampling location in the OTHER view. T is (B,4,4), cam1->cam2.

  cam_flow  : f_cam, the camera-only (rigid/static) flow view1->view2 from GT
              depth+pose. On static pixels f_cam == f_gt; the discrepancy is
              exactly object motion -> that is M_dyn (Eq 5).
  M_dyn     : ||f_cam - f_gt|| > tau           (Eq 5)
  M_occ     : ||f + warp(b, f)|| > t           (Eq 4, fwd/bwd consistency)
  sdap_target: dynamic pixels of view2's GT target are replaced by the flow-warp
              of X-bar^{1,1} (the view-1 GT pointmap) via the backward flow b;
              static pixels keep X-bar^{2,1}. Returns (target, static_mask) so the
              caller masks the plain Regr3D term by (1-M_dyn) and supervises the
              dynamic pixels against the aligned target.
"""
import torch

from cc_loss import flow_warp


def cam_flow(depth1, K1, K2, T):
    """Camera-induced (static-scene) flow view1->view2, (B,2,H,W). Back-project
    view-1 pixels with GT depth, move by the relative pose T (cam1->cam2),
    reproject with K2; f_cam = reprojected_pixel - pixel."""
    B, H, W = depth1.shape
    dev = depth1.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                            indexing='ij')
    pix = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1).float()    # (H,W,3)
    pix = pix[None].expand(B, H, W, 3)
    rays = torch.einsum('bdc,bhwc->bhwd', torch.inverse(K1), pix)       # (B,H,W,3)
    X1 = rays * depth1.unsqueeze(-1)
    R, t = T[:, :3, :3], T[:, :3, 3]
    X2 = torch.einsum('bdc,bhwc->bhwd', R, X1) + t[:, None, None, :]
    proj = torch.einsum('bdc,bhwc->bhwd', K2, X2)
    z = proj[..., 2:3].clamp(min=1e-6)
    uv2 = proj[..., :2] / z                                             # (B,H,W,2)
    f = uv2 - pix[..., :2]
    return f.permute(0, 3, 1, 2).contiguous()                          # (B,2,H,W)


def dynamic_mask(f_cam, f_gt, tau=1.5):
    """M_dyn (Eq 5): pixels whose true flow departs from the camera-only flow by
    more than tau px -> moving content. (B,1,H,W) float {0,1}."""
    err = (f_cam - f_gt).norm(dim=1, keepdim=True)
    return (err > tau).float()


def occlusion_mask(f, b, t=1.5):
    """M_occ (Eq 4): forward/backward GT-flow inconsistency. For a view-1 pixel x,
    follow f to view 2 then b back; ||f(x) + b(x+f(x))|| > t -> occluded/unreliable.
    (B,1,H,W) float {0,1}. (out-of-bounds correspondences count as occluded.)"""
    b_warp, inb = flow_warp(b, f)                       # b sampled at x+f(x)
    err = (f + b_warp).norm(dim=1, keepdim=True)
    return ((err > t) | (inb < 0.5)).float()


def sdap_target(Xbar1, Xbar2, b, M_dyn2):
    """Static-Dynamic-Aware Pointmap target for view 2 (B,H,W,3). On STATIC pixels
    keep the rigid GT target X-bar^{2,1}; on DYNAMIC pixels use the flow-warp of
    the view-1 GT pointmap X-bar^{1,1} sampled at j+b(j) (its true frame-1
    location). Also returns the in-bounds-aware static-term mask (1-M_dyn) for the
    plain Regr3D term, and the dynamic-term mask M_dyn*inbounds."""
    P1 = Xbar1.permute(0, 3, 1, 2)                      # (B,3,H,W)
    warp, inb = flow_warp(P1, b)                        # sample X-bar1 at j+b(j)
    warp = warp.permute(0, 2, 3, 1)                     # (B,H,W,3)
    md = M_dyn2.permute(0, 2, 3, 1)                     # (B,H,W,1)
    inbm = inb.permute(0, 2, 3, 1)
    use_dyn = md * inbm                                 # dynamic AND warp in-bounds
    target = (1 - use_dyn) * Xbar2 + use_dyn * warp
    static_mask = (1 - M_dyn2)                          # (B,1,H,W) for Regr3D term
    dyn_mask = (M_dyn2 * inb)                           # (B,1,H,W) for dynamic term
    return target, static_mask, dyn_mask


# ----------------------------------------------------------------------------
# Validation gates (python dynamic.py). No training, CPU-friendly.
# ----------------------------------------------------------------------------
def _gates():
    torch.manual_seed(0)
    B, H, W = 1, 40, 56

    # --- cam_flow: identity pose -> zero flow; pure translation -> matches a
    #     direct reprojection (round-trip self-consistency). ---
    depth = 3.0 + 0.5 * torch.rand(B, H, W)
    K = torch.tensor([[60., 0, W / 2], [0, 60., H / 2], [0, 0, 1.]])[None]
    T_id = torch.eye(4)[None]
    f0 = cam_flow(depth, K, K, T_id)
    A_ok = f0.abs().max() < 1e-3

    T_tr = torch.eye(4)[None].clone(); T_tr[0, 0, 3] = 0.2     # translate in x
    f_cam = cam_flow(depth, K, K, T_tr)
    moved = f_cam.abs().mean() > 1e-3                          # nonzero, sane

    # --- M_dyn: static scene (f_gt == f_cam) -> empty; inject a moving patch
    #     into f_gt -> M_dyn fires exactly there. ---
    f_gt = f_cam.clone()
    f_gt[:, 0, 10:20, 15:30] += 5.0                           # a "moving object"
    M = dynamic_mask(f_cam, f_gt, tau=1.5)
    Bdyn_ok = (M[:, :, 10:20, 15:30].mean() > 0.99 and
               M.sum() - M[:, :, 10:20, 15:30].sum() < 1)

    # --- M_occ: consistent shift (b = -f) -> no occlusion; broken b -> occluded. ---
    dx, dy = 3, 2
    f = torch.zeros(B, 2, H, W); f[:, 0] = dx; f[:, 1] = dy
    b = torch.zeros(B, 2, H, W); b[:, 0] = -dx; b[:, 1] = -dy
    occ_consistent = occlusion_mask(f, b, t=0.5)
    # inner region where x+f stays in-bounds (else correctly flagged occluded)
    inner = occ_consistent[:, :, dy + 2:H - 2 - dy, dx + 2:W - 2 - dx].mean()
    occ_broken = occlusion_mask(f, torch.zeros_like(b), t=0.5).mean()
    Cocc_ok = inner < 0.02 and occ_broken > 0.5

    # --- SDAP: static (M_dyn=0) -> target == Xbar2; a dynamic patch with known
    #     backward flow -> target there == warp of Xbar1. ---
    Xbar1 = torch.randn(B, H, W, 3)
    Xbar2 = torch.randn(B, H, W, 3)
    tgt_static, smask, dmask = sdap_target(Xbar1, Xbar2, torch.zeros(B, 2, H, W),
                                           torch.zeros(B, 1, H, W))
    static_ok = torch.allclose(tgt_static, Xbar2, atol=1e-5)

    Md = torch.zeros(B, 1, H, W); Md[:, :, 12:24, 18:34] = 1
    bflow = torch.zeros(B, 2, H, W); bflow[:, 0] = -dx; bflow[:, 1] = -dy
    Xshift = torch.roll(Xbar1, shifts=(dy, dx), dims=(1, 2))   # Xshift[j]=Xbar1[j-shift]
    tgt, _, _ = sdap_target(Xbar1, Xbar2, bflow, Md)
    # on the patch, target should equal Xbar1 sampled at j-shift == Xshift
    patch = (slice(None), slice(14, 22), slice(20, 32))
    dyn_ok = torch.allclose(tgt[patch], Xshift[patch], atol=1e-4)
    # off the patch, untouched
    off_ok = torch.allclose(tgt[:, 0:8, 0:8], Xbar2[:, 0:8, 0:8], atol=1e-5)

    print(f"Gate A  cam_flow identity=0   {'ok' if A_ok else 'BAD'}  (max {f0.abs().max():.2e})")
    print(f"Gate A' cam_flow translate>0  {'ok' if moved else 'BAD'}")
    print(f"Gate B  M_dyn localizes patch {'ok' if Bdyn_ok else 'BAD'}")
    print(f"Gate C  M_occ consistent=0    {'ok' if Cocc_ok else 'BAD'}  (inner {inner:.3f}, broken {occ_broken:.2f})")
    print(f"Gate D  SDAP static==Xbar2    {'ok' if static_ok else 'BAD'}")
    print(f"Gate E  SDAP dyn==warp(Xbar1) {'ok' if (dyn_ok and off_ok) else 'BAD'}")
    ok = all([A_ok, moved, Bdyn_ok, Cocc_ok, static_ok, dyn_ok, off_ok])
    print("DYNAMIC GATES:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _gates()
