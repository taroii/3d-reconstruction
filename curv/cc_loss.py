"""
Cross-frame curvature CONSISTENCY loss L_cc (Task 2). Self-consistency on the
*predicted* pointmaps -- no GT curvature. The idea: a 3D surface point has the
same intrinsic curvature whichever view sees it, so for corresponding pixels
(linked by optical flow) kappa(X_hat^{1,1}) and kappa(X_hat^{2,1}) should agree.
Penalizing their disagreement is a GT-free regularizer that bites hardest where
the network is least constrained -- dynamic / boundary regions.

This module is intentionally torch-ONLY (no dust3r import) so the validation
gates below run on the local PC / CPU without the heavy training env:

    python cc_loss.py        # synthetic gates: motion-invariance, warp, deform

`kappa` here is the SAME umbrella-Laplacian magnitude as
curv_loss.mean_curvature_mag (kept duplicated only to avoid importing
dust3r.losses); keep the two formulas identical if either changes.

Correspondence / masks: L_cc needs optical flow f (I1->I2) and b (I2->I1) plus
the dynamic/occlusion validity masks. In our finetune (plain ConfLoss, no
dynamic-flow loss) these are NOT already in the loss path -- they come from the
dataset's GT flow/trajectories at training time (see INTEGRATION notes). The
warp uses the brief's convention: `flow` maps a TARGET-view pixel to the
SOURCE-view (x, y) sampling location.
"""
import torch
import torch.nn.functional as F


def mean_curvature_mag(P):
    """|mean curvature| per pixel from pointmap P (B,H,W,3) via the 4-neighbour
    umbrella Laplacian. Returns (B,H,W); 1-px border = 0. MUST match
    curv_loss.mean_curvature_mag."""
    up, down = P[:, :-2, 1:-1, :], P[:, 2:, 1:-1, :]
    left, right = P[:, 1:-1, :-2, :], P[:, 1:-1, 2:, :]
    center = P[:, 1:-1, 1:-1, :]
    lap = (up + down + left + right) * 0.25 - center
    out = P.new_zeros(P.shape[:3])
    out[:, 1:-1, 1:-1] = 0.5 * lap.norm(dim=-1)
    return out


def kappa(P):
    """Teacher/student curvature channel: (B,H,W,3) -> (B,1,H,W)."""
    return mean_curvature_mag(P).unsqueeze(1)


def flow_warp(field, flow):
    """Sample `field` (B,1,H,W) on the SOURCE-view grid at the locations given by
    `flow` (B,2,H,W) which maps each TARGET-view pixel -> SOURCE (x, y). Returns
    (warped (B,1,H,W), inbounds (B,1,H,W) in {0,1})."""
    B, _, H, W = field.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=field.device),
                            torch.arange(W, device=field.device), indexing='ij')
    base = torch.stack((xs, ys)).float()[None]              # (1,2,H,W)
    coords = base + flow
    gx = 2 * coords[:, 0] / (W - 1) - 1
    gy = 2 * coords[:, 1] / (H - 1) - 1
    grid = torch.stack((gx, gy), dim=-1)                    # (B,H,W,2)
    warped = F.grid_sample(field, grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)
    inbounds = ((gx.abs() <= 1) & (gy.abs() <= 1)).unsqueeze(1).float()
    return warped, inbounds


def _term(k_student, k_teacher_src, flow, mask, conf, clamp=None):
    """One directional consistency term: warp the teacher curvature (on the
    SOURCE grid) into the student grid via `flow`, penalize |k_student - warp|
    (teacher detached), conf+mask weighted, mean-reduced. Returns scalar."""
    k_w, inb = flow_warp(k_teacher_src, flow)
    m = mask * inb                                          # (B,1,H,W)
    res = (k_student - k_w.detach()).abs()
    if clamp is not None:
        res = res.clamp(max=clamp)
    num = (conf * m * res).sum()
    den = m.sum().clamp(min=1.0)
    return num / den


def cc_loss(X1, X2, f, b, m1=None, m2=None, c1=None, c2=None, clamp=None):
    """Symmetric stop-gradient curvature-consistency loss.

      X1, X2 : (B,H,W,3) normalized predicted pointmaps  X_hat^{1,1}, X_hat^{2,1}
      f      : (B,2,H,W) flow I1->I2 (maps a view-1 pixel to view-2 (x,y))
      b      : (B,2,H,W) flow I2->I1 (maps a view-2 pixel to view-1 (x,y))
      m1, m2 : (B,1,H,W) per-view validity (Mdyn*(1-Mocc)*valid); default all-ones
      c1, c2 : (B,1,H,W) per-view confidence weights; default all-ones

    Term 2 supervises view 2 with the view-1 teacher sampled at i+b(i);
    term 1 supervises view 1 with the view-2 teacher sampled at j+f(j)."""
    k1, k2 = kappa(X1), kappa(X2)
    ones = torch.ones_like(k1)
    m1 = ones if m1 is None else m1
    m2 = ones if m2 is None else m2
    c1 = ones if c1 is None else c1
    c2 = ones if c2 is None else c2
    L2 = _term(k2, k1, b, m2, c2, clamp)        # teacher = view-1 curvature, via b
    L1 = _term(k1, k2, f, m1, c1, clamp)        # teacher = view-2 curvature, via f
    return L1 + L2, {"cc_L1": float(L1), "cc_L2": float(L2)}


# ----------------------------------------------------------------------------
# Validation gates (run with: python cc_loss.py). No training, CPU-friendly.
# ----------------------------------------------------------------------------
def _surface(H=48, W=64, kind="bump", device="cpu"):
    ys, xs = torch.meshgrid(torch.linspace(-2, 2, H), torch.linspace(-2, 2, W),
                            indexing='ij')
    if kind == "bump":
        z = 5.0 - 0.3 * (xs ** 2 + ys ** 2)
    elif kind == "saddle":
        z = 5.0 + 0.3 * (xs ** 2 - ys ** 2)
    else:
        z = torch.full_like(xs, 5.0)
    return torch.stack([xs, ys, z], dim=-1)[None].to(device)   # (1,H,W,3)


def _gates():
    torch.manual_seed(0)
    P1 = _surface(kind="bump")
    B, H, W, _ = P1.shape
    zero = torch.zeros(B, 2, H, W)

    # Gate A -- motion-invariance: rigidly rotate+translate the surface (same
    # pixel grid -> identity flow). Curvature is invariant to SO(3)+translation,
    # so L_cc must be ~0.
    ang = 0.4
    R = torch.tensor([[torch.cos(torch.tensor(ang)), 0, torch.sin(torch.tensor(ang))],
                      [0, 1, 0],
                      [-torch.sin(torch.tensor(ang)), 0, torch.cos(torch.tensor(ang))]])
    t = torch.tensor([1.3, -0.7, 2.0])
    P2_rigid = P1 @ R.T + t
    LA, _ = cc_loss(P1, P2_rigid, zero, zero)

    # Gate B -- flow realignment: shift the surface by an integer pixel offset and
    # feed the matching constant flow; the warp must realign -> residual ~0.
    dx, dy = 3, 2
    P2_shift = torch.roll(P1, shifts=(dy, dx), dims=(1, 2))     # P2[y,x]=P1[y-dy,x-dx]
    fwd = zero.clone(); fwd[:, 0] = dx; fwd[:, 1] = dy          # f: view1 i -> view2 i+shift
    bwd = zero.clone(); bwd[:, 0] = -dx; bwd[:, 1] = -dy        # b: view2 j -> view1 j-shift
    # ignore the wrapped border band so the roll's wraparound doesn't pollute it
    mask = torch.zeros(B, 1, H, W); mask[:, :, dy + 1:H - 1, dx + 1:W - 1] = 1
    LB, _ = cc_loss(P1, P2_shift, fwd, bwd, m1=mask, m2=mask)

    # Gate C -- deformation: bend the surface (scale z); curvature changes ->
    # L_cc must be clearly > Gate A.
    P2_def = P1.clone(); P2_def[..., 2] = P1[..., 2] * 1.8 - 4.0
    LC, _ = cc_loss(P1, P2_def, zero, zero)

    # Gate D -- gradient flows to the pointmap through kappa.
    Pg = _surface(kind="bump").clone().requires_grad_(True)
    Lg, _ = cc_loss(Pg, _surface(kind="saddle"), zero, zero)
    Lg.backward()
    grad_ok = Pg.grad is not None and torch.isfinite(Pg.grad).all() and Pg.grad.abs().sum() > 0

    print(f"Gate A  motion-invariance  L_cc = {LA:.3e}   (expect ~0)")
    print(f"Gate B  flow realignment   L_cc = {LB:.3e}   (expect ~0)")
    print(f"Gate C  deformation        L_cc = {LC:.3e}   (expect >> A)")
    print(f"Gate D  grad through kappa  {'finite, nonzero' if grad_ok else 'BAD'}")
    ok = (LA < 1e-4) and (LB < 1e-4) and (LC > 100 * max(LA, 1e-9)) and grad_ok
    print("CC-LOSS GATES:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _gates()
