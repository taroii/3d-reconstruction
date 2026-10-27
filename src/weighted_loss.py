"""
Geometry-weighted regression loss for VGGT, and the arms of the study.

This is the one place the method lives. It reweights the per-pixel position
residual in VGGT's published depth/point regression by a local geometric signal
read off the ground-truth pointmap, and it reduces EXACTLY to the published
objective when the strength is zero -- which `_selftest` verifies numerically
against upstream's own `regression_loss` rather than asserting in a comment.

------------------------------------------------------------------------------
Naming
------------------------------------------------------------------------------
The ACCV paper calls the weight strength `gamma`. VGGT's `loss.py` ALREADY uses
`gamma` for the confidence-loss coefficient and `alpha` for the confidence
regularizer. To keep the two from silently colliding, the weight strength is
`lam` (lambda) everywhere in this file, and `gamma`/`alpha` keep their upstream
meaning. The paper can still call it gamma.

------------------------------------------------------------------------------
The arms
------------------------------------------------------------------------------
The study factors into (signal) x (weight form) rather than a list of ad-hoc
arms, because the reviewers' objections are about these two axes separately.

SIGNAL -- what is measured at each pixel:
  none        no weighting; the published objective.
  umbrella    the ACCV signal, (1/2)||mean(4-neighbours) - centre||.
  grad        ||grad Xbar||, the published first-order control.
  cotangent   Meyer et al. area-normalized |H|, in 1/length units. The arm that
              asks whether anything SECOND-ORDER is doing work, which is what
              Reviewer CYqH asked for and nobody has run.

WEIGHT FORM -- how the measurement becomes a multiplier:
  clamp       w = 1 + lam * min(S/Sbar, tau). The published form. Note that at
              lam=1, tau=10 the ceiling is 11 and the decisive pixels saturate,
              so it behaves as a near-binary edge mask there.
  unitmean    `clamp`, then rescaled to mean exactly 1 over valid pixels.
              Reviewer CYqH: because w multiplies only the residual and not the
              -alpha*log(c) term, raising lam under `clamp` arithmetically
              LOWERS the effective confidence coefficient (measured: 0.2 -> 0.14 at
              lam=0.5, 0.10 at lam=1, 0.07 at lam=2). That confound alone could produce the
              paper's monotone trend. This form removes it, and the reviewer
              called it "a one-line change [that] would settle it".
  binary      w = 1 + lam*tau on the top `q` quantile of S, 1 elsewhere. Tests
              whether the MAGNITUDE of the signal carries any information beyond
              the identity of the pixels it selects.

------------------------------------------------------------------------------
Where the weight is applied, and the quantile filter
------------------------------------------------------------------------------
Upstream runs with `valid_range=0.98`, which DISCARDS the 2% highest-loss
pixels before averaging. Those are disproportionately the occlusion-boundary
pixels this method is trying to emphasize, so the interaction is not a detail:

  * weighting BEFORE the filter would push upweighted pixels over the quantile
    threshold and partially delete the very reweighting under test;
  * and it would make the SET of surviving pixels differ between arms, so the
    arms would no longer be a matched comparison.

The quantile mask is therefore computed on the UNWEIGHTED residual, identically
for every arm, and the weight is applied to the pixels that survive. This keeps
lam=0 bit-identical to upstream and keeps every arm supervised on the same set
of pixels, which is the only way the contrast means anything.

------------------------------------------------------------------------------
Normalization of the signal
------------------------------------------------------------------------------
Signals are computed on the GT pointmap divided by its own mean point-to-origin
norm over valid pixels (the `avg_dis` convention the ACCV loss used), so the
weight is scale-invariant. `cotangent` carries units of 1/length and is rescaled
back accordingly; `umbrella` and `grad` are dimensionless after normalization.
See src/estimator_diag.py for what each of these actually measures -- the short
version is that `umbrella` is not mean curvature, and that is the finding.

  python src/weighted_loss.py --selftest
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.environ.get("PROJ", os.path.abspath(os.path.join(_HERE, "..", "..")))
_VGGT_TRAIN = os.path.join(_PROJ, "third_party", "vggt", "training")

EPS = 1e-12
DEFAULT_TAU = 10.0          # the ACCV clamp, kept as a fixed constant
DEFAULT_Q = 0.90            # `binary` selects the top decile, matching the
                            # paper's own boundary-metric definition

# 6-neighbour ring for the cotangent estimator, in rotational order so that
# consecutive entries bound a triangle of the anti-diagonal grid split.
RING = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]


# --------------------------------------------------------------------------- #
# shift helpers (operate on (..., H, W) or (..., H, W, C))
# --------------------------------------------------------------------------- #

def _shift(x, dr, dc, fill=0.0):
    """x_out[..., r, c] = x[..., r+dr, c+dc], padded with `fill`.

    Matches the semantics of `_shift` in src/estimator_diag.py so the torch and
    numpy implementations can be checked against each other.
    """
    out = torch.full_like(x, fill)
    H, W = x.shape[-2], x.shape[-1]
    r0, r1 = max(0, dr), min(H, H + dr)
    c0, c1 = max(0, dc), min(W, W + dc)
    out[..., r0 - dr:r1 - dr, c0 - dc:c1 - dc] = x[..., r0:r1, c0:c1]
    return out


def _shift_vec(P, dr, dc):
    """P is (..., H, W, 3); shift the spatial dims."""
    return _shift(P.permute(*range(P.dim() - 3), -1, -3, -2), dr, dc).permute(
        *range(P.dim() - 3), -2, -1, -3)


def _valid_shift(valid, dr, dc):
    return _shift(valid.to(torch.float32), dr, dc, 0.0) > 0.5


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #

def avg_dis_scale(P, valid):
    """Mean point-to-origin norm over valid pixels, per (B,S) frame.

    Guarded, because the unguarded version failed SILENTLY and catastrophically.
    One non-finite -- or merely large, 1e20 is enough to overflow an fp32 sum --
    value anywhere in `world_points` made `scale` non-finite, which made the
    normalized pointmap all-zero, which made the signal zero, which made every
    weight exactly 1. The weighted arm then ran as the baseline for that frame
    with no NaN in the loss, no warning, and `isfinite(w).all()` still True.
    That is the failure mode most likely to invalidate a run while every log
    looks healthy, so it now raises rather than degrades.

    Returned with trailing singleton dims so it broadcasts against (B,S,H,W).
    """
    d = P.norm(dim=-1)                                   # (B,S,H,W)
    v = valid.to(d.dtype)
    # Sum in float64: an fp32 accumulation over ~250k pixels overflows on a
    # single 1e20 outlier, and real depth GT contains those.
    denom = v.sum(dim=(-2, -1), dtype=torch.float64).clamp_min(1.0)
    scale = (d.to(torch.float64) * v.to(torch.float64)).sum(dim=(-2, -1)) / denom
    if not torch.isfinite(scale).all():
        raise ValueError(
            "avg_dis_scale produced a non-finite scale, which means the GT "
            "pointmap contains inf/nan or values large enough to overflow. "
            "Sanitize `world_points` before computing the weight -- silently "
            "falling back to unit weights would turn a weighted arm into the "
            "baseline without any visible symptom.")
    return scale.to(P.dtype).clamp_min(EPS)[..., None, None]


# --------------------------------------------------------------------------- #
# signals -- all take GT pointmap (B,S,H,W,3) + valid (B,S,H,W), return (B,S,H,W)
# --------------------------------------------------------------------------- #

def umbrella_signal(P, valid):
    """The ACCV signal: (1/2)||mean of 4 neighbours - centre||.

    Dimensionless once P is avg_dis-normalized. src/estimator_diag.py shows this
    responds to surface slant and to depth as strongly as it responds to
    curvature; it is included because it is the published method, not because it
    measures what the paper claims.
    """
    four = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    acc = torch.zeros_like(P)
    ok = valid.clone()
    for dr, dc in four:
        acc = acc + _shift_vec(P, dr, dc)
        ok = ok & _valid_shift(valid, dr, dc)
    S = 0.5 * (acc * 0.25 - P).norm(dim=-1)
    return torch.where(ok, S, torch.zeros_like(S)), ok


def grad_signal(P, valid):
    """||grad Xbar||, the published first-order control.

    Backward differences, matching curv/curv_loss.py in the ACCV code. The
    one-sidedness is deliberate fidelity to what was published, not an
    oversight: it means this stencil can only fire on one side of a
    discontinuity, which src/estimator_diag.py accounts for explicitly.
    """
    gy = (P - _shift_vec(P, -1, 0)).norm(dim=-1)
    gx = (P - _shift_vec(P, 0, -1)).norm(dim=-1)
    ok = valid & _valid_shift(valid, -1, 0) & _valid_shift(valid, 0, -1)
    S = torch.sqrt(gx * gx + gy * gy + EPS)
    return torch.where(ok, S, torch.zeros_like(S)), ok


def _cot_at(apex, a, b):
    """cot of the angle at `apex` in triangle (apex, a, b)."""
    u, v = a - apex, b - apex
    dot = (u * v).sum(-1)
    crs = torch.cross(u, v, dim=-1).norm(dim=-1)
    return dot / crs.clamp_min(EPS)


def cotangent_signal(P, valid):
    """Meyer et al. mixed-area |H|, in units of 1/length.

    The estimator the ACCV paper cites but does not implement, and the only one
    of the three that returns mean curvature. Its correctness is checked in
    src/estimator_diag.py against closed-form spheres and cylinders; here it is
    checked for agreement with that numpy implementation.
    """
    R = [_shift_vec(P, dr, dc) for dr, dc in RING]
    ok = valid.clone()
    for dr, dc in RING:
        ok = ok & _valid_shift(valid, dr, dc)

    lap = torch.zeros_like(P)
    area = torch.zeros(P.shape[:-1], dtype=P.dtype, device=P.device)
    for k in range(6):
        prv, cur, nxt = R[(k - 1) % 6], R[k], R[(k + 1) % 6]
        w = _cot_at(prv, P, cur) + _cot_at(nxt, P, cur)
        lap = lap + w[..., None] * (P - cur)

        e1, e2 = cur - P, nxt - P
        tri = 0.5 * torch.cross(e1, e2, dim=-1).norm(dim=-1)
        obtuse_i = (e1 * e2).sum(-1) < 0
        obtuse_c = ((P - cur) * (nxt - cur)).sum(-1) < 0
        obtuse_n = ((P - nxt) * (cur - nxt)).sum(-1) < 0
        voronoi = 0.125 * ((e1 ** 2).sum(-1) * _cot_at(nxt, P, cur)
                           + (e2 ** 2).sum(-1) * _cot_at(cur, P, nxt))
        contrib = torch.where(obtuse_i, tri / 2.0,
                              torch.where(obtuse_c | obtuse_n, tri / 4.0, voronoi))
        area = area + torch.nan_to_num(contrib)

    KH = lap / (2.0 * area).clamp_min(EPS)[..., None]
    S = 0.5 * KH.norm(dim=-1)
    return torch.where(ok, S, torch.zeros_like(S)), ok


SIGNALS = {"umbrella": umbrella_signal, "grad": grad_signal,
           "cotangent": cotangent_signal}
INVERSE_LENGTH = {"cotangent"}      # rescaled back to metric units


# --------------------------------------------------------------------------- #
# weight forms
# --------------------------------------------------------------------------- #

def _masked_mean(x, m):
    mm = m.to(x.dtype)
    return (x * mm).sum(dim=(-2, -1)) / mm.sum(dim=(-2, -1)).clamp_min(1.0)


def weight_clamp(S, ok, lam, tau=DEFAULT_TAU, **kw):
    """w = 1 + lam * min(S/Sbar, tau) -- the published form."""
    Sbar = _masked_mean(S, ok).clamp_min(EPS)[..., None, None]
    w = 1.0 + lam * torch.clamp(S / Sbar, max=tau)
    return torch.where(ok, w, torch.ones_like(w))


def weight_unitmean(S, ok, lam, tau=DEFAULT_TAU, **kw):
    """`clamp`, rescaled so the mean weight over valid pixels is exactly 1.

    This is the arm that separates "supervision was redistributed toward
    boundaries" from "the effective confidence coefficient was lowered". Under
    `clamp`, E[w] = 1 + lam*E[min(S/Sbar, tau)] > 1, and since w multiplies only
    the residual, factoring that mean scale k out of the objective leaves a
    confidence coefficient of alpha/k. Holding E[w] = 1 removes the confound
    without changing where the supervision goes.
    """
    w = weight_clamp(S, ok, lam, tau)
    m = _masked_mean(w, ok).clamp_min(EPS)[..., None, None]
    return torch.where(ok, w / m, torch.ones_like(w))


def weight_binary(S, ok, lam, tau=DEFAULT_TAU, q=DEFAULT_Q, **kw):
    """w = 1 + lam*tau on the top `q` quantile of S, 1 elsewhere.

    Per-frame quantile, computed only over valid pixels. Answers whether the
    graded magnitude carries information beyond the identity of the pixels it
    picks out -- at lam=1, tau=10 the published weight already saturates at its
    ceiling on the decisive pixels, so this may be what the method has been
    doing all along.
    """
    B, Sq, H, W = S.shape
    # torch.quantile rejects half precision, so `binary` -- and only `binary` --
    # crashed under autocast while the other two forms survived. Promote.
    S = S.float() if S.dtype in (torch.float16, torch.bfloat16) else S
    flat_S, flat_ok = S.reshape(B * Sq, -1), ok.reshape(B * Sq, -1)
    thresh = torch.empty(B * Sq, dtype=S.dtype, device=S.device)
    for i in range(B * Sq):
        v = flat_S[i][flat_ok[i]]
        thresh[i] = torch.quantile(v, q) if v.numel() > 0 else float("inf")
    hi = S >= thresh.reshape(B, Sq, 1, 1)
    w = torch.where(hi & ok, torch.full_like(S, 1.0 + lam * tau), torch.ones_like(S))
    return torch.where(ok, w, torch.ones_like(w))


WEIGHTS = {"clamp": weight_clamp, "unitmean": weight_unitmean,
           "binary": weight_binary}


def compute_weight(gt_points, valid, signal="none", form="clamp", lam=0.0,
                   tau=DEFAULT_TAU, q=DEFAULT_Q):
    """Per-pixel weight (B,S,H,W) from the GT pointmap.

    `signal="none"` or `lam == 0` returns exactly ones, so the baseline arm runs
    through this same function and cannot drift from the weighted arms.
    """
    if signal == "none" or lam == 0.0:
        return torch.ones(valid.shape, dtype=gt_points.dtype, device=gt_points.device)
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; have {sorted(SIGNALS)}")
    if form not in WEIGHTS:
        raise ValueError(f"unknown form {form!r}; have {sorted(WEIGHTS)}")

    scale = avg_dis_scale(gt_points, valid)
    Pn = torch.nan_to_num(gt_points / scale[..., None], nan=0.0)
    S, ok = SIGNALS[signal](Pn, valid)
    if signal in INVERSE_LENGTH:
        S = S / scale
    S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return WEIGHTS[form](S, ok, lam, tau=tau, q=q)


# --------------------------------------------------------------------------- #
# the loss
# --------------------------------------------------------------------------- #

def _fix_inf_nan(x, hard_max=100.0):
    """Upstream `check_and_fix_inf_nan`: inf/nan -> 0, then clamp to +/-hard_max.

    Replicated rather than imported so this file runs without the upstream
    package on the path, but `_selftest` compares against the real function.
    Note the direction: upstream maps +inf to ZERO, not to hard_max. An earlier
    version here mapped it to hard_max, which is arguably saner and is not what
    the published objective does.
    """
    if x is None:
        return x
    bad = torch.isnan(x) | torch.isinf(x)
    if bad.any():
        x = torch.where(bad, torch.zeros_like(x), x)
    if hard_max is not None:
        x = torch.clamp(x, min=-hard_max, max=hard_max)
    return x


def _quantile_scalar(x, q):
    """Upstream `torch_quantile`: kthvalue with "nearest" interpolation.

    NOT `torch.quantile`, which raises above 2**24 elements -- upstream wrote
    its own precisely to dodge that, and the limit is reachable here (batch 4 x
    8 views x 518^2 is 8.6M masked pixels, and doubling the batch crosses it).
    """
    flat = x.reshape(-1)
    n = flat.numel()
    if n == 0:
        return float("inf")
    k = int(round(float(q) * (n - 1))) + 1
    return flat.kthvalue(min(max(k, 1), n)).values.item()


def _quantile_keep(loss_tensor, valid_range, min_elements=1000, hard_max=100.0):
    """Boolean mask of what upstream's `filter_by_quantile` would keep.

    Upstream returns a SHORTENED tensor; we need the mask so that the same
    surviving pixels can be selected in every arm. Three pieces of upstream's
    semantics that a naive version drops, each of which bit us:

    * the `min_elements` FALLBACK -- if filtering would leave <= min_elements,
      upstream returns the tensor UNFILTERED. Without it, a batch whose
      residuals all exceed hard_max yields an all-False mask, a zero loss and a
      dead gradient, on exactly the diverged or bad-GT batches where upstream
      deliberately keeps supervising.
    * `_quantile_scalar` rather than `torch.quantile` (see above).
    * the <= min_elements short-circuit before any work.
    """
    if valid_range is None or valid_range <= 0 or loss_tensor.numel() <= min_elements:
        return torch.ones_like(loss_tensor, dtype=torch.bool)
    clamped = loss_tensor.clamp(max=hard_max)
    thresh = min(_quantile_scalar(clamped.detach(), valid_range), hard_max)
    keep = clamped < thresh
    if keep.sum() > min_elements:
        return keep
    return torch.ones_like(loss_tensor, dtype=torch.bool)      # upstream fallback


def weighted_regression_loss(pred, gt, mask, conf, weight=None, gamma=1.0,
                             alpha=0.2, valid_range=-1, hard_max=100.0):
    """VGGT's position regression with an optional per-pixel weight.

    `weight` is (B,S,H,W) and multiplies ONLY the residual, never the
    -alpha*log(conf) term -- the placement the ACCV paper used, and the reason
    the MEAN of w matters (see `weight_unitmean`).

    Two subtleties, both of which were wrong in the first version:

    1. Upstream filters `loss_conf` and `loss_reg` INDEPENDENTLY, each by its
       own quantile -- two different masks, because a pixel can have a large
       residual but a small conf-weighted loss and vice versa. Collapsing them
       into one mask makes the lam=0 arm differ from the published objective by
       a small but SYSTEMATIC amount.
    2. Both masks are derived from the UNWEIGHTED quantities. That is the only
       way to get both properties we need: at lam=0 it is bit-identical to
       upstream, and at lam>0 every arm is supervised on the same pixel set, so
       the contrast between arms isolates the weight rather than confounding it
       with which pixels survived the filter.

    Returns (loss_conf, loss_reg) as scalars.
    """
    res = (gt[mask] - pred[mask]).norm(dim=-1)
    res = _fix_inf_nan(res, hard_max)                    # upstream's loss_reg
    c = conf[mask]

    # Unweighted conf loss: identical to upstream, and the basis of both masks.
    lc_unw = _fix_inf_nan(gamma * res * c - alpha * torch.log(c), hard_max)
    keep_conf = _quantile_keep(lc_unw, valid_range, hard_max=hard_max)
    keep_reg = _quantile_keep(res, valid_range, hard_max=hard_max)

    w = torch.ones_like(res) if weight is None else weight[mask].to(res.dtype)
    res_w = w * res
    lc_w = _fix_inf_nan(gamma * res_w * c - alpha * torch.log(c), hard_max)

    loss_conf = (_fix_inf_nan(lc_w[keep_conf], hard_max).mean()
                 if keep_conf.any() else (0.0 * pred).mean())
    loss_reg = (_fix_inf_nan(res_w[keep_reg], hard_max).mean()
                if keep_reg.any() else (0.0 * pred).mean())
    return loss_conf, loss_reg


def make_depth_loss(predictions, batch, signal="none", form="clamp", lam=0.0,
                    tau=DEFAULT_TAU, q=DEFAULT_Q, gamma=1.0, alpha=0.2,
                    valid_range=0.98):
    """Drop-in replacement for upstream `compute_depth_loss`'s conf/reg terms.

    The signal is read from the GT POINTMAP (`batch['world_points']`), not from
    the depth map, because the geometry it describes is a property of the 3D
    surface. The weight is then applied to the depth residual, which is what
    the default config actually supervises (`point: null`).
    """
    pred_depth, conf = predictions["depth"], predictions["depth_conf"]
    gt_depth = torch.nan_to_num(batch["depths"], nan=0.0)[..., None]
    valid = batch["point_masks"].clone()

    if valid.sum() < 100:
        dummy = (0.0 * pred_depth).mean()
        # Three keys, not two: upstream's MultitaskLoss sums
        # loss_conf + loss_reg + loss_grad, so dropping the gradient key either
        # raises KeyError or silently removes that term from the objective.
        return {"loss_conf_depth": dummy, "loss_reg_depth": dummy,
                "loss_grad_depth": dummy}

    w = None
    if signal != "none" and lam != 0.0:
        if "world_points" not in batch:
            raise KeyError(
                "weighted arms need batch['world_points'] to read the geometric "
                "signal from; the batch only has: " + ", ".join(sorted(batch)))
        # Sanitize the way upstream's compute_point_loss does before it touches
        # the pointmap. Without this, one bad value disables the weighting for
        # the whole frame (see avg_dis_scale).
        gt_points = _fix_inf_nan(batch["world_points"], hard_max=None)
        gt_points = torch.nan_to_num(gt_points, nan=0.0, posinf=0.0, neginf=0.0)
        w = compute_weight(gt_points, valid, signal, form, lam, tau, q)

    lc, lr = weighted_regression_loss(pred_depth, gt_depth, valid, conf, weight=w,
                                      gamma=gamma, alpha=alpha,
                                      valid_range=valid_range)
    # The gradient term is upstream's and is not reweighted; it is returned as a
    # zero here so the key set matches what MultitaskLoss expects, and the
    # caller is responsible for adding the real one if it is enabled.
    return {"loss_conf_depth": lc, "loss_reg_depth": lr,
            "loss_grad_depth": (0.0 * pred_depth).mean()}


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #

def _fake_batch(B=1, S=2, H=24, W=32, seed=0, device="cpu"):
    """iid-Gaussian pointmaps. Deliberately NON-SQUARE (24x32).

    The non-squareness is load-bearing, not cosmetic: it turns any axis-order
    mistake in `_shift_vec` into a shape error. On a square grid the same bug is
    silently wrong with the right shape. Do not "tidy" this to H == W.

    This batch has no surfaces and no boundaries, so E[S/Sbar] == 1 and nothing
    ever reaches the tau clamp -- use `_geometric_batch` for anything that needs
    the regime the method is actually about.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = torch.randn(B, S, H, W, 3, generator=g).to(device) * 0.3
    pts[..., 2] += 4.0
    valid = torch.ones(B, S, H, W, dtype=torch.bool, device=device)
    valid[..., :1, :] = False
    pred = (pts[..., 2:3] + 0.05 * torch.randn(B, S, H, W, 1, generator=g).to(device))
    conf = 1.0 + torch.rand(B, S, H, W, generator=g).to(device)
    return pts, valid, pred, conf


def _geometric_batch(B=1, S=2, H=48, W=64, device="cpu"):
    """Two fronto-parallel planes with a depth step: real geometry, real tail.

    `_fake_batch` is Gaussian noise, where every signal has E[S/Sbar] = 1 and the
    tau clamp is never exercised -- so the test data could not reach the regime
    the method exists to operate in, and a mutation removing the clamp entirely
    passed the suite. This batch has a genuine discontinuity, so S/Sbar is
    heavy-tailed and the clamp bites.
    """
    f, cx, cy = 60.0, W / 2.0, H / 2.0
    vs, us = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    z = torch.full((H, W), 3.0)
    z[:, W // 2:] = 6.0                                  # the step
    P = torch.stack([(us - cx) / f * z, (vs - cy) / f * z, z], -1)
    pts = P[None, None].repeat(B, S, 1, 1, 1).to(device)
    valid = torch.ones(B, S, H, W, dtype=torch.bool, device=device)
    valid[..., :1, :] = False
    pred = pts[..., 2:3].clone()
    conf = torch.full((B, S, H, W), 1.5, device=device)
    return pts, valid, pred, conf


def _selftest():
    def approx(a, b, tol, msg):
        assert abs(a - b) <= tol * max(abs(b), EPS), f"{msg}: {a:.8g} vs {b:.8g}"

    pts, valid, pred, conf = _fake_batch()
    gt_depth = pts[..., 2:3]

    # -- 1. lam = 0 must reproduce UPSTREAM, across the configs that matter ---
    # The first version of this check ran only at valid_range=-1 with tiny
    # residuals, which is exactly where the two implementations agree by
    # accident. The real config is 0.98, and residuals do reach hard_max.
    if not os.path.isdir(_VGGT_TRAIN):
        raise AssertionError(
            f"no upstream training code at {_VGGT_TRAIN}; cannot verify that "
            f"lam=0 reproduces the published objective.")
    for pth in (_VGGT_TRAIN, os.path.dirname(_VGGT_TRAIN)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    try:
        from loss import regression_loss as upstream_regression_loss
        from train_utils.general import check_and_fix_inf_nan as upstream_fix
    except ImportError as e:
        raise AssertionError(
            f"could not import upstream regression_loss ({e}). This is the test "
            f"that the lam=0 arm reproduces the published objective, so a silent "
            f"skip is not acceptable -- fix the path or the clone.") from e

    # our _fix_inf_nan must match upstream's element for element
    probe = torch.tensor([float("nan"), float("inf"), -float("inf"),
                          -500.0, -1.0, 0.0, 1.0, 500.0])
    assert torch.equal(_fix_inf_nan(probe.clone()), upstream_fix(probe.clone())), \
        "_fix_inf_nan diverges from upstream check_and_fix_inf_nan"

    # big enough to clear filter_by_quantile's min_elements=1000
    bp, bv, bpred, bconf = _fake_batch(B=1, S=2, H=96, W=128, seed=3)
    bgt = bp[..., 2:3]
    for vr in (-1, 0.98):
        for scale, label in ((1.0, "small residuals"), (5000.0, "past hard_max")):
            noisy = bgt + scale * (bpred - bgt)
            if scale > 1.0:
                raw = (bgt[bv] - noisy[bv]).norm(dim=-1)
                raw_conf = raw.clamp(max=100.0) * bconf[bv] - 0.2 * torch.log(bconf[bv])
                assert float(raw.max()) > 100.0 and float(raw_conf.max()) > 100.0, (
                    "the 'past hard_max' case does not actually reach hard_max, "
                    "so it cannot detect a missing clamp")
            up_conf, _, up_reg = upstream_regression_loss(
                noisy, bgt, bv, conf=bconf, gradient_loss_fn="",
                gamma=1.0, alpha=0.2, valid_range=vr)
            ours_conf, ours_reg = weighted_regression_loss(
                noisy, bgt, bv, bconf, weight=None,
                gamma=1.0, alpha=0.2, valid_range=vr)
            approx(ours_conf.item(), up_conf.item(), 1e-6,
                   f"lam=0 conf vs upstream (valid_range={vr}, {label})")
            approx(ours_reg.item(), up_reg.item(), 1e-6,
                   f"lam=0 reg vs upstream (valid_range={vr}, {label})")
    print("  lam=0 matches upstream at valid_range in {-1, 0.98}, both residual scales")

    # -- 2. an all-ones weight must change nothing ---------------------------
    a, _ = weighted_regression_loss(pred, gt_depth, valid, conf, weight=None,
                                    valid_range=-1)
    ones = torch.ones_like(valid, dtype=pred.dtype)
    b, _ = weighted_regression_loss(pred, gt_depth, valid, conf, weight=ones,
                                    valid_range=-1)
    approx(a.item(), b.item(), 1e-9, "all-ones weight must be a no-op")

    # -- 3. the quantile filter must select the SAME pixels for every arm ----
    # This replaces a tautology. The old check only proved _quantile_keep is
    # scale-equivariant (true by construction); the exact bug its comment named
    # -- deriving the mask from the WEIGHTED residual -- passed it untouched.
    # Here the count is compared through the real entry point.
    noisy = bgt + 50.0 * (bpred - bgt)
    res = (bgt[bv] - noisy[bv]).norm(dim=-1)
    c = bconf[bv]
    # A weight deliberately correlated with the residual, so that deriving the
    # mask from the weighted loss really would select a different pixel set.
    rank = res.argsort().argsort().to(res.dtype) / max(res.numel() - 1, 1)
    heavy_flat = 1.0 + 10.0 * rank
    heavy = torch.ones(bv.shape, dtype=res.dtype)
    heavy[bv] = heavy_flat

    keep_unw = _quantile_keep(_fix_inf_nan(res * c - 0.2 * torch.log(c)), 0.98)
    keep_w = _quantile_keep(_fix_inf_nan(heavy_flat * res * c - 0.2 * torch.log(c)), 0.98)
    assert not torch.equal(keep_unw, keep_w), (
        "this test is vacuous unless the weight changes which pixels the "
        "quantile filter would drop")

    # The contract, stated directly: mask from the UNWEIGHTED loss, then weight
    # the survivors. Computed independently here and compared to the function.
    expected = _fix_inf_nan(
        _fix_inf_nan(heavy_flat * res) * c - 0.2 * torch.log(c))[keep_unw].mean()
    got, _ = weighted_regression_loss(noisy, bgt, bv, bconf, weight=heavy,
                                      valid_range=0.98)
    approx(got.item(), expected.item(), 1e-6,
           "quantile mask must come from the UNWEIGHTED loss, not the weighted one")

    # -- 4. upstream's min_elements fallback: all-huge residuals must NOT ----
    # -- produce a zero loss and a dead gradient.
    huge = torch.full((5000,), 1000.0)
    keep = _quantile_keep(huge, 0.98)
    assert keep.all(), ("min_elements fallback missing: an all-huge batch "
                        "yields an empty keep-set, a zero loss and no gradient")

    # -- 5. every signal/form combination: finite, positive, right shape -----
    for sig in SIGNALS:
        for form in WEIGHTS:
            w = compute_weight(pts, valid, sig, form, lam=1.0)
            assert torch.isfinite(w).all(), f"{sig}/{form} non-finite weights"
            assert (w > 0).all(), f"{sig}/{form} non-positive weight"
            assert w.shape == valid.shape, f"{sig}/{form} wrong shape {w.shape}"
            assert torch.allclose(w[~valid], torch.ones_like(w[~valid])), \
                f"{sig}/{form} weighted an invalid pixel"

    # -- 6. unitmean has unit mean; clamp does not ---------------------------
    for sig in SIGNALS:
        wu = compute_weight(pts, valid, sig, "unitmean", lam=1.0)
        approx(float(_masked_mean(wu, valid).mean()), 1.0, 1e-5, f"unitmean({sig})")
        wc = compute_weight(pts, valid, sig, "clamp", lam=1.0)
        assert float(_masked_mean(wc, valid).mean()) > 1.0 + 1e-3, \
            f"clamp({sig}) should have mean > 1 -- that is the confound it carries"

    # -- 7. the tau clamp must actually bind, on data that can reach it ------
    # `_fake_batch` is Gaussian noise where nothing saturates, so a mutation
    # deleting the clamp passed the whole suite. Real geometry is required.
    gp, gv, _, _ = _geometric_batch()
    for sig in SIGNALS:
        w_lo = compute_weight(gp, gv, sig, "clamp", lam=1.0, tau=2.0)
        w_hi = compute_weight(gp, gv, sig, "clamp", lam=1.0, tau=10.0)
        assert float(w_lo[gv].max()) <= 1.0 + 1.0 * 2.0 + 1e-5, \
            f"{sig}: weight exceeded the tau=2 ceiling -- clamp not applied"
        assert float(w_hi[gv].max()) > float(w_lo[gv].max()) + 1e-3, \
            f"{sig}: tau made no difference, so the clamp is not binding"

    # -- 8. binary: right fraction, two values, fp16-safe --------------------
    wb = compute_weight(pts, valid, "umbrella", "binary", lam=1.0, tau=10.0, q=0.9)
    assert len(torch.unique(wb[valid])) <= 2, "binary produced >2 distinct weights"
    frac = (wb[valid] > 1.0).float().mean().item()
    assert 0.07 < frac < 0.13, f"binary selected {frac:.3f}, expected ~0.10"
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        w = compute_weight(pts.to(dt), valid, "umbrella", "binary", lam=1.0)
        assert torch.isfinite(w).all(), f"binary failed for {dt}"

    # -- 9. lam = 0 is ones for every signal ---------------------------------
    for sig in list(SIGNALS) + ["none"]:
        w = compute_weight(pts, valid, sig, "clamp", lam=0.0)
        assert torch.allclose(w, torch.ones_like(w)), f"lam=0 not ones for {sig}"
    # and the limit is approached continuously, not just short-circuited
    w_small = compute_weight(gp, gv, "umbrella", "clamp", lam=1e-6)
    assert float((w_small - 1.0).abs().max()) < 1e-4, "clamp does not tend to 1"

    # -- 10. the weight must NOT touch the -alpha*log(c) term ----------------
    zero_pred = gt_depth.clone()
    base, _ = weighted_regression_loss(zero_pred, gt_depth, valid, conf,
                                       weight=None, valid_range=-1)
    hw, _ = weighted_regression_loss(zero_pred, gt_depth, valid, conf,
                                     weight=torch.full_like(valid, 7.0, dtype=pred.dtype),
                                     valid_range=-1)
    approx(hw.item(), base.item(), 1e-9, "weight leaked into the confidence term")

    # -- 11. a poisoned pointmap must RAISE, not silently fall back to ones --
    # Previously one NaN -- or one 1e20, which overflows an fp32 sum -- turned
    # the weighted arm into the baseline for that frame, with no symptom.
    for bad in (float("nan"), float("inf"), 1e20):
        poisoned = pts.clone()
        poisoned[0, 0, 5, 5, 0] = bad
        try:
            w = compute_weight(poisoned, valid, "umbrella", "clamp", lam=1.0)
        except ValueError:
            continue                        # the intended behaviour
        assert float(_masked_mean(w, valid)[0, 0]) > 1.0 + 1e-3, (
            f"a {bad} in the pointmap silently produced unit weights: the "
            f"weighted arm would run as the baseline with no diagnostic")

    # -- 12. torch signals must agree with the validated numpy reference -----
    # Hard failure, not SKIP: this is the ONLY check that looks at what the
    # signals compute. With it skipped, mutations removing the cotangent area
    # normalization, the umbrella's /4, and the grad orientation all passed.
    sys.path.insert(0, _HERE)
    try:
        import estimator_diag as ED
        import numpy as np
    except ImportError as e:
        raise AssertionError(
            f"cannot import the numpy reference ({e}). This is the only check "
            f"that validates signal VALUES; skipping it leaves the suite unable "
            f"to detect a broken estimator.") from e
    # Both a smooth scene and one with discontinuities and degenerate triangles.
    for scene in (ED.scene_sphere(radius=0.5, dist=5.0), ED.scene_step_and_sphere()):
        P = torch.from_numpy(np.nan_to_num(scene.P)).double()[None, None]
        V = torch.from_numpy(scene.valid)[None, None]
        for name, np_fn in (("umbrella", ED.umbrella), ("cotangent", ED.cotangent),
                            ("grad", ED.pointmap_grad)):
            t_S, t_ok = SIGNALS[name](P, V)
            n_S, n_ok = np_fn(np.nan_to_num(scene.P), scene.valid)
            both = t_ok[0, 0].numpy() & n_ok & np.isfinite(n_S)
            assert both.sum() > 500, f"{name}/{scene.name}: too few pixels"
            # per-pixel relative error, not max-normalized: a max-normalized
            # tolerance hides a signal that is wrong everywhere but the peak.
            denom = np.maximum(np.abs(n_S[both]), 1e-9)
            rel = float(np.max(np.abs(t_S[0, 0].numpy()[both] - n_S[both]) / denom))
            assert rel < 1e-6, f"{name}/{scene.name}: torch vs numpy rel={rel:.2e}"
    print("  torch signals match the numpy reference on smooth AND step scenes")

    print("selftest: all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
