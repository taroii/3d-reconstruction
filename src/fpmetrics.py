r"""
Flying-pixel metric core (notes/premise_check.md Sec. 5). Pure NumPy, no I/O.

Everything is per-view, along the camera ray, in that view's own frame: inputs are
depth maps, never point clouds, so no cross-view alignment or pose enters.

Definitions implemented verbatim from the pre-registered protocol:

  rel_jump(p) = max_{q in N4(p)} |D(p)-D(q)| / min(D(p),D(q))
  B           = {p : rel_jump(p) > eta}
  B_tau       = dilate(B, tau);  I = valid \ B_tau
  S(p)        = window depths, sorted, split where consecutive rel gap > eta,
                each cluster represented by its median;  K_p = |S(p)|
  void(p)     = [d1 + beta*Delta, d2 - beta*Delta],  Delta = d2 - d1
  is_FP(p)    = D_pred(p) in void(p)
  FP          = |{p in B_eval : is_FP(p)}| / |B_eval|

Predictions in front of d1 / behind d2 are NOT flying pixels; they are counted
separately as `outside_rate`. Only K_p == 2 pixels enter B_eval; the full K_p
histogram is reported so coverage is explicit.

Self-test (Sec. 11 day 2) on hand-built cases with known answers:
    python src/fpmetrics.py --selftest
"""
from __future__ import annotations

import numpy as np

# The pre-registered sweep grid (Sec. 7). The centre of the grid is the headline.
ETA_GRID = (0.02, 0.05, 0.10)
TAU_GRID = (1, 2, 3)
BETA_GRID = (0.1, 0.2, 0.3)
ETA0, TAU0, BETA0 = 0.05, 2, 0.2


# --------------------------------------------------------------------------- #
# boundary set / band / interior
# --------------------------------------------------------------------------- #
def rel_jump(depth, valid):
    """max relative depth jump to any 4-neighbour. Invalid pixels and invalid
    neighbours contribute nothing; a pixel with no valid neighbour gets 0."""
    # A non-positive depth would make min(D(p),D(q)) zero and turn the relative
    # jump into inf; treat it as invalid rather than as an infinitely sharp edge.
    d = np.where(valid & np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float64)
    out = np.zeros(d.shape, np.float64)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        n = np.roll(d, sh, axis=ax)
        # rolled-in wrap row/col is not a real neighbour -> kill it
        if ax == 0:
            (n[:1] if sh > 0 else n[-1:])[...] = np.nan
        else:
            (n[:, :1] if sh > 0 else n[:, -1:])[...] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.abs(d - n) / np.minimum(d, n)
        out = np.fmax(out, np.nan_to_num(j, nan=0.0, posinf=0.0))
    return np.where(valid, out, 0.0)


def boundary_set(depth, valid, eta=ETA0):
    """B = pixels whose relative depth jump to a 4-neighbour exceeds eta."""
    return (rel_jump(depth, valid) > eta) & valid


def dilate(mask, tau):
    """Square dilation by `tau` pixels (Chebyshev radius), pure NumPy.

    Implemented with explicit shifted slices, NOT np.roll: roll wraps, which
    would let a boundary on row 0 dilate onto row H-1 and silently shrink the
    interior at the opposite border. Since the interior is what `align_scale`
    fits, a wrapped band corrupts the scale and therefore every FP number.
    """
    if tau <= 0:
        return mask.copy()
    t, (H, W) = int(tau), mask.shape
    out = np.zeros_like(mask)
    for dy in range(-t, t + 1):
        ys = slice(max(0, -dy), H - max(0, dy))
        yd = slice(max(0, dy), H - max(0, -dy))
        for dx in range(-t, t + 1):
            xs = slice(max(0, -dx), W - max(0, dx))
            xd = slice(max(0, dx), W - max(0, -dx))
            out[yd, xd] |= mask[ys, xs]
    return out


def interior(valid, band):
    """I = valid pixels outside the boundary band. Scale alignment uses only I."""
    return valid & ~band


# --------------------------------------------------------------------------- #
# depth support: cluster a window of GT depths into distinct surfaces
# --------------------------------------------------------------------------- #
def depth_support(depth, valid, pix, w=3, eta=ETA0, min_cluster=1):
    """Vectorised S(p) for many pixels at once.

    pix          : (N,2) int array of (row, col) pixel coordinates.
    returns dict with, per input pixel:
      K       (N,)  number of distinct surfaces in the (2w+1)^2 window
      d1, d2  (N,)  the two cluster medians when K == 2 (else nan)
      n_valid (N,)  valid depths in the window

    Clustering follows the protocol exactly: sort the valid window depths and
    split wherever the consecutive *relative* gap exceeds eta; each cluster is
    represented by its median. `min_cluster` (default 1 = as pre-registered) can
    require a cluster to hold more than one pixel; it is exposed only as a
    robustness knob and is NOT used for the headline number.
    """
    H, W = depth.shape
    m = 2 * w + 1
    N = len(pix)
    K = np.zeros(N, np.int32)
    d1 = np.full(N, np.nan)
    d2 = np.full(N, np.nan)
    nval = np.zeros(N, np.int32)
    if N == 0:
        return dict(K=K, d1=d1, d2=d2, n_valid=nval)

    # gather (N, m*m) windows with edge-safe indexing
    r = pix[:, 0][:, None] + np.arange(-w, w + 1)[None, :]        # (N,m)
    c = pix[:, 1][:, None] + np.arange(-w, w + 1)[None, :]
    rr = np.clip(r, 0, H - 1)[:, :, None].repeat(m, 2).reshape(N, -1)
    cc = np.clip(c, 0, W - 1)[:, None, :].repeat(m, 1).reshape(N, -1)
    inb = (((r >= 0) & (r < H))[:, :, None] &
           ((c >= 0) & (c < W))[:, None, :]).reshape(N, -1)

    win = depth[rr, cc].astype(np.float64)
    ok = valid[rr, cc] & inb & np.isfinite(win)
    win = np.where(ok, win, np.inf)
    win.sort(axis=1)                       # ascending, invalids (+inf) at the end
    nval = ok.sum(1).astype(np.int32)

    # relative gap between consecutive sorted depths; both entries must be valid
    a, b = win[:, :-1], win[:, 1:]
    both = np.isfinite(a) & np.isfinite(b)
    diff = np.subtract(b, a, out=np.zeros_like(a), where=both)
    gap = np.divide(diff, a, out=np.zeros_like(a), where=both & (a > 0))
    split = both & (gap > eta)             # (N, m*m-1) True = cluster ends at i
    K = np.where(nval > 0, split.sum(1) + 1, 0).astype(np.int32)

    # --- two-surface pixels: medians of the two contiguous sorted runs -------
    sel = (K == 2) & (nval >= 2)
    if sel.any():
        idx = np.flatnonzero(sel)
        j = split[idx].argmax(1)                     # unique split position
        nv = nval[idx]
        L1 = j + 1                                   # cluster 1 = win[0 .. j]
        L2 = nv - L1                                 # cluster 2 = win[j+1 .. nv-1]
        good = (L1 >= min_cluster) & (L2 >= min_cluster)

        def _med(base, L):
            """median of the contiguous sorted run win[idx, base : base+L]."""
            lo = base + (L - 1) // 2
            hi = base + L // 2
            return 0.5 * (win[idx, lo] + win[idx, hi])

        m1 = _med(np.zeros_like(L1), L1)
        m2 = _med(L1, L2)
        d1[idx] = np.where(good, m1, np.nan)
        d2[idx] = np.where(good, m2, np.nan)
        K[idx[~good]] = -1                           # excluded, not a clean pair
    return dict(K=K, d1=d1, d2=d2, n_valid=nval)


# --------------------------------------------------------------------------- #
# scale alignment (Sec. 6.1) -- interior pixels only, scale-only, no shift
# --------------------------------------------------------------------------- #
def align_scale(pred, gt, mask, mode="median"):
    """Return scalar s so that s*pred matches gt on `mask`. Scale only: the
    protocol aligns a scale-free prediction, not an affine-invariant one.
    `mask` MUST be the interior I -- aligning on the boundary band would let the
    quantity under test contaminate its own alignment."""
    p, g = pred[mask], gt[mask]
    ok = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
    if ok.sum() < 100:
        return np.nan
    p, g = p[ok], g[ok]
    if mode == "median":
        return float(np.median(g / p))
    if mode == "lstsq":
        return float((p * g).sum() / (p * p).sum())
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #
def fp_view(gt, valid, pred, *, eta=ETA0, tau=TAU0, beta=BETA0, w=3,
            delta_min=0.0, align="median", min_cluster=1, return_masks=False):
    """Flying-pixel statistics for one view.

    gt/valid/pred are HxW; `pred` is an unaligned per-view predicted depth at GT
    resolution. Returns a dict of scalars (plus masks if asked). `n_eval == 0`
    means the view contributes nothing and must be skipped by the caller, not
    counted as FP=0.
    """
    gt = gt.astype(np.float64)
    valid = valid & np.isfinite(gt) & (gt > 0)

    B = boundary_set(gt, valid, eta)
    band = dilate(B, tau)
    I = interior(valid, band)

    s = align_scale(pred, gt, I, align)
    p = pred.astype(np.float64) * (s if np.isfinite(s) else np.nan)

    pix = np.argwhere(B)
    sup = depth_support(gt, valid, pix, w=w, eta=eta, min_cluster=min_cluster)
    K, d1, d2 = sup["K"], sup["d1"], sup["d2"]

    khist = {int(k): int(v) for k, v in zip(*np.unique(K[K > 0], return_counts=True))}
    # NOTE: this is COVERAGE among the boundaries we actually measure, computed on
    # already-masked depth. It is NOT the Sec. 4 gate statistic, which must detect
    # boundaries BEFORE the dataset mask to see deleted ones -- that lives in
    # data.gate(). Do not threshold this at 0.5 and call it the gate.
    frac_both_sides = float((K >= 2).mean()) if len(K) else float("nan")

    delta = d2 - d1
    pv = p[pix[:, 0], pix[:, 1]] if len(pix) else np.zeros(0)
    # B_eval is defined by GT-side checks ONLY (Sec. 5), so every stream is scored
    # on the identical pixel set (Sec. 7). A non-finite prediction is NOT dropped
    # from the denominator -- that would let a model improve its own score by
    # returning NaN exactly where it would have been a flying pixel. Such pixels
    # count as not-FP and are surfaced separately as n_nonfinite.
    keep = (K == 2) & np.isfinite(delta) & (delta >= delta_min)
    n_eval = int(keep.sum())

    res = dict(n_boundary=int(len(pix)), n_eval=n_eval, scale=float(s),
               K_hist=khist, frac_boundary_valid=frac_both_sides,
               FP=float("nan"), outside_rate=float("nan"),
               near_rate=float("nan"), far_rate=float("nan"), n_nonfinite=0)
    if n_eval == 0:
        return (res, None) if return_masks else res

    dd1, dd2, dl, q = d1[keep], d2[keep], delta[keep], pv[keep]
    lo, hi = dd1 + beta * dl, dd2 - beta * dl
    with np.errstate(invalid="ignore"):
        is_fp = (q >= lo) & (q <= hi)      # NaN compares False -> not a flying pixel
        near = q < lo                      # in front of the near surface
        far = q > hi                       # behind the far surface
    res.update(FP=float(is_fp.mean()), outside_rate=float((near | far).mean()),
               near_rate=float(near.mean()), far_rate=float(far.mean()),
               n_nonfinite=int((~np.isfinite(q)).sum()))

    if return_masks:
        fpmask = np.zeros(gt.shape, bool)
        kp = pix[keep]
        fpmask[kp[:, 0], kp[:, 1]] = is_fp
        return res, dict(B=B, band=band, I=I, FP=fpmask, eval_pix=kp, is_fp=is_fp)
    return res


# --------------------------------------------------------------------------- #
# threshold sweep (Sec. 7) -- the mandatory (eta, tau, beta) grid
# --------------------------------------------------------------------------- #
def fp_view_grid(gt, valid, preds, *, etas=ETA_GRID, taus=TAU_GRID, betas=BETA_GRID,
                 w=3, delta_min=0.0, align="median", min_cluster=1):
    """`fp_view` over the full threshold grid for several predictions at once.

    preds: {stream name -> unaligned per-view predicted depth at GT resolution}.
    Returns [{stream, eta, tau, beta, ...}, ...].

    The boundary set and the depth support depend only on eta, the interior (and
    hence the scale) only on (eta, tau), and beta only enters the final void test.
    Exploiting that nesting costs 3 support passes instead of 27. `_selftest`
    asserts this agrees with `fp_view` cell by cell, so the fast path cannot drift
    away from the reference implementation.
    """
    gt = gt.astype(np.float64)
    valid = valid & np.isfinite(gt) & (gt > 0)
    out = []
    for eta in etas:
        B = boundary_set(gt, valid, eta)
        pix = np.argwhere(B)
        sup = depth_support(gt, valid, pix, w=w, eta=eta, min_cluster=min_cluster)
        K, d1, d2 = sup["K"], sup["d1"], sup["d2"]
        khist = {int(k): int(v) for k, v in zip(*np.unique(K[K > 0], return_counts=True))}
        fbv = float((K >= 2).mean()) if len(K) else float("nan")
        delta = d2 - d1
        base = (K == 2) & np.isfinite(delta) & (delta >= delta_min)
        for tau in taus:
            I = interior(valid, dilate(B, tau))
            for name, pred in preds.items():
                s = align_scale(pred, gt, I, align)
                q = (pred.astype(np.float64) * s)[pix[:, 0], pix[:, 1]] if len(pix) \
                    else np.zeros(0)
                keep = base                      # GT-defined; see fp_view
                n = int(keep.sum())
                rec = dict(stream=name, eta=eta, tau=tau, w=w, align=align,
                           n_boundary=int(len(pix)), n_eval=n, scale=float(s),
                           K_hist=khist, frac_boundary_valid=fbv)
                if n == 0:
                    for beta in betas:
                        out.append(dict(rec, beta=beta, FP=float("nan"),
                                        outside_rate=float("nan"),
                                        near_rate=float("nan"), far_rate=float("nan"),
                                        n_fp=0, n_outside=0, n_nonfinite=0))
                    continue
                dd1, dd2, dl, qq = d1[keep], d2[keep], delta[keep], q[keep]
                nnf = int((~np.isfinite(qq)).sum())
                for beta in betas:
                    lo, hi = dd1 + beta * dl, dd2 - beta * dl
                    with np.errstate(invalid="ignore"):
                        fp = (qq >= lo) & (qq <= hi)
                        near, far = qq < lo, qq > hi
                    out.append(dict(rec, beta=beta, FP=float(fp.mean()),
                                    outside_rate=float((near | far).mean()),
                                    near_rate=float(near.mean()),
                                    far_rate=float(far.mean()), n_fp=int(fp.sum()),
                                    n_outside=int((near | far).sum()), n_nonfinite=nnf))
    return out


# --------------------------------------------------------------------------- #
# self-test: hand-built cases with known answers (Sec. 11, day 2)
# --------------------------------------------------------------------------- #
def _step(H=64, W=64, near=2.0, far=4.0):
    """Vertical occlusion boundary: left half near, right half far."""
    d = np.full((H, W), far)
    d[:, : W // 2] = near
    return d


def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    gt = _step()
    valid = np.ones_like(gt, bool)

    # 1. a flat plane has no boundary at all
    flat = np.full((64, 64), 3.0)
    check("flat plane -> empty boundary set", boundary_set(flat, valid, 0.05).sum() == 0)

    # 2. the step is detected, and only in the two columns astride the jump
    B = boundary_set(gt, valid, 0.05)
    cols = np.unique(np.argwhere(B)[:, 1])
    check("step -> boundary at the jump only", set(cols.tolist()) == {31, 32},
          f"cols={cols.tolist()}")

    # 3. GT against itself is never a flying pixel (this is the GT control on
    #    synthetic data: FP_GT == 0 by construction)
    r = fp_view(gt, valid, gt.copy())
    check("GT vs GT -> FP == 0", r["FP"] == 0.0, f"FP={r['FP']}, n={r['n_eval']}")
    check("GT vs GT -> K == 2 everywhere", set(r["K_hist"]) == {2}, r["K_hist"])

    # 4. a prediction sitting exactly midway in the void is 100% flying
    mid = np.full_like(gt, 3.0)                       # (2+4)/2, inside the void
    r = fp_view(gt, valid, mid)
    check("midway prediction -> FP == 1", r["FP"] == 1.0, f"FP={r['FP']}")

    # 5. predictions outside [d1,d2] are counted as outside, never as FP.
    #    Perturb ONLY the boundary pixels so the interior still pins the scale
    #    at 1 -- a globally constant prediction would just be rescaled onto the
    #    surface by align_scale and would not test what it looks like it tests.
    B0 = boundary_set(gt, valid, 0.05)
    front = gt.copy(); front[B0] = 1.0                # in front of the near surface
    r = fp_view(gt, valid, front)
    check("in-front prediction -> FP 0, outside 1",
          r["FP"] == 0.0 and r["outside_rate"] == 1.0 and r["near_rate"] == 1.0,
          f"FP={r['FP']}, out={r['outside_rate']}")
    behind = gt.copy(); behind[B0] = 9.0              # behind the far surface
    r = fp_view(gt, valid, behind)
    check("behind prediction -> FP 0, outside 1",
          r["FP"] == 0.0 and r["outside_rate"] == 1.0 and r["far_rate"] == 1.0,
          f"FP={r['FP']}, out={r['outside_rate']}")

    # 6. the beta margin excludes points hugging either surface
    eps = np.where(np.arange(64)[None, :] < 32, 2.0, 4.0) * np.ones((64, 1))
    r = fp_view(gt, valid, eps + 0.05)                # just off each surface
    check("beta margin protects the surfaces", r["FP"] == 0.0, f"FP={r['FP']}")

    # 7. scale alignment is computed on the interior and recovers a known factor
    r = fp_view(gt, valid, gt * 0.5)
    check("scale recovered from interior", abs(r["scale"] - 2.0) < 1e-9,
          f"s={r['scale']}")
    check("scaled GT still FP == 0", r["FP"] == 0.0)

    # 8. delta_min rejects trivial jumps
    small = np.full((64, 64), 2.0)
    small[:, 32:] = 2.2                               # 10% jump, absolute 0.2
    r = fp_view(small, valid, small.copy(), eta=0.05, delta_min=1.0)
    check("delta_min rejects a trivial jump", r["n_eval"] == 0, f"n={r['n_eval']}")

    # 9. invalid pixels never enter the boundary set or the interior
    v = valid.copy()
    v[:, 30:34] = False
    B2 = boundary_set(gt, v, 0.05)
    check("invalid pixels excluded from B", not B2[:, 30:34].any())

    # 10. a three-surface staircase is reported as K=3 and excluded from B_eval.
    #     The treads must be narrow enough that one (2w+1) window spans all three
    #     surfaces, otherwise no pixel ever sees more than two.
    st = np.full((64, 64), 2.0)
    st[:, 30:33] = 4.0
    st[:, 33:] = 8.0
    r = fp_view(st, valid, st.copy(), w=3)
    check("staircase exposes K=3 in the histogram", 3 in r["K_hist"], r["K_hist"])
    check("K=3 pixels are kept out of B_eval",
          r["n_eval"] == r["K_hist"].get(2, 0), f"n_eval={r['n_eval']}, {r['K_hist']}")
    check("staircase GT vs GT is still FP == 0", r["FP"] == 0.0, f"FP={r['FP']}")

    # 11. REGRESSION: dilate must not wrap across image borders. A band that
    #     wraps eats interior pixels at the opposite edge, which shifts the
    #     scale fit and silently changes every FP number in the view.
    seed = np.zeros((7, 7), bool); seed[0, 3] = True
    d1_ = dilate(seed, 1)
    check("dilate does not wrap top->bottom", not d1_[6, 3])
    seed2 = np.zeros((7, 7), bool); seed2[3, 0] = True
    check("dilate does not wrap left->right", not dilate(seed2, 1)[3, 6])
    pt = np.zeros((11, 11), bool); pt[5, 5] = True
    check("dilate(tau=2) is a 5x5 square", dilate(pt, 2).sum() == 25,
          f"got {dilate(pt, 2).sum()}")
    check("dilate(tau=0) is identity", (dilate(pt, 0) == pt).all())

    # 12. REGRESSION: B_eval is GT-defined, so a stream cannot shrink its own
    #     denominator by predicting NaN where it would have been a flying pixel.
    #     The NaNs go ONLY on boundary pixels, which are excluded from the
    #     interior anyway, so the scale fit is byte-identical between the two
    #     runs and the only thing under test is the denominator.
    mid_pred = np.full_like(gt, 3.0)                  # midway = flying everywhere
    holes = mid_pred.copy()
    hb = boundary_set(gt, valid, 0.05)
    hb[32:, :] = False                                # blank half the boundary rows
    holes[hb] = np.nan
    r_full = fp_view(gt, valid, mid_pred)
    r_hole = fp_view(gt, valid, holes)
    check("NaN holes leave the scale untouched", r_hole["scale"] == r_full["scale"],
          f"{r_hole['scale']} vs {r_full['scale']}")
    check("NaN holes do not shrink n_eval",
          r_hole["n_eval"] == r_full["n_eval"],
          f"{r_hole['n_eval']} vs {r_full['n_eval']}")
    check("NaN holes are reported, not hidden", r_hole["n_nonfinite"] > 0,
          f"n_nonfinite={r_hole['n_nonfinite']}")
    check("NaN pixels are not counted as flying", r_hole["FP"] < r_full["FP"],
          f"{r_hole['FP']:.3f} vs {r_full['FP']:.3f}")

    # 13. the fast sweep path must reproduce the reference implementation exactly
    rng = np.random.default_rng(0)
    noisy = gt + rng.normal(0, 0.35, gt.shape)          # lands some points in the void
    recs = fp_view_grid(gt, valid, {"m": noisy})
    agree = True
    for r in recs:
        ref = fp_view(gt, valid, noisy, eta=r["eta"], tau=r["tau"], beta=r["beta"])
        for f in ("FP", "outside_rate", "n_eval", "scale"):
            a, b = ref[f], r[f]
            if not (a == b or (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12):
                agree = False
                print(f"      grid/ref mismatch {f} at {r['eta']},{r['tau']},{r['beta']}: {a} vs {b}")
    check(f"grid path == reference over all {len(recs)} cells", agree)
    check("grid actually exercises the void", any(r["FP"] > 0 for r in recs),
          f"max FP={max(r['FP'] for r in recs):.3f}")

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_selftest() if a.selftest else
                     print(__doc__) or 0)
