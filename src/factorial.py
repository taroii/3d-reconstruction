r"""
Phase 1B — the controlled synthetic experiment (notes/diagnosis.md §3).

One file, because this is one experiment: the scene construction, the objective
variants and the factorial are only ever used together and never independently.

The only controlled experiment in Phase 1, and the only place a causal claim may
originate. Anything determining a learned predictor sits in exactly one of
TARGET, OBJECTIVE, or PREDICTOR-AND-OPTIMIZATION — a partition by construction,
with the third bucket defined as the residual. **[ASSUMED]** only that the
partition is USEFUL, i.e. that effects localize rather than smearing across all
three. If the residual dominates we learn that the cause is not the target or the
objective as implemented here; the residual cannot name a mechanism.

--- 1. SCENE CONSTRUCTION (§3.2) -------------------------------------------

A controlled occlusion boundary at a KNOWN location, so "did the predictor put
depth in the void" has a ground answer rather than an estimated one.

* **Rendered high-resolution, then downsampled two ways.** Nearest-neighbour
  gives a clean target (every pixel belongs to exactly one surface);
  area-average gives a mixed target (boundary pixels are blends sitting in the
  void). That makes "dirty target" a controlled FACTOR rather than a nuisance.

* **Sub-pixel jitter across realizations.** This is what makes the §3.4
  falsifier meaningful. Claim (a) concerns the minimizer of `Σ_k w_k ‖x − d_k‖`
  for a target DISTRIBUTION over candidate depths along a ray. With one
  deterministic target and free per-pixel parameters, any predictor trivially
  reproduces it and the test is vacuous. Jitter makes a boundary pixel's target
  genuinely vary between the near and far surface, with weights set by sub-pixel
  coverage. **[ASSUMED]** that this is a fair stand-in for the target ambiguity
  a real model faces at a boundary.

--- 2. OBJECTIVE VARIANTS (§3.3) -------------------------------------------

Each term is a reduction of the published loss to the scalar-depth setting.
**[ASSUMED]** that reducing a 3-channel pointmap term to its depth analogue
preserves the behaviour under test; along a pixel's ray the pointmap residual is
collinear with the depth residual, which is what claim (a) relies on, but the
reduction is an assumption and is stated as one.

CONFIDENCE: `C·ℓ − α log C` is minimized at `C* = α/ℓ`, giving `α log ℓ`, which
is unbounded below as `ℓ → 0` — §1.1(b)'s caveat. Every real codebase constrains
`C`, so both variants are provided: §1.3 lists the clamp itself as a candidate
mechanism, which makes it a factor rather than a fixed choice.

--- 3. PREDICTORS ----------------------------------------------------------

  free       one parameter per pixel, no shared weights and no input. Its
             optimum IS the pointwise minimizer, so this level removes the
             predictor by construction and measures what the loss alone prefers.
             This is where the §3.4 falsifier lives.
  cnn_small  3 conv layers, receptive field 7
  cnn_large  6 dilated conv layers, receptive field 35

The CNNs map an ANTIALIASED IMAGE to depth, the kind of input a real model gets.
Fitting uses the jittered realizations; scoring uses the un-jittered reference,
so a predictor that merely memorizes one target cannot score well.

SCOPE, to carry into any write-up: results describe THIS construction, THESE
objective implementations and THESE small predictors. **[ASSUMED]** that they
transfer to billion-parameter models trained at scale — an extrapolation, not a
measurement. Phase 1C is the partial and confounded check on it.

    python src/factorial.py --selftest
    python src/factorial.py --preregister --out results/phase1
    python src/factorial.py --run --seeds 5 --out results/phase1
"""
from __future__ import annotations

import os
import csv
import json
import time
import argparse
import itertools
import collections
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fpmetrics as FM

SUPER = 8                      # hi-res supersampling factor before downsampling


@dataclass
class Scene:
    """One occlusion-boundary configuration. Depth increases away from camera."""
    h: int = 64
    w: int = 64
    near: float = 2.0
    gap: float = 1.0           # Delta between the near and far surface
    slant_near: float = 0.0    # metres of depth change across the full width
    slant_far: float = 0.0
    k: int = 2                 # 2 or 3 surfaces
    mid_frac: float = 0.10     # width of the middle surface when k == 3
    boundary: float = 0.5      # boundary position as a fraction of width
    px_offset: float = 0.5     # extra sub-pixel shift, in OUTPUT pixels

    @property
    def far(self):
        return self.near + self.gap

    def depths(self):
        """The distinct surface depths, near to far."""
        if self.k == 2:
            return [self.near, self.far]
        return [self.near, self.near + 0.5 * self.gap, self.far]


def render(sc: Scene, jitter=0.0, super_=SUPER):
    """Hi-res depth map. `jitter` shifts the boundary by that many OUTPUT pixels.

    Surfaces are slanted along x by `slant_*` metres across the full width, so a
    pixel's own depth can differ from its surface's median -- the situation that
    made iBims' control non-zero in Phase 0.
    """
    H, W = sc.h * super_, sc.w * super_
    x = (np.arange(W) + 0.5) / super_                    # in output-pixel units
    # The half-pixel offset is load-bearing, not cosmetic: with the boundary
    # exactly on an output-pixel edge every hi-res sample in a pixel belongs to
    # one surface, area-averaging blends nothing, and the "mixed target" factor
    # silently becomes a no-op. Keep the boundary generically off-grid.
    b = sc.boundary * sc.w + sc.px_offset + jitter
    t = np.clip(x / sc.w, 0, 1)

    near = sc.near + sc.slant_near * t
    far = sc.far + sc.slant_far * t
    row = np.where(x < b, near, far)

    if sc.k >= 3:
        mid_w = sc.mid_frac * sc.w
        if b + mid_w > sc.w:
            raise SystemExit(
                f"k=3 middle strip runs off the right edge (boundary {b:.1f} + width "
                f"{mid_w:.1f} > {sc.w}). The far surface would vanish and the scene "
                f"would silently become k=2 while depths() still reports three.")
        mid = sc.near + 0.5 * sc.gap + 0.5 * (sc.slant_near + sc.slant_far) * t
        row = np.where((x >= b) & (x < b + mid_w), mid, row)

    return np.repeat(row[None, :], H, axis=0).astype(np.float64)


def down_nearest(hi, super_=SUPER):
    """Clean target: every output pixel takes one hi-res sample, so it belongs to
    exactly one surface. No value that was not in the scene can appear."""
    off = super_ // 2
    return hi[off::super_, off::super_].copy()


def down_area(hi, super_=SUPER):
    """Mixed target: area average. A boundary pixel becomes a blend of the two
    surfaces -- a value belonging to neither, sitting in the void. This is the
    'GT is itself averaged at boundary pixels' candidate from §1.3, made explicit."""
    H, W = hi.shape
    h, w = H // super_, W // super_
    return hi[:h * super_, :w * super_].reshape(h, super_, w, super_).mean((1, 3))


def render_image(sc: Scene, jitter=0.0, super_=SUPER, albedo=(0.25, 0.75)):
    """Intensity image for the same scene, area-averaged to output resolution.

    The CNN predictors need an INPUT, and it has to be the kind of input a real
    model gets: a camera integrates over a pixel footprint, so the edge in the
    image is ANTIALIASED even when the depth step is perfectly sharp. Handing the
    network a sharp edge would quietly remove the difficulty the "predictor
    smoothness" hypothesis (§1.3) is about.
    """
    H, W = sc.h * super_, sc.w * super_
    x = (np.arange(W) + 0.5) / super_
    b = sc.boundary * sc.w + sc.px_offset + jitter
    row = np.where(x < b, albedo[0], albedo[1])
    if sc.k >= 3:
        mid_w = sc.mid_frac * sc.w
        if b + mid_w > sc.w:
            raise SystemExit(f"k=3 middle strip runs off the right edge in the image "
                             f"({b:.1f} + {mid_w:.1f} > {sc.w}).")
        row = np.where((x >= b) & (x < b + mid_w),
                       0.5 * (albedo[0] + albedo[1]), row)
    hi = np.repeat(row[None, :], H, axis=0)
    return down_area(hi, super_).astype(np.float64)


def realizations(sc: Scene, n=16, jitter_px=1.0, mode="nearest", seed=0, super_=SUPER):
    """`n` target maps with the boundary jittered sub-pixel.

    Returns (targets, images, reference_depth, reference_image), all float64.
    The reference is the UN-JITTERED clean map and image; scoring there keeps the
    measurement independent of which target variant the predictor was fitted to.
    """
    rng = np.random.default_rng(seed)
    down = down_nearest if mode == "nearest" else down_area
    js = rng.uniform(-jitter_px / 2, jitter_px / 2, size=n)
    tg = np.stack([down(render(sc, j, super_), super_) for j in js])
    im = np.stack([render_image(sc, j, super_) for j in js])
    ref = down_nearest(render(sc, 0.0, super_), super_)
    ref_im = render_image(sc, 0.0, super_)
    return (tg.astype(np.float64), im.astype(np.float64),
            ref.astype(np.float64), ref_im.astype(np.float64))


# --------------------------------------------------------------------------- #
# closed form: what the position term alone prefers
# --------------------------------------------------------------------------- #
def weighted_median(vals, w=None, axis=0):
    """Minimizer of `Σ_k w_k |x − v_k|`.

    For collinear atoms this IS the geometric median, so it is the closed-form
    answer to claim (a) for the position-only / free-parameter cell -- available
    without fitting anything, and therefore a check on the optimizer rather than
    a substitute for it.
    """
    v = np.moveaxis(np.asarray(vals, float), axis, 0)
    n = v.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, float)
    order = np.argsort(v, axis=0)
    vs = np.take_along_axis(v, order, 0)
    ws = w[order] if w.ndim == 1 else np.take_along_axis(w, order, 0)
    cw = np.cumsum(ws, axis=0)
    half = cw[-1] / 2.0
    idx = (cw < half).sum(0)
    idx = np.clip(idx, 0, n - 1)
    return np.take_along_axis(vs, idx[None], 0)[0]


EPS = 1e-8


# --------------------------------------------------------------------------- #
# confidence
# --------------------------------------------------------------------------- #
def conf_from_raw(raw, clamp=1.0, mode="softplus"):
    """Map an unconstrained parameter to C > 0.

    `clamp` is the floor applied to C. Passing None gives the unclamped form the
    §1.1(b) derivation assumes, which is degenerate but is exactly what we want
    to be able to run in order to test whether the clamp matters.
    """
    c = F.softplus(raw) + EPS if mode == "softplus" else raw.exp()
    return c if clamp is None else c.clamp(min=clamp)


def apply_conf(per_pixel, conf, alpha=0.2):
    """`C·ℓ − α log C`, reduced over all pixels. `conf=None` leaves ℓ untouched."""
    if conf is None:
        return per_pixel.mean()
    return (per_pixel * conf - alpha * torch.log(conf)).mean()


# --------------------------------------------------------------------------- #
# terms
# --------------------------------------------------------------------------- #
def position(pred, tgt, squared=False):
    """Per-pixel position residual. Unsquared is the published form."""
    d = pred - tgt
    return d.pow(2) if squared else d.abs()


def _grads(x):
    """Forward differences, zero-padded so the shape is preserved."""
    gy = torch.zeros_like(x)
    gx = torch.zeros_like(x)
    gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
    return gy, gx


def _safe_norm(sq):
    """sqrt of a squared magnitude, exactly 0 at 0 and differentiable there.

    The usual `sqrt(x + eps)` dodge is not good enough here: it puts a constant
    `sqrt(eps)` floor on EVERY pixel, so the term never reads zero even on a
    perfect match and 'where does this term fire' becomes unanswerable.
    Subtracting the floor keeps the gradient finite at the origin and restores
    the true zero.
    """
    return (sq + EPS).sqrt() - EPS ** 0.5


def gradient(pred, tgt):
    """VGGT's gradient term, per pixel. Couples neighbours, so an objective that
    includes it is NOT separable across pixels and the pointwise minimizer of
    §1.1(a) no longer describes it."""
    py, px = _grads(pred)
    ty, tx = _grads(tgt)
    return _safe_norm((py - ty).pow(2) + (px - tx).pow(2))


def normals_from_depth(d, fx=1.0, fy=1.0):
    """Surface normal per pixel from a depth map, via `n ∝ (-fx·∂z/∂x, -fy·∂z/∂y, 1)`."""
    gy, gx = _grads(d)
    n = torch.stack([-fx * gx, -fy * gy, torch.ones_like(d)], dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + EPS)   # +z component is 1, never 0


def normal(pred, tgt, fx=1.0, fy=1.0):
    """π³'s normal loss, per pixel: 1 − cos between predicted and target normals."""
    return 1.0 - (normals_from_depth(pred, fx, fy)
                  * normals_from_depth(tgt, fx, fy)).sum(-1)


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def total(pred, tgt, conf=None, *, w_grad=0.0, w_normal=0.0, alpha=0.2,
          squared=False, fx=1.0, fy=1.0):
    """Full objective for one factorial cell.

    Confidence weights the POSITION term only, matching the published losses; the
    auxiliary terms are added outside it. Weight 0 removes a term exactly, so the
    'position only' level is the same code path as the others rather than a
    separate implementation that could drift.
    """
    loss = apply_conf(position(pred, tgt, squared), conf, alpha)
    if w_grad:
        loss = loss + w_grad * gradient(pred, tgt).mean()
    if w_normal:
        loss = loss + w_normal * normal(pred, tgt, fx, fy).mean()
    return loss


# Objective levels for the §3.3 factorial. `clamp` is carried alongside because
# it changes the objective, not the predictor.
LEVELS = {
    "position":            dict(w_grad=0.0, w_normal=0.0, use_conf=False),
    "position+grad":       dict(w_grad=1.0, w_normal=0.0, use_conf=False),
    "position+normal":     dict(w_grad=0.0, w_normal=1.0, use_conf=False),
    "position+both":       dict(w_grad=1.0, w_normal=1.0, use_conf=False),
    "position+conf":       dict(w_grad=0.0, w_normal=0.0, use_conf=True, clamp=1.0),
    "position+conf_noclip": dict(w_grad=0.0, w_normal=0.0, use_conf=True, clamp=None),
    "position_squared":    dict(w_grad=0.0, w_normal=0.0, use_conf=False, squared=True),
}


DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Scene levels. Gap is swept over an order of magnitude; slant and k=3 make the
# two-surface restriction and the median-stand-in question into variables.
SCENES = {
    "fronto_d1.0": Scene(h=48, w=64, near=2.0, gap=1.0),
    "fronto_d0.2": Scene(h=48, w=64, near=2.0, gap=0.2),
    "fronto_d3.0": Scene(h=48, w=64, near=2.0, gap=3.0),
    "slanted_d1.0": Scene(h=48, w=64, near=2.0, gap=1.0, slant_near=0.5, slant_far=0.5),
    "k3_d2.0": Scene(h=48, w=64, near=2.0, gap=2.0, k=3, mid_frac=0.12),
}
TARGETS = ("clean", "mixed")          # nearest-neighbour vs area-average downsample
PREDICTORS = ("free", "cnn_small", "cnn_large")


# --------------------------------------------------------------------------- #
# predictors
# --------------------------------------------------------------------------- #
class Free(nn.Module):
    """One parameter per pixel. No input, no sharing: the optimum is the
    pointwise minimizer of whatever objective is applied."""

    def __init__(self, h, w, init):
        super().__init__()
        self.p = nn.Parameter(torch.full((h, w), float(init)))

    def forward(self, img):
        return self.p.expand(img.shape[0], -1, -1)


class CNN(nn.Module):
    """Image -> depth. `dilations` sets the receptive field."""

    def __init__(self, ch=32, dilations=(1, 1, 1), bias=0.0):
        super().__init__()
        layers, c_in = [], 1
        for d in dilations:
            layers += [nn.Conv2d(c_in, ch, 3, padding=d, dilation=d), nn.ReLU()]
            c_in = ch
        layers += [nn.Conv2d(c_in, 1, 1)]
        self.net = nn.Sequential(*layers)
        self.bias = bias
        self.rf = 1 + 2 * sum(dilations)

    def forward(self, img):
        return self.net(img[:, None])[:, 0] + self.bias


def make_predictor(kind, h, w, init):
    if kind == "free":
        return Free(h, w, init)
    if kind == "cnn_small":
        return CNN(32, (1, 1, 1), init)
    if kind == "cnn_large":
        return CNN(48, (1, 2, 4, 8, 1, 1), init)
    raise KeyError(kind)


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #
def fit_cell(scene, objective, predictor, target, seed, *, n_real=25, steps=1500,
             lr=0.05, jitter_px=1.0):
    """Fit one factorial cell and score it with the SAME flying-pixel metric used
    on the real models, so Phase 1B numbers are commensurable with Phase 0.

    `n_real` MUST BE ODD, and that is a correctness precondition rather than a
    preference. With an even number of symmetric jitter draws, a boundary
    pixel's targets can split exactly half near / half far. At such a tie the L1
    objective is FLAT across the entire interval `[near, far]` -- the subgradient
    is identically zero -- so the argmin set is the whole void and the optimizer
    simply stays wherever it was initialized. Claim (a) pins the minimizer to a
    surface only when the tie is broken; at a tie it makes no prediction at all.
    Measured tie rate at n_real=24 was ~14% per seed, which gave a default
    5-seed run a better-than-even chance of writing "FALSIFIED" against the
    project's central claim on the strength of an optimizer accident.
    """
    if n_real % 2 == 0:
        raise SystemExit(
            f"n_real must be ODD (got {n_real}). An even count lets a boundary "
            f"pixel's targets tie exactly, which makes the L1 argmin the whole "
            f"void interval and turns the falsifier into a coin flip.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = SCENES[scene]
    tg, im, ref, ref_im = realizations(sc, n=n_real, jitter_px=jitter_px,
                                          mode="nearest" if target == "clean" else "area",
                                          seed=seed)
    T = torch.tensor(tg, dtype=torch.float32, device=DEV)
    I = torch.tensor(im, dtype=torch.float32, device=DEV)
    RI = torch.tensor(ref_im, dtype=torch.float32, device=DEV)[None]

    cfg = dict(LEVELS[objective])
    use_conf = cfg.pop("use_conf", False)
    clamp = cfg.pop("clamp", 1.0)
    squared = cfg.pop("squared", False)

    net = make_predictor(predictor, sc.h, sc.w, init=float(ref.mean())).to(DEV)
    params = list(net.parameters())
    raw_c = None
    if use_conf:
        raw_c = torch.zeros(sc.h, sc.w, device=DEV, requires_grad=True)
        params.append(raw_c)
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        pred = net(I)
        conf = (conf_from_raw(raw_c, clamp=clamp).expand_as(pred)
                if raw_c is not None else None)
        loss = total(pred, T, conf, squared=squared, **cfg)
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = net(RI)[0].double().cpu().numpy()

    valid = np.ones_like(ref, bool)
    # delta_min scales with the gap so a 0.2 m scene is not rejected wholesale
    r = FM.fp_view(ref, valid, out, eta=FM.ETA0, tau=FM.TAU0, beta=FM.BETA0,
                   w=3, delta_min=0.05 * sc.gap)

    # For the one cell with a closed form -- position-only, free parameters --
    # compare the fit against `weighted_median`, which IS the pointwise minimizer
    # for collinear atoms. Any gap is an OPTIMIZER artefact, not evidence about
    # the loss, and the falsifier must never be reported off a bad fit. This is
    # the "check on the optimizer" the weighted_median docstring promises.
    opt_gap = float("nan")
    if objective == "position" and predictor == "free":
        closed = weighted_median(tg, axis=0)
        opt_gap = float(np.abs(out - closed).max())

    return dict(scene=scene, objective=objective, predictor=predictor,
                target=target, seed=seed, FP=r["FP"], outside=r["outside_rate"],
                n_eval=r["n_eval"], final_loss=float(loss.item()),
                rmse=float(np.sqrt(np.mean((out - ref) ** 2))),
                optimizer_gap=opt_gap)


# --------------------------------------------------------------------------- #
# variance decomposition
# --------------------------------------------------------------------------- #
def decompose(rows, factors=("objective", "predictor", "target"), y="FP"):
    """Type-I-style sum-of-squares partition: the share of variance in `y`
    attributable to each factor's main effect, with the rest left as residual.

    This is deliberately a MAIN-EFFECTS decomposition. Interactions are real and
    are reported separately as cell means; folding them in would let one factor
    absorb variance that belongs to a pairing.
    """
    vals = np.array([r[y] for r in rows if np.isfinite(r[y])], float)
    keep = [r for r in rows if np.isfinite(r[y])]
    if vals.size < 2:
        return {}
    grand = vals.mean()
    sst = ((vals - grand) ** 2).sum()
    out = {}
    for f in factors:
        ss = 0.0
        for lvl in sorted({r[f] for r in keep}):
            g = np.array([r[y] for r in keep if r[f] == lvl])
            ss += g.size * (g.mean() - grand) ** 2
        out[f] = dict(ss=ss, eta2=(ss / sst if sst > 0 else float("nan")),
                      levels={lvl: float(np.mean([r[y] for r in keep if r[f] == lvl]))
                              for lvl in sorted({r[f] for r in keep})})
    expl = sum(v["ss"] for v in out.values())
    out["_residual"] = dict(ss=max(sst - expl, 0.0),
                            eta2=max(sst - expl, 0.0) / sst if sst > 0 else float("nan"))
    out["_total_ss"] = sst
    return out


def boot_ci(vals, n=10000, seed=0):
    """95% CI on the mean. Fewer than two finite values has no CI -- the caller
    must not print a point estimate beside [nan, nan] as though it were one."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, v.size), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------- #
# which branch of notes/outcome_tree.md the numbers point to
# --------------------------------------------------------------------------- #
# Which decomposition bucket corresponds to which Test 2 branch of the tree.
BUCKET_TO_BRANCH = {"objective": "OBJECTIVE", "target": "TARGET",
                    "predictor": "RESIDUAL"}
DOMINANT_ABS = 0.15      # a factor must explain at least this much on its own
DOMINANT_REL = 0.50      # ...and at least this share of the named factors


def decide_branch(dec):
    """Map the decomposition onto Test 2 of notes/outcome_tree.md.

    ONE DISTINCTION THAT MUST NOT BE BLURRED. The tree's RESIDUAL branch means
    "smoothness only", i.e. the PREDICTOR bucket -- networks cannot represent a
    step. The decomposition's `_residual` is something else entirely: variance
    left unexplained by the three main effects (interactions, scene, seed).
    A large `_residual` does NOT put us on the tree's RESIDUAL branch; it means
    the design did not localize the effect and no branch can be selected. These
    are reported separately below and never summed.

    The thresholds are fixed here, in code, rather than chosen after seeing the
    numbers.
    """
    named = {k: dec[k]["eta2"] for k in ("objective", "predictor", "target")
             if k in dec}
    if not named:
        return "UNDETERMINED", "No decomposition available.", named
    tot = sum(named.values())
    unexplained = dec.get("_residual", {}).get("eta2", float("nan"))
    dom = [k for k, v in named.items()
           if v >= DOMINANT_ABS and tot > 0 and v / tot >= DOMINANT_REL]

    if np.isfinite(unexplained) and unexplained > 0.60:
        return ("UNDETERMINED",
                f"Main effects explain only {tot:.2f} of the variance while "
                f"{unexplained:.2f} is unexplained (interactions, scene, seed). The "
                f"design did not localize the effect. This is NOT the tree's "
                f"RESIDUAL branch -- that branch is a claim about the predictor, "
                f"and this is an absence of attribution.", named)
    if len(dom) == 1:
        k = dom[0]
        branch = BUCKET_TO_BRANCH[k]
        why = {"objective": "auxiliary coupling terms dominate",
               "target": "the supervision itself is dirty at boundaries",
               "predictor": "predictor smoothness/capacity dominates"}[k]
        return branch, f"{k} explains {named[k]:.2f} of the variance ({why}).", named
    if tot >= 0.25:
        return ("MIXED",
                "several factors contribute and none dominates: "
                + ", ".join(f"{k}={v:.2f}" for k, v in sorted(named.items())), named)
    return ("UNDETERMINED",
            f"No factor reaches the {DOMINANT_ABS:.2f} threshold and the named "
            f"effects total only {tot:.2f}.", named)


def write_decision(out, dec, fal_fp, lo, hi, falsified, n_cells, seeds):
    branch, why, named = decide_branch(dec)
    tree = {"OBJECTIVE": "STRONG PAPER — corrects a stated belief in the field",
            "TARGET": "SOLID PAPER — implicates the benchmarks, not one loss term",
            "MIXED": "SOLID PAPER — attribution with effect sizes",
            "RESIDUAL": "THIN — pivot to benchmark or workshop",
            "UNDETERMINED": "no branch selected"}[branch]
    md = f"""# Phase 1 — decision

Generated {time.strftime('%F %T')} from {n_cells} factorial cells, {seeds} seeds.
Branches are those of `notes/outcome_tree.md`. Every claim carries a
`notes/diagnosis.md` §0 label; none is promoted without the evidence for it.

## Test 2 branch: **{branch}**

{why}

Tree consequence: _{tree}_

**[MEASURED]** — this holds for this synthetic construction, these objective
implementations and these small predictors, and for nothing else yet.
**[ASSUMED]** that it transfers to billion-parameter models trained at scale;
that is an extrapolation, and Phase 1C is the partial and confounded check on it.

## The pre-registered falsifier (§3.4)

> position-only + free parameters + clean GT -> FP ~ 0

**[MEASURED]** FP = {np.nanmean(fal_fp):.4f}, 95% CI [{lo:.4f}, {hi:.4f}].

{"**NOT EVALUATED.** No falsifier cell produced a finite result; this run says nothing about claim (a)." if falsified is None else "**FALSIFIED.** The derivation in §1.1(a), or its collinearity condition, does not hold here. §1.1 must be rewritten before anything downstream is trusted." if falsified else "**Not falsified.** Consistent with claim (a): with the unsquared norm and no coupling terms the pointwise optimum lands on a surface. A consistency check, not a proof."}

## Variance attributed

| bucket | share | reading |
|---|---|---|
""" + "".join(
        "| {} | {:.3f} | tree branch {} |\n".format(k, v, BUCKET_TO_BRANCH[k])
        for k, v in sorted(named.items())
    ) + f"""| _unexplained_ | {dec.get('_residual', {}).get('eta2', float('nan')):.3f} | interactions, scene, seed — **not** the tree's RESIDUAL branch |

## Scope and what is not claimed

The candidate list in `notes/diagnosis.md` §1.3 is **open**. Selecting a branch
says which of the three buckets carried variance in THIS design; it does not
show the list was exhaustive, and the unexplained bucket cannot name a mechanism.

Per the standing rule, no fix is attempted on the strength of this.
"""
    p = os.path.join(out, "decision.md")
    open(p, "w").write(md)
    print("wrote", p)
    print(f"\n=== Test 2 branch: {branch} ===\n{why}")
    return branch


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
PREREG = """# Phase 1B — pre-registered predictions (notes/diagnosis.md §3.4)

Recorded {date}, BEFORE the factorial was run. Do not edit after seeing results;
`--run` refuses to start without this file so the order cannot be reversed.

| # | Prediction | Label going in |
|---|---|---|
| 1 | position-only + free params + clean GT -> `FP ~ 0` | **falsifier for claim (a)** |
| 2 | adding gradient and/or normal terms at published weights | **[UNTESTED]** — direction and magnitude both unknown. A null is informative and must be reported as such. |
| 3 | mixed GT (area-average) raises `FP` independent of objective | **[UNTESTED]**, plausible |
| 4 | smoother / larger-receptive-field predictor raises `FP` independent of objective | **[UNTESTED]**, plausible |

Prediction 1 is the only one with a committed direction. If it fails, the
derivation in §1.1(a) or its collinearity condition is wrong and §1.1 must be
rewritten before anything else proceeds.

Explicitly NOT predicted: that the objective factor dominates. The candidate list
in §1.3 is open and none of its entries is privileged going in.

## Factor levels

- objective: {objectives}
- predictor: {predictors}
- target: clean (nearest-neighbour downsample), mixed (area-average downsample)
- scene: {scenes}
- seeds: {seeds}
"""


def preregister(out, seeds, scenes, objectives, predictors, force=False):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "1b_prereg.md")
    if os.path.exists(p) and not force:
        raise SystemExit(f"{p} already exists. Pre-registration is write-once: the "
                         f"§3.4 predictions must precede the run.")
    open(p, "w").write(PREREG.format(
        date=time.strftime("%F"), seeds=seeds,
        objectives=", ".join(objectives), predictors=", ".join(predictors),
        scenes=", ".join(scenes)))
    print("wrote", p)


def run(out, seeds=5, scenes=None, objectives=None, predictors=None,
        steps=1500, n_real=24):
    scenes = scenes or list(SCENES)
    objectives = objectives or list(LEVELS)
    predictors = predictors or list(PREDICTORS)
    if not os.path.exists(os.path.join(out, "1b_prereg.md")):
        raise SystemExit(
            f"No pre-registration at {out}/1b_prereg.md.\n"
            f"§3.4 requires the predictions to be recorded BEFORE the run:\n"
            f"  python src/factorial.py --preregister --out {out}")
    cells = list(itertools.product(scenes, objectives, predictors, TARGETS,
                                   range(seeds)))
    print(f"{len(cells)} cells on {DEV}", flush=True)
    rows, t0 = [], time.time()
    for i, (sc, ob, pr, tg, sd) in enumerate(cells, 1):
        try:
            rows.append(fit_cell(sc, ob, pr, tg, sd, steps=steps, n_real=n_real))
        except Exception as e:
            print(f"  ! {sc}/{ob}/{pr}/{tg}/s{sd}: {type(e).__name__}: {e}", flush=True)
        if i % 25 == 0 or i == len(cells):
            print(f"  {i}/{len(cells)}  ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("Every factorial cell failed; nothing to write. Re-run with "
                         "a smaller grid and read the per-cell errors above.")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "1b_factorial.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", p)

    # ---- the falsifier, checked explicitly (§3.4) ----
    fal = [r for r in rows if r["objective"] == "position"
           and r["predictor"] == "free" and r["target"] == "clean"]
    fal_fp = [r["FP"] for r in fal]
    lo, hi = boot_ci(fal_fp)
    # None means NOT EVALUATED. It must not fall through to the negative branch,
    # which would stamp zero evidence as a confirmation of claim (a). An all-NaN
    # set is equally uninformative: nanmean of NaNs is NaN and `nan > 0.05` is
    # False, which would have read as "not falsified" too.
    finite = [x for x in fal_fp if np.isfinite(x)]
    falsified = bool(np.mean(finite) > 0.05) if finite else None

    dec = decompose(rows)
    md = [
        "# Phase 1B — variance decomposition",
        "",
        f"Generated {time.strftime('%F %T')} · {len(rows)} cells · {seeds} seeds · device {DEV}",
        "",
        "## The pre-registered falsifier (§3.4)",
        "",
        "> position-only + free parameters + clean GT -> FP ~ 0",
        "",
        f"Measured: **FP = {np.nanmean(fal_fp):.4f}** over {len(fal_fp)} cells, "
        f"95% CI [{lo:.4f}, {hi:.4f}].",
        "",
        ("**NOT EVALUATED.** No falsifier cell produced a finite result, so this run "
         "says nothing about claim (a) in either direction." if falsified is None else
         "**FALSIFIED.** The derivation in §1.1(a) or its collinearity condition does not "
         "hold here, and §1.1 must be rewritten." if falsified else
         "**Not falsified.** Consistent with claim (a): with the unsquared norm and no "
         "coupling terms, the pointwise optimum lands on a surface rather than between "
         "surfaces. This is a consistency check, not a proof of (a)."),
        "",
        "## Main-effects variance decomposition",
        "",
        "| factor | share of variance | level means |",
        "|---|---|---|",
    ]
    for f in ("objective", "predictor", "target"):
        if f not in dec:
            continue
        lv = "; ".join(f"{k}={v:.3f}" for k, v in dec[f]["levels"].items())
        md.append(f"| {f} | {dec[f]['eta2']:.3f} | {lv} |")
    md.append(f"| _residual (incl. interactions, scene, seed)_ | "
              f"{dec.get('_residual', {}).get('eta2', float('nan')):.3f} | — |")
    md += [
        "",
        "Main effects only: interactions are left in the residual on purpose, so no "
        "factor absorbs variance that belongs to a pairing. Scene and seed are also "
        "in the residual.",
        "",
        "## Scope",
        "",
        "**[ASSUMED]** These results describe this synthetic construction, these "
        "objective implementations, and these small predictors. Transfer to "
        "billion-parameter models trained on real data at scale is an extrapolation, "
        "not a measurement; Phase 1C is the partial and confounded check on it.",
        "",
        "**The residual bucket cannot name a mechanism.** If it dominates, we know only "
        "that the cause is not the target or the objective as implemented here.",
    ]
    q = os.path.join(out, "1b_decomposition.md")
    open(q, "w").write("\n".join(md) + "\n")
    print("wrote", q)
    json.dump(dec, open(os.path.join(out, "1b_decomposition.json"), "w"),
              indent=1, default=float)
    write_decision(out, dec, fal_fp, lo, hi, falsified, len(rows), seeds)
    print("\n".join(md[4:14]))
    return rows, dec


# --------------------------------------------------------------------------- #


def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    # ---- scene construction ----


    sc = Scene(h=32, w=64, near=2.0, gap=1.0)

    hi = render(sc)
    check("render has exactly the two surface depths",
          set(np.round(np.unique(hi), 6)) == {2.0, 3.0}, sorted(np.unique(hi)))

    cl = down_nearest(hi)
    check("clean target introduces no new values",
          set(np.round(np.unique(cl), 6)) == {2.0, 3.0}, sorted(np.unique(cl)))

    ar = down_area(hi)
    inter = ((ar > 2.001) & (ar < 2.999))
    check("area target DOES blend at the boundary", inter.any(),
          f"{inter.sum()} blended px, values {np.unique(np.round(ar[inter],3))[:3]}")
    check("area blending is confined to the boundary column",
          len(np.unique(np.argwhere(inter)[:, 1])) <= 2)
    grid = down_area(render(Scene(h=32, w=64, near=2.0, gap=1.0, px_offset=0.0)))
    check("a grid-aligned boundary would blend nothing (why px_offset exists)",
          not ((grid > 2.001) & (grid < 2.999)).any())

    # jitter must make a boundary pixel's target genuinely vary
    tg, im, ref, ref_im = realizations(sc, n=32, jitter_px=1.0, mode="nearest", seed=0)
    varies = (tg.std(0) > 1e-9)
    check("jitter creates a target distribution at the boundary", varies.any(),
          f"{varies.sum()} varying px")
    check("interior targets do not vary", varies.sum() < 0.1 * ref.size)

    # the falsifier, in closed form: the L1 optimum sits ON a surface,
    # the L2 optimum sits in the void
    col = np.argmax(varies.any(0))
    v = tg[:, :, col]
    med = weighted_median(v, axis=0)
    mean = v.mean(0)
    check("L1 (position-term) optimum lands on a surface",
          np.all(np.isclose(med[:, None], [2.0, 3.0]).any(1)), f"{np.unique(med)}")
    check("L2 optimum lands strictly inside the void",
          np.all((mean > 2.05) & (mean < 2.95)), f"mean={mean[0]:.3f}")

    # slant: a pixel's own value should differ from its surface median
    ss = Scene(h=32, w=64, gap=1.0, slant_near=0.4, slant_far=0.4)
    r = down_nearest(render(ss))
    check("image edge is antialiased (what a camera would give)",
          ((ref_im > 0.26) & (ref_im < 0.74)).any(),
          f"{int(((ref_im>0.26)&(ref_im<0.74)).sum())} intermediate px")
    check("image and depth share the same boundary column",
          abs(int(np.argmax(np.abs(np.diff(ref_im[0]))))
              - int(np.argmax(np.abs(np.diff(ref[0]))))) <= 1)

    check("slant produces within-surface depth variation",
          r[:, :20].std() > 1e-3, f"std={r[:, :20].std():.4f}")

    # K=3
    s3 = Scene(h=32, w=64, gap=2.0, k=3, mid_frac=0.12)
    r3 = down_nearest(render(s3))
    check("k=3 yields three distinct depths", len(np.unique(r3)) == 3,
          f"{np.unique(r3)}")

    # ---- objective variants ----


    torch.manual_seed(0)
    t = torch.zeros(1, 8, 8)
    p = torch.zeros(1, 8, 8)

    check("identical inputs give zero position loss", position(p, t).sum() == 0)
    check("identical inputs give zero gradient loss",
          gradient(p, t).max() < 1e-6)
    check("identical inputs give zero normal loss", normal(p, t).abs().max() < 1e-6)

    # a step: the gradient term must see it, and see it only at the step
    st = torch.zeros(1, 8, 8); st[..., 4:] = 1.0
    g = gradient(torch.zeros(1, 8, 8), st)
    check("gradient term fires at a step", g.max() > 0.9, f"max={g.max():.3f}")
    check("gradient term fires ONLY at the step",
          (g > 1e-6).sum() == 8, f"{int((g>1e-6).sum())} px")

    # normals: a flat plane must be exactly +z, a slope must not
    flat = normals_from_depth(torch.zeros(1, 8, 8))
    check("flat depth -> +z normal", torch.allclose(flat[..., 2], torch.ones(1, 8, 8)))
    ramp = torch.arange(8, dtype=torch.float32).repeat(8, 1)[None] * 0.1
    check("sloped depth -> tilted normal", normals_from_depth(ramp)[0, 4, 4, 0].abs() > 0.05)

    # confidence: the analytic optimum is C* = alpha / l
    l = torch.tensor(0.5)
    raw = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.1)
    for _ in range(2000):
        opt.zero_grad()
        c = conf_from_raw(raw, clamp=None)
        (l * c - 0.2 * torch.log(c)).mean().backward()
        opt.step()
    c_star = conf_from_raw(raw, clamp=None).item()
    check("unclamped confidence converges to alpha/l", abs(c_star - 0.2 / 0.5) < 0.02,
          f"C*={c_star:.4f} vs {0.2/0.5}")

    cl = conf_from_raw(torch.tensor([-20.0]), clamp=1.0).item()
    check("clamp floors C where the analysis would send it to ~0", cl == 1.0, f"C={cl}")

    # weight 0 must remove a term exactly
    a = total(p + 0.3, t, w_grad=0.0, w_normal=0.0)
    b = total(p + 0.3, t)
    check("weight 0 removes a term exactly", torch.allclose(a, b))

    # squared vs unsquared must actually differ on a spread target
    pred = torch.full((1, 1, 1), 0.4, requires_grad=True)
    tg = torch.tensor([0.0, 0.0, 1.0]).reshape(3, 1, 1)
    for sq, want in ((False, 0.0), (True, 1 / 3)):
        v = torch.full((1, 1, 1), 0.5, requires_grad=True)
        o = torch.optim.Adam([v], lr=0.05)
        for _ in range(4000):
            o.zero_grad()
            position(v.expand(3, 1, 1), tg, squared=sq).mean().backward()
            o.step()
        got = v.item()
        check(f"{'L2' if sq else 'L1'} optimum over {{0,0,1}} -> {want:.3f}",
              abs(got - want) < 0.03, f"got {got:.4f}")

    # ---- predictors, cells, decomposition ----


    n = make_predictor("cnn_small", 8, 8, 0.0)
    check("cnn_small receptive field is small", n.rf == 7, f"rf={n.rf}")
    n2 = make_predictor("cnn_large", 8, 8, 0.0)
    check("cnn_large receptive field is much larger", n2.rf > 25, f"rf={n2.rf}")
    f = make_predictor("free", 8, 8, 2.0)
    check("free predictor has one parameter per pixel",
          f.p.numel() == 64, f"{f.p.numel()}")
    check("free predictor ignores its input",
          torch.allclose(f(torch.zeros(3, 8, 8)), f(torch.ones(3, 8, 8))))

    # The falsifier, at the parameters `run()` ACTUALLY uses and over several
    # seeds. Testing it at reduced settings with one hard-coded seed is how a
    # tie-degeneracy that fired on ~14% of seeds went unnoticed: the check
    # passed by luck on the seed it happened to use.
    fps, gaps = [], []
    for sd in range(4):
        r = fit_cell("fronto_d1.0", "position", "free", "clean", sd)
        fps.append(r["FP"]); gaps.append(r["optimizer_gap"])
    check("falsifier cell evaluates pixels", r["n_eval"] > 0, f"n={r['n_eval']}")
    check("position-only + free + clean gives FP ~ 0 on EVERY seed",
          max(fps) < 0.05, f"FP per seed={[round(x,4) for x in fps]}")
    check("fit agrees with the closed-form weighted median",
          max(gaps) < 0.15, f"max gap={max(gaps):.3f} m")

    # an even n_real makes the L1 argmin the whole void; it must be refused
    try:
        fit_cell("fronto_d1.0", "position", "free", "clean", 0, n_real=24)
        check("even n_real is refused", False, "it was accepted")
    except SystemExit:
        check("even n_real is refused", True)

    # the L2 contrast must behave differently -- otherwise the rig cannot detect
    # the very thing claim (a) is about
    r2 = fit_cell("fronto_d1.0", "position_squared", "free", "clean", 0)
    check("squared position drives FP up (rig is sensitive)", r2["FP"] > max(fps),
          f"L2 FP={r2['FP']:.4f} vs L1 max={max(fps):.4f}")

    # a falsifier set with no finite value must read as NOT EVALUATED
    from unittest.mock import patch
    d0 = decompose([dict(objective="a", predictor="p", target="clean", FP=0.1),
                    dict(objective="b", predictor="p", target="clean", FP=0.2)])
    check("decomposition survives a missing _residual key",
          "_residual" in d0)
    b, why, _ = decide_branch({})
    check("empty decomposition gives UNDETERMINED, not a branch", b == "UNDETERMINED")

    d = decompose([dict(objective="a", predictor="p", target="clean", FP=0.0),
                   dict(objective="a", predictor="p", target="mixed", FP=0.0),
                   dict(objective="b", predictor="p", target="clean", FP=1.0),
                   dict(objective="b", predictor="p", target="mixed", FP=1.0)])
    check("decomposition attributes a pure objective effect to objective",
          d["objective"]["eta2"] > 0.99 and d["target"]["eta2"] < 0.01,
          f"obj={d['objective']['eta2']:.3f} tgt={d['target']['eta2']:.3f}")

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _main():
    ap = argparse.ArgumentParser(description="Phase 1B factorial")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--preregister", action="store_true",
                    help="record the §3.4 predictions; required before --run")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--objectives", default="")
    ap.add_argument("--predictors", default="")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    sc = [x for x in a.scenes.split(",") if x] or list(SCENES)
    ob = [x for x in a.objectives.split(",") if x] or list(LEVELS)
    pr = [x for x in a.predictors.split(",") if x] or list(PREDICTORS)
    if a.preregister:
        preregister(a.out, a.seeds, sc, ob, pr, a.force)
        return 0
    if a.run:
        run(a.out, a.seeds,
            [s for s in a.scenes.split(",") if s] or None,
            [o for o in a.objectives.split(",") if o] or None,
            [p for p in a.predictors.split(",") if p] or None,
            steps=a.steps, n_real=a.n_real)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
