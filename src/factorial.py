r"""
Phase 1B — the controlled synthetic experiment (notes/diagnosis.md §3). Version 2.

One file, because this is one experiment: the scene construction, the objective
variants, the pilot, the pre-registration and the analysis are only ever used
together.

The only controlled experiment in Phase 1, and the only place a causal claim may
originate. Anything determining a learned predictor sits in exactly one of
TARGET, OBJECTIVE, or PREDICTOR-AND-OPTIMIZATION — a partition by construction,
with the third bucket defined as the residual. **[ASSUMED]** only that the
partition is USEFUL, i.e. that effects localize rather than smearing across all
three.

    python src/factorial.py --selftest
    python src/factorial.py --pilot       --out results/phase1b_v2   # CNN learning rates, FP-blind
    python src/factorial.py --preregister --out results/phase1b_v2   # write-once; READ IT before running
    python src/factorial.py --run         --out results/phase1b_v2   # resumable

--- WHY A VERSION 2 --------------------------------------------------------

The first run (results/phase1/, 2026-08-31) cannot support an attribution. Four
defects, each fixed here and each covered by a self-test:

  1. TWO PIXELS PER CELL. Every row of the v1 scene was identical, so the
     boundary set was two columns x 48 copies of the same two pixels, and FP
     could only take the values {0, 0.5, 1}. The "near-categorical rates" and
     "bistable conditions" reported from v1 were largely one pixel flipping.
     Fix: the boundary is TILTED (`TILT` px per row), so each row's boundary
     pixel sees a different sub-pixel coverage and FP is a fraction over ~100
     genuinely distinct cases.
  2. COLLAPSE SCORED AS VOID-LANDING. A constant prediction, scale-aligned on an
     interior that was exactly half near and half far, landed at the void's
     centre and scored FP = 1.0. Fix: the boundary sits at 0.4 of the width, so
     an aligned constant lands on the far surface and an UNDETECTED collapse
     biases toward a null rather than toward an effect; detected collapses are
     excluded as before.
  3. THE CNNs NEVER FITTED. 97% of cnn_large and 75% of cnn_small cells
     collapsed at the free predictor's learning rate (0.05). Fix: CNN learning
     rates come from a pilot on disjoint seeds, chosen by interior fit quality
     alone — the pilot never computes FP (`--pilot`).
  4. A VERDICT FROM VARIANCE SHARES. Shares depend on which levels the design
     includes (the squared-norm control alone inflated the objective share).
     Fix: the decision rests on pre-registered PAIRED CONTRASTS with effect
     sizes and simultaneous confidence intervals; the decomposition is kept as
     description only.

--- 1. SCENE CONSTRUCTION (§3.2) -------------------------------------------

A controlled occlusion boundary at a KNOWN location. Rendered at 8x, then
downsampled two ways: nearest-neighbour gives a clean target (every pixel on
exactly one surface), area-average a mixed one (boundary pixels are blends in the
void). Sub-pixel jitter across realizations gives each boundary pixel a genuine
target DISTRIBUTION over the near and far depth, which is what claim (a) is about.
**[ASSUMED]** that this is a fair stand-in for the ambiguity a real model faces.

--- 2. OBJECTIVE VARIANTS (§3.3) -------------------------------------------

Each term is a reduction of the published loss to scalar depth. **[ASSUMED]**
that the reduction preserves the behaviour under test. CONFIDENCE:
`C·ℓ − α log C` is unbounded below as written (§1.1(b)). The published heads put
a smooth floor under C (`1 + exp`); the unfloored `softplus` form is run too.

--- 3. PREDICTORS ----------------------------------------------------------

  free       one parameter per pixel, no input. Its optimum IS the pointwise
             minimizer, so it measures what the loss alone prefers.
  cnn_small  3 conv layers, receptive field 7, antialiased image -> depth
  cnn_large  6 dilated conv layers, receptive field 35

SCOPE, to carry into any write-up: results describe THIS construction, THESE
objective implementations and THESE small predictors. **[ASSUMED]** that they
transfer to billion-parameter models trained at scale. Phase 1C is the partial
and confounded check on it.
"""
from __future__ import annotations

import os
# cuBLAS reads this when the CUDA context is created. It must be set before torch
# touches the GPU; v1 set it inside fit_cell, which was only correct by luck of
# call order.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import json
import time
import hashlib
import argparse
import itertools
import collections
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fpmetrics as FM

SUPER = 8                      # hi-res supersampling factor before downsampling
TILT = 0.137                   # boundary shift in output px per row (see defect 1)


@dataclass
class Scene:
    """One occlusion-boundary configuration. Depth increases away from camera."""
    h: int = 48
    w: int = 64
    near: float = 2.0
    gap: float = 1.0           # Delta between the near and far surface
    slant_near: float = 0.0    # metres of depth change across the full width
    slant_far: float = 0.0
    k: int = 2                 # 2 or 3 surfaces
    mid_frac: float = 0.10     # width of the middle surface when k == 3
    boundary: float = 0.4      # boundary position as a fraction of width (defect 2)
    px_offset: float = 0.5     # extra sub-pixel shift, in OUTPUT pixels
    tilt: float = TILT         # output px of boundary shift per row (defect 1)

    @property
    def far(self):
        return self.near + self.gap

    def depths(self):
        """The distinct surface depths, near to far."""
        if self.k == 2:
            return [self.near, self.far]
        return [self.near, self.near + 0.5 * self.gap, self.far]


def _edge(sc: Scene, jitter, super_):
    """Boundary x-position in output pixels for every hi-res row, shape (H, 1).

    TILTED, not vertical. With a vertical boundary every row is a copy of every
    other, so a 48-row scene carries exactly one boundary configuration and FP is
    quantized to {0, 0.5, 1}. With `tilt` irrational-ish in pixel units, the
    sub-pixel phase of the boundary differs row to row and the boundary set
    covers the whole range of near/far coverage weights.
    """
    y = (np.arange(sc.h * super_) + 0.5) / super_
    b = sc.boundary * sc.w + sc.px_offset + jitter + sc.tilt * (y - sc.h / 2)
    mid_w = sc.mid_frac * sc.w if sc.k >= 3 else 0.0
    if b.min() < 1 or b.max() + mid_w > sc.w - 1:
        raise SystemExit(
            f"boundary leaves the frame (x in [{b.min():.1f}, {b.max() + mid_w:.1f}] "
            f"for width {sc.w}). A surface would vanish from some rows while depths() "
            f"still reports it.")
    return b[:, None]


def render(sc: Scene, jitter=0.0, super_=SUPER):
    """Hi-res depth map. `jitter` shifts the boundary by that many OUTPUT pixels.

    Surfaces are slanted along x by `slant_*` metres across the full width, so a
    pixel's own depth can differ from its surface's median.
    """
    x = ((np.arange(sc.w * super_) + 0.5) / super_)[None, :]
    # The half-pixel offset keeps the boundary generically off-grid: on an exact
    # pixel edge area-averaging blends nothing and "mixed target" is a no-op.
    b = _edge(sc, jitter, super_)
    t = np.clip(x / sc.w, 0, 1)
    near = sc.near + sc.slant_near * t
    far = sc.far + sc.slant_far * t
    d = np.where(x < b, near, far)
    if sc.k >= 3:
        mid = sc.near + 0.5 * sc.gap + 0.5 * (sc.slant_near + sc.slant_far) * t
        d = np.where((x >= b) & (x < b + sc.mid_frac * sc.w), mid, d)
    return np.ascontiguousarray(d, dtype=np.float64)


def down_nearest(hi, super_=SUPER):
    """Clean target: every output pixel takes one hi-res sample, so it belongs to
    exactly one surface. No value that was not in the scene can appear."""
    off = super_ // 2
    return hi[off::super_, off::super_].copy()


def down_area(hi, super_=SUPER):
    """Mixed target: area average. A boundary pixel becomes a blend of the two
    surfaces -- a value belonging to neither, sitting in the void."""
    H, W = hi.shape
    h, w = H // super_, W // super_
    return hi[:h * super_, :w * super_].reshape(h, super_, w, super_).mean((1, 3))


def render_image(sc: Scene, jitter=0.0, super_=SUPER, albedo=(0.25, 0.75)):
    """Intensity image for the same scene, area-averaged to output resolution.

    A camera integrates over a pixel footprint, so the image edge is ANTIALIASED
    even when the depth step is sharp. Handing the CNN a sharp edge would remove
    the difficulty the "predictor smoothness" hypothesis (§1.3) is about.
    """
    x = ((np.arange(sc.w * super_) + 0.5) / super_)[None, :]
    b = _edge(sc, jitter, super_)
    row = np.where(x < b, albedo[0], albedo[1])
    if sc.k >= 3:
        row = np.where((x >= b) & (x < b + sc.mid_frac * sc.w),
                       0.5 * (albedo[0] + albedo[1]), row)
    return down_area(np.ascontiguousarray(row, dtype=np.float64), super_)


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
    return tg, im, ref, ref_im


# --------------------------------------------------------------------------- #
# closed form: what the position term alone prefers
# --------------------------------------------------------------------------- #
def weighted_median(vals, w=None, axis=0):
    """Minimizer of `Σ_k w_k |x − v_k|`. For collinear atoms this IS the geometric
    median, so it is the closed-form answer to claim (a) for position/free -- a
    check on the optimizer rather than a substitute for it."""
    v = np.moveaxis(np.asarray(vals, float), axis, 0)
    n = v.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, float)
    order = np.argsort(v, axis=0)
    vs = np.take_along_axis(v, order, 0)
    ws = w[order] if w.ndim == 1 else np.take_along_axis(w, order, 0)
    cw = np.cumsum(ws, axis=0)
    idx = np.clip((cw < cw[-1] / 2.0).sum(0), 0, n - 1)
    return np.take_along_axis(vs, idx[None], 0)[0]


EPS = 1e-8


# --------------------------------------------------------------------------- #
# objective terms
# --------------------------------------------------------------------------- #
def conf_from_raw(raw, floor=True):
    """Map an unconstrained parameter to C > 0.

    `floor=True` is the PUBLISHED form, `C = 1 + exp(raw)` (DUSt3R's conf head
    with vmin=1; VGGT's `expp1`): a smooth floor at 1 with a gradient everywhere.
    `floor=False` is `softplus(raw)`, the unfloored form the §1.1(b) derivation
    assumes and which is degenerate as ℓ → 0.

    NOT a hard `clamp(min=1)`. v1 used one with C initialized at softplus(0) =
    0.69, below the floor, where clamp passes zero gradient: C stayed exactly 1,
    the loss reduced to ℓ, and "position+conf" was the position level under
    another name. Its "confidence changes nothing" result was that bug.
    """
    return 1.0 + raw.exp() if floor else F.softplus(raw) + EPS


def apply_conf(per_pixel, conf, alpha=0.2):
    """`C·ℓ − α log C`, reduced over all pixels. `conf=None` leaves ℓ untouched."""
    if conf is None:
        return per_pixel.mean()
    return (per_pixel * conf - alpha * torch.log(conf)).mean()


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
    `sqrt(x + eps)` alone would put a constant floor on every pixel."""
    return (sq + EPS).sqrt() - EPS ** 0.5


def gradient(pred, tgt):
    """VGGT's gradient term, per pixel. Couples neighbours, so the objective is
    NOT separable and the pointwise minimizer of §1.1(a) no longer describes it."""
    py, px = _grads(pred)
    ty, tx = _grads(tgt)
    return _safe_norm((py - ty).pow(2) + (px - tx).pow(2))


def normals_from_depth(d, fx=1.0, fy=1.0):
    """Surface normal per pixel from a depth map, via `n ∝ (-fx·∂z/∂x, -fy·∂z/∂y, 1)`."""
    gy, gx = _grads(d)
    n = torch.stack([-fx * gx, -fy * gy, torch.ones_like(d)], dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + EPS)


def normal(pred, tgt, fx=1.0, fy=1.0):
    """π³'s normal loss, per pixel: 1 − cos between predicted and target normals."""
    return 1.0 - (normals_from_depth(pred, fx, fy)
                  * normals_from_depth(tgt, fx, fy)).sum(-1)


def total(pred, tgt, conf=None, *, w_grad=0.0, w_normal=0.0, alpha=0.2,
          squared=False, fx=1.0, fy=1.0):
    """Full objective for one cell. Confidence weights the POSITION term only, as
    in the published losses. Weight 0 removes a term exactly, so every level runs
    the same code path."""
    loss = apply_conf(position(pred, tgt, squared), conf, alpha)
    if w_grad:
        loss = loss + w_grad * gradient(pred, tgt).mean()
    if w_normal:
        loss = loss + w_normal * normal(pred, tgt, fx, fy).mean()
    return loss


LEVELS = {
    "position":             dict(w_grad=0.0, w_normal=0.0, use_conf=False),
    "position+grad":        dict(w_grad=1.0, w_normal=0.0, use_conf=False),
    "position+normal":      dict(w_grad=0.0, w_normal=1.0, use_conf=False),
    "position+both":        dict(w_grad=1.0, w_normal=1.0, use_conf=False),
    "position+conf":        dict(w_grad=0.0, w_normal=0.0, use_conf=True, floor=True),
    "position+conf_noclip": dict(w_grad=0.0, w_normal=0.0, use_conf=True, floor=False),
    "position_squared":     dict(w_grad=0.0, w_normal=0.0, use_conf=False, squared=True),
}

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Gap swept over an order of magnitude; slant and k=3 make the two-surface
# restriction and the median-stand-in question into variables.
SCENES = {
    "fronto_d0.2": Scene(gap=0.2),
    "fronto_d0.5": Scene(gap=0.5),
    "fronto_d1.0": Scene(gap=1.0),
    "fronto_d2.0": Scene(gap=2.0),
    "fronto_d3.0": Scene(gap=3.0),
    "slanted_d1.0": Scene(gap=1.0, slant_near=0.5, slant_far=0.5),
    "k3_d2.0": Scene(gap=2.0, k=3, mid_frac=0.12),
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


class EdgePad(nn.Module):
    """Replicate-pad by `d` pixels using slicing and concatenation.

    Replicate padding, because a tilted boundary meets the top and bottom of the
    frame and zero padding would manufacture a spurious edge exactly there. Built
    by hand, because the CUDA backward of nn.ReplicationPad2d (and of
    padding_mode="replicate") has no deterministic implementation, and the run's
    reproducibility gate requires bitwise-identical re-fits.
    """

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        d = self.d
        x = torch.cat([x[:, :, :1].repeat(1, 1, d, 1), x, x[:, :, -1:].repeat(1, 1, d, 1)], 2)
        return torch.cat([x[..., :1].repeat(1, 1, 1, d), x, x[..., -1:].repeat(1, 1, 1, d)], 3)


class CNN(nn.Module):
    """Image -> depth. `dilations` sets the receptive field."""

    def __init__(self, ch=32, dilations=(1, 1, 1), bias=0.0):
        super().__init__()
        layers, c_in = [], 1
        for d in dilations:
            layers += [EdgePad(d), nn.Conv2d(c_in, ch, 3, dilation=d), nn.ReLU()]
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
# design constants — every one of these is copied into 1b_prereg.md, and --run
# refuses to start if the live values no longer match the recorded ones
# --------------------------------------------------------------------------- #
STEPS = 1500
N_REAL = 25              # MUST be odd, see fit_cell
JITTER_PX = 1.0
FREE_LR = 0.05           # peak rate; decays to zero (see fit_cell)

PILOT_SEEDS = (900, 901)                             # disjoint from the main run
PILOT_SCENES = ("fronto_d0.2", "fronto_d1.0", "k3_d2.0")
PILOT_LRS = (1e-2, 3e-3, 1e-3, 3e-4)
PILOT_MAX_EXCLUDED = 0.10

MAIN_SEED_OFFSET = 1000  # fresh: v1 used 0-4, the stopped confirmation 5-14
MAIN_SEEDS = 10

DELTA = 0.05             # smallest effect worth a claim, in FP units (Phase 0's 5 pp)
ALPHA = 0.05             # family-wise over PRIMARY, Bonferroni
N_BOOT = 20000
MAX_EXCLUDED = 0.20      # per predictor; above this its contrasts are NOT EVALUATED
MIN_PAIR_FRAC = 0.50     # a contrast needs this share of its design pairs
FALSIFIER_MAX = 0.05
OPT_GAP_FRAC = 0.10      # position/free fit this far (x gap) from closed form = unconverged
COLLAPSE_STD = 0.02      # prediction spatial std below this (x gap) = collapsed
FAILED_RMSE = 0.40       # interior RMSE above this (x gap) = failed fit
UNSETTLED_FRAC = 0.05    # sensitivity analysis only, never an exclusion
N_REPRO = 6              # cells re-fitted after the run; must reproduce exactly

LR = {"free": FREE_LR, "cnn_small": None, "cnn_large": None}


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #
FIELDS = ("scene", "objective", "predictor", "target", "seed", "lr", "FP", "outside",
          "n_eval", "final_loss", "rmse_interior", "optimizer_gap", "settle_frac",
          "pred_std", "collapsed", "failed")


def fit_cell(scene, objective, predictor, target, seed, *, n_real=N_REAL, steps=STEPS,
             lr=None, jitter_px=JITTER_PX, score=True):
    """Fit one cell and score it with the SAME flying-pixel metric used on the
    real models, so Phase 1B numbers are commensurable with Phase 0.

    `score=False` never computes FP. The pilot uses it, so choosing a learning
    rate cannot be informed by the quantity under test.

    `n_real` MUST BE ODD. With an even count a boundary pixel's targets can tie
    exactly half near / half far, the L1 objective is then flat across the whole
    void, and the optimizer stays wherever it was initialized -- which turned the
    v1 falsifier into a coin flip on ~14% of seeds.
    """
    if n_real % 2 == 0:
        raise SystemExit(
            f"n_real must be ODD (got {n_real}). An even count lets a boundary "
            f"pixel's targets tie exactly, which makes the L1 argmin the whole "
            f"void interval and turns the falsifier into a coin flip.")
    lr = LR[predictor] if lr is None else lr
    if lr is None:
        raise SystemExit(f"no learning rate for {predictor}: run --pilot, then "
                         f"load_pilot(), before fitting a CNN.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Conv backward is nondeterministic on GPU by default; v1's same cell re-run
    # gave FP 0.031 / 0.469 / 0.500.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sc = SCENES[scene]
    tg, im, ref, ref_im = realizations(sc, n=n_real, jitter_px=jitter_px,
                                       mode="nearest" if target == "clean" else "area",
                                       seed=seed)
    T = torch.tensor(tg, dtype=torch.float32, device=DEV)
    I = torch.tensor(im, dtype=torch.float32, device=DEV)
    RI = torch.tensor(ref_im, dtype=torch.float32, device=DEV)[None]

    cfg = dict(LEVELS[objective])
    use_conf = cfg.pop("use_conf", False)
    floor = cfg.pop("floor", True)
    squared = cfg.pop("squared", False)

    net = make_predictor(predictor, sc.h, sc.w, init=float(ref.mean())).to(DEV)
    params = list(net.parameters())
    raw_c = None
    if use_conf:
        raw_c = torch.zeros(sc.h, sc.w, device=DEV, requires_grad=True)
        params.append(raw_c)
    opt = torch.optim.Adam(params, lr=lr)
    # Cosine decay to zero. At a constant rate Adam keeps a free parameter
    # oscillating by roughly `lr` around an L1 optimum -- 0.05 m, which is a
    # quarter of the 0.2 m scene's gap: enough to be excluded as unconverged, or
    # to be scored as a flying pixel when it is only optimizer jitter.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # `settle`: movement of the prediction over the final 10% of optimization.
    settle_from = int(steps * 0.9)
    snapshot = None
    for i in range(steps):
        if i == settle_from:
            with torch.no_grad():
                snapshot = net(RI)[0].detach().clone()
        opt.zero_grad()
        pred = net(I)
        conf = (conf_from_raw(raw_c, floor=floor).expand_as(pred)
                if raw_c is not None else None)
        loss = total(pred, T, conf, squared=squared, **cfg)
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        out_t = net(RI)[0]
        settle = (float((out_t - snapshot).abs().max()) if snapshot is not None
                  else float("nan"))
        out = out_t.double().cpu().numpy()

    # Fit quality is judged on the INTERIOR only -- the pixels away from the
    # boundary band, where no loss variant should disagree. That keeps the
    # exclusion rules (and the pilot's choice of learning rate) blind to exactly
    # the behaviour the experiment measures.
    valid = np.ones_like(ref, bool)
    interior = ~FM.dilate(FM.boundary_set(ref, valid), FM.TAU0)
    rmse_int = float(np.sqrt(np.mean((out - ref)[interior] ** 2)))
    pred_std = float(np.std(out))
    # A non-finite prediction must be EXCLUDED, not scored: every comparison
    # with NaN is False, so it would otherwise pass both checks and fp_view would
    # count its NaN pixels as "not a flying pixel", i.e. a perfect FP = 0.
    collapsed = bool(not np.isfinite(pred_std) or pred_std < COLLAPSE_STD * sc.gap)
    failed = bool(not np.isfinite(rmse_int) or rmse_int > FAILED_RMSE * sc.gap)

    fp = outside = float("nan")
    n_eval = 0
    if score:
        r = FM.fp_view(ref, valid, out, eta=FM.ETA0, tau=FM.TAU0, beta=FM.BETA0,
                       w=3, delta_min=0.05 * sc.gap)
        fp, outside, n_eval = r["FP"], r["outside_rate"], r["n_eval"]

    opt_gap = float("nan")
    if objective == "position" and predictor == "free":
        opt_gap = float(np.abs(out - weighted_median(tg, axis=0)).max())

    return dict(scene=scene, objective=objective, predictor=predictor, target=target,
                seed=int(seed), lr=float(lr), FP=fp, outside=outside, n_eval=int(n_eval),
                final_loss=float(loss.item()), rmse_interior=rmse_int,
                optimizer_gap=opt_gap, settle_frac=float(settle / sc.gap),
                pred_std=pred_std, collapsed=collapsed, failed=failed)


def _parse(row):
    """A CSV row back into typed values."""
    out = dict(row)
    out["seed"] = int(row["seed"])
    out["n_eval"] = int(row["n_eval"])
    for k in ("lr", "FP", "outside", "final_loss", "rmse_interior", "optimizer_gap",
              "settle_frac", "pred_std"):
        out[k] = float(row[k])
    for k in ("collapsed", "failed"):
        out[k] = row[k] == "True"
    return out


def _read_rows(path):
    """Rows already on disk, after repairing a crash mid-write.

    A kill during `writerow` can leave a truncated last line (a final `True` cut
    to `Tr` parses silently as False) and the next append would continue on that
    same line. So an unterminated tail is cut off -- that cell is simply re-fitted
    -- and a row missing any field is dropped.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "rb+") as f:
        data = f.read()
        if not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [_parse(r) for r in rows
            if None not in r and all(r.get(k) not in (None, "") for k in FIELDS)]


def _key(r):
    return (r["scene"], r["objective"], r["predictor"], r["target"], int(r["seed"]))


def _code_md5():
    return hashlib.md5(open(__file__, "rb").read()).hexdigest()


# --------------------------------------------------------------------------- #
# pilot: CNN learning rates, chosen without ever computing FP
# --------------------------------------------------------------------------- #
def choose_lr(rows, lrs=PILOT_LRS):
    """Per CNN: among learning rates whose exclusion rate is at most
    PILOT_MAX_EXCLUDED, the one with the lowest median interior RMSE (as a
    fraction of the gap). None if no rate qualifies.

    The criterion reads only `collapsed`, `failed` and `rmse_interior`, all of
    which are computed away from the boundary band.
    """
    choice, table = {}, []
    for pr in ("cnn_small", "cnn_large"):
        best = None
        for lr in lrs:
            rs = [r for r in rows if r["predictor"] == pr and np.isclose(r["lr"], lr)]
            if not rs:
                continue
            excl = float(np.mean([r["collapsed"] or r["failed"]
                                  or not np.isfinite(r["rmse_interior"]) for r in rs]))
            kept = [r["rmse_interior"] / SCENES[r["scene"]].gap for r in rs
                    if not (r["collapsed"] or r["failed"]) and np.isfinite(r["rmse_interior"])]
            med = float(np.median(kept)) if kept else float("inf")
            settle = float(np.median([r["settle_frac"] for r in rs]))
            ok = excl <= PILOT_MAX_EXCLUDED
            table.append(dict(predictor=pr, lr=lr, n=len(rs), excluded=excl,
                              median_rmse_frac=med, median_settle_frac=settle,
                              eligible=ok))
            if ok and (best is None or med < best[1]):
                best = (lr, med)
        choice[pr] = best[0] if best else None
    return choice, table


def pilot(out, force=False):
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "1b_prereg.md")):
        raise SystemExit("A pre-registration already exists here. Re-running the pilot "
                         "after committing to a design is tuning after the fact.")
    jp = os.path.join(out, "1b_pilot.json")
    if os.path.exists(jp) and not force:
        raise SystemExit(f"{jp} exists; pass --force to redo the pilot.")
    cells = list(itertools.product(PILOT_SCENES, LEVELS, ("cnn_small", "cnn_large"),
                                   TARGETS, PILOT_SEEDS, PILOT_LRS))
    print(f"pilot: {len(cells)} CNN fits on {DEV}, FP is never computed", flush=True)
    rows, t0 = [], time.time()
    for i, (sc, ob, pr, tg, sd, lr) in enumerate(cells, 1):
        rows.append(fit_cell(sc, ob, pr, tg, sd, lr=lr, score=False))
        if i % 25 == 0 or i == len(cells):
            print(f"  {i}/{len(cells)}  ({time.time()-t0:.0f}s)", flush=True)
    keep = [f for f in FIELDS if f not in ("FP", "outside", "n_eval")]
    with open(os.path.join(out, "1b_pilot.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    choice, table = choose_lr(rows)
    passed = all(v is not None for v in choice.values())
    json.dump(dict(passed=passed, lr=choice, table=table, date=time.strftime("%F %T"),
                   code_md5=_code_md5(), seeds=list(PILOT_SEEDS),
                   scenes=list(PILOT_SCENES), lrs=list(PILOT_LRS),
                   criterion=f"exclusion rate <= {PILOT_MAX_EXCLUDED}, then lowest "
                             f"median interior RMSE / gap; FP never computed"),
              open(jp, "w"), indent=1)
    md = ["# Phase 1B v2 — pilot (CNN learning rates)", "",
          f"{time.strftime('%F %T')} · {len(rows)} fits · seeds {list(PILOT_SEEDS)} "
          f"(disjoint from the main run) · FP was never computed.", "",
          f"Criterion: exclusion (collapsed or failed) at most {PILOT_MAX_EXCLUDED:.0%}, "
          f"then the lowest median interior RMSE as a fraction of the gap. Interior "
          f"means away from the boundary band, so nothing here sees boundary behaviour.", "",
          "| predictor | lr | fits | excluded | median interior RMSE / gap | median settle / gap | eligible |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {t['predictor']} | {t['lr']:g} | {t['n']} | {t['excluded']:.1%} | "
           f"{t['median_rmse_frac']:.4f} | {t['median_settle_frac']:.4f} | {t['eligible']} |"
           for t in table]
    md += ["", f"**{'PASSED' if passed else 'FAILED'}** — chosen: "
           + ", ".join(f"{k}={v}" for k, v in choice.items())]
    if not passed:
        md += ["", "No learning rate in the grid fits a CNN reliably. Do not pre-register; "
               "the predictor factor would again measure optimization failure."]
    open(os.path.join(out, "1b_pilot.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[-3:]))
    return 0 if passed else 5


def load_pilot(path):
    d = json.load(open(path))
    if not d.get("passed"):
        raise SystemExit(f"{path}: the pilot did not pass; no CNN learning rate is usable.")
    LR.update({k: float(v) for k, v in d["lr"].items()})
    return dict(LR)


# --------------------------------------------------------------------------- #
# pre-registered contrasts
# --------------------------------------------------------------------------- #
# (id, bucket, fixed factors, varied factor, level A, level B); effect = FP_A - FP_B,
# paired on (scene, seed) -- the same seed gives the same realizations.
def _obj(term, pr, tgt="clean"):
    return (f"O:{term}/{pr}/{tgt}", "objective", {"predictor": pr, "target": tgt},
            "objective", f"position+{term}", "position")


PRIMARY = tuple(
    [_obj(t, p) for t in ("grad", "normal", "conf", "conf_noclip") for p in PREDICTORS]
    + [(f"T:mixed/{p}", "target", {"predictor": p, "objective": "position"},
        "target", "mixed", "clean") for p in PREDICTORS]
    + [(f"P:{p}", "predictor", {"objective": "position", "target": "clean"},
        "predictor", p, "free") for p in ("cnn_small", "cnn_large")])
CONTROL = ("C:squared/free", "control", {"predictor": "free", "target": "clean"},
           "objective", "position_squared", "position")
SECONDARY = tuple(
    [_obj("both", p) for p in PREDICTORS]
    + [_obj(t, p, "mixed") for t in ("grad", "normal", "conf", "conf_noclip")
       for p in PREDICTORS])

TREE = {"OBJECTIVE": "STRONG PAPER — corrects a stated belief in the field",
        "TARGET": "SOLID PAPER — implicates the supervision, not one loss term",
        "MIXED": "SOLID PAPER — attribution with effect sizes",
        "RESIDUAL": "THIN — pivot to benchmark or workshop",
        "NULL": "no leaf: this construction does not reproduce the artifact",
        "NO VERDICT": "no branch selected", "FALSIFIED": "rewrite §1.1 first"}


def exclude(rows):
    """Mark `_excluded` on every row: collapsed, failed, or unscored.

    NOT the optimizer gap. On clean targets the closed-form optimum lies on a
    surface, so any position/free pixel scored as flying is at least β·Δ from it
    -- more than OPT_GAP_FRAC·gap. Excluding on the gap would therefore drop
    exactly the cells with FP > 0: the falsifier could never fail, and every
    contrast with position/free as a baseline would lose its high-FP pairs. The
    gap is used only to diagnose a falsifier breach (see `decide`).
    """
    for r in rows:
        r["_excluded"] = bool(r["collapsed"] or r["failed"] or not np.isfinite(r["FP"]))
    return rows


def strat_boot(groups, a, n_boot=N_BOOT, seed=0):
    """Mean over all pairs, with a percentile CI at two-sided level `a`, resampling
    pairs WITHIN each scene so the scene mix is held fixed."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    n = sum(len(g) for g in groups)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    tot = np.zeros(n_boot)
    for g in groups:
        tot += g[rng.integers(0, len(g), (n_boot, len(g)))].sum(1)
    lo, hi = np.quantile(tot / n, [a / 2, 1 - a / 2])
    return float(np.concatenate(groups).mean()), float(lo), float(hi)


def classify(mean, lo, hi, delta=DELTA):
    if not np.isfinite(mean):
        return "NOT EVALUATED"
    if lo > 0 and mean >= delta:
        return "PRESENT+"
    if hi < 0 and mean <= -delta:
        return "PRESENT-"
    if lo >= -delta and hi <= delta:
        return "NEGLIGIBLE"
    return "INCONCLUSIVE"


def contrast(rows, spec, a, scenes, seeds, not_evaluated=(), drop_unsettled=False):
    cid, bucket, fixed, factor, la, lb = spec
    involved = {fixed["predictor"]} if "predictor" in fixed else {la, lb}
    res = dict(id=cid, bucket=bucket, A=la, B=lb, n_pairs=0,
               n_design=len(scenes) * len(seeds), mean=float("nan"),
               lo=float("nan"), hi=float("nan"), FP_A=float("nan"), FP_B=float("nan"),
               status="NOT EVALUATED", per_scene={})
    if involved & set(not_evaluated):
        res["why"] = "predictor exclusion rate above MAX_EXCLUDED"
        return res
    idx = {}
    for r in rows:
        if r[factor] in (la, lb) and all(r[k] == v for k, v in fixed.items()):
            idx[(r["scene"], r["seed"], r[factor])] = r

    def usable(r):
        return (r is not None and not r["_excluded"]
                and not (drop_unsettled and r["settle_frac"] > UNSETTLED_FRAC))

    by_scene, fa, fb = collections.defaultdict(list), [], []
    for s in scenes:
        for sd in seeds:
            ra, rb = idx.get((s, sd, la)), idx.get((s, sd, lb))
            if usable(ra) and usable(rb):
                by_scene[s].append(ra["FP"] - rb["FP"])
                fa.append(ra["FP"]); fb.append(rb["FP"])
    res["n_pairs"] = len(fa)
    if res["n_pairs"] < MIN_PAIR_FRAC * res["n_design"]:
        res["why"] = f"only {len(fa)} of {res['n_design']} pairs usable"
        return res
    m, lo, hi = strat_boot(list(by_scene.values()), a)
    res.update(mean=m, lo=lo, hi=hi, FP_A=float(np.mean(fa)), FP_B=float(np.mean(fb)),
               status=classify(m, lo, hi),
               per_scene={s: float(np.mean(v)) for s, v in by_scene.items()})
    return res


def analyze(rows, scenes, seeds):
    """Gates, then the pre-registered contrasts, then the branch."""
    exclude(rows)
    rate = {p: float(np.mean([r["_excluded"] for r in rows if r["predictor"] == p]))
            for p in PREDICTORS if any(r["predictor"] == p for r in rows)}
    not_eval = [p for p, v in rate.items() if v > MAX_EXCLUDED]
    a_fam = ALPHA / len(PRIMARY)

    fal_rows = [r for r in rows if r["objective"] == "position" and r["predictor"] == "free"
                and r["target"] == "clean" and not r["_excluded"]]
    fal = [r["FP"] for r in fal_rows]
    fal_mean = float(np.mean(fal)) if fal else float("nan")
    falsified = None if not fal else bool(fal_mean > FALSIFIER_MAX)
    unconverged = sum(1 for r in fal_rows
                      if r["optimizer_gap"] > OPT_GAP_FRAC * SCENES[r["scene"]].gap)

    control = contrast(rows, CONTROL, ALPHA, scenes, seeds, not_eval)
    primary = [contrast(rows, c, a_fam, scenes, seeds, not_eval) for c in PRIMARY]
    sensitivity = [contrast(rows, c, a_fam, scenes, seeds, not_eval, drop_unsettled=True)
                   for c in PRIMARY]
    secondary = [contrast(rows, c, ALPHA, scenes, seeds, not_eval) for c in SECONDARY]
    return dict(exclusion_rate=rate, not_evaluated=not_eval, falsifier_mean=fal_mean,
                falsifier_n=len(fal), falsified=falsified, unconverged=unconverged,
                control=control,
                primary=primary, sensitivity=sensitivity, secondary=secondary,
                alpha_family=a_fam)


def decide(res, repro_ok=True):
    """The pre-registered rule. Returns (branch, clean_attribution, reasons)."""
    why = []
    if res["falsified"] is None:
        return "NO VERDICT", "—", ["falsifier not evaluated: no usable position/free/clean cell"]
    if res["falsified"]:
        head = f"position/free/clean FP = {res['falsifier_mean']:.4f} > {FALSIFIER_MAX}"
        if res.get("unconverged", 0):
            # On clean targets the closed form is on a surface by construction, so
            # a breach with fits far from it is the optimizer, not claim (a).
            return "NO VERDICT", "—", [
                f"{head}, but {res['unconverged']} of {res['falsifier_n']} fits are "
                f"further than {OPT_GAP_FRAC}·gap from the closed-form optimum: the "
                f"optimizer did not reach the pointwise minimizer, so the apparatus "
                f"failed and claim (a) was not tested"]
        return "FALSIFIED", "—", [
            f"{head} with fits AT the closed-form optimum: the metric scores a "
            f"surface-valued minimizer as flying, or claim (a) fails here"]
    if res["control"]["status"] != "PRESENT+":
        why.append(f"positive control {CONTROL[0]} is {res['control']['status']}: the "
                   f"apparatus did not detect void-landing where theory guarantees it")
    if not repro_ok:
        why.append("re-fitted cells did not reproduce exactly")
    if why:
        return "NO VERDICT", "—", why

    by = collections.defaultdict(list)
    for c in res["primary"]:
        by[c["bucket"]].append(c["status"])
    # A bucket is RESOLVED when every contrast in it has a definite reading.
    # PRESENT- is definite: a measured reduction says the term does not cause
    # void-landing, which is a result, not a failure to measure.
    definite = {"PRESENT+", "PRESENT-", "NEGLIGIBLE"}
    obj, prd = "PRESENT+" in by["objective"], "PRESENT+" in by["predictor"]
    obj_res = all(s in definite for s in by["objective"])
    prd_res = all(s in definite for s in by["predictor"])
    tgt = "PRESENT+" in by["target"]
    for b in ("objective", "predictor", "target"):
        why.append(f"{b}: " + ", ".join(f"{s}×{by[b].count(s)}" for s in sorted(set(by[b]))))
    neg = [c["id"] for c in res["primary"] if c["status"] == "PRESENT-"]
    if neg:
        why.append("measured reductions (not causes): " + ", ".join(neg))

    # The branch is decided on CLEAN targets, where only the loss and the
    # predictor can act. Target contrasts are PRESENT+ nearly by construction (a
    # predictor fitted to blended targets reproduces the blend), so letting them
    # promote a clean result to MIXED would make OBJECTIVE and RESIDUAL
    # unreachable. They decide only when nothing acts on clean targets.
    if obj and prd:
        return "MIXED", "OBJECTIVE+PREDICTOR", why
    if obj and prd_res:
        return "OBJECTIVE", "OBJECTIVE", why
    if prd and obj_res:
        return "RESIDUAL", "PREDICTOR", why
    if obj or prd:
        present, other = ("objective", "predictor") if obj else ("predictor", "objective")
        return "NO VERDICT", f"{present.upper()} + UNRESOLVED {other}", why + [
            f"{present} is PRESENT+ but some {other} contrast is inconclusive or not "
            f"evaluated, so the branch could be {'OBJECTIVE' if obj else 'RESIDUAL'} or "
            f"MIXED; the {present} effect itself stands"]
    if obj_res and prd_res:
        return ("TARGET" if tgt else "NULL"), "NONE", why
    return "NO VERDICT", "UNRESOLVED", why + [
        "nothing is PRESENT+ on clean targets and some contrast is inconclusive or not "
        "evaluated: underpowered or not measured"]


def terms_for_1c(res):
    """§4: 1C runs only on a non-trivial objective effect, and only for the terms
    that showed one."""
    out = set()
    for c in res["primary"]:
        if c["bucket"] == "objective" and c["status"] == "PRESENT+":
            if c["A"] == "position+grad":
                out.add("VGGT gradient term (vggt_grad_on/off)")
            if c["A"] == "position+normal":
                out.add("π³ normal loss (pi3_normal_1.0/0.5/0.0)")
    return sorted(out)


# --------------------------------------------------------------------------- #
# descriptive variance decomposition (NOT used by the decision)
# --------------------------------------------------------------------------- #
def anova(rows, factors, y="FP"):
    """Sums of squares for every factor subset, with lower-order effects removed,
    plus within-cell replicate noise. Group sizes are counted from the rows, so
    exclusions (which unbalance the design) are handled -- but an unbalanced
    design is not orthogonal and the terms need not sum to the total. Description
    only."""
    keep = [r for r in rows if np.isfinite(r[y])]
    if len(keep) < 2:
        return {}
    grand = float(np.mean([r[y] for r in keep]))
    total = float(sum((r[y] - grand) ** 2 for r in keep))
    cells = collections.defaultdict(list)
    for r in keep:
        cells[tuple(r[f] for f in factors)].append(r[y])
    within = float(sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in cells.values()))
    out, lower = {}, {}
    for size in range(1, len(factors) + 1):
        for sub in itertools.combinations(factors, size):
            g = collections.defaultdict(list)
            for r in keep:
                g[tuple(r[f] for f in sub)].append(r[y])
            adj = {}
            for k, v in g.items():
                a = np.mean(v) - grand
                for s2 in range(1, size):
                    for sub2 in itertools.combinations(sub, s2):
                        a -= lower[sub2].get(tuple(k[sub.index(f)] for f in sub2), 0.0)
                adj[k] = a
            lower[sub] = adj
            out["*".join(sub)] = float(sum(len(g[k]) * a * a for k, a in adj.items()))
    out["within"], out["total"] = within, total
    return out


def blocked_shares(rows, block=("scene",)):
    """Shares of systematic variance over objective/predictor/target, with scene
    (a stimulus sweep we chose, not a mechanism) removed from both sides and
    interactions split equally across the buckets they involve."""
    ss = anova(rows, ("scene", "objective", "predictor", "target"))
    if not ss:
        return {}
    tot, within = ss.pop("total"), ss.pop("within")
    blocked = sum(v for k, v in ss.items() if set(k.split("*")) & set(block))
    denom = tot - within - blocked
    b = collections.defaultdict(float)
    for k, v in ss.items():
        parts = k.split("*")
        if not set(parts) & set(block):
            for f in parts:
                b[f] += v / len(parts)
    return dict(shares={k: v / denom if denom > 0 else float("nan") for k, v in b.items()},
                noise=within / tot if tot else float("nan"),
                blocked=blocked / tot if tot else float("nan"))


# --------------------------------------------------------------------------- #
# pre-registration
# --------------------------------------------------------------------------- #
PREREG_TAG = "PREREG_1B_V2"


def design(lr):
    """Everything the run and the decision depend on. Written into the
    pre-registration and compared, value for value, before --run starts."""
    return json.loads(json.dumps(dict(
        tag=PREREG_TAG,
        # The constants below do not capture everything the run depends on (CNN
        # widths, the confidence form, the metric's code, the rule's code). The
        # hashes do: after registering, ANY edit to these files blocks --run. A
        # bug found after registration means a new registration in a new
        # directory, with the reason written down -- not a silent patch.
        code_md5={"factorial.py": _code_md5(),
                  "fpmetrics.py": hashlib.md5(open(FM.__file__, "rb").read()).hexdigest()},
        scenes={k: asdict(v) for k, v in SCENES.items()},
        objectives=LEVELS, predictors=list(PREDICTORS), targets=list(TARGETS),
        seeds=list(range(MAIN_SEED_OFFSET, MAIN_SEED_OFFSET + MAIN_SEEDS)),
        steps=STEPS, schedule="adam+cosine-to-zero", n_real=N_REAL,
        jitter_px=JITTER_PX, super=SUPER,
        lr={k: lr[k] for k in PREDICTORS},
        metric=dict(eta=FM.ETA0, tau=FM.TAU0, beta=FM.BETA0, w=3, delta_min="0.05*gap"),
        delta=DELTA, alpha=ALPHA, n_boot=N_BOOT, max_excluded=MAX_EXCLUDED,
        min_pair_frac=MIN_PAIR_FRAC, falsifier_max=FALSIFIER_MAX,
        opt_gap_frac=OPT_GAP_FRAC, collapse_std=COLLAPSE_STD, failed_rmse=FAILED_RMSE,
        unsettled_frac=UNSETTLED_FRAC, n_repro=N_REPRO,
        primary=[list(c) for c in PRIMARY], control=list(CONTROL),
        secondary=[c[0] for c in SECONDARY])))


def _spec_row(c):
    cid, bucket, fixed, factor, la, lb = c
    fx = ", ".join(f"{k}={v}" for k, v in fixed.items())
    return f"| `{cid}` | {bucket} | `{la}` − `{lb}` | {fx} |"


def preregister(out, force=False):
    p = os.path.join(out, "1b_prereg.md")
    if os.path.exists(p) and not force:
        raise SystemExit(f"{p} already exists. Pre-registration is write-once.")
    if os.path.exists(os.path.join(out, "1b_factorial.csv")):
        raise SystemExit("Results already exist in this directory; a pre-registration "
                         "written after data is not one.")
    jp = os.path.join(out, "1b_pilot.json")
    if not os.path.exists(jp):
        raise SystemExit(f"No pilot at {jp}. Run --pilot first: the CNN learning rates "
                         f"are part of the registered design.")
    lr = load_pilot(jp)
    d = design(lr)
    m = len(PRIMARY)
    n_cells = len(SCENES) * len(LEVELS) * len(PREDICTORS) * len(TARGETS) * MAIN_SEEDS
    md = f"""# Phase 1B v2 — pre-registration

<!-- {PREREG_TAG} {json.dumps(d, sort_keys=True)} -->

Recorded {time.strftime('%F %T')}, BEFORE the run. `--run` parses the marker above
and refuses to start if any design constant differs from it, or if a single byte of
`src/factorial.py` or `src/fpmetrics.py` has changed. Pilot: `1b_pilot.md` (FP never
computed). Commit this file together with the code it names.

## Why this is a second run

The first run (`results/phase1/`) is not used for any claim. Its cells scored two
distinct boundary pixels each, a collapsed CNN scored as maximal void-landing,
most CNN fits never left initialization, and its verdict came from variance shares
whose size depended on which levels were included. Each is fixed in the code
(see the `factorial.py` docstring) and covered by a self-test.

## Design

- scenes ({len(SCENES)}): {", ".join(SCENES)} — boundary at 0.4 of the width, tilted
  {TILT} px per row, so a row-invariant scene cannot quantize FP.
- objectives ({len(LEVELS)}): {", ".join(LEVELS)}
- predictors: free (lr {lr['free']}), cnn_small (lr {lr['cnn_small']}), cnn_large (lr {lr['cnn_large']})
- targets: clean (nearest-neighbour), mixed (area-average)
- seeds: {MAIN_SEED_OFFSET}–{MAIN_SEED_OFFSET + MAIN_SEEDS - 1} (fresh; v1 used 0–4, the stopped
  confirmation run 5–14, the pilot {PILOT_SEEDS[0]}–{PILOT_SEEDS[-1]})
- {n_cells} cells, {STEPS} Adam steps with cosine decay to zero, {N_REAL} jittered
  realizations each
- metric: η={FM.ETA0}, τ={FM.TAU0}, β={FM.BETA0}, identical to Phase 0

## Exclusions (fixed now, all blind to boundary behaviour)

A cell is excluded from every contrast if its prediction is **collapsed** (spatial std
< {COLLAPSE_STD}·gap, or non-finite) or **failed** (interior RMSE > {FAILED_RMSE}·gap, or
non-finite). Both are computed away from the boundary band. A predictor with more than {MAX_EXCLUDED:.0%} of its cells excluded has all of
its contrasts marked NOT EVALUATED. A contrast needs at least {MIN_PAIR_FRAC:.0%} of its
{len(SCENES) * MAIN_SEEDS} (scene, seed) pairs.

## Gates (checked before any branch)

1. **Falsifier.** position + free + clean, over every fitted cell (the optimizer gap
   is NOT an exclusion — on clean targets it would remove exactly the cells with
   FP > 0): mean FP ≤ {FALSIFIER_MAX}. On a breach, if any of those fits is further
   than {OPT_GAP_FRAC}·gap from the closed-form weighted median the optimizer failed
   and there is NO VERDICT; if all are at the closed form, the run reports
   FALSIFIED and §1.1 is rewritten before anything else. Note what this can test:
   on clean targets the closed form is on a surface by construction, so the gate
   checks that the fit reaches it and the metric scores it correctly. Claim (a)
   itself is checked in closed form by the self-test.
2. **Positive control.** `{CONTROL[0]}` (squared − unsquared norm) must be PRESENT+.
   Otherwise the apparatus cannot detect void-landing and there is NO VERDICT.
3. **Reproducibility.** {N_REPRO} cells (two per predictor) are re-fitted after the run
   and must match their recorded FP, final loss and interior RMSE exactly.

## Primary contrasts (family of {m}, Bonferroni)

Effect = mean over (scene, seed) pairs of FP_A − FP_B. CI: percentile bootstrap,
{N_BOOT} resamples within scene, two-sided level α/m = {ALPHA}/{m}.

| id | bucket | effect | held fixed |
|---|---|---|---|
""" + "\n".join(_spec_row(c) for c in PRIMARY) + f"""

Each contrast is classified with δ = {DELTA} (Phase 0's 5-point margin):

| status | condition |
|---|---|
| PRESENT+ | CI lower bound > 0 **and** effect ≥ δ |
| PRESENT− | CI upper bound < 0 **and** effect ≤ −δ |
| NEGLIGIBLE | CI inside [−δ, δ] |
| INCONCLUSIVE | anything else |

## Decision rule

The target contrasts are expected to be PRESENT+ nearly by construction: a free
predictor fitted to blended targets reproduces the blend, which sits in the void.
They size the target effect; they do not discover it. So attribution is decided
first on CLEAN targets, where only the objective and the predictor can act.

A bucket (objective: the 12 `O:` contrasts; predictor: the 2 `P:` contrasts) is
**resolved** when every contrast in it is PRESENT+, PRESENT− or NEGLIGIBLE. Only
PRESENT+ implicates a bucket as a cause; PRESENT− is a definite reading that the
term does not cause void-landing, and is listed separately.

| outcome-tree branch | condition |
|---|---|
| MIXED | an objective contrast AND a predictor contrast are PRESENT+ |
| OBJECTIVE | an objective contrast is PRESENT+ and the predictor bucket is resolved with none PRESENT+ |
| RESIDUAL | a predictor contrast is PRESENT+ and the objective bucket is resolved with none PRESENT+ |
| TARGET | both buckets resolved with nothing PRESENT+, and a target contrast is PRESENT+ |
| NULL | both buckets resolved with nothing PRESENT+, and no target contrast PRESENT+ |
| NO VERDICT | anything else (one bucket PRESENT+ with the other unresolved is reported as that effect plus "branch undetermined"), or a gate failed |

Target contrasts never promote a clean-target result to MIXED: since they are
PRESENT+ nearly by construction, that would make OBJECTIVE and RESIDUAL
unreachable. **Phase 1C** runs only for the objective terms (gradient, normal) with a
PRESENT+ contrast on at least one predictor.

## Stated before seeing anything

- Any branch involving TARGET is conditional on the real TRAINING ground truth being
  blended at boundaries. 1B cannot measure that; the Phase 0 GT controls on the
  training sets (Hypersim, TartanAir, PointOdyssey) are the available evidence.
- A PREDICTOR effect compares a CNN that sees an image with a free map that does
  not, under a different learning rate. It is "smoothness or optimization",
  not smoothness alone.
- Not predicted: that the objective dominates. The §1.3 list is open.

## Reported but not used by the rule

Secondary contrasts at unadjusted α = {ALPHA} (the `both` level; every objective
contrast under mixed targets); a sensitivity re-analysis dropping cells whose
prediction still moved more than {UNSETTLED_FRAC}·gap over the last 10% of steps;
per-scene effects; exclusion counts by predictor and objective; the blocked
variance decomposition.
"""
    open(p, "w").write(md)
    print("wrote", p)
    return 0


def read_prereg(out):
    p = os.path.join(out, "1b_prereg.md")
    if not os.path.exists(p):
        raise SystemExit(f"No pre-registration at {p}. Run --pilot, then --preregister.")
    txt = open(p).read()
    tag = f"<!-- {PREREG_TAG} "
    if tag not in txt:
        raise SystemExit(f"{p} carries no {PREREG_TAG} marker.")
    stored = json.loads(txt.split(tag, 1)[1].split(" -->", 1)[0])
    lr = load_pilot(os.path.join(out, "1b_pilot.json"))
    live = design(lr)
    if stored != live:
        diff = sorted(k for k in set(stored) | set(live) if stored.get(k) != live.get(k))
        raise SystemExit(f"The live design differs from the pre-registration in: "
                         f"{', '.join(diff)}. Nothing may change after registering; a "
                         f"necessary fix means a new registration in a new --out, with "
                         f"the reason recorded.")
    return stored


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(out):
    d = read_prereg(out)
    LR.update(d["lr"])
    scenes, objectives, seeds = list(d["scenes"]), list(d["objectives"]), d["seeds"]
    cells = list(itertools.product(scenes, objectives, d["predictors"], d["targets"], seeds))
    path = os.path.join(out, "1b_factorial.csv")
    done = {_key(r): r for r in _read_rows(path)}
    todo = [c for c in cells if c not in done]
    print(f"{len(cells)} cells on {DEV}: {len(done)} already on disk, {len(todo)} to fit",
          flush=True)

    # An empty file (killed before the header was flushed) needs the header too.
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    failures, t0 = [], time.time()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader(); f.flush()
        for attempt in (1, 2):
            retry = []
            for i, c in enumerate(todo, 1):
                try:
                    r = fit_cell(*c)
                except Exception as e:           # noqa: BLE001 - recorded, retried once
                    print(f"  ! {'/'.join(map(str, c))}: {type(e).__name__}: {e}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    retry.append(c)
                    continue
                w.writerow(r); f.flush()
                done[c] = r
                if i % 25 == 0 or i == len(todo):
                    print(f"  pass {attempt}: {i}/{len(todo)}  ({time.time()-t0:.0f}s)",
                          flush=True)
            todo = retry
            if not todo:
                break
        failures = todo

    rows = [done[c] for c in cells if c in done]
    repro, repro_ok = reproduce(rows)
    res = analyze(rows, scenes, seeds)
    branch, clean, why = decide(res, repro_ok)
    write_results(out, rows, res, branch, clean, why, repro, failures, len(cells))
    return 0 if not failures and repro_ok else 3


def reproduce(rows, n=N_REPRO):
    """Re-fit n cells (evenly across predictors, chosen by a fixed rng) and compare
    with what was written. Bitwise-deterministic fits should match exactly."""
    rng = np.random.default_rng(12345)
    out = []
    for pr in PREDICTORS:
        pool = [r for r in rows if r["predictor"] == pr]
        if not pool:
            continue
        for j in rng.choice(len(pool), min(n // len(PREDICTORS), len(pool)), replace=False):
            r = pool[j]
            again = fit_cell(*_key(r))
            same = all(np.isclose(r[k], again[k], rtol=1e-9, atol=1e-12, equal_nan=True)
                       for k in ("FP", "final_loss", "rmse_interior"))
            out.append(dict(cell="/".join(map(str, _key(r))), FP=r["FP"],
                            FP_again=again["FP"], same=bool(same)))
    return out, bool(out) and all(x["same"] for x in out)


def _fmt(c):
    return (f"| `{c['id']}` | {c['FP_A']:.3f} | {c['FP_B']:.3f} | **{c['mean']:+.3f}** | "
            f"[{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['n_pairs']}/{c['n_design']} | "
            f"{c['status']} |")


def write_results(out, rows, res, branch, clean, why, repro, failures, n_cells):
    with open(os.path.join(out, "1b_contrasts.csv"), "w", newline="") as f:
        keys = ("id", "bucket", "A", "B", "FP_A", "FP_B", "mean", "lo", "hi",
                "n_pairs", "n_design", "status")
        w = csv.DictWriter(f, fieldnames=("family",) + keys, extrasaction="ignore")
        w.writeheader()
        for fam in ("control", "primary", "sensitivity", "secondary"):
            items = [res[fam]] if fam == "control" else res[fam]
            for c in items:
                w.writerow(dict(c, family=fam))

    head = ("| contrast | FP_A | FP_B | effect | CI | pairs | status |\n"
            "|---|---|---|---|---|---|---|")
    excl = collections.Counter((r["predictor"], r["objective"]) for r in rows if r["_excluded"])
    changed = [(a["id"], a["status"], b["status"]) for a, b in
               zip(res["primary"], res["sensitivity"]) if a["status"] != b["status"]]
    bs = blocked_shares([r for r in rows if not r["_excluded"]
                         and r["objective"] != "position_squared"])
    present = [c for c in res["primary"] if c["status"].startswith("PRESENT")]
    md = [
        "# Phase 1B v2 — effect sizes", "",
        f"Generated {time.strftime('%F %T')} · {len(rows)}/{n_cells} cells · device {DEV}"
        + (f" · **{len(failures)} cells failed to fit twice and are missing**" if failures else ""),
        "",
        "## Exclusions", "",
        "| predictor | excluded | status |", "|---|---|---|"]
    md += [f"| {p} | {v:.1%} | {'NOT EVALUATED' if p in res['not_evaluated'] else 'ok'} |"
           for p, v in res["exclusion_rate"].items()]
    if excl:
        md += ["", "By predictor and objective: " + ", ".join(
            f"{p}/{o} {n}" for (p, o), n in sorted(excl.items()))]
    md += ["", "## Gates", "",
           f"- falsifier (position/free/clean): FP = {res['falsifier_mean']:.4f} over "
           f"{res['falsifier_n']} cells (max {FALSIFIER_MAX}); {res['unconverged']} fits "
           f"further than {OPT_GAP_FRAC}·gap from the closed form",
           f"- positive control: {res['control']['status']} "
           f"({res['control']['mean']:+.3f} [{res['control']['lo']:+.3f}, "
           f"{res['control']['hi']:+.3f}])",
           f"- reproducibility: " + ", ".join(
               f"{x['cell']} {'same' if x['same'] else 'DIFFERENT'}" for x in repro), "",
           f"## Primary contrasts (simultaneous {1 - ALPHA:.0%} CIs, α/m = {res['alpha_family']:.4f})",
           "", head] + [_fmt(c) for c in res["primary"]]
    md += ["", "### Sensitivity: unsettled cells dropped", "",
           ("No primary status changes." if not changed else
            "Status changes: " + "; ".join(f"`{i}` {a} → {b}" for i, a, b in changed))]
    if present:
        md += ["", "### Per-scene effects for PRESENT contrasts", "",
               "| contrast | " + " | ".join(SCENES) + " |",
               "|---|" + "---|" * len(SCENES)]
        md += [f"| `{c['id']}` | " + " | ".join(
            f"{c['per_scene'].get(s, float('nan')):+.3f}" for s in SCENES) + " |"
            for c in present]
    md += ["", f"## Secondary contrasts (unadjusted {1 - ALPHA:.0%} CIs, not used by the rule)",
           "", head] + [_fmt(c) for c in res["secondary"]]
    if bs:
        md += ["", "## Blocked variance decomposition (description only)", "",
               f"Scene blocked; position_squared and excluded cells removed. Replicate "
               f"noise {bs['noise']:.3f} of total, scene and its interactions "
               f"{bs['blocked']:.3f}. Unbalanced after exclusions, so approximate.", "",
               "| bucket | share of systematic variance |", "|---|---|"]
        md += [f"| {k} | {v:.3f} |" for k, v in sorted(bs["shares"].items())]
    md += ["", "## Scope", "",
           "**[MEASURED]** for this synthetic construction, these objective implementations "
           "and these small predictors. **[ASSUMED]** that any of it transfers to "
           "billion-parameter models trained at scale; Phase 1C is the partial, confounded "
           "check."]
    open(os.path.join(out, "1b_decomposition.md"), "w").write("\n".join(md) + "\n")

    c1 = terms_for_1c(res)
    dec = [
        "# Phase 1B v2 — decision", "",
        f"Generated {time.strftime('%F %T')} under the rule in `1b_prereg.md`. Branches are "
        f"those of `notes/outcome_tree.md`; every claim carries a `notes/diagnosis.md` §0 label.",
        "", f"## Test 2 branch: **{branch}**", "",
        f"Tree consequence: _{TREE[branch]}_", ""] + [f"- {x}" for x in why] + [
        "", "## Phase 1C", "",
        ("Triggered for: " + "; ".join(c1) + "." if c1 else
         "**Not triggered.** No gradient or normal contrast is PRESENT+ on any predictor, "
         "so §4's condition for finetuning is not met."),
        "", "## Carried with any branch", "",
        "- **[MEASURED]** only for this construction, these loss reductions and these "
        "predictors. **[ASSUMED]** transfer to models at scale.",
        "- Target effects show sufficiency. Whether real training GT is blended at "
        "boundaries is a separate measurement 1B does not make.",
        "- A predictor effect is smoothness **or** optimization; the design does not "
        "separate them.",
        "- The candidate list in §1.3 is open. Per the standing rule, no fix is "
        "attempted on the strength of this.",
        "", "Effect sizes and CIs: `1b_decomposition.md`. Per-contrast rows: `1b_contrasts.csv`."]
    open(os.path.join(out, "1b_decision.md"), "w").write("\n".join(dec) + "\n")
    print(f"\n=== Test 2 branch: {branch} (clean-target attribution: {clean}) ===")
    print("\n".join(f"  {x}" for x in why))
    print("wrote 1b_factorial.csv, 1b_contrasts.csv, 1b_decomposition.md, 1b_decision.md")


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest():
    global DELTA
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    # ---- scene construction ----
    sc = Scene(gap=1.0)
    hi = render(sc)
    check("render has exactly the two surface depths",
          set(np.round(np.unique(hi), 6)) == {2.0, 3.0}, sorted(np.unique(hi)))
    check("clean target introduces no new values",
          set(np.round(np.unique(down_nearest(hi)), 6)) == {2.0, 3.0})
    ar = down_area(hi)
    inter = (ar > 2.001) & (ar < 2.999)
    check("area target DOES blend at the boundary", inter.any(), f"{inter.sum()} px")
    check("area blending is confined to <= 2 columns in every row",
          max(inter[y].sum() for y in range(sc.h)) <= 2)
    flat = down_area(render(Scene(gap=1.0, px_offset=0.0, boundary=0.5, tilt=0.0)))
    check("an untilted grid-aligned boundary blends nothing (why px_offset exists)",
          not ((flat > 2.001) & (flat < 2.999)).any())

    # defect 1: rows must not be copies of one another
    tg, im, ref, ref_im = realizations(sc, n=N_REAL, mode="nearest", seed=0)
    far_share = (tg > 2.5).mean(0)                       # per-pixel P(far) across jitter
    amb = far_share[(far_share > 0) & (far_share < 1)]
    check("boundary pixels span many distinct near/far weightings (defect 1)",
          len(np.unique(np.round(amb, 3))) >= 8, f"{len(np.unique(np.round(amb, 3)))} distinct")
    check("interior targets do not vary", ((tg.std(0) > 1e-9).sum()) < 0.1 * ref.size)
    old = realizations(Scene(gap=1.0, boundary=0.5, tilt=0.0), n=N_REAL, seed=0)[2]
    r_old = FM.fp_view(old, np.ones_like(old, bool), old, delta_min=0.05)
    r_new = FM.fp_view(ref, np.ones_like(ref, bool), ref, delta_min=0.05)
    check("v1 geometry had only 2 distinct boundary columns",
          len(np.unique(FM.boundary_set(old, np.ones_like(old, bool)).nonzero()[1])) == 2)
    check("v2 boundary set covers many columns",
          len(np.unique(FM.boundary_set(ref, np.ones_like(ref, bool)).nonzero()[1])) >= 6,
          f"n_eval old={r_old['n_eval']} new={r_new['n_eval']}")

    # defect 2: a constant prediction must not be scored at the void centre
    worst = 0.0
    for name, s in SCENES.items():
        r0 = down_nearest(render(s))
        for c in (0.5, float(r0.mean()), 7.3):
            fp = FM.fp_view(r0, np.ones_like(r0, bool), np.full_like(r0, c),
                            delta_min=0.05 * s.gap)["FP"]
            worst = max(worst, fp)
    check("a constant prediction scores FP <= 0.05 on every scene (defect 2)",
          worst <= 0.05, f"worst={worst:.3f}")
    check("GT vs GT scores FP == 0 on every scene", all(
        FM.fp_view(g, np.ones_like(g, bool), g, delta_min=0.05 * s.gap)["FP"] == 0.0
        for s in SCENES.values() for g in [down_nearest(render(s))]))

    # the falsifier in closed form, row by row
    col_rows = [(y, x) for y in range(sc.h) for x in range(sc.w)
                if 0 < far_share[y, x] < 1]
    med = np.array([weighted_median(tg[:, y, x]) for y, x in col_rows])
    mean = np.array([tg[:, y, x].mean() for y, x in col_rows])
    check("L1 optimum lands on a surface for every ambiguous pixel",
          np.all(np.isclose(med[:, None], [2.0, 3.0]).any(1)))
    check("L2 optimum lands strictly inside the void for every ambiguous pixel",
          np.all((mean > 2.0) & (mean < 3.0)))
    check("image edge is antialiased", ((ref_im > 0.26) & (ref_im < 0.74)).any())
    check("image and depth share the boundary column in every row",
          all(abs(int(np.argmax(np.abs(np.diff(ref_im[y]))))
                  - int(np.argmax(np.abs(np.diff(ref[y]))))) <= 1 for y in range(sc.h)))
    r = down_nearest(render(Scene(gap=1.0, slant_near=0.4, slant_far=0.4)))
    check("slant produces within-surface depth variation", r[:, :15].std() > 1e-3)
    check("k=3 yields three distinct depths",
          len(np.unique(down_nearest(render(SCENES["k3_d2.0"])))) == 3)
    try:
        render(Scene(gap=1.0, boundary=0.95, k=3, mid_frac=0.2))
        check("a boundary leaving the frame is refused", False)
    except SystemExit:
        check("a boundary leaving the frame is refused", True)

    # ---- objective terms ----
    t = torch.zeros(1, 8, 8)
    p = torch.zeros(1, 8, 8)
    check("identical inputs give zero position loss", position(p, t).sum() == 0)
    check("identical inputs give zero gradient loss", gradient(p, t).max() < 1e-6)
    check("identical inputs give zero normal loss", normal(p, t).abs().max() < 1e-6)
    st = torch.zeros(1, 8, 8); st[..., 4:] = 1.0
    g = gradient(torch.zeros(1, 8, 8), st)
    check("gradient term fires only at the step", g.max() > 0.9 and (g > 1e-6).sum() == 8)
    check("flat depth -> +z normal",
          torch.allclose(normals_from_depth(torch.zeros(1, 8, 8))[..., 2], torch.ones(1, 8, 8)))
    raw = torch.zeros(1, requires_grad=True)
    o = torch.optim.Adam([raw], lr=0.1)
    for _ in range(2000):
        o.zero_grad()
        c = conf_from_raw(raw, floor=False)
        (0.5 * c - 0.2 * torch.log(c)).mean().backward()
        o.step()
    check("unfloored confidence converges to alpha/l",
          abs(conf_from_raw(raw, floor=False).item() - 0.4) < 0.02)
    lo_raw = torch.tensor([-20.0], requires_grad=True)
    cf = conf_from_raw(lo_raw)
    cf.backward()
    check("floored C stays >= 1 and still has a gradient near the floor (v1 bug)",
          cf.item() >= 1.0 and lo_raw.grad.item() > 0)
    raw = torch.zeros(1, requires_grad=True)
    o = torch.optim.Adam([raw], lr=0.1)
    for _ in range(2000):
        o.zero_grad()
        c = conf_from_raw(raw)
        (0.05 * c - 0.2 * torch.log(c)).mean().backward()
        o.step()
    check("floored confidence reaches alpha/l when that is above the floor",
          abs(conf_from_raw(raw).item() - 4.0) < 0.1, f"C={conf_from_raw(raw).item():.3f}")
    check("weight 0 removes a term exactly",
          torch.allclose(total(p + 0.3, t, w_grad=0.0, w_normal=0.0), total(p + 0.3, t)))

    # ---- predictors and cells ----
    check("cnn_small receptive field 7", make_predictor("cnn_small", 8, 8, 0.0).rf == 7)
    check("cnn_large receptive field > 25", make_predictor("cnn_large", 8, 8, 0.0).rf > 25)
    fr = make_predictor("free", 8, 8, 2.0)
    check("free predictor ignores its input",
          torch.allclose(fr(torch.zeros(3, 8, 8)), fr(torch.ones(3, 8, 8))))
    try:
        fit_cell("fronto_d1.0", "position", "cnn_small", "clean", 0, steps=2)
        check("a CNN without a pilot learning rate is refused", False)
    except SystemExit:
        check("a CNN without a pilot learning rate is refused", True)
    orig_mp = make_predictor
    globals()["make_predictor"] = lambda k, h, w, init: orig_mp(k, h, w, float("nan"))
    try:
        nan_row = fit_cell("fronto_d1.0", "position", "free", "clean", 0, steps=3)
    finally:
        globals()["make_predictor"] = orig_mp
    check("a non-finite fit is excluded, not scored FP = 0 (review #4)",
          nan_row["collapsed"] and nan_row["failed"], f"FP={nan_row['FP']}")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cp_ = os.path.join(tmp, "x.csv")
        good = dict(fit_cell("fronto_d1.0", "position", "free", "clean", 0, steps=3))
        with open(cp_, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerow(good)
        with open(cp_, "a") as f:
            f.write(",".join(str(good[k]) for k in FIELDS)[:-2])   # "...Fal" unterminated
        back = _read_rows(cp_)
        check("resume drops a half-written last row and repairs the file (review #7)",
              len(back) == 1 and open(cp_).read().endswith("\n"))
        open(cp_, "w").close()
        check("resume treats an empty file as no rows", _read_rows(cp_) == [])
    try:
        fit_cell("fronto_d1.0", "position", "free", "clean", 0, n_real=24)
        check("even n_real is refused", False)
    except SystemExit:
        check("even n_real is refused", True)

    fps, gaps = [], []
    for sd in range(3):
        rr = fit_cell("fronto_d1.0", "position", "free", "clean", sd)
        fps.append(rr["FP"]); gaps.append(rr["optimizer_gap"])
    check("falsifier: position/free/clean FP <= 0.05 on every seed",
          max(fps) <= FALSIFIER_MAX, f"FP={[round(x, 4) for x in fps]} n_eval={rr['n_eval']}")
    check("falsifier fits agree with the closed form", max(gaps) < 0.01,
          f"max gap={max(gaps):.4f} m")
    small = fit_cell("fronto_d0.2", "position", "free", "clean", 0)
    check("...including on the smallest gap", small["optimizer_gap"] < OPT_GAP_FRAC * 0.2
          and small["FP"] <= FALSIFIER_MAX,
          f"gap={small['optimizer_gap']:.4f} m, FP={small['FP']:.3f}")
    check("FP is no longer quantized to halves",
          rr["n_eval"] > 60, f"n_eval={rr['n_eval']}")
    r2 = fit_cell("fronto_d1.0", "position_squared", "free", "clean", 0)
    check("squared position drives FP up (rig is sensitive)",
          r2["FP"] > max(fps) + DELTA, f"L2 FP={r2['FP']:.3f}")
    cp = fit_cell("fronto_d1.0", "position+conf", "free", "clean", 0, score=False, steps=200)
    pp = fit_cell("fronto_d1.0", "position", "free", "clean", 0, score=False, steps=200)
    check("position+conf is a different fit from position (v1 bug)",
          cp["final_loss"] != pp["final_loss"],
          f"{cp['final_loss']!r} vs {pp['final_loss']!r}")
    blind = fit_cell("fronto_d1.0", "position", "free", "clean", 0, score=False, steps=5)
    check("score=False never computes FP", not np.isfinite(blind["FP"]))
    again = fit_cell("fronto_d1.0", "position", "free", "clean", 0)
    check("a re-fit reproduces exactly", again["final_loss"] == fit_cell(
        "fronto_d1.0", "position", "free", "clean", 0)["final_loss"])
    c1, c2 = (fit_cell("k3_d2.0", "position+both", "cnn_large", "mixed", 3, lr=1e-3,
                       steps=40, score=False) for _ in range(2))
    check(f"a CNN re-fit reproduces exactly on {DEV}",
          c1["final_loss"] == c2["final_loss"] and c1["rmse_interior"] == c2["rmse_interior"],
          f"{c1['final_loss']!r} vs {c2['final_loss']!r}")

    # ---- pilot choice ----
    def prow(pr, lr, collapsed, rmse):
        return dict(predictor=pr, lr=lr, scene="fronto_d1.0", collapsed=collapsed,
                    failed=False, rmse_interior=rmse, settle_frac=0.0)
    pr_rows = ([prow("cnn_small", 1e-2, True, 0.5)] * 5 + [prow("cnn_small", 1e-2, False, 0.01)] * 5
               + [prow("cnn_small", 1e-3, False, 0.05)] * 10
               + [prow("cnn_large", 1e-3, True, 0.5)] * 10)
    ch, _ = choose_lr(pr_rows, lrs=(1e-2, 1e-3))
    check("pilot skips a low-RMSE rate that collapses too often", ch["cnn_small"] == 1e-3, ch)
    check("pilot returns None when nothing qualifies", ch["cnn_large"] is None)

    # ---- contrasts and the rule, on synthetic rows with known effects ----
    rng = np.random.default_rng(0)
    scn, sds = list(SCENES), list(range(10))

    def rows_with(effects):
        out = []
        for s in scn:
            for sd in sds:
                for pr in PREDICTORS:
                    for tgt in TARGETS:
                        for ob in LEVELS:
                            fp = 0.1 + 0.01 * rng.standard_normal()
                            fp += effects.get(ob, 0.0) + (effects.get("mixed", 0.0) if tgt == "mixed" else 0)
                            fp += effects.get(pr, 0.0) if ob == "position" and tgt == "clean" else 0
                            out.append(dict(scene=s, seed=sd, predictor=pr, target=tgt,
                                            objective=ob, FP=fp, collapsed=False, failed=False,
                                            optimizer_gap=0.0 if (ob, pr) == ("position", "free") else float("nan"),
                                            settle_frac=0.0))
        return out

    base = {"position_squared": 0.4}
    res = analyze(rows_with(dict(base, **{"position+normal": 0.2})), scn, sds)
    st_ = {c["id"]: c["status"] for c in res["primary"]}
    check("a real +0.2 normal effect reads PRESENT+", st_["O:normal/free/clean"] == "PRESENT+")
    check("a zero grad effect reads NEGLIGIBLE", st_["O:grad/free/clean"] == "NEGLIGIBLE")
    res["falsifier_mean"], res["falsified"] = 0.0, False   # base FP 0.1 is synthetic
    check("objective only -> OBJECTIVE", decide(res)[0] == "OBJECTIVE", decide(res)[0])
    res2 = analyze(rows_with(dict(base, **{"position+normal": 0.2, "mixed": 0.3})), scn, sds)
    res2["falsified"] = False
    check("objective with a by-construction target effect stays OBJECTIVE (review #2)",
          decide(res2)[0] == "OBJECTIVE", decide(res2)[0])
    res2b = analyze(rows_with(dict(base, **{"position+normal": 0.2, "cnn_small": 0.2})), scn, sds)
    res2b["falsified"] = False
    check("objective plus predictor -> MIXED", decide(res2b)[0] == "MIXED")
    res3 = analyze(rows_with(dict(base, mixed=0.3)), scn, sds)
    res3["falsified"] = False
    check("target only -> TARGET", decide(res3)[0] == "TARGET", decide(res3)[1])
    res4 = analyze(rows_with(dict(base, cnn_large=0.25)), scn, sds)
    res4["falsified"] = False
    check("predictor only -> RESIDUAL", decide(res4)[0] == "RESIDUAL")
    res5 = analyze(rows_with({"position+normal": 0.2}), scn, sds)
    res5["falsified"] = False
    check("no positive control -> NO VERDICT", decide(res5)[0] == "NO VERDICT")
    check("failed reproducibility -> NO VERDICT", decide(res, repro_ok=False)[0] == "NO VERDICT")
    res6 = dict(res, falsified=True, falsifier_mean=0.3, unconverged=0)
    check("falsifier breach at the closed form -> FALSIFIED", decide(res6)[0] == "FALSIFIED")
    check("falsifier breach with unconverged fits -> NO VERDICT",
          decide(dict(res6, unconverged=2))[0] == "NO VERDICT")
    fr_rows = rows_with(base)
    for r in fr_rows:
        if (r["objective"], r["predictor"], r["target"]) == ("position", "free", "clean"):
            r["FP"], r["optimizer_gap"] = 0.3, 0.3
    check("the optimizer gap does not exclude high-FP falsifier cells (review #3)",
          analyze(fr_rows, scn, sds)["falsifier_mean"] > 0.25)
    res8 = analyze(rows_with(dict(base, **{"position+grad": -0.2})), scn, sds)
    res8["falsified"] = False
    check("a measured reduction alone reads NULL, not NO VERDICT (review #5a)",
          decide(res8)[0] == "NULL", decide(res8))
    res9 = analyze(rows_with(dict(base, **{"position+normal": 0.2, "mixed": 0.3})), scn, sds)
    for c in res9["primary"]:
        if c["id"] == "P:cnn_large":
            c["status"] = "NOT EVALUATED"
    res9["falsified"] = False
    check("an unresolved predictor blocks a clean OBJECTIVE call (review #5b)",
          decide(res9)[0] == "NO VERDICT" and decide(res9)[1].startswith("OBJECTIVE"))
    cr = rows_with(base)
    for r in cr:
        if r["predictor"] == "cnn_large":
            r["collapsed"] = True
    res7 = analyze(cr, scn, sds)
    check("a predictor over MAX_EXCLUDED is NOT EVALUATED",
          all(c["status"] == "NOT EVALUATED" for c in res7["primary"]
              if "cnn_large" in c["id"]))
    res7["falsified"] = False
    check("NOT EVALUATED contrasts block NULL", decide(res7)[0] == "NO VERDICT")
    m, lo, hi = strat_boot([np.full(10, 0.2), np.full(10, 0.2)], 0.05)
    check("bootstrap of a constant effect is exact", np.allclose([m, lo, hi], 0.2))
    x = EdgePad(2)(torch.arange(12.0).reshape(1, 1, 3, 4))
    check("EdgePad replicates edges", x.shape == (1, 1, 7, 8) and x[0, 0, 0, 0] == 0
          and x[0, 0, -1, -1] == 11)
    check("classify is exhaustive at the margin", classify(0.05, 0.01, 0.09) == "PRESENT+"
          and classify(0.04, -0.01, 0.049) == "NEGLIGIBLE"
          and classify(0.04, 0.01, 0.07) == "INCONCLUSIVE")

    # ---- pre-registration guard ----
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        json.dump(dict(passed=True, lr={"cnn_small": 1e-3, "cnn_large": 3e-4}),
                  open(os.path.join(tmp, "1b_pilot.json"), "w"))
        preregister(tmp)
        check("a fresh pre-registration reads back", read_prereg(tmp)["tag"] == PREREG_TAG)
        check("the registration pins the code, not only the constants (review #6)",
              set(read_prereg(tmp)["code_md5"]) == {"factorial.py", "fpmetrics.py"})
        saved, DELTA = DELTA, 0.03
        try:
            read_prereg(tmp)
            check("a moved constant is refused", False)
        except SystemExit:
            check("a moved constant is refused", True)
        finally:
            DELTA = saved
        try:
            preregister(tmp)
            check("pre-registration is write-once", False)
        except SystemExit:
            check("pre-registration is write-once", True)
    LR.update({"cnn_small": None, "cnn_large": None})

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _main():
    ap = argparse.ArgumentParser(description="Phase 1B factorial, version 2")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="choose CNN learning rates, FP-blind")
    ap.add_argument("--preregister", action="store_true", help="write-once; needs the pilot")
    ap.add_argument("--run", action="store_true", help="needs the pre-registration; resumable")
    ap.add_argument("--force", action="store_true", help="redo the pilot / pre-registration")
    ap.add_argument("--out", default="results/phase1b_v2")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.pilot:
        return pilot(a.out, a.force)
    if a.preregister:
        return preregister(a.out, a.force)
    if a.run:
        return run(a.out)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
