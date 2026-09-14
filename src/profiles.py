r"""
Phase 1B deliverable §5.5 — one-dimensional depth profiles across the occlusion
boundary, one per objective variant.

`notes/diagnosis.md` §5 lists this as: "figs/ — one-dimensional depth profiles
across the boundary per objective variant. Inspect before trusting numbers."
The instruction is the point. `1b_factorial.csv` reports a scalar `FP` per cell;
a scalar cannot distinguish a smooth ramp across the void from an overshoot from
a displaced step, and all three would be reported as "flying pixels". The profile
can. This module produces the picture that has to be looked at before the
`1b_decomposition.md` numbers mean anything.

WHAT IS PLOTTED, and why each series is there

  reference GT      the un-jittered nearest-neighbour depth map -- the surface
                    the metric scores against. A clean step by construction.
  target envelope   min/max over the `n_real` jittered realizations the cell was
                    actually FITTED to. This is the target DISTRIBUTION claim (a)
                    is about; where the envelope has width, the pixel's target
                    genuinely varies between near and far and the pointwise
                    minimizer is a real question rather than a copy.
  fitted prediction the SAME predictor object `fit_cell` fitted, evaluated on the
                    same un-jittered reference image, then multiplied by the same
                    median alignment scale `fp_view` applies before scoring. What
                    is drawn is what was scored, not a re-derivation of it.
  void band         per evaluated boundary pixel, the interval
                    `[d1 + beta*delta, d2 - beta*delta]` from `fpmetrics`. A
                    prediction inside this band IS a flying pixel by definition.
                    Drawing it makes the metric's verdict visible rather than
                    asserted.

HOW THE PREDICTION IS OBTAINED WITHOUT TOUCHING `factorial.py`

`fit_cell` returns metrics, not the fitted map, and `factorial.py` is not ours to
edit. Rather than re-implementing its optimizer loop -- which would silently
drift from the code that produced `1b_factorial.csv` and make every figure a
picture of a DIFFERENT experiment -- this module temporarily wraps
`factorial.make_predictor` to keep a reference to the network `fit_cell` builds,
runs `fit_cell` unmodified, and evaluates that same network afterwards.
`--selftest` asserts that the FP recomputed from the captured map equals the FP
`fit_cell` returned, so a drift between figure and number cannot go unnoticed.

LEGIBILITY. Panels carry their own opaque near-white background and dark ink, so
they read on a dark page as well as a light one. Every series is separated by
LINESTYLE and MARKER as well as hue, and the void band by hatching, so nothing
depends on colour discrimination.

    python src/profiles.py --selftest
    python src/profiles.py --all --out results/phase1b_v2/figs   # after factorial --pilot
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import factorial as FA
import fpmetrics as FM


# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
# Okabe-Ito, which is colour-vision-safe, on a near-white panel that the figure
# paints itself. Hue is never the only carrier: see LINESTYLE/MARKER below.
INK = "#141414"
PANEL = "#FCFCFA"
GRID = "#C9C9C4"
C_GT = "#0072B2"      # reference GT       solid, circle
C_PRED = "#D55E00"    # fitted prediction  dashed, square
C_TGT = "#009E73"     # target envelope    fill + dotted edge
C_VOID = "#8A8A8A"    # void band          hatched
C_MARK = "#CC79A7"    # boundary location  vertical rule


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for lbl in (ax.xaxis.label, ax.yaxis.label):
        lbl.set_color(INK)


# --------------------------------------------------------------------------- #
# fitting: reuse factorial.fit_cell verbatim, capture the network it builds
# --------------------------------------------------------------------------- #
_CACHE: dict = {}

N_REAL = FA.N_REAL   # the run's values, so a figure is a cell of the run
JITTER_PX = FA.JITTER_PX
STEPS = FA.STEPS


def fit_and_profile(scene, objective, predictor, target, seed,
                    n_real=N_REAL, steps=STEPS, jitter_px=JITTER_PX):
    """Fit one factorial cell and return everything a profile needs.

    The fit is `factorial.fit_cell`, unmodified. We only intercept the predictor
    construction so the fitted network survives the call; the loss, the optimizer,
    the realizations and the scoring are all its own code, so the figure and the
    CSV row describe the same fit.
    """
    key = (scene, objective, predictor, target, seed, n_real, steps, jitter_px)
    if key in _CACHE:
        return _CACHE[key]

    captured = {}
    orig = FA.make_predictor

    def spy(kind, h, w, init):
        net = orig(kind, h, w, init)
        captured["net"] = net
        return net

    FA.make_predictor = spy
    try:
        row = FA.fit_cell(scene, objective, predictor, target, seed,
                          n_real=n_real, steps=steps, jitter_px=jitter_px)
    finally:
        FA.make_predictor = orig

    sc = FA.SCENES[scene]
    # Same arguments fit_cell used, so the same seeded rng gives the same maps.
    tg, im, ref, ref_im = FA.realizations(
        sc, n=n_real, jitter_px=jitter_px,
        mode="nearest" if target == "clean" else "area", seed=seed)

    RI = torch.tensor(ref_im, dtype=torch.float32, device=FA.DEV)[None]
    with torch.no_grad():
        out = captured["net"](RI)[0].double().cpu().numpy()

    valid = np.ones_like(ref, bool)
    delta_min = 0.05 * sc.gap
    r, masks = FM.fp_view(ref, valid, out, eta=FM.ETA0, tau=FM.TAU0,
                          beta=FM.BETA0, w=3, delta_min=delta_min,
                          return_masks=True)

    # The metric SCALES the prediction before testing it. Plotting the unscaled
    # map beside a band computed on the scaled one would show a verdict the
    # figure contradicts, so the plotted series is the scaled one.
    scale = FM.align_scale(out, ref, masks["I"], "median")
    scale = float(scale) if np.isfinite(scale) else 1.0

    # Per-boundary-pixel void band, recomputed exactly as fp_view defines it.
    B = FM.boundary_set(ref, valid, FM.ETA0)
    pix = np.argwhere(B)
    sup = FM.depth_support(ref, valid, pix, w=3, eta=FM.ETA0, min_cluster=1)
    K, d1, d2 = sup["K"], sup["d1"], sup["d2"]
    delta = d2 - d1
    keep = (K == 2) & np.isfinite(delta) & (delta >= delta_min)
    lo = d1 + FM.BETA0 * delta
    hi = d2 - FM.BETA0 * delta

    res = dict(row=row, scene=scene, sc=sc, objective=objective,
               predictor=predictor, target=target, seed=seed,
               ref=ref, targets=tg, pred=out, pred_scaled=out * scale,
               scale=scale, fp=r["FP"], outside=r["outside_rate"],
               n_eval=r["n_eval"], pix=pix, keep=keep, lo=lo, hi=hi,
               d1=d1, d2=d2, masks=masks,
               # The boundary is tilted (factorial.TILT), so its column depends on
               # the row; this is its position on the mid row every panel draws.
               boundary_x=(sc.boundary * sc.w + sc.px_offset
                           + sc.tilt * (sc.h // 2 + 0.5 - sc.h / 2) - 0.5))
    _CACHE[key] = res
    return res


# --------------------------------------------------------------------------- #
# one panel
# --------------------------------------------------------------------------- #
def draw_profile(ax, profs, *, row=None, zoom=9, title=None, legend=False,
                 show_targets=True):
    """Draw one 1-D profile panel. `profs` is a list of cells (one per seed);
    all are drawn, because a single seed cannot show whether a shape is a
    property of the objective or an accident of one optimizer run."""
    p0 = profs[0]
    sc, ref = p0["sc"], p0["ref"]
    r = sc.h // 2 if row is None else row
    W = sc.w
    x = np.arange(W)
    bx = p0["boundary_x"]
    lo_x, hi_x = max(0, int(bx) - zoom), min(W - 1, int(bx) + zoom)

    _style(ax)

    # --- target envelope actually fitted to ---
    if show_targets:
        tg = p0["targets"][:, r, :]
        ax.fill_between(x, tg.min(0), tg.max(0), color=C_TGT, alpha=0.20,
                        linewidth=0, zorder=1)
        ax.plot(x, tg.mean(0), color=C_TGT, linestyle=":", linewidth=1.2,
                zorder=2)

    # --- void band, per evaluated boundary pixel ---
    pix, keep, lo, hi = p0["pix"], p0["keep"], p0["lo"], p0["hi"]
    sel = keep & (pix[:, 0] == r)
    for c, a, b in zip(pix[sel, 1], lo[sel], hi[sel]):
        ax.fill_between([c - 0.5, c + 0.5], [a, a], [b, b], color=C_VOID,
                        alpha=0.28, hatch="///", edgecolor=C_VOID,
                        linewidth=0.0, zorder=0)

    # --- reference GT ---
    ax.step(x, ref[r], where="mid", color=C_GT, linewidth=2.0, zorder=3)
    ax.plot(x, ref[r], color=C_GT, linestyle="none", marker="o", markersize=3.0,
            zorder=3)

    # --- fitted predictions, one line per seed ---
    for i, p in enumerate(profs):
        y = p["pred_scaled"][r]
        ax.plot(x, y, color=C_PRED, linestyle="--", linewidth=1.6 if i == 0 else 1.0,
                alpha=1.0 if i == 0 else 0.55, zorder=4)
        if i == 0:
            ax.plot(x, y, color=C_PRED, linestyle="none", marker="s",
                    markersize=3.2, zorder=5)
        # filled marker where the metric calls this pixel a flying pixel
        s = p["keep"] & (p["pix"][:, 0] == r)
        cols = p["pix"][s, 1]
        inside = (y[cols] >= p["lo"][s]) & (y[cols] <= p["hi"][s])
        if inside.any():
            ax.plot(cols[inside], y[cols][inside], linestyle="none", marker="X",
                    markersize=8, markerfacecolor=C_PRED, markeredgecolor=INK,
                    markeredgewidth=0.8, zorder=6)

    ax.axvline(bx, color=C_MARK, linestyle="-.", linewidth=1.1, alpha=0.9, zorder=2)

    ax.set_xlim(lo_x - 0.5, hi_x + 0.5)
    ys = np.concatenate([ref[r][lo_x:hi_x + 1]] +
                        [p["pred_scaled"][r][lo_x:hi_x + 1] for p in profs])
    pad = 0.18 * max(np.ptp(ys), 1e-3)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)

    if title:
        ax.set_title(title, fontsize=9, color=INK)
    if legend:
        ax.legend(handles=LEGEND_HANDLES, fontsize=7, loc="best",
                  facecolor=PANEL, edgecolor=INK, labelcolor=INK)
    return ax


LEGEND_HANDLES = [
    Line2D([], [], color=C_GT, lw=2, marker="o", ms=3, label="reference GT"),
    Line2D([], [], color=C_PRED, lw=1.6, ls="--", marker="s", ms=3,
           label="fitted prediction (scaled)"),
    Line2D([], [], color=C_TGT, lw=1.2, ls=":", label="mean fitted target"),
    Patch(facecolor=C_TGT, alpha=0.20, label="target envelope over jitter"),
    Patch(facecolor=C_VOID, alpha=0.28, hatch="///", label="void band (FP if inside)"),
    Line2D([], [], color=C_PRED, ls="none", marker="X", ms=8, mec=INK,
           label="scored a flying pixel"),
    Line2D([], [], color=C_MARK, ls="-.", lw=1.1, label="true boundary"),
]


def _fig(nrows, ncols, w=3.6, h=2.7):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(PANEL)
    return fig, axes


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=140, facecolor=PANEL, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p, flush=True)
    return p


# --------------------------------------------------------------------------- #
# the figures
# --------------------------------------------------------------------------- #
OBJ_ORDER = ["position", "position+grad", "position+normal", "position+both",
             "position+conf", "position+conf_noclip", "position_squared"]


def fig_objectives(out, scene="fronto_d1.0", predictor="free", target="clean",
                   seeds=(0, 1, 2, 3, 4), **kw):
    """Every objective level at one fixed scene/predictor/target. The deliverable's
    core panel: what does each loss term alone do to the shape of the step?"""
    fig, axes = _fig(2, 4)
    for i, ob in enumerate(OBJ_ORDER):
        ax = axes[i // 4][i % 4]
        ps = [fit_and_profile(scene, ob, predictor, target, s, **kw) for s in seeds]
        fp = np.mean([p["fp"] for p in ps])
        draw_profile(ax, ps, title=f"{ob}\nFP={fp:.2f}  (n={len(seeds)} seeds)",
                     legend=(i == 0))
        if i % 4 == 0:
            ax.set_ylabel("depth (m)", fontsize=8)
        if i // 4 == 1:
            ax.set_xlabel("column", fontsize=8)
    axes[1][3].axis("off")
    axes[1][3].set_facecolor(PANEL)
    fig.suptitle(f"§5.5 depth profile across the boundary — every objective level\n"
                 f"scene={scene}  predictor={predictor}  target={target}  row={FA.SCENES[scene].h//2}",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, f"profiles_objectives_{predictor}_{target}_{scene}.png")


def fig_objective_x_predictor(out, scene="fronto_d1.0", target="clean",
                              seeds=(0, 1, 2, 3, 4), **kw):
    """Objective down the rows, predictor across the columns.

    This is the figure that separates "the loss prefers a flying pixel" from
    "the predictor cannot express a step": the `free` column has no shared
    parameters, so its shape is the loss's own preference."""
    preds = list(FA.PREDICTORS)
    fig, axes = _fig(len(OBJ_ORDER), len(preds), w=3.6, h=2.5)
    for i, ob in enumerate(OBJ_ORDER):
        for j, pr in enumerate(preds):
            ps = [fit_and_profile(scene, ob, pr, target, s, **kw) for s in seeds]
            fp = np.mean([p["fp"] for p in ps])
            draw_profile(axes[i][j], ps, title=f"{ob} · {pr} · FP={fp:.2f}",
                         legend=(i == 0 and j == 0))
            if j == 0:
                axes[i][j].set_ylabel("depth (m)", fontsize=8)
            if i == len(OBJ_ORDER) - 1:
                axes[i][j].set_xlabel("column", fontsize=8)
    fig.suptitle(f"§5.5 objective × predictor — scene={scene}, target={target}\n"
                 f"`free` has one parameter per pixel: its shape is what the LOSS prefers",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    return _save(fig, out, f"profiles_objective_x_predictor_{target}_{scene}.png")


def fig_target_contrast(out, scene="fronto_d1.0", predictor="free",
                        seeds=(0, 1, 2, 3, 4), **kw):
    """clean (nearest) vs mixed (area-average) target, per objective."""
    fig, axes = _fig(2, len(OBJ_ORDER), w=3.2, h=2.5)
    for j, ob in enumerate(OBJ_ORDER):
        for i, tg in enumerate(FA.TARGETS):
            ps = [fit_and_profile(scene, ob, predictor, tg, s, **kw) for s in seeds]
            fp = np.mean([p["fp"] for p in ps])
            draw_profile(axes[i][j], ps, title=f"{ob}\n{tg} · FP={fp:.2f}",
                         legend=(i == 0 and j == 0))
            if j == 0:
                axes[i][j].set_ylabel(f"{tg}\ndepth (m)", fontsize=8)
    fig.suptitle(f"§5.5 target contrast — scene={scene}, predictor={predictor}. "
                 f"The mixed target's own value sits IN the void band.",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save(fig, out, f"profiles_target_contrast_{predictor}_{scene}.png")


def fig_single(out, scene, objective, predictor, target, seeds=(0, 1, 2, 3, 4), **kw):
    """One objective variant, full width beside the boundary zoom."""
    ps = [fit_and_profile(scene, objective, predictor, target, s, **kw) for s in seeds]
    fp = np.mean([p["fp"] for p in ps])
    fig, axes = _fig(1, 2, w=5.4, h=3.6)
    draw_profile(axes[0][0], ps, zoom=FA.SCENES[scene].w, title="full width")
    draw_profile(axes[0][1], ps, zoom=7, title="boundary zoom", legend=True)
    for a in axes[0]:
        a.set_xlabel("column", fontsize=8)
    axes[0][0].set_ylabel("depth (m)", fontsize=8)
    fig.suptitle(f"{objective} · {predictor} · {target} · {scene} · "
                 f"FP={fp:.2f} over {len(seeds)} seeds", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out, f"profile_{scene}_{predictor}_{target}_"
                           f"{objective.replace('+', '_')}.png")


# --------------------------------------------------------------------------- #
# the numbers behind the pictures
# --------------------------------------------------------------------------- #
def report(profs, fh=None):
    """Per-evaluated-pixel table: where the prediction landed relative to the
    void band. The figure is the evidence; this is the figure read out loud, so
    an interpretation can be checked rather than eyeballed."""
    lines = []
    for p in profs:
        r = p["sc"].h // 2
        sel = p["keep"] & (p["pix"][:, 0] == r)
        y = p["pred_scaled"][r]
        lines.append(
            f"{p['objective']:22s} {p['predictor']:10s} {p['target']:6s} "
            f"s{p['seed']} FP={p['fp']:.3f} out={p['outside']:.3f} "
            f"scale={p['scale']:.4f} n_eval={p['n_eval']}")
        for c, a, b, dd1, dd2 in zip(p["pix"][sel, 1], p["lo"][sel], p["hi"][sel],
                                     p["d1"][sel], p["d2"][sel]):
            v = y[c]
            where = ("IN VOID" if a <= v <= b else
                     "in front of near" if v < a else "behind far")
            t = (v - dd1) / (dd2 - dd1) if dd2 > dd1 else float("nan")
            lines.append(f"    col {c:3d}  gt={p['ref'][r, c]:.4f}  "
                         f"pred={v:.4f}  t={t:+.3f}  band=[{a:.3f},{b:.3f}]  "
                         f"support=[{dd1:.3f},{dd2:.3f}]  {where}")
    s = "\n".join(lines)
    print(s, flush=True)
    if fh:
        fh.write(s + "\n")
    return s


# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    # The whole module rests on this: the map we plot must be the map that was
    # scored. If capture drifted from fit_cell, every figure would be a picture
    # of a different experiment and the deliverable would be worse than absent.
    p = fit_and_profile("fronto_d1.0", "position", "free", "clean", 0)
    check("captured prediction reproduces fit_cell's FP",
          abs(p["fp"] - p["row"]["FP"]) < 1e-12,
          f"{p['fp']} vs {p['row']['FP']}")
    inner = ~FM.dilate(FM.boundary_set(p["ref"], np.ones_like(p["ref"], bool)), FM.TAU0)
    check("captured prediction reproduces fit_cell's interior rmse",
          abs(np.sqrt(np.mean((p["pred"] - p["ref"])[inner] ** 2))
              - p["row"]["rmse_interior"]) < 1e-9)
    check("reference is a clean two-surface step",
          len(np.unique(np.round(p["ref"], 6))) == 2, f"{np.unique(p['ref'])}")
    check("some pixels are evaluated", p["n_eval"] > 0, f"n={p['n_eval']}")
    check("the void band is non-empty where it is drawn",
          np.all(p["hi"][p["keep"]] > p["lo"][p["keep"]]))

    # The evaluated boundary pixels must straddle the true boundary, or the
    # profile would be drawn against the wrong location.
    sel = p["keep"] & (p["pix"][:, 0] == p["sc"].h // 2)
    cols = p["pix"][sel, 1]
    check("evaluated pixels sit at the true boundary",
          np.all(np.abs(cols - p["boundary_x"]) <= 2.0), f"cols={cols}")

    # Since factorial v2 the boundary is tilted, so rows are NOT copies of one
    # another and one mid-row profile shows one of many boundary phases. Stated
    # as a check so a figure is never read as the whole story for a cell.
    check("scene rows differ (a profile is one row, not the cell)",
          not np.allclose(p["ref"], p["ref"][:1]))

    # A cache hit must not refit.
    q = fit_and_profile("fronto_d1.0", "position", "free", "clean", 0)
    check("cache returns the identical object", q is p)

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _main():
    ap = argparse.ArgumentParser(description="Phase 1B §5.5 boundary profiles")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="results/phase1b_v2/figs")
    ap.add_argument("--pilot", default="results/phase1b_v2/1b_pilot.json",
                    help="CNN learning rates from factorial --pilot; needed for CNN panels")
    ap.add_argument("--scene", default="fronto_d1.0")
    ap.add_argument("--predictor", default="free")
    ap.add_argument("--target", default="clean")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if os.path.exists(a.pilot):
        FA.load_pilot(a.pilot)
    if not a.all:
        print(__doc__)
        return 0

    # The run's own seeds, so every panel is a cell that is in 1b_factorial.csv.
    seeds = tuple(range(FA.MAIN_SEED_OFFSET, FA.MAIN_SEED_OFFSET + a.seeds))
    kw = dict(steps=a.steps)
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, "profiles_report.txt"), "w")
    fh.write("Phase 1B §5.5 — per-evaluated-pixel readout behind the figures.\n"
             "t is the position of the prediction between the two supporting\n"
             "surfaces: t=0 on the near surface, t=1 on the far surface, and\n"
             "0.2 < t < 0.8 is the void band at beta=0.2.\n\n")

    # 1. every objective level, free predictor, clean target
    fig_objectives(a.out, a.scene, "free", "clean", seeds, **kw)
    # 2. free vs the CNNs, every objective
    fig_objective_x_predictor(a.out, a.scene, "clean", seeds, **kw)
    # 3. clean vs mixed target
    fig_target_contrast(a.out, a.scene, "free", seeds, **kw)
    # 4. per-objective singles for the free predictor and for cnn_large
    for ob in OBJ_ORDER:
        for pr in ("free", "cnn_large"):
            fig_single(a.out, a.scene, ob, pr, "clean", seeds, **kw)

    fh.write("=== free / clean ===\n")
    for ob in OBJ_ORDER:
        report([fit_and_profile(a.scene, ob, "free", "clean", s, **kw)
                for s in seeds], fh)
    fh.write("\n=== cnn_small / cnn_large, clean ===\n")
    for ob in OBJ_ORDER:
        for pr in ("cnn_small", "cnn_large"):
            report([fit_and_profile(a.scene, ob, pr, "clean", s, **kw)
                    for s in seeds], fh)
    fh.write("\n=== free / mixed ===\n")
    for ob in OBJ_ORDER:
        report([fit_and_profile(a.scene, ob, "free", "mixed", s, **kw)
                for s in seeds], fh)
    fh.close()
    print("wrote", os.path.join(a.out, "profiles_report.txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
