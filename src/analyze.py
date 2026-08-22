r"""
Statistics, the pre-registered decision rule, and the figures (Sec. 2, 7, 8).

Writes into <out>/:
    decision.md     pre-registration (--preregister), then the outcome (--decide)
    table_main.csv  per stream: FP_model, FP_GT, diff, ratio, CI, p
    sensitivity.csv the full (eta, tau, beta) grid
    figs/           per-scene panels + a silhouette point-cloud render

Analysis rules that are easy to get wrong, and how they are handled here:
  * The unit is the SCENE (Sec. 6.7). Every test is paired across scenes on the
    identical pixel set; pixels are never pooled across scenes.
  * Effect size governs the decision, not the p-value (Sec. 7). With enough
    scenes a trivial difference becomes significant, so the GO thresholds are
    absolute (>= 5 pp) and relative (>= 2x), and the CI only has to exclude 0.
  * Holm correction is applied across prediction streams.
  * Sign stability over the whole sweep is a GO requirement: one flipped cell
    downgrades the outcome to WEAK no matter how good the headline looks.

Order of operations matters. Run --preregister BEFORE any measurement; it refuses
to overwrite an existing decision.md so the thresholds cannot be edited after the
numbers are in.

    python src/analyze.py --preregister
    python src/analyze.py --decide
    python src/analyze.py --figs --n-scenes 5
"""
from __future__ import annotations

import os
import csv
import json
import re
import argparse
from datetime import date

import numpy as np

import fpmetrics as FM

GO_MIN_PP = 0.05          # FP_model - FP_GT >= 5 percentage points
GO_MIN_RATIO = 2.0        # FP_model >= 2 x FP_GT
NOGO_GT_MAX = 0.15        # FP_GT >= 15% -> GT too contaminated to measure against
N_BOOT = 10000
GT = "gt_raw"


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def paired_bootstrap(d, n=N_BOOT, seed=0):
    """95% CI on the mean paired difference, resampling SCENES with replacement."""
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _avg_rank(a):
    """Ranks with ties averaged (what the signed-rank test actually requires)."""
    try:
        from scipy.stats import rankdata
        return rankdata(a)
    except Exception:
        order = np.argsort(a, kind="mergesort")
        sa, r, i = a[order], np.empty(len(a), float), 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sa[j + 1] == sa[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r


def wilcoxon(d):
    """Two-sided signed-rank p. scipy if present, else a normal approximation."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 3:
        return float("nan")
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(d).pvalue)
    except Exception:
        r = _avg_rank(np.abs(d))
        wp = r[d > 0].sum()
        n = len(d)
        mu, sd = n * (n + 1) / 4.0, np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        from math import erfc, sqrt
        return float(erfc(abs(wp - mu) / sd / sqrt(2)))


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation: (W+ - W-) / (W+ + W-) in [-1,1]."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if not len(d):
        return float("nan")
    r = _avg_rank(np.abs(d))
    pos, neg = r[d > 0].sum(), r[d < 0].sum()
    return float((pos - neg) / (pos + neg)) if pos + neg else float("nan")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    idx = np.flatnonzero(ok)[np.argsort(p[ok])]
    m, run = len(idx), 0.0
    for i, j in enumerate(idx):
        run = max(run, (m - i) * p[j])
        out[j] = min(1.0, run)
    return out


# --------------------------------------------------------------------------- #
# reshaping results.json
# --------------------------------------------------------------------------- #
def cell_table(res, eta, tau, beta):
    """{stream: {scene_key: FP}} for one grid cell, restricted to scenes where
    both the stream and the GT control produced an evaluable number."""
    out: dict[str, dict[str, float]] = {}
    for sc in res["scenes"]:
        key = f"{sc['dataset']}/{sc['scene']}"
        for c in sc["cells"]:
            if (c["eta"], c["tau"], c["beta"]) == (eta, tau, beta) and c["n_eval"] > 0:
                out.setdefault(c["stream"], {})[key] = c["FP"]
    return out


def paired(tab, stream, only=None):
    """Aligned (model, gt) FP vectors over the scenes both could evaluate.
    `only` restricts to a set of scene keys (used for the Tier A requirement)."""
    a, b = tab.get(stream, {}), tab.get(GT, {})
    ks = set(a) & set(b)
    if only is not None:
        ks &= set(only)
    ks = sorted(ks)
    return ks, np.array([a[k] for k in ks]), np.array([b[k] for k in ks])


def tier_keys(res, tier="A"):
    """Scene keys of a given tier, in the SAME 'dataset/scene' form `paired`
    returns -- the two must match or the tier restriction silently matches
    nothing."""
    return {f"{sc['dataset']}/{sc['scene']}" for sc in res["scenes"]
            if sc.get("tier") == tier}


def streams_of(res):
    return [s for s in res["config"]["streams"] if s != GT]


# --------------------------------------------------------------------------- #
# the decision rule (Sec. 2) -- fixed in advance, evaluated mechanically
# --------------------------------------------------------------------------- #
def stream_stats(res, stream, eta, tau, beta, only=None):
    ks, m, g = paired(cell_table(res, eta, tau, beta), stream, only)
    if len(ks) < 2:
        return None
    d = m - g
    lo, hi = paired_bootstrap(d)
    return dict(stream=stream, n_scenes=len(ks), scenes=ks, FP_model=float(m.mean()),
                FP_GT=float(g.mean()), diff=float(d.mean()), ci_lo=lo, ci_hi=hi,
                ratio=float(m.mean() / g.mean()) if g.mean() > 0 else float("inf"),
                p=wilcoxon(d), effect=rank_biserial(d))


def sweep_complete(res):
    """Did the run actually cover the pre-registered (eta, tau, beta) grid?
    Sec. 7 makes the sweep mandatory, so a --no-grid run must not be allowed to
    claim 'sign-stable across the full sweep' on the strength of one cell."""
    c = res["config"]
    return (sorted(c["etas"]) == sorted(FM.ETA_GRID)
            and sorted(c["taus"]) == sorted(FM.TAU_GRID)
            and sorted(c["betas"]) == sorted(FM.BETA_GRID))


def sign_stable(res, stream, only=None):
    """Does the paired difference keep one sign across the ENTIRE sweep?

    A cell whose mean difference is exactly 0 is not a sign *flip*, so zeros are
    dropped rather than counted as a third sign. A cell that could not be
    measured is reported via `missing` -- the caller must not treat an unmeasured
    grid as a stable one.
    """
    signs, measured, missing = set(), 0, 0
    for eta in res["config"]["etas"]:
        for tau in res["config"]["taus"]:
            for beta in res["config"]["betas"]:
                ks, m, g = paired(cell_table(res, eta, tau, beta), stream, only)
                if len(ks) < 2:
                    missing += 1
                    continue
                measured += 1
                sg = int(np.sign((m - g).mean()))
                if sg != 0:
                    signs.add(sg)
    return (len(signs) <= 1), measured, missing, signs


PREREG_TAG = "prereg-constants"


def prereg_constants(out):
    """Thresholds as recorded at pre-registration time, parsed back from
    decision.md. `decide()` compares these against the live module constants so
    that editing a constant after the fact cannot silently change the verdict
    while the written record still shows the original numbers."""
    p = os.path.join(out, "decision.md")
    if not os.path.exists(p):
        return None
    m = re.search(rf"<!-- {PREREG_TAG} (.*?) -->", open(p).read())
    if not m:
        return None
    return {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split())}


def decide(res, gate_rows=None, prereg=None):
    """Evaluate the Sec. 2 rule. Returns (outcome, justification, rows, warnings)."""
    cfg = res["config"]["centre"]
    warn = []

    live = dict(GO_MIN_PP=GO_MIN_PP, GO_MIN_RATIO=GO_MIN_RATIO, NOGO_GT_MAX=NOGO_GT_MAX)
    if prereg:
        drift = {k: (prereg[k], v) for k, v in live.items()
                 if k in prereg and abs(prereg[k] - v) > 1e-12}
        if drift:
            raise SystemExit(
                "Thresholds changed after pre-registration: "
                + "; ".join(f"{k} was {a} now {b}" for k, (a, b) in drift.items())
                + "\nSec. 2 fixes these in advance. Restore them, or start a new "
                  "pre-registration in a fresh output directory.")

    # full-sample stats for the record ...
    rows = [r for r in (stream_stats(res, s, cfg["eta"], cfg["tau"], cfg["beta"])
                        for s in streams_of(res)) if r]
    # ... and Tier A only, which is what the GO decision is allowed to use.
    ta = tier_keys(res, "A")
    if not ta:
        warn.append("No scene is labelled tier A; GO is unreachable (Sec. 2 requires "
                    "the effect on the synthetic set).")
    if rows:
        for r, ap in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = float(ap)

    if gate_rows:
        usable = [g for g in gate_rows if g.get("usable")]
        absent = [g for g in gate_rows if g.get("status") == "absent"]
        if not usable and len(absent) < len(gate_rows):
            return "NO-GO", ("Every dataset that had data on disk failed the Sec. 4 "
                             "measurability gate: the boundaries needed to answer the "
                             "question are not present in any available data."), rows, warn
        if absent:
            warn.append(f"{len(absent)} dataset(s) had no data on disk and were "
                        f"neither gated nor measured: "
                        f"{', '.join(g['dataset'] for g in absent)}.")
    if not rows:
        return "NO-GO", "No scene produced an evaluable paired measurement.", rows, warn

    if all(r["FP_GT"] >= NOGO_GT_MAX for r in rows):
        return "NO-GO", (f"The GT control is at or above {NOGO_GT_MAX:.0%} flying pixels "
                         f"for every stream, so the ground truth is too contaminated to "
                         f"measure a model against."), rows, warn

    complete = sweep_complete(res)
    if not complete:
        warn.append("The run did not cover the pre-registered (eta, tau, beta) grid, "
                    "so sign stability is unverified and GO is withheld (Sec. 7).")

    # cross-stream balance: Sec. 7 wants the streams compared on one scene set
    sets = {r["stream"]: set(r["scenes"]) for r in rows}
    if len({frozenset(v) for v in sets.values()}) > 1:
        warn.append("Streams were paired on different scene sets: "
                    + "; ".join(f"{k} n={len(v)}" for k, v in sets.items())
                    + ". Cross-stream comparisons are not like-for-like.")

    go = []
    for r in rows:
        st, meas, missing, _ = sign_stable(res, r["stream"])
        r["sign_stable"] = bool(st)
        r["sweep_cells"] = f"{meas} measured / {missing} unmeasurable"
        if missing:
            warn.append(f"{r['stream']}: {missing} sweep cell(s) had too few scenes to "
                        f"evaluate; those cannot support a stability claim.")
        ra = stream_stats(res, r["stream"], cfg["eta"], cfg["tau"], cfg["beta"], only=ta)
        r["FP_model_tierA"] = ra["FP_model"] if ra else float("nan")
        r["n_scenes_tierA"] = ra["n_scenes"] if ra else 0
        if ra is None:
            continue
        if ra["FP_GT"] >= NOGO_GT_MAX:      # per-stream disqualification, not global
            warn.append(f"{r['stream']}: its own GT control is at {ra['FP_GT']:.1%} "
                        f"flying pixels (>= {NOGO_GT_MAX:.0%}); disqualified from GO.")
            continue
        if (ra["diff"] >= GO_MIN_PP - 1e-9 and ra["ratio"] >= GO_MIN_RATIO - 1e-9
                and np.isfinite(ra["ci_lo"]) and ra["ci_lo"] > 0
                and st and missing == 0 and complete):
            go.append(r["stream"])
    if go:
        return "GO", (f"{', '.join(go)} clears every pre-registered threshold on the "
                      f"Tier A synthetic set with a complete, sign-stable sweep."), rows, warn

    excl = [r for r in rows if np.isfinite(r["ci_lo"])
            and (r["ci_lo"] > 0 or r["ci_hi"] < 0)]
    if excl:
        neg = [r["stream"] for r in excl if r["ci_hi"] < 0]
        bits = []
        if neg:
            bits.append(f"{', '.join(neg)} differs from the GT control in the OPPOSITE "
                        f"direction (fewer flying pixels than the ground truth)")
        pos = [r for r in excl if r["ci_lo"] > 0]
        if pos:
            why = ("below the GO thresholds" if all(r["sign_stable"] for r in pos)
                   else "sign-unstable somewhere in the (eta, tau, beta) sweep")
            bits.append(f"{', '.join(r['stream'] for r in pos)} shows a positive effect "
                        f"whose CI excludes 0, but it is {why}")
        return "WEAK", ("; ".join(bits) + ". Do not build a diagnosis paper on this."), rows, warn
    return "NO-GO", "No stream's 95% bootstrap CI on the paired difference excludes 0.", rows, warn


# --------------------------------------------------------------------------- #
# deliverables
# --------------------------------------------------------------------------- #
def preregister(out, force=False):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "decision.md")
    if os.path.exists(p) and not force:
        raise SystemExit(f"{p} already exists. Pre-registration is write-once by "
                         f"design -- thresholds must not move after seeing results.")
    open(p, "w").write(f"""# Premise check - decision record

<!-- {PREREG_TAG} GO_MIN_PP={GO_MIN_PP} GO_MIN_RATIO={GO_MIN_RATIO} NOGO_GT_MAX={NOGO_GT_MAX} -->

**Pre-registered:** {date.today().isoformat()}
**Protocol:** `notes/premise_check.md` (Sec. 2 decision rule, copied here before any run)

## Thresholds, fixed in advance

| Outcome | Condition |
|---|---|
| GO | `FP_model - FP_GT >= {GO_MIN_PP:.2f}` (5 pp) **and** `FP_model >= {GO_MIN_RATIO}x FP_GT`, 95% bootstrap CI on the paired difference excludes 0, for >=1 stream on Tier A, **and** the sign is stable across the full (eta, tau, beta) sweep |
| WEAK | CI excludes 0 but the effect is below the GO thresholds, or the sign flips anywhere in the sweep |
| NO-GO | CI includes 0, **or** `FP_GT >= {NOGO_GT_MAX:.0%}`, **or** the Sec. 4 gate fails on every dataset |

Centre of the sweep: eta={FM.ETA0}, tau={FM.TAU0}, beta={FM.BETA0}.
Sweep: eta in {list(FM.ETA_GRID)}, tau in {list(FM.TAU_GRID)}, beta in {list(FM.BETA_GRID)}.
Unit of analysis: the scene. Test: Wilcoxon signed-rank, paired bootstrap CI
({N_BOOT} resamples), Holm-corrected across streams.

A result landing near a boundary is WEAK. Thresholds are not to be adjusted.

## Outcome

_Not yet run. `python src/analyze.py --decide` fills this in._
""")
    print("wrote", p)


def write_decision(out, res, outcome, why, rows, warnings=()):
    p = os.path.join(out, "decision.md")
    txt = open(p).read() if os.path.exists(p) else ""
    head = txt.split("## Outcome")[0]
    if not head.strip() or PREREG_TAG not in head:
        raise SystemExit(f"{p} is not a valid pre-registration (no {PREREG_TAG} "
                         f"marker). Run --preregister first.")
    lines = [head, "## Outcome\n", f"**{outcome}** - measured {date.today().isoformat()}\n",
             f"{why}\n",
             "| stream | scenes | tierA scenes | FP_model | FP_GT | diff (pp) | ratio | "
             "95% CI (pp) | p (Holm) | sign stable | sweep |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['stream']} | {r['n_scenes']} | {r.get('n_scenes_tierA', 0)} | "
            f"{r['FP_model']:.4f} | {r['FP_GT']:.4f} | "
            f"{100*r['diff']:+.2f} | {r['ratio']:.2f} | "
            f"[{100*r['ci_lo']:+.2f}, {100*r['ci_hi']:+.2f}] | "
            f"{r.get('p_holm', float('nan')):.4g} | {r.get('sign_stable', '?')} | "
            f"{r.get('sweep_cells', '?')} |")
    if warnings:
        lines.append("\n### Caveats recorded at decision time\n")
        lines += [f"- {w}" for w in warnings]
    lines.append("\nGenerated by `src/analyze.py --decide`. The threshold table above is "
                 "the one written at pre-registration; `decide()` refuses to run if the "
                 "live constants no longer match it.\n")
    open(p, "w").write("\n".join(lines))
    print("wrote", p)


def write_tables(out, res, rows):
    p = os.path.join(out, "table_main.csv")
    cols = ["stream", "n_scenes", "n_scenes_tierA", "FP_model", "FP_model_tierA",
            "FP_GT", "diff", "ratio", "ci_lo", "ci_hi", "p", "p_holm", "effect",
            "sign_stable", "sweep_cells"]
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", p)

    q = os.path.join(out, "sensitivity.csv")
    with open(q, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["stream", "eta", "tau", "beta", "n_scenes",
                     "FP_model", "FP_GT", "diff", "ci_lo", "ci_hi"])
        for s in streams_of(res):
            for eta in res["config"]["etas"]:
                for tau in res["config"]["taus"]:
                    for beta in res["config"]["betas"]:
                        ks, m, g = paired(cell_table(res, eta, tau, beta), s)
                        if len(ks) < 2:
                            continue
                        d = m - g
                        lo, hi = paired_bootstrap(d)
                        wr.writerow([s, eta, tau, beta, len(ks), f"{m.mean():.6f}",
                                     f"{g.mean():.6f}", f"{d.mean():.6f}",
                                     f"{lo:.6f}", f"{hi:.6f}"])
    print("wrote", q)


# --------------------------------------------------------------------------- #
# figures (Sec. 8.5) -- LOOK AT THESE before trusting any number
# --------------------------------------------------------------------------- #
def figures(out, res, n_scenes=5, n_per=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import data as D
    import infer as INF

    fd = os.path.join(out, "figs")
    os.makedirs(fd, exist_ok=True)
    cfg = res["config"]
    made = 0
    for sc in res["scenes"]:
        if made >= n_scenes:
            break
        ds, scene = sc["dataset"], sc["scene"]
        for v in D.views(ds, scene, cfg["n_views"])[:n_per]:
            try:
                vd = D.load(v, with_rgb=True)
            except Exception:
                continue
            preds = {s: INF.load_pred(out, s, v.key, vd.depth.shape)
                     for s in streams_of(res)}
            preds = {k: p for k, p in preds.items() if p is not None}
            if not preds:
                continue
            stream, pred = next(iter(preds.items()))
            r, mk = FM.fp_view(vd.depth, vd.valid, pred, eta=cfg["centre"]["eta"],
                               tau=cfg["centre"]["tau"], beta=cfg["centre"]["beta"],
                               w=cfg["w"], return_masks=True)
            if mk is None:
                continue
            pa = pred * (r["scale"] if np.isfinite(r["scale"]) else 1.0)

            fig, ax = plt.subplots(2, 3, figsize=(16, 8))
            ax = ax.ravel()
            ax[0].imshow(vd.rgb); ax[0].set_title(f"RGB  {ds}/{scene} #{v.idx}")
            vmin, vmax = np.nanpercentile(vd.depth, [2, 98])
            ax[1].imshow(vd.depth, cmap="turbo", vmin=vmin, vmax=vmax); ax[1].set_title("GT depth")
            ax[2].imshow(pa, cmap="turbo", vmin=vmin, vmax=vmax)
            ax[2].set_title(f"pred depth ({stream}, aligned)")
            ax[3].imshow(mk["B"], cmap="gray"); ax[3].set_title("boundary set B")
            ov = np.asarray(vd.rgb, float) / 255.0
            ov[mk["FP"]] = [1, 0, 0]
            ax[4].imshow(ov); ax[4].set_title(f"FP mask (FP={r['FP']:.3f}, n={r['n_eval']})")

            # silhouette point cloud: unproject a crop around the densest boundary
            # column and look down the y axis -- flying pixels appear as a bridge
            # of points strung across the empty space between the two surfaces.
            K = vd.K
            ys, xs = np.nonzero(mk["B"])
            if len(xs):
                cx = int(np.median(xs)); cy = int(np.median(ys))
                sl = (slice(max(0, cy - 40), cy + 40), slice(max(0, cx - 40), cx + 40))
                uu, vv = np.meshgrid(np.arange(*sl[1].indices(vd.depth.shape[1])),
                                     np.arange(*sl[0].indices(vd.depth.shape[0])))
                for lbl, dm, col in (("GT", vd.depth[sl], "tab:blue"),
                                     (stream, pa[sl], "tab:red")):
                    m = np.isfinite(dm) & (dm > 0)
                    X = (uu[m] - K[0, 2]) / K[0, 0] * dm[m]
                    ax[5].scatter(X, dm[m], s=1, alpha=.35, label=lbl, c=col)
                ax[5].set_xlabel("X (m)"); ax[5].set_ylabel("Z (m)")
                ax[5].set_title("silhouette, viewed from above"); ax[5].legend(markerscale=6)
            for a in ax[:5]:
                a.axis("off")
            fig.tight_layout()
            fp = os.path.join(fd, f"{ds}_{scene.replace('/', '_')}_{v.idx:06d}.png")
            fig.savefig(fp, dpi=110); plt.close(fig)
            print("wrote", fp)
            made += 1
            break
    if not made:
        print("no figures: no cached predictions matched the measured scenes")


def _main():
    ap = argparse.ArgumentParser(description="premise-check statistics and figures")
    ap.add_argument("--out", default="results/premise_check")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow re-writing decision.md")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--n-scenes", type=int, default=5)
    a = ap.parse_args()

    if a.preregister:
        preregister(a.out, a.force)
        return 0
    rp = os.path.join(a.out, "results.json")
    if not os.path.exists(rp):
        raise SystemExit(f"{rp} not found -- run src/measure.py first.")
    res = json.load(open(rp))
    if a.figs:
        figures(a.out, res, a.n_scenes)
    if a.decide or not a.figs:
        gp = os.path.join(a.out, "gate.csv")
        gate_rows = None
        if os.path.exists(gp):
            gate_rows = [dict(r, usable=r["usable"] in ("True", "true", True))
                         for r in csv.DictReader(open(gp))]
        outcome, why, rows, warns = decide(res, gate_rows, prereg_constants(a.out))
        write_decision(a.out, res, outcome, why, rows, warns)
        write_tables(a.out, res, rows)
        print(f"\n=== {outcome} ===\n{why}")
        for w in warns:
            print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
