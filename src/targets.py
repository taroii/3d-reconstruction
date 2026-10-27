r"""
Target audit — is the models' own supervision blended at occlusion boundaries?

Phase 1B's largest effect by far is the TARGET one: fit a predictor to targets
that were area-averaged at boundaries and it reproduces the blend, which sits in
the void (+0.20 to +0.30, `results/phase1b_v2/1b_decomposition.md`). That is a
SUFFICIENCY result about a synthetic construction. It says nothing about whether
the data VGGT, π³, MASt3R and DUSt3R actually trained on is blended that way.

This module measures that, with no model and no inference: run the Phase 0
ground-truth control on the TRAINING sets we hold, and ask how often the training
depth itself puts a value in the void between the two surfaces it separates.


THE STATISTIC, AND WHY IT IS NOT THE FLYING-PIXEL RATE

The obvious move is to report the Phase 0 control rate per training set. It is
reported, but it CANNOT be the headline, and the self-test is what showed why:
the metric only scores pixels whose window holds exactly two surfaces, and a
blended edge is itself a third cluster. Blend the boundary and those pixels leave
`B_eval` entirely -- the rate goes DOWN, not up, because the evidence removes
itself. The same blindness applies to every Phase 0 and Phase 1A number, and is
worth stating in any write-up.

So the headline here is measured over ALL boundary pixels: cluster the window at
the fine threshold, find the cluster holding the pixel's own depth, and ask
whether it lies strictly BETWEEN the nearest and furthest clusters. That is
"this training target sits between two surfaces", which is the thing a model
would be learning, and it is blind to nothing.

WHAT THIS CAN AND CANNOT SETTLE

The RATE is unambiguous and is what the pre-registered reading rule uses: a
training target that lands in the void is a target that teaches void-landing,
whatever put it there.

The MECHANISM is not cleanly separable here, and the shape statistics below are
descriptive only. A one-pixel antialiased edge and a genuine one-pixel-wide
surface at intermediate depth can look alike in a 7x7 window: both leave an
intermediate cluster. `support` is the most useful hint -- a one-pixel blend
leaves a lone sample, a real surface brings its neighbours -- but a straight
antialiased edge also brings neighbours, at slightly different coverages. Nothing
beyond the rate is claimed. **[ASSUMED]** nothing else.

    python src/targets.py --selftest
    python src/targets.py --preregister --out results/target_audit
    python src/targets.py --run --out results/target_audit
"""
from __future__ import annotations

import os
import csv
import json
import time
import argparse
import collections

import numpy as np

import data as D
import fpmetrics as FM

# Training sets we hold on disk, and who trained on them (src/data.py SPECS).
TRAINING = ("hypersim", "tartanair", "pointodyssey")
# Held out of every mix; the Phase 0 controls that anchor the comparison.
COMPARATORS = ("middlebury", "infinigen", "spring", "sintel", "ibims", "eth3d")

FINE_ETA = 0.01          # §11's finer clustering threshold
DENS_MAX = 0.25          # §11's silhouette cut: local boundary density
DELTA_MIN = 0.05         # metres; Phase 0's minimum absolute gap
N_SCENES, N_VIEWS = 40, 2
MAX_PX = 300_000         # boundary pixels kept per view, sampled at a fixed seed

# Pre-registered reading rule. The margin is Phase 0's 5 points, reused so the
# answer is on the same scale as every other number in this project.
BLENDED, SUBSTANTIAL = 0.05, 0.02


def box_density(mask, r=3):
    """Share of pixels within a (2r+1)^2 box that are boundary pixels."""
    c = np.cumsum(np.cumsum(np.pad(mask.astype(np.float64), ((1, 0), (1, 0))), 0), 1)
    H, W = mask.shape
    y, x = np.mgrid[0:H, 0:W]
    y0, y1 = np.clip(y - r, 0, H), np.clip(y + r + 1, 0, H)
    x0, x1 = np.clip(x - r, 0, W), np.clip(x + r + 1, 0, W)
    tot = c[y1, x1] - c[y0, x1] - c[y1, x0] + c[y0, x0]
    return tot / ((y1 - y0) * (x1 - x0))


def between(depth, valid, pix, w=3, eta=FINE_ETA):
    """Per boundary pixel: is its own depth strictly between two other surfaces?

    Cluster the window at the fine threshold, locate the cluster holding the
    pixel's own value, and compare its rank with the nearest and furthest
    clusters. Returns, per pixel:

      k          number of clusters in the window
      is_between own cluster is neither the nearest nor the furthest
      t          where the value sits between the clusters either side of it
                 (0 on the nearer, 1 on the further)
      support    how many samples share the pixel's own cluster, itself
                 excluded. 0 is a lone intermediate sample, the signature of a
                 one-pixel blend; a genuine surface brings its neighbours.
    """
    N, m = len(pix), 2 * w + 1
    if N == 0:
        return (np.zeros(0, int),) + (np.zeros(0, bool),) + (np.zeros(0),) * 2
    H, W = depth.shape
    r = np.clip(pix[:, 0][:, None] + np.arange(-w, w + 1)[None, :], 0, H - 1)
    c = np.clip(pix[:, 1][:, None] + np.arange(-w, w + 1)[None, :], 0, W - 1)
    rr = r[:, :, None].repeat(m, 2).reshape(N, -1)
    cc = c[:, None, :].repeat(m, 1).reshape(N, -1)
    win = depth[rr, cc].astype(np.float64)
    ok = valid[rr, cc] & np.isfinite(win) & (win > 0)
    win = np.where(ok, win, np.inf)
    win.sort(1)
    nval = ok.sum(1)
    a, b = win[:, :-1], win[:, 1:]
    both = np.isfinite(a) & np.isfinite(b)
    gap = np.divide(b - a, a, out=np.zeros_like(a), where=both & (a > 0))
    split = both & (gap > eta)
    cid = np.concatenate([np.zeros((N, 1), int), np.cumsum(split, 1)], 1)   # per sorted sample
    k = np.where(nval > 0, split.sum(1) + 1, 0)

    own = depth[pix[:, 0], pix[:, 1]].astype(np.float64)
    # index of the pixel's own value among its sorted window samples
    j = np.clip(np.argmin(np.abs(win - own[:, None]), 1), 0, win.shape[1] - 1)
    mine = cid[np.arange(N), j]
    is_between = (k >= 3) & (mine > 0) & (mine < k - 1)

    # Vectorised: a cluster is contiguous in the sorted window, so its start and
    # size give its median, and the samples either side of it are the clusters
    # below and above. A per-pixel loop here is unusable at a few million
    # boundary pixels per view.
    idx = np.arange(N)
    here = cid == mine[:, None]
    start = np.argmax(here, 1)
    cnt = here.sum(1)
    last = win.shape[1] - 1
    med = 0.5 * (win[idx, np.clip(start + (cnt - 1) // 2, 0, last)]
                 + win[idx, np.clip(start + cnt // 2, 0, last)])
    below = win[idx, np.clip(start - 1, 0, last)]
    above = win[idx, np.clip(start + cnt, 0, last)]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(is_between & (above > below), (med - below) / (above - below), np.nan)
    return k, is_between, t, np.where(is_between, cnt - 1, 0)


def audit_view(vd, *, eta=FM.ETA0, tau=FM.TAU0, beta=FM.BETA0, w=3,
               delta_min=DELTA_MIN, fine_eta=FINE_ETA, dens_max=DENS_MAX,
               max_px=MAX_PX, seed=0):
    """Per-view ground-truth control, in the three §11 variants, plus shapes."""
    gt = vd.depth.astype(np.float64)
    valid = vd.valid & np.isfinite(gt) & (gt > 0)
    B = FM.boundary_set(gt, valid, eta)
    pix = np.argwhere(B)
    if len(pix) == 0:
        return None
    # Each pixel carries a 7x7 window, so a view with 5 M boundary pixels (this
    # happens on Infinigen) would allocate ~2 GB per gather. Subsample at a fixed
    # seed instead; the statistic is a rate, so a uniform sample is unbiased.
    if len(pix) > max_px:
        pix = pix[np.random.default_rng(seed).choice(len(pix), max_px, replace=False)]
    sup = FM.depth_support(gt, valid, pix, w=w, eta=eta)
    K, d1, d2 = sup["K"], sup["d1"], sup["d2"]
    delta = d2 - d1
    base = (K == 2) & np.isfinite(delta) & (delta >= delta_min)

    # The control is the dataset's own PRE-MASK depth, aligned on the interior —
    # identical to Phase 0, so these rates sit on the same scale as the model ones.
    raw = vd.depth_raw.astype(np.float64)
    I = FM.interior(valid, FM.dilate(B, tau))
    s = FM.align_scale(raw, gt, I, "median")
    if not np.isfinite(s):
        return None
    q_all = raw[pix[:, 0], pix[:, 1]] * s

    fine_K = FM.depth_support(gt, valid, pix, w=w, eta=fine_eta)["K"]
    dens = box_density(B)[pix[:, 0], pix[:, 1]]
    masks = {"all": base, "fine": base & (fine_K == 2),
             "silhouette": base & (dens < dens_max)}

    out = {"K_hist": collections.Counter(K[K > 0].tolist())}
    for name, mk in masks.items():
        if mk.sum() < 20:
            continue
        dd1, dd2, dl, q = d1[mk], d2[mk], delta[mk], q_all[mk]
        lo, hi = dd1 + beta * dl, dd2 - beta * dl
        with np.errstate(invalid="ignore"):
            fp = (q >= lo) & (q <= hi)
        out[name] = dict(n=int(fp.size), fp=int(fp.sum()))

    # The headline, over EVERY boundary pixel: blend-sensitive, and not subject
    # to the two-surface restriction that a blend itself breaks.
    _, is_between, t, support = between(gt, valid, pix, w=w, eta=fine_eta)
    out["between"] = dict(n=int(is_between.size), fp=int(is_between.sum()))
    if is_between.any():
        out["shape"] = dict(t=t[is_between], support=support[is_between],
                            third=int((fine_K[is_between] >= 3).sum()))
    return out


def run(out, datasets, n_scenes=N_SCENES, n_views=N_VIEWS, delta_min=DELTA_MIN):
    if not os.path.exists(os.path.join(out, "audit_prereg.md")):
        raise SystemExit(f"No pre-registration at {out}/audit_prereg.md. Write it "
                         f"first: python src/targets.py --preregister --out {out}")
    acc = {ds: collections.defaultdict(lambda: dict(n=0, fp=0)) for ds in datasets}
    shapes = {ds: collections.defaultdict(list) for ds in datasets}
    khist = {ds: collections.Counter() for ds in datasets}
    used = collections.Counter()
    skipped = collections.Counter()
    for ds in datasets:
        t0 = time.time()
        for sc in D.scenes(ds)[:n_scenes]:
            for v in D.views(ds, sc, n_views):
                try:
                    vd = D.load(v)
                except Exception as e:                       # noqa: BLE001
                    skipped[f"{ds}:load:{type(e).__name__}"] += 1
                    continue
                r = audit_view(vd, delta_min=delta_min)
                if r is None:
                    skipped[f"{ds}:unusable"] += 1
                    continue
                khist[ds] += r.pop("K_hist")
                sh = r.pop("shape", None)
                if sh:
                    for k in ("t", "support"):
                        shapes[ds][k].append(sh[k])
                    shapes[ds]["third"].append(sh["third"])
                for name, a in r.items():
                    acc[ds][name]["n"] += a["n"]
                    acc[ds][name]["fp"] += a["fp"]
                used[ds] += 1
        n = acc[ds]["all"]["n"]
        print(f"  {ds}: {used[ds]} views, {n} evaluated boundary px "
              f"({time.time()-t0:.0f}s)", flush=True)
    write(out, acc, shapes, khist, used, skipped, datasets)
    return acc


def verdict(rate):
    if not np.isfinite(rate):
        return "NOT MEASURED"
    if rate >= BLENDED:
        return "BLENDED"
    if rate >= SUBSTANTIAL:
        return "SUBSTANTIAL"
    return "NEGLIGIBLE"


def write(out, acc, shapes, khist, used, skipped, datasets):
    os.makedirs(out, exist_ok=True)
    rows = []
    for ds in datasets:
        a = acc[ds]
        row = dict(dataset=ds, trains=",".join(D.SPECS[ds]["contam"]) or "none",
                   views=used[ds], n_eval=a["all"]["n"], n_boundary=0)
        for name in ("between", "all", "fine", "silhouette"):
            row[f"FP_{name}"] = (a[name]["fp"] / a[name]["n"]) if a[name]["n"] else float("nan")
        row["n_boundary"] = a["between"]["n"]
        s = shapes[ds]
        t = np.concatenate(s["t"]) if s["t"] else np.zeros(0)
        sup = np.concatenate(s["support"]) if s["support"] else np.zeros(0)

        row.update(
            t_median=float(np.median(t)) if t.size else float("nan"),
            t_iqr_lo=float(np.percentile(t, 25)) if t.size else float("nan"),
            t_iqr_hi=float(np.percentile(t, 75)) if t.size else float("nan"),
            frac_midvoid=float(np.mean(np.abs(t - 0.5) < 0.1)) if t.size else float("nan"),
            support_median=float(np.median(sup)) if sup.size else float("nan"),
            frac_isolated=float(np.mean(sup <= 1)) if sup.size else float("nan"),
            frac_third_surface=(sum(s["third"]) / t.size) if t.size else float("nan"),
            verdict=verdict(row["FP_between"]))
        rows.append(row)
    p = os.path.join(out, "audit.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump({ds: {str(k): v for k, v in sorted(khist[ds].items())} for ds in datasets},
              open(os.path.join(out, "audit_k_histogram.json"), "w"), indent=1)

    md = ["# Target audit — is the training supervision blended at boundaries?", "",
          f"Generated {time.strftime('%F %T')} under the rule in `audit_prereg.md`. "
          f"No model and no inference: this is the Phase 0 ground-truth control, run "
          f"on the sets the models trained on.", "",
          "| dataset | trains | views | boundary px | **between surfaces** | FP all | FP fine | FP silhouette | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['trains']} | {r['views']} | {r['n_boundary']} | "
                  f"**{r['FP_between']:.4f}** | {r['FP_all']:.4f} | {r['FP_fine']:.4f} | "
                  f"{r['FP_silhouette']:.4f} | {r['verdict']} |")
    md += ["", "`between surfaces` is the headline: the share of boundary pixels whose own "
           "depth sits strictly between two other surfaces in its window. The `FP` columns "
           "are the Phase 0 control on the two-surface pixels only, for comparability — "
           "and they UNDERSTATE blending, because a blend is itself a third cluster and "
           "leaves that pixel set."]
    md += ["", "## Shape of the flying ground-truth values (descriptive only)", "",
           "`t` is where the value sits between the surfaces (0 near, 1 far). "
           "`support` counts other samples in the window in the same cluster — a lone "
           "sample suggests a one-pixel blend, a large cluster a real surface, and the "
           "two are separated only by degree. `third surface` is the share whose window "
           "holds three or more clusters at the finer threshold.", "",
           "| dataset | t median | t IQR | mid-void | support median | isolated | third surface |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['t_median']:.3f} | "
                  f"[{r['t_iqr_lo']:.2f}, {r['t_iqr_hi']:.2f}] | {r['frac_midvoid']:.3f} | "
                  f"{r['support_median']:.1f} | {r['frac_isolated']:.3f} | "
                  f"{r['frac_third_surface']:.3f} |")
    train_rows = [r for r in rows if r["dataset"] in TRAINING]
    worst = max((r["FP_between"] for r in train_rows), default=float("nan"))
    md += ["", "## Reading", "",
           f"Highest training-set rate: **{worst:.4f}** ({verdict(worst)}).", ""]
    if np.isfinite(worst):
        md += [f"**[ASSUMED]** that a blended fraction contributes about in proportion: "
               f"1B blended *every* boundary pixel and moved the rate by 0.20–0.30, so a "
               f"training set blended on a fraction `f` of its boundary pixels would, on a "
               f"linear reading, contribute roughly `f × 0.25`. At {worst:.4f} that is "
               f"≈ {0.25 * worst:.4f} against the 0.19–0.30 the models actually show. "
               f"Linearity is an assumption and is not tested here.", ""]
    md += ["**What this does not settle.** Only the sets on this disk are audited, and "
           "each model trained on many more (MASt3R and π³ each list an undisclosed "
           "internal set). A rate measured on the boundaries this metric can see says "
           "nothing about sets it cannot. Mechanism is not separated: see the module "
           "docstring.", ""]
    if skipped:
        md += [f"Views skipped: {dict(skipped)}", ""]
    open(os.path.join(out, "audit.md"), "w").write("\n".join(md) + "\n")
    print("wrote", p, "and audit.md")
    print("\n".join(md[4:6 + len(rows)]))


PREREG = """# Target audit — pre-registration

Recorded {date}, BEFORE the measurement. `--run` refuses to start without this file.

## The question

Phase 1B found that blended targets produce flying pixels (+0.20 to +0.30 when
every boundary pixel is blended). That is sufficiency on synthetic data. This
asks whether the supervision the models actually trained on is blended at
boundaries, using the Phase 0 ground-truth control and no model at all.

## What is measured

**Headline:** over every boundary pixel (relative jump > η={eta}), the share whose own
depth lies strictly between two other surfaces in its 7×7 window, clustered at {fine}.
This is deliberately NOT the flying-pixel rate: the metric scores only pixels whose
window holds exactly two surfaces, and a blended edge is a third cluster, so blending
removes its own evidence from that pixel set.

**Also reported, for comparability:** `gt_raw` — the dataset's own pre-mask depth,
aligned on the interior — scored by the Phase 0 metric (η={eta}, τ={tau}, β={beta},
minimum gap {delta} m) in the three nested variants of §11.

At most {max_px} boundary pixels per view, uniformly subsampled at a fixed seed
where a view has more (Infinigen views carry millions). The statistic is a rate,
so this costs precision, not correctness.

Training sets held on this disk: {training}. Comparators, in no training mix:
{comparators}. {scenes} scenes × {views} views each.

## Reading rule, fixed now

Per dataset, on the `between surfaces` share (all boundary pixels):

| verdict | condition |
|---|---|
| BLENDED | share ≥ {blended} — the same 5-point margin Phase 0 used |
| SUBSTANTIAL | {substantial} ≤ share < {blended} |
| NEGLIGIBLE | share < {substantial} |

A BLENDED training set makes the 1B target effect a live explanation for the real
models. NEGLIGIBLE on every set we hold makes it a sufficiency result that does
not transfer, and the weight shifts back to the objective and predictor effects.

## Stated before seeing the numbers

- Not predicted either way. Phase 0 measured near-zero controls on held-out
  synthetic sets (Middlebury 0.00028, Hypersim 0.0067), which is evidence about
  those sets, not about a training mix as a whole.
- The shape statistics (where in the void, how isolated the sample, whether a
  third surface is present) are DESCRIPTIVE. No verdict depends on them, because
  a one-pixel antialiased edge and a genuine one-pixel surface are not reliably
  separable in a 7×7 window.
- Coverage is partial by construction: three training sets out of the many each
  model used, and two of the four models list undisclosed internal data.
- Per the standing rule, no fix is attempted on the strength of this.
"""


def preregister(out, force=False):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "audit_prereg.md")
    if os.path.exists(p) and not force:
        raise SystemExit(f"{p} already exists. Pre-registration is write-once.")
    open(p, "w").write(PREREG.format(
        date=time.strftime("%F %T"), eta=FM.ETA0, tau=FM.TAU0, beta=FM.BETA0,
        delta=DELTA_MIN, fine=FINE_ETA, dens=DENS_MAX, scenes=N_SCENES, views=N_VIEWS,
        training=", ".join(TRAINING), comparators=", ".join(COMPARATORS),
        blended=BLENDED, substantial=SUBSTANTIAL, max_px=MAX_PX))
    print("wrote", p)
    return 0


def w_default():
    return 3


def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    class V:
        def __init__(self, depth, raw=None):
            self.depth = depth
            self.depth_raw = depth if raw is None else raw
            self.valid = np.ones_like(depth, bool)

    H, W = 40, 40
    step = np.full((H, W), 2.0); step[:, 20:] = 3.0

    r = audit_view(V(step))
    check("a clean step has no between-surfaces ground truth",
          r["between"]["fp"] == 0, f"{r['between']['fp']}/{r['between']['n']}")
    check("...and no flying ground truth either", r["all"]["fp"] == 0)

    blend = step.copy(); blend[:, 19] = 2.0 + 1.0 * np.linspace(0.15, 0.85, H)
    rb = audit_view(V(blend))
    fb = rb["between"]["fp"] / rb["between"]["n"]
    check("a blended edge column IS caught by the between-surfaces share",
          fb > 0.2, f"{fb:.3f}")
    check("...with values spread across the gap, tracking coverage",
          np.ptp(rb["shape"]["t"]) > 0.3, f"ptp={np.ptp(rb['shape']['t']):.2f}")
    # NOT "isolated": where coverage varies slowly along an edge, neighbouring
    # blended pixels fall within the fine threshold of each other and cluster
    # together. Support separates a blend from a surface only by DEGREE, which
    # is why no verdict rests on it.
    check("blend samples cluster far more weakly than a real surface's",
          np.median(rb["shape"]["support"]) < 0.5 * (2 * w_default() + 1) ** 2 / 4,
          f"support={np.median(rb['shape']['support'])}")
    check("THE POINT: the flying-pixel rate does NOT catch it, because the blend "
          "is itself a third cluster",
          rb.get("all", {"fp": 0})["fp"] == 0 or
          rb["all"]["fp"] / rb["all"]["n"] < fb / 2,
          f"FP_all={rb.get('all', {}).get('fp')}/{rb.get('all', {}).get('n')} vs between={fb:.3f}")

    third = step.copy(); third[:, 18:21] = 2.5
    rt = audit_view(V(third))
    check("a genuine intermediate surface is caught too", rt["between"]["fp"] > 0)
    check("...and its samples support each other, unlike a varying blend",
          np.median(rt["shape"]["support"]) > np.median(rb["shape"]["support"]),
          f"{np.median(rt['shape']['support'])} vs {np.median(rb['shape']['support'])}")

    scaled = V(step, raw=step * 4.0)
    check("the Phase 0 control is scale-aligned like every stream",
          audit_view(scaled)["all"]["fp"] == 0)

    flat = np.full((H, W), 2.0)
    check("a scene with no boundary returns nothing", audit_view(V(flat)) is None)

    d = box_density(np.zeros((9, 9), bool))
    check("density of an empty mask is 0", d.max() == 0)
    check("density of a full mask is 1", np.allclose(box_density(np.ones((9, 9), bool)), 1))

    check("the reading rule is ordered", verdict(0.06) == "BLENDED"
          and verdict(0.03) == "SUBSTANTIAL" and verdict(0.001) == "NEGLIGIBLE"
          and verdict(float("nan")) == "NOT MEASURED")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        preregister(tmp)
        try:
            preregister(tmp)
            check("pre-registration is write-once", False)
        except SystemExit:
            check("pre-registration is write-once", True)
        os.remove(os.path.join(tmp, "audit_prereg.md"))
        try:
            run(tmp, ["middlebury"])
            check("--run refuses without a pre-registration", False)
        except SystemExit:
            check("--run refuses without a pre-registration", True)

    print("\nself-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _main():
    ap = argparse.ArgumentParser(description="ground-truth audit of the training sets")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/target_audit")
    ap.add_argument("--datasets", default=",".join(TRAINING + COMPARATORS))
    ap.add_argument("--scenes", type=int, default=N_SCENES)
    ap.add_argument("--views", type=int, default=N_VIEWS)
    ap.add_argument("--delta-min", type=float, default=DELTA_MIN)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.preregister:
        return preregister(a.out, a.force)
    if a.run:
        run(a.out, [d for d in a.datasets.split(",") if d], a.scenes, a.views, a.delta_min)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
