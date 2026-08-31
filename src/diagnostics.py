r"""
Robustness diagnostics for the premise check. Not part of the pre-registered
decision rule -- these exist to attack a positive result before it is believed,
and their output is written to disk so the record does not live in a chat log.

Two checks, both prompted by eyeballing the Sec. 8 figures:

1. ALIGNMENT (`--alignment`). The registered analysis fits scale only (Sec. 6.1).
   If a model is affine-invariant rather than scale-invariant, a scale-only fit
   leaves a residual depth offset that displaces the whole prediction, and near
   an occlusion boundary a displaced surface lands inside the void and scores as
   a flying pixel. Comparing FP under scale-only vs scale+shift separates "the
   model puts points in empty space" from "the model has a global depth offset".
   If FP collapses under the affine fit, the headline is an alignment artefact.

2. BOUNDARY DENSITY (`--density`). A wire mesh or foliage produces thousands of
   legitimate K=2 boundaries no feed-forward model could resolve. Those are real
   discontinuities but they are not the smeared silhouette the study is about,
   and if they dominate B_eval the headline means something else. Local boundary
   density separates the two: a clean silhouette puts a thin arc of boundary
   pixels in a window, a mesh saturates it. If FP is concentrated in the
   saturated band, the effect is about unresolvable texture, not silhouettes.

    python src/diagnostics.py --alignment --density --out results/premise_check
"""
from __future__ import annotations

import os
import csv
import argparse
import collections

import numpy as np

import data as D
import infer as INF
import fpmetrics as FM

# Derived from the inference registry, NOT hardcoded. A literal list here silently
# omitted DUSt3R and MASt3R from every diagnostic the moment Phase 1A added them --
# the recalibration would have reported only the VGGT-lineage streams and nothing
# would have flagged the absence. Streams with no cached prediction are skipped
# per view anyway, so listing them all is safe.
STREAMS = tuple(INF.STREAMS)
DENSITY_BINS = [(0.00, 0.10, "sparse (clean silhouette)"),
                (0.10, 0.20, "moderate"),
                (0.20, 0.35, "dense"),
                (0.35, 1.01, "saturated (mesh/foliage)")]
DENSITY_R = 7


def box_density(mask, r=DENSITY_R):
    """Fraction of boundary pixels within a (2r+1)^2 window, via a summed-area
    table so it stays O(HW) rather than O(HW r^2)."""
    S = np.pad(mask.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    H, W = mask.shape
    ys, xs = np.arange(H), np.arange(W)
    y0, y1 = np.clip(ys - r, 0, H), np.clip(ys + r + 1, 0, H)
    x0, x1 = np.clip(xs - r, 0, W), np.clip(xs + r + 1, 0, W)
    tot = (S[np.ix_(y1, x1)] - S[np.ix_(y0, x1)]
           - S[np.ix_(y1, x0)] + S[np.ix_(y0, x0)])
    return tot / np.maximum(np.outer(y1 - y0, x1 - x0), 1)


def _views(datasets, n_scenes, n_views):
    for ds in datasets:
        for sc in D.scenes(ds)[:n_scenes]:
            for v in D.views(ds, sc, n_views):
                yield ds, sc, v


def _prep(vd, eta, tau, delta_min):
    """Shared setup: boundary set, interior, and the K==2 pixels with their void."""
    gt, valid = vd.depth.astype(np.float64), vd.valid
    B = FM.boundary_set(gt, valid, eta)
    pix = np.argwhere(B)
    if len(pix) < 50:
        return None
    I = FM.interior(valid, FM.dilate(B, tau))
    sup = FM.depth_support(gt, valid, pix, w=3, eta=eta)
    delta = sup["d2"] - sup["d1"]
    keep = (sup["K"] == 2) & np.isfinite(delta) & (delta >= delta_min)
    if keep.sum() < 20:
        return None
    return dict(gt=gt, B=B, I=I, pix=pix, keep=keep,
                d1=sup["d1"][keep], d2=sup["d2"][keep], delta=delta[keep])


def run(datasets, n_scenes, n_views, out, eta=FM.ETA0, tau=FM.TAU0, beta=FM.BETA0,
        delta_min=0.05, do_align=True, do_density=True):
    align_acc = collections.defaultdict(lambda: dict(fp_s=0, fp_a=0, n=0, shift=[]))
    dens_acc = {(s, b[:2]): [0, 0] for s in STREAMS for b in DENSITY_BINS}
    per_scene = []

    for ds, sc, v in _views(datasets, n_scenes, n_views):
        try:
            vd = D.load(v)
        except Exception:
            continue
        preds = {s: INF.load_pred(out, s, v.key, vd.depth.shape) for s in STREAMS}
        preds = {k: p for k, p in preds.items() if p is not None}
        if not preds:
            continue
        P = _prep(vd, eta, tau, delta_min)
        if P is None:
            continue
        lo = P["d1"] + beta * P["delta"]
        hi = P["d2"] - beta * P["delta"]
        dens = (box_density(P["B"])[P["pix"][:, 0], P["pix"][:, 1]][P["keep"]]
                if do_density else None)

        for name, pr in preds.items():
            s_only = FM.align_scale(pr, P["gt"], P["I"], "median")
            qs = (pr * s_only)[P["pix"][:, 0], P["pix"][:, 1]][P["keep"]]
            with np.errstate(invalid="ignore"):
                fp_s = (qs >= lo) & (qs <= hi)
            if do_align:
                a, b = FM.align_affine(pr, P["gt"], P["I"])
                qa = (pr * a + b)[P["pix"][:, 0], P["pix"][:, 1]][P["keep"]]
                with np.errstate(invalid="ignore"):
                    fp_a = (qa >= lo) & (qa <= hi)
                A = align_acc[(ds, name)]
                A["fp_s"] += int(fp_s.sum()); A["fp_a"] += int(fp_a.sum())
                A["n"] += int(fp_s.size); A["shift"].append(float(b))
            if do_density:
                for b0, b1, _ in DENSITY_BINS:
                    m = (dens >= b0) & (dens < b1)
                    cell = dens_acc[(name, (b0, b1))]
                    cell[0] += int(fp_s[m].sum()); cell[1] += int(m.sum())
                if name == STREAMS[0]:
                    low = dens < 0.20
                    per_scene.append(dict(
                        dataset=ds, scene=sc, stream=name, n_eval=int(fp_s.size),
                        FP_all=float(fp_s.mean()), mean_density=float(dens.mean()),
                        FP_silhouette_like=(float(fp_s[low].mean()) if low.sum() > 20
                                            else float("nan")),
                        n_silhouette_like=int(low.sum())))

    os.makedirs(out, exist_ok=True)
    written = []
    if do_align:
        p = os.path.join(out, "diag_alignment.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "stream", "n_eval", "FP_scale_only", "FP_affine",
                        "delta_FP", "median_fitted_shift_m"])
            for (ds, name), A in sorted(align_acc.items()):
                if not A["n"]:
                    continue
                fs, fa = A["fp_s"] / A["n"], A["fp_a"] / A["n"]
                w.writerow([ds, name, A["n"], f"{fs:.6f}", f"{fa:.6f}",
                            f"{fa-fs:+.6f}", f"{np.median(A['shift']):.4f}"])
        written.append(p)
    if do_density:
        p = os.path.join(out, "diag_density.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stream", "density_lo", "density_hi", "band", "n_eval", "FP"])
            for s in STREAMS:
                for b0, b1, lab in DENSITY_BINS:
                    fp, n = dens_acc[(s, (b0, b1))]
                    if n:
                        w.writerow([s, b0, b1, lab, n, f"{fp/n:.6f}"])
        written.append(p)
        q = os.path.join(out, "diag_density_by_scene.csv")
        with open(q, "w", newline="") as f:
            if per_scene:
                w = csv.DictWriter(f, fieldnames=list(per_scene[0]))
                w.writeheader(); w.writerows(per_scene)
        written.append(q)

    for p in written:
        print("wrote", p)
    if do_align:
        print("\n=== alignment sensitivity (FP should NOT collapse under affine) ===")
        for (ds, name), A in sorted(align_acc.items()):
            if A["n"]:
                print(f"  {ds:11s} {name:12s} scale-only {A['fp_s']/A['n']:.4f}  "
                      f"affine {A['fp_a']/A['n']:.4f}  shift {np.median(A['shift']):+.3f} m")
    if do_density:
        print("\n=== FP by local boundary density (flat => not a fine-structure artefact) ===")
        for s in STREAMS:
            bits = []
            for b0, b1, lab in DENSITY_BINS:
                fp, n = dens_acc[(s, (b0, b1))]
                bits.append(f"{lab.split()[0]}={fp/n:.3f}" if n else f"{lab.split()[0]}=--")
            print(f"  {s:12s} " + "  ".join(bits))
    return written


def void_position(datasets, n_scenes, n_views, out):
    """Why is a dataset's GT CONTROL non-zero when its boundaries are exact?

    Writes, per dataset, where the control's own flying pixels sit inside the
    void: t = (value - d1)/(d2 - d1). Plus how well a cluster MEDIAN stands in
    for the surface a pixel actually belongs to. Together these separate three
    explanations that are easy to confuse:

      t centred near 0.5, medians faithful  -> genuine intermediate geometry the
                                               K=2 restriction failed to exclude
      t piled near the beta margins         -> the median is a poor proxy for a
                                               slanted surface (a metric artefact)
      medians unfaithful (large p90/p99)    -> same, and visible directly
    """
    tpos = collections.defaultdict(list)
    spread = collections.defaultdict(list)
    for ds, sc, v in _views(datasets, n_scenes, n_views):
        try:
            vd = D.load(v)
        except Exception:
            continue
        P = _prep(vd, FM.ETA0, FM.TAU0, 0.05)
        if P is None:
            continue
        d1, d2, dl = P["d1"], P["d2"], P["delta"]
        lo, hi = d1 + FM.BETA0 * dl, d2 - FM.BETA0 * dl
        kp = P["pix"][P["keep"]]
        raw = vd.depth_raw.astype(np.float64)[kp[:, 0], kp[:, 1]]
        with np.errstate(invalid="ignore"):
            gfp = (raw >= lo) & (raw <= hi)
        if gfp.any():
            tpos[ds].extend(((raw[gfp] - d1[gfp]) / dl[gfp]).tolist()[:4000])
        nearest = np.where(np.abs(raw - d1) < np.abs(raw - d2), d1, d2)
        spread[ds].extend((np.abs(raw - nearest) / dl).tolist()[:4000])

    p = os.path.join(out, "diag_void_position.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_gt_flying", "median_t", "t_q25", "t_q75",
                    "share_mid_0.4_0.6", "median_abs_dev_from_cluster_median",
                    "p90_dev", "p99_dev"])
        for ds in datasets:
            t, sp = np.array(tpos[ds]), np.array(spread[ds])
            if not sp.size:
                continue
            row = [ds, t.size]
            row += ([f"{np.median(t):.4f}", f"{np.percentile(t,25):.4f}",
                     f"{np.percentile(t,75):.4f}",
                     f"{((t>0.4)&(t<0.6)).mean():.4f}"] if t.size >= 20
                    else ["", "", "", ""])
            row += [f"{np.median(sp):.5f}", f"{np.percentile(sp,90):.5f}",
                    f"{np.percentile(sp,99):.5f}"]
            w.writerow(row)
    print("wrote", p)
    return p


def landing(datasets, n_scenes, n_views, out):
    """Decompose each stream into near / FP / far.

    FP alone cannot distinguish 'the model fills the void' from 'the model puts
    everything behind the far surface'. It also exposes how thin a dataset's
    eval set is, which is what makes a per-stream number unstable.
    """
    acc = collections.defaultdict(lambda: dict(near=0, fp=0, far=0, n=0))
    for ds, sc, v in _views(datasets, n_scenes, n_views):
        try:
            vd = D.load(v)
        except Exception:
            continue
        P = _prep(vd, FM.ETA0, FM.TAU0, 0.05)
        if P is None:
            continue
        lo = P["d1"] + FM.BETA0 * P["delta"]
        hi = P["d2"] - FM.BETA0 * P["delta"]
        kp = P["pix"][P["keep"]]
        for s in STREAMS:
            pr = INF.load_pred(out, s, v.key, vd.depth.shape)
            if pr is None:
                continue
            sc_ = FM.align_scale(pr, vd.depth.astype(np.float64), P["I"], "median")
            q = (pr * sc_)[kp[:, 0], kp[:, 1]]
            a = acc[(ds, s)]
            with np.errstate(invalid="ignore"):
                a["near"] += int((q < lo).sum()); a["fp"] += int(((q >= lo) & (q <= hi)).sum())
                a["far"] += int((q > hi).sum()); a["n"] += int(q.size)
    p = os.path.join(out, "diag_landing.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "stream", "n_eval", "near_rate", "FP", "far_rate"])
        for (ds, s), a in sorted(acc.items()):
            if a["n"]:
                w.writerow([ds, s, a["n"], f"{a['near']/a['n']:.4f}",
                            f"{a['fp']/a['n']:.4f}", f"{a['far']/a['n']:.4f}"])
    print("wrote", p)
    return p


def recalibrate(datasets, n_scenes, n_views, out, eta=FM.ETA0, tau=FM.TAU0,
                beta=FM.BETA0, delta_min=0.05, fine_eta=0.01, dens_max=0.20):
    """Phase 1A2 (notes/diagnosis.md §2): three variants of every rate, plus the
    K_p histogram.

    Phase 0 found the ground truth itself scoring as flying pixels on Infinigen,
    and traced it to genuine third surfaces that the two-cluster restriction
    merged at eta = 0.05. That attribution is **[UNTESTED]** -- profile inspection
    was suggestive, not conclusive. These variants make it testable:

      all         every K_p == 2 boundary pixel, as Phase 0 reported it
      fine        support re-clustered at a FINER jump threshold, keeping only
                  pixels that are STILL K_p == 2. A pixel whose surfaces merged
                  at the coarse threshold separates here and drops out, so if the
                  Infinigen attribution is right this variant should shrink the
                  GT control most on the datasets with fine structure.
      silhouette  low local boundary density only, i.e. clean silhouettes

    GATE (§2): if the recalibrated model-minus-GT difference falls below the
    pre-registered 5-point margin on the clean sets, stop and re-plan before 1B.
    """
    if fine_eta >= eta:
        raise SystemExit(f"--fine-eta ({fine_eta}) must be SMALLER than eta ({eta}); "
                         f"a finer jump threshold is what splits merged surfaces.")
    acc = collections.defaultdict(lambda: collections.defaultdict(
        lambda: dict(fp=0, n=0)))
    khist = collections.defaultdict(collections.Counter)
    skipped = collections.Counter()

    for ds, sc, v in _views(datasets, n_scenes, n_views):
        try:
            vd = D.load(v)
        except Exception:
            continue
        gt, valid = vd.depth.astype(np.float64), vd.valid
        B = FM.boundary_set(gt, valid, eta)
        pix = np.argwhere(B)
        if len(pix) < 50:
            continue
        I = FM.interior(valid, FM.dilate(B, tau))
        sup = FM.depth_support(gt, valid, pix, w=3, eta=eta)
        K, d1, d2 = sup["K"], sup["d1"], sup["d2"]
        delta = d2 - d1
        base = (K == 2) & np.isfinite(delta) & (delta >= delta_min)
        if base.sum() < 20:
            continue
        # counted AFTER the skip, so the histogram covers exactly the views the
        # rates cover rather than a superset
        for k, c in collections.Counter(K[K > 0].tolist()).items():
            khist[ds][int(k)] += int(c)

        # variant masks, all subsets of `base` so the three rates are nested
        fine_K = FM.depth_support(gt, valid, pix, w=3, eta=fine_eta)["K"]
        masks = {"all": base,
                 "fine": base & (fine_K == 2),
                 "silhouette": base & (box_density(B)[pix[:, 0], pix[:, 1]] < dens_max)}

        # A view is used only when EVERY requested stream has a prediction.
        # Without this, a stream missing on some views accumulates over a
        # strictly smaller view set than the GT control, and the gate below
        # subtracts two rates computed on different pixels -- which can
        # manufacture a breach on a clean set. measure.py already guards this;
        # omitting it here was the same bug in a second place.
        preds = {s: INF.load_pred(out, s, v.key, gt.shape) for s in STREAMS}
        missing = [k for k, p in preds.items() if p is None]
        if missing:
            for k in missing:
                skipped[f"missing:{k}"] += 1
            skipped["view_dropped_incomplete"] += 1
            continue
        preds["gt_raw"] = vd.depth_raw.astype(np.float64)

        # Phase 0 aligns EVERY entry including the control (fpmetrics.fp_view_grid),
        # so the control is aligned here too. It is a no-op wherever raw == depth
        # on the interior, but not for nyuv2, whose raw map is the sensor output
        # and whose depth map is in-painted.
        scales = {st: FM.align_scale(pr, gt, I, "median") for st, pr in preds.items()}
        if not all(np.isfinite(v) for v in scales.values()):
            # An undefined scale would turn every comparison False and be written
            # out as a real FP of exactly 0. Drop the view and say so.
            skipped["view_dropped_scale_undefined"] += 1
            continue

        for name, m in masks.items():
            if m.sum() < 20:
                continue
            dd1, dd2, dl = d1[m], d2[m], delta[m]
            lo, hi = dd1 + beta * dl, dd2 - beta * dl
            kp = pix[m]
            for st, pr in preds.items():
                q = pr[kp[:, 0], kp[:, 1]] * scales[st]
                with np.errstate(invalid="ignore"):
                    fp = (q >= lo) & (q <= hi)
                a = acc[(ds, st)][name]
                a["fp"] += int(fp.sum()); a["n"] += int(fp.size)

    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "1a_calibration.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "stream", "variant", "n_eval", "FP"])
        for (ds, st), byv in sorted(acc.items()):
            for name in ("all", "fine", "silhouette"):
                a = byv.get(name)
                if a and a["n"]:
                    w.writerow([ds, st, name, a["n"], f"{a['fp']/a['n']:.6f}"])
    print("wrote", p)

    q = os.path.join(out, "1a_k_histogram.csv")
    with open(q, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "K", "n_boundary_px", "share"])
        for ds, c in sorted(khist.items()):
            tot = sum(c.values())
            for k in sorted(c):
                w.writerow([ds, k, c[k], f"{c[k]/tot:.4f}"])
    print("wrote", q)

    # ---- the gate ----
    print("\n=== Phase 1A2 recalibration: model minus GT control, by variant ===")
    print(f"{'dataset':12s} {'stream':12s} " + "".join(f"{v:>13s}" for v in
          ("all", "fine", "silhouette")))
    breaches = []
    for (ds, st) in sorted(acc):
        if st == "gt_raw":
            continue
        cells = []
        for name in ("all", "fine", "silhouette"):
            m = acc[(ds, st)].get(name)
            g = acc[(ds, "gt_raw")].get(name)
            if m and g and m["n"] and g["n"]:
                diff = m["fp"] / m["n"] - g["fp"] / g["n"]
                cells.append(f"{diff:+.4f}")
                if diff < 0.05:
                    breaches.append((ds, st, name, diff))
            else:
                cells.append("--")
        print(f"{ds:12s} {st:12s} " + "".join(f"{c:>13s}" for c in cells))
    # §2 scopes the stop-gate to the CLEAN sets. A breach on a contaminated
    # stratum is not a reason to stop, and mixing the two buries the one line
    # that matters under caveats the reader has to apply by hand.
    # A gate that says PASS having measured nothing is worse than no gate: this
    # is a stop-or-replan decision, and "no model stream accumulated a single
    # pixel" must never render as a clear.
    measured = sum(a["n"] for (ds, st), byv in acc.items() if st != "gt_raw"
                   for a in byv.values())
    if not measured:
        print("\n!! GATE NOT EVALUATED: no model stream accumulated any evaluated "
              "pixels.")
        print(f"   skipped: {dict(skipped)}")
        print("   This is an ABSENCE of evidence, not a pass. Populate the "
              "prediction cache (src/infer.py) and re-run.")
        return acc, 5

    clean = {d for d, sp in D.SPECS.items() if not sp["contam"]}
    hard = [b for b in breaches if b[0] in clean]
    soft = [b for b in breaches if b[0] not in clean]
    if hard:
        print("\n!! GATE BREACH on CLEAN datasets -- model-minus-GT is below the "
              "pre-registered 5-point margin:")
        for ds, st, name, d in hard:
            print(f"     {ds}/{st}/{name}: {100*d:+.2f} pp")
        print("   notes/diagnosis.md §2: STOP and re-plan before 1B.")
    else:
        print("\nGate: every CLEAN-dataset cell clears the 5-point margin.")
    if soft:
        print(f"\n   ({len(soft)} cell(s) below the margin on CONTAMINATED strata "
              f"({', '.join(sorted({b[0] for b in soft}))}) -- recorded, but §2 does "
              f"not gate on these.)")
    if skipped:
        print(f"\n   views skipped: {dict(skipped)}")
    # A clean-set breach is a real §2 stop signal and must halt the pipeline,
    # not scroll past as one line among many.
    return acc, (6 if hard else 0)


def _main():
    ap = argparse.ArgumentParser(description="premise-check robustness diagnostics")
    ap.add_argument("--datasets", default="middlebury,infinigen,ibims")
    ap.add_argument("--streams", default="",
                    help="comma list; default is every stream infer.py knows")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--out", default="results/premise_check")
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--alignment", action="store_true")
    ap.add_argument("--density", action="store_true")
    ap.add_argument("--void-position", action="store_true",
                    help="why a dataset's GT control is non-zero")
    ap.add_argument("--landing", action="store_true",
                    help="near / FP / far decomposition per stream")
    ap.add_argument("--recalibrate", action="store_true",
                    help="Phase 1A2: three rate variants + K_p histogram")
    ap.add_argument("--fine-eta", type=float, default=0.01)
    a = ap.parse_args()
    dss = [d for d in a.datasets.split(",") if d]
    global STREAMS
    if a.streams:
        bad = [x for x in a.streams.split(",") if x and x not in INF.STREAMS]
        if bad:
            raise SystemExit(f"unknown streams {bad}; choose from {list(INF.STREAMS)}")
        STREAMS = tuple(x for x in a.streams.split(",") if x)
    extra = a.void_position or a.landing or a.recalibrate
    if not (a.alignment or a.density or extra):
        a.alignment = a.density = True
    if a.alignment or a.density:
        run(dss, a.scenes, a.views, a.out, delta_min=a.delta_min,
            do_align=a.alignment, do_density=a.density)
    if a.void_position:
        void_position(dss, a.scenes, a.views, a.out)
    if a.landing:
        landing(dss, a.scenes, a.views, a.out)
    if a.recalibrate:
        _, rc = recalibrate(dss, a.scenes, a.views, a.out, delta_min=a.delta_min,
                            fine_eta=a.fine_eta)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    # Propagate the exit code: recalibrate() returns 5 when the §2 gate could not
    # be evaluated and 6 on a clean-set breach, and both must reach the runner.
    raise SystemExit(_main())
