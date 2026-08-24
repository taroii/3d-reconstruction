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

STREAMS = ("vggt_point", "vggt_depth", "pi3_local")
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


def _main():
    ap = argparse.ArgumentParser(description="premise-check robustness diagnostics")
    ap.add_argument("--datasets", default="middlebury,infinigen,ibims")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--out", default="results/premise_check")
    ap.add_argument("--delta-min", type=float, default=0.05)
    ap.add_argument("--alignment", action="store_true")
    ap.add_argument("--density", action="store_true")
    a = ap.parse_args()
    if not (a.alignment or a.density):
        a.alignment = a.density = True
    run([d for d in a.datasets.split(",") if d], a.scenes, a.views, a.out,
        delta_min=a.delta_min, do_align=a.alignment, do_density=a.density)


if __name__ == "__main__":
    _main()
