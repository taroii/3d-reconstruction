"""
Scale Formulation A across Sintel -> R1 drift-invariance as a DISTRIBUTION over
scenes (planning-agent §2). Converts the single-scene R1 (raw 0.78->0.48,
harmonic flat ~0.81) into a spread. DUSt3R loaded once; MASt3R matching batched.

Per scene (first N frames): single-shot harmonic & raw AUROC at clean poses and
at high added drift (the drift-invariance contrast), over scored pixels. Scenes
with too few dynamic scored pixels are EXCLUDED and LOGGED (no silent caps).

    conda run -n sheaf python experiments/batch_r1.py --nframes 5 --drift_hi 0.12

Outputs: results/r1_sintel.csv, results/r1_sintel_distribution.png
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backbone as B
import sheaf as S
import eval as EV
import sintel as SI
from datatypes import Matches
from pipeline import clip_edges, align_to_grid


def perturb(poses, rng, rot, trans):
    p = poses.copy(); aR = np.eye(3); aT = np.zeros(3)
    for v in range(len(p.s)):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-12
        aR = S.so3_exp(ax * rng.uniform(0, rot)) @ aR
        aT = aT + rng.normal(scale=trans, size=3)
        p.R[v] = aR @ p.R[v]; p.t[v] = p.t[v] + aT
    return p


def auroc_at(pointmaps, edges, matches, poses, shapes, gt, base_valid):
    res = S.solve_harmonic(pointmaps, edges, matches, poses, S.SolveCfg(n_iters=0))
    em, ec = S.splat_to_pixels(res.eps_k, res.index, shapes, reduce="median")
    rm, _ = S.splat_to_pixels(res.raw_k, res.index, shapes, reduce="median")
    valid = {v: base_valid[v] & (ec[v] > 0) for v in gt}
    ndyn = sum(int((gt[v] & valid[v]).sum()) for v in gt)
    return EV.map_auroc(em, gt, valid), EV.map_auroc(rm, gt, valid), valid, ndyn


def run_scene(model, scene, frames, root, cache, niter, drift_hi, seeds, tau):
    paths = [SI.frame_paths(root, scene, "clean")[k] for k in frames]
    edges = clip_edges(len(frames))
    matches = {e: Matches.load(os.path.join(cache, scene, "matches", f"{i}_{j}.npz"))
               for e, (i, j) in enumerate(edges)}
    out = B.run_clip(model, paths, niter=niter)
    poses0 = S.Poses(out["s"], out["R"], out["t"])
    pm, shapes = out["localpts"], out["shapes"]
    t_scale = float(np.std(out["t"], axis=0).mean()) + 1e-6
    gt, base_valid = {}, {}
    for vi, k in enumerate(frames):
        m, _, v = SI.motion_gt(root, scene, k + 1, k + 2, tau=tau)
        gt[vi] = align_to_grid(m, nearest=True)
        base_valid[vi] = align_to_grid(v, nearest=True)
    h0, r0, valid, ndyn = auroc_at(pm, edges, matches, poses0, shapes, gt, base_valid)
    dynfrac = np.mean([gt[v][valid[v]].mean() for v in gt if valid[v].any()])
    hh, rr = [], []
    for s in range(seeds):
        pp = perturb(poses0, np.random.default_rng(s), drift_hi, drift_hi * t_scale)
        ah, ar, _, _ = auroc_at(pm, edges, matches, pp, shapes, gt, base_valid)
        hh.append(ah); rr.append(ar)
    return dict(scene=scene, dynfrac=float(dynfrac), ndyn=ndyn,
                h0=float(h0), r0=float(r0), hhi=float(np.mean(hh)),
                rhi=float(np.mean(rr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nframes", type=int, default=5)
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--drift_hi", type=float, default=0.12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--min_dyn", type=int, default=80, help="min dynamic scored px")
    args = ap.parse_args()
    root, cache = "../data", "../cache"
    scenes = sorted(os.listdir(os.path.join(root, "training", "clean")))
    frames = list(range(args.nframes))
    print(f"{len(scenes)} Sintel scenes: {scenes}")

    # batch MASt3R matching (one model load)
    print(">> batch matching (MASt3R) ...")
    subprocess.run([sys.executable, "match_mast3r.py", "--scenes", *scenes,
                    "--root", root, "--out_root", cache,
                    "--nframes", str(args.nframes)], check=True)

    print(">> DUSt3R backbone (loaded once) ...")
    model = B.load_backbone("dust3r")
    rows, skipped = [], []
    for sc in scenes:
        try:
            r = run_scene(model, sc, frames, root, cache, args.niter,
                          args.drift_hi, args.seeds, args.tau)
        except Exception as e:
            skipped.append((sc, f"error: {type(e).__name__}: {e}")); continue
        if r["ndyn"] < args.min_dyn:
            skipped.append((sc, f"too few dynamic scored px ({r['ndyn']})")); continue
        rows.append(r)
        print(f"  {sc:<14} dynfrac={r['dynfrac']:.3f} ndyn={r['ndyn']:<5} "
              f"h0={r['h0']:.3f} r0={r['r0']:.3f} | hi: harm={r['hhi']:.3f} "
              f"raw={r['rhi']:.3f}")

    # CSV
    os.makedirs("../results", exist_ok=True)
    import csv
    with open("../results/r1_sintel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "dynfrac", "ndyn", "h0", "r0",
                                          "hhi", "rhi"])
        w.writeheader(); w.writerows(rows)
    print(f"\nincluded {len(rows)} scenes; skipped {len(skipped)}:")
    for sc, why in skipped:
        print(f"   - {sc}: {why}")

    if rows:
        h0 = np.array([r["h0"] for r in rows]); hhi = np.array([r["hhi"] for r in rows])
        rhi = np.array([r["rhi"] for r in rows]); r0 = np.array([r["r0"] for r in rows])
        print(f"\n  harmonic AUROC @clean: {h0.mean():.3f} ± {h0.std():.3f}")
        print(f"  harmonic AUROC @drift: {hhi.mean():.3f} ± {hhi.std():.3f}")
        print(f"  raw AUROC @clean:      {r0.mean():.3f} ± {r0.std():.3f}")
        print(f"  raw AUROC @drift:      {rhi.mean():.3f} ± {rhi.std():.3f}")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        data = [h0, hhi, rhi]
        labels = ["harmonic\n@clean", "harmonic\n@drift", "raw\n@drift"]
        bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#c0392b", "#c0392b", "0.6"]):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-.06, .06, len(d)),
                       d, s=14, c="k", alpha=0.5, zorder=3)
        ax.axhline(0.5, ls="--", c="0.5", lw=1, label="chance")
        ax.set_ylabel("AUROC: moving vs static")
        ax.set_title(f"R1 across Sintel (n={len(rows)} scenes): harmonic is "
                     f"drift-invariant; raw collapses under drift")
        ax.set_ylim(0.3, 1.02); ax.legend()
        fig.tight_layout()
        fig.savefig("../results/r1_sintel_distribution.png", dpi=150,
                    bbox_inches="tight")
        print("  -> results/r1_sintel_distribution.png, results/r1_sintel.csv")


if __name__ == "__main__":
    main()
