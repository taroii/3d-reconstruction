"""
R1 headline (real data): drift-invariance. One DUSt3R global-alignment solve,
then sweep *added* camera-pose drift and compare localization AUROC of the raw
residual (what BA sees) vs the single-shot harmonic H^1. Expectation (from the
synthetic PoC): raw decays toward chance as drift grows; harmonic stays high.

Single-shot harmonic = SolveCfg(n_iters=0): projection at the given poses, no
Gauss-Newton pose update — isolates the projection's drift-invariance.

    conda run -n sheaf python experiments/r1_drift.py --scene alley_1 --nframes 5
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
    """Accumulating camera drift: rot (rad/step) on rotation, trans (world units
    /step) on translation."""
    p = poses.copy(); aR = np.eye(3); aT = np.zeros(3)
    for v in range(len(p.s)):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-12
        aR = S.so3_exp(ax * rng.uniform(0, rot)) @ aR
        aT = aT + rng.normal(scale=trans, size=3)
        p.R[v] = aR @ p.R[v]; p.t[v] = p.t[v] + aT
    return p


def auroc_at(pointmaps, edges, matches, poses, shapes, gt, base_valid):
    res = S.solve_harmonic(pointmaps, edges, matches, poses,
                           S.SolveCfg(n_iters=0))   # single-shot, no GN
    emaps, ecnt = S.splat_to_pixels(res.eps_k, res.index, shapes, reduce="median")
    rmaps, _ = S.splat_to_pixels(res.raw_k, res.index, shapes, reduce="median")
    valid = {v: base_valid[v] & (ecnt[v] > 0) for v in gt}
    return EV.map_auroc(emaps, gt, valid), EV.map_auroc(rmaps, gt, valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="alley_1")
    ap.add_argument("--nframes", type=int, default=5)
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--tau", type=float, default=3.0)
    args = ap.parse_args()
    root, cache = "../data", "../cache"
    frames = list(range(args.nframes))
    paths = [SI.frame_paths(root, args.scene, "clean")[k] for k in frames]
    edges = clip_edges(len(frames))

    # matching (reuse cache if present)
    mdir = os.path.join(cache, args.scene, "matches")
    if not os.path.exists(os.path.join(mdir, f"{edges[0][0]}_{edges[0][1]}.npz")):
        subprocess.run([sys.executable, "match_mast3r.py", "--clip",
                        os.path.join(root, "training", "clean", args.scene),
                        "--out", mdir, "--frames", *[str(k) for k in frames]],
                       check=True)
    matches = {e: Matches.load(os.path.join(mdir, f"{i}_{j}.npz"))
               for e, (i, j) in enumerate(edges)}

    # backbone once
    print(">> backbone + global alignment (once) ...")
    model = B.load_backbone("dust3r")
    out = B.run_clip(model, paths, niter=args.niter)
    poses0 = S.Poses(out["s"], out["R"], out["t"])
    pointmaps, shapes = out["localpts"], out["shapes"]
    t_scale = float(np.std(out["t"], axis=0).mean()) + 1e-6

    # GT per view (once)
    gt, base_valid = {}, {}
    for vi, k in enumerate(frames):
        m, _, v = SI.motion_gt(root, args.scene, k + 1, k + 2, tau=args.tau)
        gt[vi] = align_to_grid(m, nearest=True)
        base_valid[vi] = align_to_grid(v, nearest=True)

    drifts = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    har_m, har_s, raw_m, raw_s = [], [], [], []
    print(f"  {'drift(rad)':<12}{'AUROC harm':>14}{'AUROC raw':>12}")
    for d in drifts:
        hs, rs = [], []
        for seed in range(3):
            pp = perturb(poses0, np.random.default_rng(seed), rot=d,
                         trans=d * t_scale) if d > 0 else poses0
            ah, ar = auroc_at(pointmaps, edges, matches, pp, shapes, gt, base_valid)
            hs.append(ah); rs.append(ar)
        har_m.append(np.mean(hs)); har_s.append(np.std(hs))
        raw_m.append(np.mean(rs)); raw_s.append(np.std(rs))
        print(f"  {d:<12.2f}{np.mean(hs):>14.3f}{np.mean(rs):>12.3f}")

    os.makedirs("../results", exist_ok=True)
    har_m, har_s = np.array(har_m), np.array(har_s)
    raw_m, raw_s = np.array(raw_m), np.array(raw_s)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(drifts, har_m, "o-", color="#c0392b", lw=2, label="harmonic $H^1$ (ours)")
    ax.fill_between(drifts, har_m - har_s, har_m + har_s, color="#c0392b", alpha=0.2)
    ax.plot(drifts, raw_m, "s-", color="0.5", lw=2, label="raw residual (BA)")
    ax.fill_between(drifts, raw_m - raw_s, raw_m + raw_s, color="0.5", alpha=0.25)
    ax.axhline(0.5, ls="--", c="0.6", lw=1)
    ax.set_xlabel("added camera drift per step (rad)")
    ax.set_ylabel("AUROC: moving vs static")
    ax.set_title(f"Real data ({args.scene}): harmonic is drift-invariant; raw is not")
    ax.set_ylim(0.45, 1.0); ax.legend()
    fig.tight_layout()
    fig.savefig(f"../results/r1_drift_{args.scene}.png", dpi=150, bbox_inches="tight")
    print(f"  -> results/r1_drift_{args.scene}.png")


if __name__ == "__main__":
    main()
