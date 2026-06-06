"""
Flow-vs-sheaf head-to-head (plan §3). Same geometry (DUSt3R), same GT, same
pixels, same metric — only the motion cue differs:
  - A (ours, flow-free): harmonic H^1 over MASt3R correspondences.
  - flow baseline: RAFT flow vs DUSt3R camera-induced flow (D2USt3R Eq.5 form).

Swept over temporal stride dt in {1,3,5,7,9}: dt=1 is the §3a parity point;
larger dt is the §3b divergence regime (dense flow gets ill-posed at large
displacement; appearance matching + geometry should degrade more gracefully).

GT is the per-frame dynamic mask (consecutive-frame scene flow, stride-
independent). Both methods are scored on A's matched pixels (common support) for
a fair comparison, pooled over a clip's consecutive frames. Static-dominant
scenes only (the operating regime; R1 precondition).

    conda run -n sheaf python experiments/flow_vs_sheaf.py

Outputs: results/flow_vs_sheaf.csv, results/flow_vs_sheaf.png
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
import flow_baseline as FB
from datatypes import Matches
from pipeline import clip_edges, align_to_grid

DEFAULT_SCENES = ["alley_1", "cave_2", "market_2", "market_5", "ambush_4", "temple_3"]


def g4(R, t):
    g = np.eye(4); g[:3, :3] = R; g[:3, 3] = t; return g


def eval_scene_stride(dust3r, raft, scene, stride, nframes, root, cache, niter, tau):
    clean = os.path.join(root, "training", "clean", scene)
    all_paths = sorted(__import__("glob").glob(os.path.join(clean, "frame_*.png")))
    paths = all_paths[::stride][:nframes]
    nf = len(paths)
    if nf < 3:
        return None
    edges = clip_edges(nf)
    sub = "matches" if stride == 1 else f"matches_s{stride}"
    mdir = os.path.join(cache, scene, sub)
    try:
        matches = {e: Matches.load(os.path.join(mdir, f"{i}_{j}.npz"))
                   for e, (i, j) in enumerate(edges)}
    except FileNotFoundError:
        return None

    out = B.run_clip(dust3r, paths, niter=niter)
    poses = S.Poses(out["s"], out["R"], out["t"])
    pm, shapes, depths, foc = out["localpts"], out["shapes"], out["depths"], out["focals"]

    # A: harmonic H^1 -> per-clip-frame energy maps
    res = S.solve_harmonic(pm, edges, matches, poses, S.SolveCfg(n_iters=0))
    emaps, ecnt = S.splat_to_pixels(res.eps_k, res.index, shapes, reduce="median")

    # per consecutive clip-frame: GT mask, valid, A score, flow residual
    a_sc, f_sc, labels = [], [], []
    for i in range(nf - 1):
        a_i = i * stride                                   # 0-based absolute frame
        m, _, v = SI.motion_gt(root, scene, a_i + 1, a_i + 2, tau=tau)
        gt = align_to_grid(m, nearest=True)
        valid = align_to_grid(v, nearest=True) & (ecnt[i] > 0)
        if valid.sum() < 30:
            continue
        flow_res = FB.flow_residual_map(raft, paths[i], paths[i + 1],
                                        depths[i], float(foc[i]), float(foc[i + 1]),
                                        g4(out["R"][i], out["t"][i]),
                                        g4(out["R"][i + 1], out["t"][i + 1]))
        a_sc.append(np.sqrt(emaps[i][valid] + 1e-15))
        f_sc.append(flow_res[valid])
        labels.append(gt[valid])
    if not labels:
        return None
    a_sc = np.concatenate(a_sc); f_sc = np.concatenate(f_sc)
    labels = np.concatenate(labels)
    if labels.sum() < 30 or (~labels).sum() < 30:
        return None
    return dict(scene=scene, stride=stride, nf=nf, ndyn=int(labels.sum()),
                auroc_A=EV.auroc(a_sc, labels), auroc_flow=EV.auroc(f_sc, labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--strides", type=int, nargs="*", default=[1, 3, 5, 7, 9])
    ap.add_argument("--nframes", type=int, default=5)
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--tau", type=float, default=3.0)
    args = ap.parse_args()
    root, cache = "../data", "../cache"

    # batch MASt3R matching per stride (one model load each)
    for st in args.strides:
        print(f">> batch matching stride {st} ...")
        subprocess.run([sys.executable, "match_mast3r.py", "--scenes", *args.scenes,
                        "--root", root, "--out_root", cache,
                        "--nframes", str(args.nframes), "--stride", str(st)],
                       check=True)

    print(">> loading DUSt3R + RAFT (once) ...")
    dust3r = B.load_backbone("dust3r")
    raft = FB.load_raft()

    rows = []
    for st in args.strides:
        for sc in args.scenes:
            r = eval_scene_stride(dust3r, raft, sc, st, args.nframes, root, cache,
                                  args.niter, args.tau)
            if r is None:
                print(f"  stride {st} {sc}: skipped"); continue
            rows.append(r)
            print(f"  stride {st} {sc:<12} ndyn={r['ndyn']:<5} "
                  f"A={r['auroc_A']:.3f}  flow={r['auroc_flow']:.3f}")

    import csv
    os.makedirs("../results", exist_ok=True)
    with open("../results/flow_vs_sheaf.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "stride", "nf", "ndyn",
                                          "auroc_A", "auroc_flow"])
        w.writeheader(); w.writerows(rows)

    # divergence curve: mean±sd over scenes per stride
    print("\n  stride |   A (sheaf)   |  flow (RAFT)")
    As, Fs, xs = [], [], []
    for st in args.strides:
        a = np.array([r["auroc_A"] for r in rows if r["stride"] == st])
        fl = np.array([r["auroc_flow"] for r in rows if r["stride"] == st])
        if len(a) == 0:
            continue
        xs.append(st); As.append((a.mean(), a.std())); Fs.append((fl.mean(), fl.std()))
        print(f"   {st:<5} | {a.mean():.3f} ± {a.std():.3f} | {fl.mean():.3f} ± {fl.std():.3f}")

    As, Fs = np.array(As), np.array(Fs)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(xs, As[:, 0], yerr=As[:, 1], fmt="o-", color="#c0392b", lw=2,
                capsize=4, label="A: sheaf H¹ (flow-free)")
    ax.errorbar(xs, Fs[:, 0], yerr=Fs[:, 1], fmt="s-", color="#2166ac", lw=2,
                capsize=4, label="flow baseline (RAFT)")
    ax.axhline(0.5, ls="--", c="0.5", lw=1)
    ax.set_xlabel("temporal stride Δt (frames)")
    ax.set_ylabel("AUROC: moving vs static (A's scored px)")
    ax.set_title(f"Flow vs sheaf across stride ({len(args.scenes)} static-dominant scenes)")
    ax.set_ylim(0.45, 1.0); ax.legend()
    fig.tight_layout()
    fig.savefig("../results/flow_vs_sheaf.png", dpi=150, bbox_inches="tight")
    print("  -> results/flow_vs_sheaf.png, results/flow_vs_sheaf.csv")


if __name__ == "__main__":
    main()
