"""
R2 / week-one go-no-go (plan S1a, S10): DUSt3R-vs-D2USt3R H^1 differential.

Same clip, same MASt3R correspondences; only the sheaf backbone changes. DUSt3R
models the scene rigidly, so motion shows up as H^1. D2USt3R's SDAP warps dynamic
pixels to align across time, so it should SUPPRESS the dynamic H^1. Confirming
that suppression validates the interpretation "H^1 = the inconsistency a rigid
backbone cannot absorb = dynamics".

Metrics are scale-invariant (the two backbones have different world scales):
  - AUROC(H^1; dynamic vs static) over scored pixels.
  - dynamic/static median-energy ratio.
GO if D2USt3R's AUROC and ratio drop markedly vs DUSt3R's.

    conda run -n sheaf python experiments/r2_differential.py --scene alley_1 --nframes 5
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


def run_backbone_h1(name, paths, edges, matches, niter):
    import torch, gc
    model = B.load_backbone(name)
    out = B.run_clip(model, paths, niter=niter)
    poses = S.Poses(out["s"], out["R"], out["t"])
    res = S.solve_harmonic(out["localpts"], edges, matches, poses,
                           S.SolveCfg(n_iters=0))
    emaps, ecnt = S.splat_to_pixels(res.eps_k, res.index, out["shapes"],
                                    reduce="median")
    imgs = out["imgs"]
    del model, out, res
    gc.collect(); torch.cuda.empty_cache()
    return emaps, ecnt, imgs


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

    mdir = os.path.join(cache, args.scene, "matches")
    if not os.path.exists(os.path.join(mdir, f"{edges[0][0]}_{edges[0][1]}.npz")):
        subprocess.run([sys.executable, "match_mast3r.py", "--clip",
                        os.path.join(root, "training", "clean", args.scene),
                        "--out", mdir, "--frames", *[str(k) for k in frames]],
                       check=True)
    matches = {e: Matches.load(os.path.join(mdir, f"{i}_{j}.npz"))
               for e, (i, j) in enumerate(edges)}

    gt, base_valid = {}, {}
    for vi, k in enumerate(frames):
        m, _, v = SI.motion_gt(root, args.scene, k + 1, k + 2, tau=args.tau)
        gt[vi] = align_to_grid(m, nearest=True)
        base_valid[vi] = align_to_grid(v, nearest=True)

    results = {}
    for name in ["dust3r", "d2ust3r"]:
        print(f">> {name} backbone ...")
        emaps, ecnt, imgs = run_backbone_h1(name, paths, edges, matches, args.niter)
        valid = {v: base_valid[v] & (ecnt[v] > 0) for v in gt}
        au = EV.map_auroc(emaps, gt, valid)
        # scale-invariant dynamic/static energy ratio over scored pixels
        dyn, sta = [], []
        for v in gt:
            sc = valid[v]
            e = np.sqrt(emaps[v][sc] + 1e-15); g = gt[v][sc]
            dyn.append(e[g]); sta.append(e[~g])
        dyn, sta = np.concatenate(dyn), np.concatenate(sta)
        ratio = np.median(dyn) / (np.median(sta) + 1e-12)
        results[name] = dict(au=au, ratio=ratio, emaps=emaps, valid=valid, imgs=imgs)
        print(f"   AUROC={au:.3f}  dyn/static energy ratio={ratio:.2f}")

    d_au, dd_au = results["dust3r"]["au"], results["d2ust3r"]["au"]
    d_r, dd_r = results["dust3r"]["ratio"], results["d2ust3r"]["ratio"]
    print("\n=== R2 week-one gate ===")
    print(f"  DUSt3R  : AUROC {d_au:.3f}  ratio {d_r:.2f}")
    print(f"  D2USt3R : AUROC {dd_au:.3f}  ratio {dd_r:.2f}")
    go = (d_au - dd_au > 0.1) and (dd_r < d_r * 0.7)
    print("  --> GO: SDAP suppresses the dynamic H^1 (interpretation validated)."
          if go else
          "  --> NO-GO / inspect: SDAP did not suppress H^1 as predicted.")

    # figure: H^1 on one frame, DUSt3R vs D2USt3R, GT-dynamic outline
    vi = max(gt, key=lambda v: gt[v][base_valid[v]].mean())  # most-dynamic view
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    img = results["dust3r"]["imgs"][vi]
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    axes[0].imshow(img); axes[0].set_title("frame"); axes[0].axis("off")
    for ax, name, ttl in [(axes[1], "dust3r", f"DUSt3R H¹ (AUROC {d_au:.2f})"),
                          (axes[2], "d2ust3r", f"D²USt3R H¹ (AUROC {dd_au:.2f})")]:
        r = results[name]
        e = np.sqrt(r["emaps"][vi] + 1e-15)
        e = np.where(r["valid"][vi], e, np.nan)
        axes_im = ax.imshow(e, cmap="inferno"); ax.set_title(ttl); ax.axis("off")
        ax.contour(gt[vi], levels=[0.5], colors="cyan", linewidths=1.0)
    fig.suptitle(f"R2 differential ({args.scene}): SDAP suppresses the dynamic H¹",
                 fontweight="bold")
    fig.tight_layout()
    os.makedirs("../results", exist_ok=True)
    fig.savefig(f"../results/r2_differential_{args.scene}.png", dpi=150,
                bbox_inches="tight")
    print(f"  -> results/r2_differential_{args.scene}.png")


if __name__ == "__main__":
    main()
