"""
Figure A: what the loss upweights. For each of several frames, a 4-panel strip of
RGB, ground-truth depth, the curvature magnitude |K| that drives the weight, and
the per-pixel weight w = 1 + gamma * min(|K|/mean|K|, tau). No model, no GPU: this
only visualizes the ground-truth signal, so it cannot overclaim.

  cd curv
  python viz_signal.py --dataset sintel --split val --n 12 --gamma 1 --tau 10
outputs signal_<scene>_<frame>.png (+ .pdf) into figs/signal/.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets as DS
import curvature as CV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sintel")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--only", default="",
                    help="render only frames whose key contains this substring "
                         "(e.g. ambush_6_000001); empty = the sampled set")
    ap.add_argument("--grid", action="store_true",
                    help="lay the four panels out as a 2x2 grid (larger panels)")
    ap.add_argument("--out", default="figs/signal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = DS.build_frames([args.dataset], args.split)
    if not frames:
        raise SystemExit(f"no {args.dataset}/{args.split} frames")
    if args.only:
        picks = [f for f in frames if args.only in f.key.replace("/", "_")]
        if not picks:
            raise SystemExit(f"no frame matching '{args.only}'")
    else:
        step = max(1, len(frames) // args.n)
        picks = frames[::step][:args.n]
    cfg = DS.CFG[args.dataset]

    for fr in picks:
        depth, K = DS.load_depth_K(fr)
        Kmap, valid = CV.curvature_from_depth_K(
            depth, K, mode="mean", rel_thresh=cfg["rel_thresh"],
            max_depth=cfg["max_depth"], normalize=True)
        S = np.abs(np.where(valid, Kmap, 0.0))
        Sbar = S[valid].mean() if valid.any() else 1.0
        w = 1.0 + args.gamma * np.minimum(S / max(Sbar, 1e-8), args.tau)
        w = np.where(valid, w, 1.0)

        import imageio.v2 as imageio
        rgb = imageio.imread(fr.rgb_path)
        d_show = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)

        if args.grid:
            fig, axg = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
            ax = axg.ravel()                                 # [rgb, depth, |K|, weight]
        else:
            fig, ax = plt.subplots(1, 4, figsize=(16, 3.2), constrained_layout=True)
        ax[0].imshow(rgb);                                   ax[0].set_title("RGB")
        im1 = ax[1].imshow(d_show, cmap="turbo");            ax[1].set_title("GT depth")
        im2 = ax[2].imshow(np.where(valid, S, np.nan), cmap="magma",
                           vmax=np.nanpercentile(S[valid], 98) if valid.any() else 1)
        ax[2].set_title(r"$|K_i|$ (curvature)")
        im3 = ax[3].imshow(w, cmap="viridis", vmin=1.0,
                           vmax=1.0 + args.gamma * args.tau)
        ax[3].set_title(r"weight $w_i = 1+\gamma\,\min(|K_i|/\bar S,\tau)$")
        for a in ax:
            a.axis("off")
        for im, a, lab in [(im1, ax[1], "depth"),
                           (im2, ax[2], r"curvature $|K_i|$"),
                           (im3, ax[3], r"weight $w_i$")]:
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label=lab)
        base = os.path.join(args.out, f"signal_{fr.key.replace('/', '_')}")
        fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {base}.png")


if __name__ == "__main__":
    main()
