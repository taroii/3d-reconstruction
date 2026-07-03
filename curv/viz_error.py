"""
Figure C: where the improvement lives. For each Sintel frame, per-pixel AbsRel of
the baseline (gamma=0) and curvature (gamma=1) arms under scale-and-shift
alignment, and the signed difference (baseline - ours; blue where curvature is
better), with the boundary decile outlined. Also writes a CSV ranking frames by
boundary-region error reduction so you can pick the strongest examples.

  cd curv
  python viz_error.py --g0 ../DDUSt3R/results/g0_s0/checkpoint-best.pth \
                      --g1 ../DDUSt3R/results/g1_s0/checkpoint-best.pth \
                      --max_frames 300
outputs figs/error/err_<rank>_<scene>_<frame>_d{reduction}.png and figs/error/ranking.csv
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_common as VC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", default="../DDUSt3R/results/g0_s0/checkpoint-best.pth")
    ap.add_argument("--g1", default="../DDUSt3R/results/g1_s0/checkpoint-best.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--out", default="figs/error")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = VC.load_models([args.g0, args.g1], args.device)
    rows = []
    for fr in VC.frame_iter(VC.SINTEL, models, args.device, args.max_frames):
        gt, valid, K = fr["gt"], fr["valid"], fr["K"]
        d0 = VC.ssi_align(fr["preds"][0][..., 2], gt, valid & (gt > 0))
        d1 = VC.ssi_align(fr["preds"][1][..., 2], gt, valid & (gt > 0))
        bnd, _ = VC.boundary_mask(gt, K, valid)
        e0, m0all, mall = VC.abs_rel(d0, gt, valid)
        e1, m1all, _ = VC.abs_rel(d1, gt, valid)
        if bnd.sum() < 50:
            continue
        red_bnd = float(e0[bnd].mean() - e1[bnd].mean())     # >0 means curvature better
        red_all = float(e0[mall].mean() - e1[mall].mean())
        rows.append((red_bnd, red_all, fr["scene"], fr["frame"], fr["rgb"], e0, e1, bnd))

    rows.sort(key=lambda r: r[0], reverse=True)              # best boundary reduction first
    with open(os.path.join(args.out, "ranking.csv"), "w") as f:
        f.write("rank,scene,frame,boundary_reduction,all_reduction\n")
        for i, r in enumerate(rows):
            f.write(f"{i},{r[2]},{r[3]},{r[0]:.5f},{r[1]:.5f}\n")

    vmax = np.percentile([np.abs(r[0]) for r in rows], 90) if rows else 0.1
    for i, (rb, ra, scene, frame, rgb, e0, e1, bnd) in enumerate(rows):
        emax = max(np.percentile(e0[e0 > 0], 95) if (e0 > 0).any() else 1,
                   np.percentile(e1[e1 > 0], 95) if (e1 > 0).any() else 1)
        diff = e0 - e1
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.0), constrained_layout=True)
        ax[0].imshow(rgb);                                            ax[0].set_title("RGB")
        ax[1].imshow(e0, cmap="inferno", vmin=0, vmax=emax);          ax[1].set_title(r"AbsRel $\gamma{=}0$")
        ax[2].imshow(e1, cmap="inferno", vmin=0, vmax=emax);          ax[2].set_title(r"AbsRel $\gamma{=}1$")
        im = ax[3].imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
        ax[3].contour(bnd.astype(float), levels=[0.5], colors="k", linewidths=0.3)
        ax[3].set_title(r"$\gamma{=}0 - \gamma{=}1$ (blue: ours better)")
        for a in ax:
            a.axis("off")
        fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
        fig.suptitle(f"{scene}/{frame}   boundary reduction {rb:+.4f}", fontsize=9)
        base = os.path.join(args.out, f"err_{i:03d}_{scene}_{frame}_d{rb:+.4f}")
        fig.savefig(base + ".png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {len(rows)} error figures + ranking.csv to {args.out}")


if __name__ == "__main__":
    main()
