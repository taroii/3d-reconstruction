"""
Figure C: where the improvement lives. For each Sintel frame, three panels:
RGB, the curvature-arm (gamma=1) predicted depth (scale-and-shift aligned), and
the per-pixel AbsRel reduction (baseline gamma=0 minus curvature gamma=1; blue
where curvature is better) with the high-curvature boundary decile outlined.
Also writes a CSV ranking frames by boundary-region error reduction so you can
pick the strongest examples.

  cd curv
  python viz_error.py --g0 ../DDUSt3R/results/g0_s0/checkpoint-best.pth \
                      --g1 ../DDUSt3R/results/g1_s0/checkpoint-best.pth \
                      --max_frames 300 --scenes ambush_2,ambush_4,alley_1
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
    ap.add_argument("--top_k", type=int, default=40,
                    help="save only the top-K figures by boundary reduction (0 = all); "
                         "ranking.csv always lists every frame")
    ap.add_argument("--scenes", default="",
                    help="comma-separated scene-name substrings to keep (e.g. "
                         "ambush_2,alley_1); empty = all scenes")
    ap.add_argument("--out", default="figs/error")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    keep = [s.strip() for s in args.scenes.split(",") if s.strip()]

    models = VC.load_models([args.g0, args.g1], args.device)
    rows = []
    for fr in VC.frame_iter(VC.SINTEL, models, args.device, args.max_frames):
        if keep and not any(k in str(fr["scene"]) for k in keep):
            continue
        gt, valid, K = fr["gt"], fr["valid"], fr["K"]
        vg = valid & (gt > 0)
        d0 = VC.ssi_align(fr["preds"][0][..., 2], gt, vg)
        d1 = VC.ssi_align(fr["preds"][1][..., 2], gt, vg)
        bnd, _ = VC.boundary_mask(gt, K, valid)
        e0, _, _ = VC.abs_rel(d0, gt, valid)
        e1, _, mall = VC.abs_rel(d1, gt, valid)
        if bnd.sum() < 50:
            continue
        red_bnd = float(e0[bnd].mean() - e1[bnd].mean())     # >0 means curvature better
        red_all = float(e0[mall].mean() - e1[mall].mean())
        rows.append((red_bnd, red_all, fr["scene"], fr["frame"], fr["rgb"], d1, e0, e1, bnd, vg))

    rows.sort(key=lambda r: r[0], reverse=True)              # best boundary reduction first
    with open(os.path.join(args.out, "ranking.csv"), "w") as f:
        f.write("rank,scene,frame,boundary_reduction,all_reduction\n")
        for i, r in enumerate(rows):
            f.write(f"{i},{r[2]},{r[3]},{r[0]:.5f},{r[1]:.5f}\n")

    vmax = np.percentile([np.abs(r[0]) for r in rows], 90) if rows else 0.1
    save = rows[:args.top_k] if args.top_k else rows
    for i, (rb, ra, scene, frame, rgb, d1, e0, e1, bnd, vg) in enumerate(save):
        # curvature-arm predicted depth, shown only on valid pixels (colors = depth)
        dshow = np.where(vg, d1, np.nan)
        dv = d1[vg]
        dlo = np.percentile(dv, 2) if dv.size else 0.0
        dhi = np.percentile(dv, 98) if dv.size else 1.0
        diff = e0 - e1                                       # >0 = curvature better

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
        ax[0].imshow(rgb)
        ax[0].set_title("RGB")
        dd = ax[1].imshow(dshow, cmap="turbo", vmin=dlo, vmax=dhi)
        ax[1].set_title(r"Predicted depth ($\gamma{=}1$)")
        fig.colorbar(dd, ax=ax[1], fraction=0.046, pad=0.04, label="relative depth")
        # RdBu: positive (curvature better) -> blue, matching the caption
        im = ax[2].imshow(diff, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax[2].contour(bnd.astype(float), levels=[0.5], colors="k", linewidths=0.35)
        ax[2].set_title(r"AbsRel reduction $\gamma{=}0-\gamma{=}1$")
        fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{scene}/{frame}   boundary reduction {rb:+.4f}", fontsize=9)
        base = os.path.join(args.out, f"err_{i:03d}_{scene}_{frame}_d{rb:+.4f}")
        fig.savefig(base + ".png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print(f"ranked {len(rows)} frames (ranking.csv); saved top {len(save)} error figures "
          f"(err_000 = biggest boundary win) to {args.out}")


if __name__ == "__main__":
    main()
