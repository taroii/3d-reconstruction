"""
Figure B: novel-view renders of the predicted pointmaps. For each Sintel frame,
a strip of RGB, the ground-truth pointmap, and the baseline (gamma=0) and
curvature (gamma=1) predicted pointmaps, each rendered from a rotated viewpoint so
that depth-edge bleeding is visible. Renders EVERY frame (across all Sintel scenes)
so there are many candidates, and writes a CSV ranking them by boundary-region
error reduction; the filename also carries the reduction, so you can sort visually
and pick the strongest and most diverse examples.

  cd curv
  python viz_pointmap.py --g0 ../DDUSt3R/results/g0_s0/checkpoint-best.pth \
                         --g1 ../DDUSt3R/results/g1_s0/checkpoint-best.pth \
                         --max_frames 400 --yaw 25
outputs figs/pointmap/pm_<scene>_<frame>_d{reduction}.png + figs/pointmap/ranking.csv
Rerun with a different --yaw to get other viewpoints.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_common as VC
import curvature as CV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", default="../DDUSt3R/results/g0_s0/checkpoint-best.pth")
    ap.add_argument("--g1", default="../DDUSt3R/results/g1_s0/checkpoint-best.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--yaw", type=float, default=25.0)
    ap.add_argument("--pitch", type=float, default=12.0)
    ap.add_argument("--out", default="figs/pointmap")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = VC.load_models([args.g0, args.g1], args.device)
    rank = []
    for fr in VC.frame_iter(VC.SINTEL, models, args.device, args.max_frames):
        rgb, gt, valid, K = fr["rgb"], fr["gt"], fr["valid"], fr["K"]
        vg = valid & (gt > 0)

        # boundary error reduction (for ranking): ssi-aligned AbsRel, g0 vs g1
        d0 = VC.ssi_align(fr["preds"][0][..., 2], gt, vg)
        d1 = VC.ssi_align(fr["preds"][1][..., 2], gt, vg)
        bnd, _ = VC.boundary_mask(gt, K, valid)
        e0, _, _ = VC.abs_rel(d0, gt, valid)
        e1, _, _ = VC.abs_rel(d1, gt, valid)
        red = float(e0[bnd].mean() - e1[bnd].mean()) if bnd.sum() > 50 else 0.0

        gt_pts = CV.backproject(np.nan_to_num(gt, nan=0.0), K)
        r_gt = VC.render_points(gt_pts, rgb, vg, args.yaw, args.pitch)
        r0 = VC.render_points(fr["preds"][0], rgb, valid, args.yaw, args.pitch)
        r1 = VC.render_points(fr["preds"][1], rgb, valid, args.yaw, args.pitch)

        fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
        ax[0].imshow(rgb);   ax[0].set_title("RGB")
        ax[1].imshow(r_gt);  ax[1].set_title("GT pointmap")
        ax[2].imshow(r0);    ax[2].set_title(r"$\gamma{=}0$ (baseline)")
        ax[3].imshow(r1);    ax[3].set_title(r"$\gamma{=}1$ (ours)")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{fr['scene']}/{fr['frame']}   boundary reduction {red:+.4f}  "
                     f"(yaw {args.yaw:g})", fontsize=9)
        fig.tight_layout()
        base = os.path.join(args.out, f"pm_{fr['scene']}_{fr['frame']}_d{red:+.4f}")
        fig.savefig(base + ".png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        rank.append((red, fr["scene"], fr["frame"]))

    rank.sort(reverse=True)
    with open(os.path.join(args.out, "ranking.csv"), "w") as f:
        f.write("rank,scene,frame,boundary_reduction\n")
        for i, (red, scene, frame) in enumerate(rank):
            f.write(f"{i},{scene},{frame},{red:+.5f}\n")
    scenes = sorted(set(s for _, s, _ in rank))
    print(f"wrote {len(rank)} pointmap figures across {len(scenes)} scenes to {args.out}")
    print(f"scenes: {', '.join(scenes)}")
    print("top-5 by boundary reduction:")
    for red, scene, frame in rank[:5]:
        print(f"  {red:+.4f}  {scene}/{frame}")


if __name__ == "__main__":
    main()
