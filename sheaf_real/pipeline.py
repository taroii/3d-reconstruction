"""
End-to-end real-data pipeline (Formulation A). Runs in the `sheaf` env.

  frames --[MASt3R matching, subprocess]--> cache/matches
         --[DDUSt3R backbone, this process]--> per-view pointmaps/poses/conf
         --[sheaf]--> harmonic H^1 energy --> per-view maps
         --[eval]--> AUROC vs Sintel motion GT

MASt3R matching is a subprocess so its bundled `dust3r` never clashes with
DDUSt3R's MonST3R-fork `dust3r` imported here.

    conda run -n sheaf python pipeline.py --scene alley_1 --nframes 5
"""

import os
import sys
import argparse
import subprocess
import numpy as np

import backbone as B
import sheaf as S
import eval as EV
import sintel as SI
from datatypes import Matches


def clip_edges(n):
    """Temporal chain + two long-range loop edges (>=3 views, cycles)."""
    edges = [(k, k + 1) for k in range(n - 1)] + [(0, n - 1), (0, n // 2)]
    return [(a, b) for a, b in edges if a != b]


def align_to_grid(arr, nearest=True):
    """Apply the backbone's resize+crop (crop_img) to a full-res HxW array so it
    lands on the (208,512)-style grid the pointmaps live on."""
    import PIL.Image
    from dust3r.utils.image import crop_img
    mode = "uint8" if arr.dtype == bool else "F"
    a = (arr.astype(np.uint8) * 255) if arr.dtype == bool else arr.astype(np.float32)
    pil = PIL.Image.fromarray(a, mode=("L" if arr.dtype == bool else "F"))
    pil = crop_img(pil, 512, nearest=nearest)
    out = np.asarray(pil)
    return out > 127 if arr.dtype == bool else out


def run(scene, frames, backbone="dust3r", tau=3.0, niter=300,
        root="../data", cache="../cache"):
    paths_all = SI.frame_paths(root, scene, "clean")
    paths = [paths_all[k] for k in frames]
    n = len(paths)
    edges = clip_edges(n)

    # 1. MASt3R matching (subprocess) -> cache/<scene>/matches/<i>_<j>.npz
    mdir = os.path.join(cache, scene, "matches")
    fr = " ".join(str(k) for k in frames)
    cmd = [sys.executable, "match_mast3r.py", "--clip",
           os.path.join(root, "training", "clean", scene),
           "--out", mdir, "--frames", *[str(k) for k in frames]]
    print(">> matching (MASt3R subprocess) ...")
    subprocess.run(cmd, check=True)

    # 2. backbone (DDUSt3R) -> per-view geometry + poses
    print(f">> backbone ({backbone}) + global alignment ...")
    model = B.load_backbone(backbone)
    out = B.run_clip(model, paths, niter=niter)
    poses = S.Poses(out["s"], out["R"], out["t"])
    pointmaps, shapes = out["localpts"], out["shapes"]

    # 3. load matches for the chosen edges
    matches = {}
    for e_idx, (i, j) in enumerate(edges):
        matches[e_idx] = Matches.load(os.path.join(mdir, f"{i}_{j}.npz"))

    # 4. sheaf solve -> per-correspondence harmonic + raw energy
    print(">> sheaf solve ...")
    res = S.solve_harmonic(pointmaps, edges, matches, poses, S.SolveCfg())
    emaps, ecnt = S.splat_to_pixels(res.eps_k, res.index, shapes, reduce="median")
    rmaps, _ = S.splat_to_pixels(res.raw_k, res.index, shapes, reduce="median")

    # 5. motion GT per view (consecutive-frame scene flow), aligned to grid.
    # The detector only produces a score at matched pixels, so AUROC is evaluated
    # over *scored* pixels (hit-count>0); dense masks come later via diffusion.
    gt, valid = {}, {}
    for vi, k in enumerate(frames):
        fa = k + 1                                   # Sintel files are 1-based
        m, _, v = SI.motion_gt(root, scene, fa, fa + 1, tau=tau)
        gt[vi] = align_to_grid(m, nearest=True)
        valid[vi] = align_to_grid(v, nearest=True) & (ecnt[vi] > 0)

    au_h = EV.map_auroc(emaps, gt, valid)
    au_r = EV.map_auroc(rmaps, gt, valid)
    dynfrac = np.mean([gt[v][valid[v]].mean() for v in gt])
    print(f"\n  scene={scene} frames={frames} backbone={backbone}")
    print(f"  dynamic fraction (GT, aligned): {dynfrac:.3f}")
    print(f"  AUROC harmonic H^1 : {au_h:.3f}")
    print(f"  AUROC raw residual : {au_r:.3f}")
    return dict(au_h=au_h, au_r=au_r, emaps=emaps, rmaps=rmaps, gt=gt, valid=valid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="alley_1")
    ap.add_argument("--nframes", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--backbone", default="dust3r")
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--niter", type=int, default=300)
    args = ap.parse_args()
    frames = list(range(args.start, args.start + args.nframes))
    run(args.scene, frames, backbone=args.backbone, tau=args.tau, niter=args.niter)
