"""
Complete the fair comparison: ours (harmonic H^1) on EACH backbone's geometry,
so it can be compared to flow-residual on the same geometry (dynamic_mask_
comparison.py gave the flow row). MASt3R matches are backbone-independent, so the
scored-pixel set and GT are identical across geometries -> directly comparable.

    conda run -n sheaf python experiments/ours_by_geometry.py
Outputs: results/ours_by_geometry.csv (combine with dyn_mask_comparison.csv).
"""
import os
import sys
import gc
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backbone as B
import sheaf as S
import eval as EV
import sintel as SI
from datatypes import Matches
from pipeline import clip_edges, align_to_grid

SCENES = ["alley_1", "cave_2", "market_2", "market_5", "ambush_4", "temple_3"]
NFRAMES, NITER, TAU = 5, 300, 3.0
ROOT, CACHE = "../data", "../cache"


def scene_paths(scene):
    import glob
    return sorted(glob.glob(os.path.join(ROOT, "training", "clean", scene,
                                         "frame_*.png")))[:NFRAMES]


def ours_auroc(out, edges, matches, scene):
    res = S.solve_harmonic(out["localpts"], edges, matches,
                           S.Poses(out["s"], out["R"], out["t"]),
                           S.SolveCfg(n_iters=0))
    emaps, ecnt = S.splat_to_pixels(res.eps_k, res.index, out["shapes"], reduce="median")
    sc, lab = [], []
    for i in range(len(out["localpts"]) - 1):
        m, _, v = SI.motion_gt(ROOT, scene, i + 1, i + 2, tau=TAU)
        g = align_to_grid(m, nearest=True); vv = align_to_grid(v, nearest=True) & (ecnt[i] > 0)
        if vv.sum() < 30 or g[vv].sum() < 15 or (~g[vv]).sum() < 15:
            continue
        sc.append(np.sqrt(emaps[i][vv] + 1e-15)); lab.append(g[vv])
    if not lab:
        return float("nan")
    return EV.auroc(np.concatenate(sc), np.concatenate(lab))


def main():
    rows = {g: {} for g in ["dust3r", "monst3r", "d2ust3r"]}
    for geom in ["dust3r", "monst3r", "d2ust3r"]:
        print(f">> ours on {geom} geometry ...")
        model = B.load_backbone(geom)
        for sc in SCENES:
            paths = scene_paths(sc)
            edges = clip_edges(len(paths))
            matches = {e: Matches.load(os.path.join(CACHE, sc, "matches", f"{i}_{j}.npz"))
                       for e, (i, j) in enumerate(edges)}
            out = B.run_clip(model, paths, niter=NITER)
            rows[geom][sc] = ours_auroc(out, edges, matches, sc)
            print(f"  {sc:<12} ours@{geom}={rows[geom][sc]:.3f}")
        del model; gc.collect()
        import torch; torch.cuda.empty_cache()

    os.makedirs("../results", exist_ok=True)
    with open("../results/ours_by_geometry.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "ours@dust3r", "ours@monst3r", "ours@d2ust3r"])
        for sc in SCENES:
            w.writerow([sc] + [f"{rows[g].get(sc, float('nan')):.4f}" for g in
                               ["dust3r", "monst3r", "d2ust3r"]])
    print("\n  ours mean AUROC by geometry:")
    for g in ["dust3r", "monst3r", "d2ust3r"]:
        v = np.array([rows[g][sc] for sc in SCENES if sc in rows[g] and not np.isnan(rows[g][sc])])
        print(f"  ours@{g:<8} {v.mean():.3f} ± {v.std():.3f}")
    print("  -> results/ours_by_geometry.csv")


if __name__ == "__main__":
    main()
