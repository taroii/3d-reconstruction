"""
Ours vs the learned dynamic detectors (plan §3a / positioning). KEY FACT: the
released MonST3R/D2USt3R dynamic mask IS the flow residual ||ego_flow - RAFT||
(get_dynamic_mask_from_pairviewer / get_motion_mask_from_pairs). There is no
separate SDAP dynamic mask. So "their detector" = flow residual; the meaningful
ladder holds the flow method fixed and varies the GEOMETRY that feeds ego-flow:

  ours        : harmonic H^1 on frozen DUSt3R + MASt3R (flow-free, no dyn. training)
  flow-DUSt3R : ego_flow(DUSt3R geom) vs RAFT      (generic flow baseline)
  flow-MonST3R: ego_flow(MonST3R geom) vs RAFT     (dynamic-trained backbone)
  flow-D2USt3R: ego_flow(D2USt3R geom) vs RAFT     (SOTA dynamic backbone)

All scored on A's matched pixels, same GT (per-frame dynamic mask), same metric,
same static-dominant scenes, stride 1. RAFT flow is geometry-independent so it is
computed once per scene and reused across the three geometries.

    conda run -n sheaf python experiments/dynamic_mask_comparison.py
Outputs: results/dyn_mask_comparison.csv, results/dyn_mask_comparison.png
"""

import os
import sys
import gc
import csv
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

SCENES = ["alley_1", "cave_2", "market_2", "market_5", "ambush_4", "temple_3"]
NFRAMES, NITER, TAU = 5, 300, 3.0
ROOT, CACHE = "../data", "../cache"


def g4(R, t):
    g = np.eye(4); g[:3, :3] = R; g[:3, 3] = t; return g


def scene_paths(scene):
    import glob
    return sorted(glob.glob(os.path.join(ROOT, "training", "clean", scene,
                                         "frame_*.png")))[:NFRAMES]


def flow_residual_scores(out, raft_flows):
    """Per-frame flow-residual scores for a backbone's geometry, only on the
    valid frames present in raft_flows."""
    depths, foc, R, t = out["depths"], out["focals"], out["R"], out["t"]
    scores = {}
    for i in raft_flows:
        induced = FB.induced_flow_dust3r(depths[i], float(foc[i]), float(foc[i + 1]),
                                         g4(R[i], t[i]), g4(R[i + 1], t[i + 1]))
        scores[i] = np.linalg.norm(raft_flows[i] - induced, axis=-1)
    return scores


def pooled_auroc(score_by_frame, ref):
    """AUROC over A's scored pixels (ref['valid'][i]) vs GT (ref['gt'][i])."""
    sc, lab = [], []
    for i, s in score_by_frame.items():
        v = ref["valid"][i]
        sc.append(s[v]); lab.append(ref["gt"][i][v])
    sc, lab = np.concatenate(sc), np.concatenate(lab)
    return EV.auroc(sc, lab)


def main():
    raft = FB.load_raft()
    refs = {}            # per scene: A scores, valid (scored∩occ), gt, raft flows, paths

    # ---- DUSt3R pass: ours (H^1) + DUSt3R-geometry flow + RAFT flows ----
    print(">> DUSt3R: ours (H^1) + flow(DUSt3R) ...")
    dust3r = B.load_backbone("dust3r")
    rows = {m: {} for m in ["ours", "flow-DUSt3R", "flow-MonST3R", "flow-D2USt3R"]}
    for sc in SCENES:
        paths = scene_paths(sc)
        edges = clip_edges(len(paths))
        matches = {e: Matches.load(os.path.join(CACHE, sc, "matches", f"{i}_{j}.npz"))
                   for e, (i, j) in enumerate(edges)}
        out = B.run_clip(dust3r, paths, niter=NITER)
        res = S.solve_harmonic(out["localpts"], edges, matches, S.Poses(out["s"], out["R"], out["t"]),
                               S.SolveCfg(n_iters=0))
        emaps, ecnt = S.splat_to_pixels(res.eps_k, res.index, out["shapes"], reduce="median")
        gt, valid, raft_flows, A_scores = {}, {}, {}, {}
        for i in range(len(paths) - 1):
            m, _, v = SI.motion_gt(ROOT, sc, i + 1, i + 2, tau=TAU)
            g = align_to_grid(m, nearest=True); vv = align_to_grid(v, nearest=True) & (ecnt[i] > 0)
            if vv.sum() < 30 or g[vv].sum() < 15 or (~g[vv]).sum() < 15:
                continue
            gt[i] = g; valid[i] = vv
            raft_flows[i] = FB.predict_flow(raft, paths[i], paths[i + 1])
            A_scores[i] = np.sqrt(emaps[i] + 1e-15)
        if not gt:
            print(f"  {sc}: no usable frames, skip"); continue
        refs[sc] = dict(paths=paths, gt=gt, valid=valid, raft_flows=raft_flows)
        rows["ours"][sc] = pooled_auroc({i: A_scores[i] for i in gt}, refs[sc])
        rows["flow-DUSt3R"][sc] = pooled_auroc(
            {i: s for i, s in flow_residual_scores(out, raft_flows).items() if i in gt},
            refs[sc])
        print(f"  {sc:<12} ours={rows['ours'][sc]:.3f}  flow-DUSt3R={rows['flow-DUSt3R'][sc]:.3f}")
    del dust3r; gc.collect()
    import torch; torch.cuda.empty_cache()

    # ---- MonST3R / D2USt3R passes: flow residual with their geometry ----
    for name, key in [("monst3r", "flow-MonST3R"), ("d2ust3r", "flow-D2USt3R")]:
        print(f">> {name}: {key} ...")
        model = B.load_backbone(name)
        for sc in SCENES:
            if sc not in refs:
                continue
            paths = refs[sc]["paths"]
            out = B.run_clip(model, paths, niter=NITER)
            fr = flow_residual_scores(out, refs[sc]["raft_flows"])
            rows[key][sc] = pooled_auroc({i: s for i, s in fr.items() if i in refs[sc]["gt"]},
                                         refs[sc])
            print(f"  {sc:<12} {key}={rows[key][sc]:.3f}")
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- table + bar chart ----
    methods = ["ours", "flow-DUSt3R", "flow-MonST3R", "flow-D2USt3R"]
    os.makedirs("../results", exist_ok=True)
    with open("../results/dyn_mask_comparison.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene"] + methods)
        for sc in SCENES:
            if sc in refs:
                w.writerow([sc] + [f"{rows[m].get(sc, float('nan')):.4f}" for m in methods])
    print("\n  method        mean AUROC ± sd (scored px, stride 1)")
    means = {}
    for m in methods:
        v = np.array([rows[m][sc] for sc in refs if sc in rows[m]])
        means[m] = (v.mean(), v.std())
        print(f"  {m:<14} {v.mean():.3f} ± {v.std():.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = np.arange(len(methods))
    vals = [means[m][0] for m in methods]; errs = [means[m][1] for m in methods]
    colors = ["#c0392b", "0.6", "#6699cc", "#2166ac"]
    ax.bar(xs, vals, yerr=errs, capsize=5, color=colors, alpha=0.85)
    for sc in refs:
        ax.plot(xs, [rows[m].get(sc, np.nan) for m in methods], "o-", color="k",
                alpha=0.3, lw=0.8, ms=4)
    ax.axhline(0.5, ls="--", c="0.5", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("AUROC: moving vs static (A's scored px)")
    ax.set_title("Dynamic detection: ours (flow-free) vs flow-residual by geometry\n"
                 f"(stride 1, {len(refs)} static-dominant Sintel scenes)")
    ax.set_ylim(0.3, 1.02)
    fig.tight_layout()
    fig.savefig("../results/dyn_mask_comparison.png", dpi=150, bbox_inches="tight")
    print("  -> results/dyn_mask_comparison.{png,csv}")


if __name__ == "__main__":
    main()
