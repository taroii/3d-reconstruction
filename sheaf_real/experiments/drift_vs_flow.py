"""
DECISIVE experiment (notes/instruction_drift_vs_flow.md): does our H^1 detector
overtake the FLOW detector as camera poses degrade, within a realistic range and
against flow's best defense (pose re-estimation)?

Setup (each method at its best backbone):
  ours        : harmonic H^1 on DUSt3R geometry (flow-free).
  flow-naive  : ego_flow(MonST3R geom, drifted poses) vs RAFT(C_T, no Sintel).
  flow-refined: re-fit poses on static (same robust GN ours uses) THEN ego_flow.

Axis: injected accumulating camera drift (deg/frame; translation ∝). drift=0 =
backbone GA poses = REALISTIC operating point (marked). Same perturbation model
applied to each method's own backbone poses (drift in each backbone's own frame;
depth+poses stay self-consistent; GT mask is frame-independent). GT mask from GT
flow vs GT-ego-flow (GT poses/depth/K) — independent of the degradation. Scored
on A's matched pixels. Metric AUROC.

Guards: (1) realistic anchor marked + GA-vs-GT pose error reported; (2) flow gets
its best defense (flow-refined); (3) shared-failure check on dynamic-heavy scenes.

    conda run -n sheaf python experiments/drift_vs_flow.py
Outputs: results/drift_vs_flow.{png,csv}, results/drift_vs_flow_guard3.csv
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

STATIC = ["alley_1", "cave_2", "market_2", "market_5", "ambush_4", "temple_3"]
DYN = ["ambush_2", "ambush_5", "bandage_1", "cave_4", "shaman_2"]
DRIFT_DEG = [0, 1, 2, 4, 8, 16]
NFRAMES, NITER, TAU = 5, 300, 3.0
ROOT, CACHE = "../data", "../cache"
os.environ.setdefault("RAFT_WEIGHTS", "C_T_V2")   # bias-free flow


def g4(R, t):
    g = np.eye(4); g[:3, :3] = R; g[:3, 3] = t; return g


def scene_paths(scene):
    import glob
    return sorted(glob.glob(os.path.join(ROOT, "training", "clean", scene,
                                         "frame_*.png")))[:NFRAMES]


def perturb(poses, deg, t_scale, seed):
    """Accumulating camera drift: deg/frame rotation, translation ∝ deg."""
    rng = np.random.default_rng(seed)
    p = poses.copy(); aR = np.eye(3); aT = np.zeros(3)
    rot = np.deg2rad(deg)
    for v in range(len(p.s)):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-12
        aR = S.so3_exp(ax * rng.uniform(0, rot)) @ aR
        aT = aT + rng.normal(scale=rot * t_scale, size=3)
        p.R[v] = aR @ p.R[v]; p.t[v] = p.t[v] + aT
    return p


def ours_maps(localpts, edges, matches, poses, shapes):
    res = S.solve_harmonic(localpts, edges, matches, poses, S.SolveCfg(n_iters=0))
    em, ec = S.splat_to_pixels(res.eps_k, res.index, shapes, reduce="median")
    return em, ec


def flow_maps(depths, foc, poses, raft_flows):
    out = {}
    for i in raft_flows:
        induced = FB.induced_flow_dust3r(depths[i], float(foc[i]), float(foc[i + 1]),
                                         g4(poses.R[i], poses.t[i]),
                                         g4(poses.R[i + 1], poses.t[i + 1]))
        out[i] = np.linalg.norm(raft_flows[i] - induced, axis=-1)
    return out


def auroc(score_by_frame, gt, valid):
    sc, lab = [], []
    for i in score_by_frame:
        if i not in valid:
            continue
        sc.append(score_by_frame[i][valid[i]]); lab.append(gt[i][valid[i]])
    if not sc:
        return float("nan")
    sc, lab = np.concatenate(sc), np.concatenate(lab)
    if lab.sum() < 15 or (~lab).sum() < 15:
        return float("nan")
    return EV.auroc(sc, lab)


def prep_scene(dust, monst, raft, scene):
    """Run both backbones once; cache geometry, matches, RAFT flow, GT, scored px."""
    paths = scene_paths(scene)
    edges = clip_edges(len(paths))
    matches = {e: Matches.load(os.path.join(CACHE, scene, "matches", f"{i}_{j}.npz"))
               for e, (i, j) in enumerate(edges)}
    od = B.run_clip(dust, paths, niter=NITER)
    om = B.run_clip(monst, paths, niter=NITER)
    # scored pixels (pose-independent) + GT per frame
    _, ec0 = ours_maps(od["localpts"], edges, matches,
                       S.Poses(od["s"], od["R"], od["t"]), od["shapes"])
    gt, valid, raft_flows = {}, {}, {}
    for i in range(len(paths) - 1):
        m, _, v = SI.motion_gt(ROOT, scene, i + 1, i + 2, tau=TAU)
        gtm = align_to_grid(m, nearest=True)
        vv = align_to_grid(v, nearest=True) & (ec0[i] > 0)
        if vv.sum() < 30 or gtm[vv].sum() < 15 or (~gtm[vv]).sum() < 15:
            continue
        gt[i] = gtm; valid[i] = vv
        raft_flows[i] = FB.predict_flow(raft, paths[i], paths[i + 1])
    return dict(paths=paths, edges=edges, matches=matches, od=od, om=om,
                gt=gt, valid=valid, raft_flows=raft_flows)


def main():
    raft = FB.load_raft()                       # C_T_V2 via env
    dust = B.load_backbone("dust3r")
    monst = B.load_backbone("monst3r")

    # ---- main sweep on static-dominant scenes ----
    curves = {"ours": {d: [] for d in DRIFT_DEG},
              "flow-naive": {d: [] for d in DRIFT_DEG},
              "flow-refined": {d: [] for d in DRIFT_DEG}}
    ga_rot_err = []
    for scene in STATIC:
        S_ = prep_scene(dust, monst, raft, scene)
        if not S_["gt"]:
            print(f"  {scene}: no usable frames"); continue
        od, om = S_["od"], S_["om"]
        edges, matches = S_["edges"], S_["matches"]
        poses_d0 = S.Poses(od["s"], od["R"], od["t"])
        poses_m0 = S.Poses(om["s"], om["R"], om["t"])
        ts_d = float(np.std(od["t"], 0).mean()) + 1e-6
        ts_m = float(np.std(om["t"], 0).mean()) + 1e-6
        # realism grounding: GA(DUSt3R)-vs-GT mean consecutive rotation error
        ga_rot_err.append(ga_vs_gt_rot(scene, od))
        for d in DRIFT_DEG:
            pd = perturb(poses_d0, d, ts_d, seed=hash((scene, d)) % 2**31) if d else poses_d0
            pm = perturb(poses_m0, d, ts_m, seed=hash((scene, d)) % 2**31) if d else poses_m0
            em, _ = ours_maps(od["localpts"], edges, matches, pd, od["shapes"])
            ours_a = auroc({i: np.sqrt(em[i] + 1e-15) for i in S_["gt"]}, S_["gt"], S_["valid"])
            fn = flow_maps(om["depths"], om["focals"], pm, S_["raft_flows"])
            flow_naive_a = auroc(fn, S_["gt"], S_["valid"])
            ref = S.solve_harmonic(om["localpts"], edges, matches, pm,
                                   S.SolveCfg(n_iters=5, use_scale=False)).poses
            fr = flow_maps(om["depths"], om["focals"], ref, S_["raft_flows"])
            flow_ref_a = auroc(fr, S_["gt"], S_["valid"])
            curves["ours"][d].append(ours_a)
            curves["flow-naive"][d].append(flow_naive_a)
            curves["flow-refined"][d].append(flow_ref_a)
            print(f"  {scene:<12} drift={d:>2}°  ours={ours_a:.3f} "
                  f"flow-naive={flow_naive_a:.3f} flow-refined={flow_ref_a:.3f}")

    # ---- Guard 3: dynamic-heavy scenes, clean poses (drift 0) ----
    print(">> Guard 3 (dynamic-heavy, drift 0): shared-failure check ...")
    g3 = []
    for scene in DYN:
        try:
            S_ = prep_scene(dust, monst, raft, scene)
        except Exception as e:
            print(f"  {scene}: err {e}"); continue
        if not S_["gt"]:
            continue
        od, om = S_["od"], S_["om"]
        em, _ = ours_maps(od["localpts"], S_["edges"], S_["matches"],
                          S.Poses(od["s"], od["R"], od["t"]), od["shapes"])
        ours_a = auroc({i: np.sqrt(em[i] + 1e-15) for i in S_["gt"]}, S_["gt"], S_["valid"])
        ref = S.solve_harmonic(om["localpts"], S_["edges"], S_["matches"],
                               S.Poses(om["s"], om["R"], om["t"]),
                               S.SolveCfg(n_iters=5, use_scale=False)).poses
        fr = flow_maps(om["depths"], om["focals"], ref, S_["raft_flows"])
        flow_a = auroc(fr, S_["gt"], S_["valid"])
        g3.append((scene, ours_a, flow_a))
        print(f"  {scene:<12} ours={ours_a:.3f}  flow-refined={flow_a:.3f}")

    # ---- save + plot ----
    os.makedirs("../results", exist_ok=True)
    realistic_deg = float(np.nanmean(ga_rot_err)) if ga_rot_err else float("nan")
    with open("../results/drift_vs_flow.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["drift_deg", "ours", "ours_sd", "flow_naive",
                                       "flow_naive_sd", "flow_refined", "flow_refined_sd"])
        for d in DRIFT_DEG:
            row = [d]
            for m in ["ours", "flow-naive", "flow-refined"]:
                v = np.array([x for x in curves[m][d] if not np.isnan(x)])
                row += [round(v.mean(), 4), round(v.std(), 4)]
            w.writerow(row)
    with open("../results/drift_vs_flow_guard3.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "ours", "flow_refined"])
        w.writerows([(s, round(o, 4), round(fl, 4)) for s, o, fl in g3])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    col = {"ours": "#c0392b", "flow-naive": "#999999", "flow-refined": "#2166ac"}
    for m in ["ours", "flow-naive", "flow-refined"]:
        mean = [np.nanmean(curves[m][d]) for d in DRIFT_DEG]
        sd = [np.nanstd(curves[m][d]) for d in DRIFT_DEG]
        ax.errorbar(DRIFT_DEG, mean, yerr=sd, fmt="o-", color=col[m], lw=2,
                    capsize=4, label=m)
    ax.axhline(0.5, ls="--", c="0.6", lw=1)
    ax.axvline(0, ls=":", c="green", lw=1.5)
    ax.text(0.2, 0.47, f"realistic (GA poses)\n~{realistic_deg:.1f}°/frame vs GT",
            fontsize=8, color="green")
    ax.set_xlabel("injected camera drift (deg/frame, accumulating)")
    ax.set_ylabel("AUROC: moving vs static")
    ax.set_title("Drift-vs-flow: does H¹ overtake the flow detector under pose error?\n"
                 f"(each at best backbone; flow=MonST3R+C_T-RAFT; {len(STATIC)} scenes)")
    ax.set_ylim(0.3, 1.0); ax.legend()
    fig.tight_layout()
    fig.savefig("../results/drift_vs_flow.png", dpi=150, bbox_inches="tight")
    print(f"\n  realistic GA-vs-GT pose error ≈ {realistic_deg:.2f}°/frame")
    print("  -> results/drift_vs_flow.{png,csv}, drift_vs_flow_guard3.csv")


def ga_vs_gt_rot(scene, od):
    """Mean consecutive-frame relative-rotation error (deg) between DUSt3R GA
    poses and Sintel GT poses — grounds the realistic operating point."""
    try:
        N = len(od["R"])
        gt_R = []
        for i in range(N):
            _, Ncam = SI.read_cam(os.path.join(ROOT, "training", "camdata_left",
                                                scene, f"frame_{i+1:04d}.cam"))
            gt_R.append(Ncam[:3, :3].T)            # world2cam->cam2world rotation
        errs = []
        for i in range(N - 1):
            rel_est = od["R"][i + 1].T @ od["R"][i]
            rel_gt = gt_R[i + 1].T @ gt_R[i]
            dR = rel_est.T @ rel_gt
            errs.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        return float(np.mean(errs))
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
