"""
Vector-level discriminator on the D2USt3R 2.6x own-frame finding (planning-agent
instruction). Soundness check on Formulation A's world-frame metric; NOT an
attempt to rescue R2.

For GT-dynamic matched points (restricted to the top tertile by own-frame
geometry change), form A's world-frame cross-frame discrepancy vector
    d_k = w_i_k - w_j_k,   w_v_k = g_v . X_vv[p_v]
under DUSt3R and under D2USt3R, and compare them as VECTORS:
  - magnitude ratio ||d^D2|| / ||d^DUSt3R||
  - direction angle(d^D2, d^DUSt3R)
The two reconstructions have independent GA gauges, so D2USt3R's vectors are
first mapped into DUSt3R's frame by a similarity (Umeyama) fit on STATIC matched
points (translation cancels in the difference d; apply s*R).

Branches:
  Reading 1 (PASS, scale A): magnitudes ~preserved AND directions rotated
    (angles >> 0) -> geometry moved within the inconsistency's null space; A
    correctly still sees motion.
  Reading 2 (FAIL, fix A): d vectors near-identical (ratio~1, angle~0) despite
    the 2.6x own-frame change -> world-frame composition eats the change ->
    pipeline defect; localize by peeling stages.
  Ambiguous: ||d^D2|| << ||d^DUSt3R|| (magnitude collapse) -> flag, stop.

    conda run -n sheaf python experiments/r2_vector_discriminator.py --scene alley_1 --nframes 5
"""

import os
import sys
import argparse
import subprocess
import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backbone as B
import sheaf as S
import sintel as SI
from datatypes import Matches
from pipeline import clip_edges, align_to_grid


def umeyama_sim(src, dst):
    """Similarity (s,R,t) mapping src->dst (both (N,3)); standard Umeyama."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    Sigma = (Xd.T @ Xs) / len(src)
    U, D, Vt = np.linalg.svd(Sigma)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var_s = (Xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ W) / (var_s + 1e-12)
    t = mu_d - s * R @ mu_s
    return s, R, t


def collect(out, edges, matches):
    """Per matched point: d=w_i-w_j, w_i, w_j (world), pix_i (view,row,col)."""
    poses = S.Poses(out["s"], out["R"], out["t"])
    lp = out["localpts"]
    D, Wi, refs = [], [], []
    for e, (i, j) in enumerate(edges):
        m = matches[e]
        for k in range(len(m.conf)):
            yi, xi = m.pix_i[k]; yj, xj = m.pix_j[k]
            wi = poses.world(i, lp[i][yi, xi]); wj = poses.world(j, lp[j][yj, xj])
            D.append(wi - wj); Wi.append(wi); refs.append((i, yi, xi, j, yj, xj))
    return np.array(D), np.array(Wi), refs


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

    # GT-dynamic mask + own-frame depth per view, per backbone (for tertile)
    gt = {}
    for vi, k in enumerate(frames):
        m, _, v = SI.motion_gt(root, args.scene, k + 1, k + 2, tau=args.tau)
        gt[vi] = align_to_grid(m, nearest=True)

    outs, depths = {}, {}
    for name in ["dust3r", "d2ust3r"]:
        print(f">> {name} ...")
        model = B.load_backbone(name)
        out = B.run_clip(model, paths, niter=args.niter)
        outs[name] = {kk: out[kk] for kk in ("localpts", "s", "R", "t")}
        depths[name] = [d.copy() for d in out["depths"]]
        del model, out; gc.collect()
        import torch; torch.cuda.empty_cache()

    D1, W1, refs = collect(outs["dust3r"], edges, matches)
    D2, W2, _ = collect(outs["d2ust3r"], edges, matches)

    # labels + per-point own-frame change (scale-normalized log-depth, mean of endpoints)
    nv = {n: [d / np.median(d[d > 1e-6]) for d in depths[n]] for n in depths}
    is_dyn = np.array([gt[i][yi, xi] for (i, yi, xi, j, yj, xj) in refs])
    ofc = []
    for (i, yi, xi, j, yj, xj) in refs:
        a = abs(np.log(nv["dust3r"][i][yi, xi] + 1e-9) - np.log(nv["d2ust3r"][i][yi, xi] + 1e-9))
        b = abs(np.log(nv["dust3r"][j][yj, xj] + 1e-9) - np.log(nv["d2ust3r"][j][yj, xj] + 1e-9))
        ofc.append(0.5 * (a + b))
    ofc = np.array(ofc)

    # align D2USt3R world frame -> DUSt3R via similarity on STATIC matched points
    stat = ~is_dyn
    s_al, R_al, _ = umeyama_sim(W2[stat], W1[stat])
    print(f"   static-Umeyama D2->DUSt3R: scale={s_al:.3f}  (rot det "
          f"{np.linalg.det(R_al):+.2f}, n_static={stat.sum()})")
    D2a = (s_al * (R_al @ D2.T)).T              # d is a difference -> only s,R

    # high-change dynamic subset (top tertile of own-frame change among dynamic)
    dyn = is_dyn.copy()
    thr = np.percentile(ofc[dyn], 66.7)
    sub = dyn & (ofc >= thr)
    n1 = np.linalg.norm(D1[sub], axis=1); n2 = np.linalg.norm(D2a[sub], axis=1)
    keep = (n1 > 1e-9) & (n2 > 1e-9)
    n1, n2 = n1[keep], n2[keep]
    ratio = n2 / n1
    cos = np.sum(D1[sub][keep] * D2a[sub][keep], axis=1) / (n1 * n2)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))

    def pct(a): return np.round(np.percentile(a, [10, 50, 90]), 3)
    print(f"\n  high-change dynamic subset: n={keep.sum()} "
          f"(tertile thr own-frame-change={thr:.2f})")
    print(f"  magnitude ratio ||d^D2||/||d^DUSt3R||  (10/50/90): {pct(ratio)}")
    print(f"  direction angle(d^D2, d^DUSt3R) [deg]  (10/50/90): {pct(ang)}")

    # --- reconciling DATA (not a story): does the AUROC tie survive at vector level? ---
    def med_norm(M, mask):
        n = np.linalg.norm(M[mask], axis=1); return float(np.median(n[n > 1e-9]))
    # all-dynamic magnitude ratio
    nd1 = np.linalg.norm(D1[dyn], axis=1); nd2 = np.linalg.norm(D2a[dyn], axis=1)
    kk = (nd1 > 1e-9) & (nd2 > 1e-9)
    print(f"  [all dynamic] magnitude ratio (10/50/90): {pct((nd2[kk]/nd1[kk]))}")
    # per-backbone dynamic/static |d| contrast in COMMON scale (vector analog of AUROC ratio)
    du_dyn, du_sta = med_norm(D1, dyn), med_norm(D1, stat)
    d2_dyn, d2_sta = med_norm(D2a, dyn), med_norm(D2a, stat)
    print(f"  median |d| (common scale): DUSt3R dyn={du_dyn:.4f} static={du_sta:.4f} "
          f"contrast={du_dyn/du_sta:.2f}")
    print(f"  median |d| (common scale): D2USt3R dyn={d2_dyn:.4f} static={d2_sta:.4f} "
          f"contrast={d2_dyn/d2_sta:.2f}")
    print(f"  (if both contrasts similar -> AUROC tie reconciled; static collapsed too)")

    mr, ma = np.median(ratio), np.median(ang)
    if mr < 0.5:
        verdict = "AMBIGUOUS: magnitude collapse (||d^D2||<<||d^DUSt3R||) -- flag, stop."
    elif ma > 25 and mr > 0.6:
        verdict = "READING 1 (PASS): magnitudes preserved, directions rotated -> A sound, SCALE."
    elif ma < 15 and 0.85 < mr < 1.15:
        verdict = "READING 2 (FAIL): d near-identical despite 2.6x own-frame change -> A pipeline defect; localize."
    else:
        verdict = f"INCONCLUSIVE (median ratio={mr:.2f}, angle={ma:.1f}deg) -- human call."
    print(f"\n  --> {verdict}")

    # scatter ||d^D2|| vs ||d^DUSt3R|| colored by angle
    os.makedirs("../results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(n1, n2, c=ang, cmap="viridis", s=10, alpha=0.6, vmin=0, vmax=180)
    lim = np.percentile(np.concatenate([n1, n2]), 99)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x (identical magnitude)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("‖d‖ DUSt3R"); ax.set_ylabel("‖d‖ D²USt3R (aligned)")
    ax.set_title(f"{args.scene}: cross-frame discrepancy vectors,\nhigh-change "
                 f"dynamic subset (median ratio {mr:.2f}, angle {ma:.0f}°)")
    fig.colorbar(sc, ax=ax, label="angle(d^D², d^DUSt3R) [deg]")
    ax.legend(fontsize=9)
    fig.savefig(f"../results/r2_vecdiscrim_{args.scene}.png", dpi=150,
                bbox_inches="tight")
    print(f"  -> results/r2_vecdiscrim_{args.scene}.png")


if __name__ == "__main__":
    main()
