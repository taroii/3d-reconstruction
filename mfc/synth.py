"""
Synthetic cache + end-to-end test of the downstream pipeline.

Generates a clip whose per-view pointmaps and per-edge pixel correspondences
have the SAME interface the real backbone+cache will produce, from a known
scene (static shell + moving object). Running the real-data downstream
(build_coboundary -> solve_harmonic -> splat -> AUROC) on it must reproduce the
PoC result: harmonic H^1 localizes the mover and survives camera drift that
defeats the raw residual. This validates the plumbing before any network exists.

    conda run -n sheaf-poc python synth.py
"""

import numpy as np

from types import SimpleNamespace   # stdlib

import sheaf as S
import eval as E
from datatypes import Matches


def _cameras(n, rng, radius=5.0):
    R = np.zeros((n, 3, 3)); t = np.zeros((n, 3))
    for i in range(n):
        a = 2 * np.pi * i / n * 0.6
        pos = radius * np.array([np.cos(a), 0.3 * np.sin(2 * a), np.sin(a)])
        fwd = -pos / (np.linalg.norm(pos) + 1e-12)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, fwd); right /= np.linalg.norm(right) + 1e-12
        R[i] = np.stack([right, np.cross(fwd, right), fwd], axis=1); t[i] = pos
    return R, t


def make_synth_clip(seed=0, n_views=10, n_static=90, n_dyn=24,
                    H=64, W=96, obj_speed=0.08, obj_spin=0.05):
    rng = np.random.default_rng(seed)
    # static shell
    d = rng.normal(size=(n_static, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    static = d * rng.uniform(1.2, 2.0, size=(n_static, 1))
    # moving object (object-local + per-view pose)
    local = rng.normal(scale=0.18, size=(n_dyn, 3))
    vel = rng.normal(size=3); vel = obj_speed * vel / np.linalg.norm(vel)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    objR = np.zeros((n_views, 3, 3)); objt = np.zeros((n_views, 3)); Racc = np.eye(3)
    for i in range(n_views):
        objR[i] = Racc; objt[i] = vel * i; Racc = S.so3_exp(axis * obj_spin) @ Racc
    camR, camt = _cameras(n_views, rng)
    P = n_static + n_dyn
    is_dyn = np.concatenate([np.zeros(n_static, bool), np.ones(n_dyn, bool)])

    def world(k, v):
        if k < n_static:
            return static[k]
        return objR[v] @ local[k - n_static] + objt[v]

    # assign a distinct pixel to each point in each view; build pointmaps
    pointmaps, valid, gt_motion, pix = [], [], [], np.zeros((n_views, P, 2), int)
    for v in range(n_views):
        X = np.zeros((H, W, 3)); vm = np.zeros((H, W), bool); gm = np.zeros((H, W), bool)
        flat = rng.choice(H * W, size=P, replace=False)
        for k in range(P):
            y, x = divmod(int(flat[k]), W)
            pix[v, k] = (y, x)
            Xw = world(k, v)
            X[y, x] = camR[v].T @ (Xw - camt[v])     # camera-frame point
            vm[y, x] = True; gm[y, x] = is_dyn[k]
        pointmaps.append(X); valid.append(vm); gt_motion.append(gm)

    poses_gt = S.Poses(np.ones(n_views), camR.copy(), camt.copy())
    return SimpleNamespace(pointmaps=pointmaps, valid=valid, gt_motion=gt_motion,
                           pix=pix, P=P, is_dyn=is_dyn, n_views=n_views,
                           shapes=[(H, W)] * n_views, poses_gt=poses_gt)


def view_graph(n, loops=True):
    edges = [(i, i + 1) for i in range(n - 1)]
    if loops:
        for e in [(0, n - 1), (0, n // 2), (n // 4, 3 * n // 4)]:
            if e[0] != e[1] and e not in edges:
                edges.append(e)
    return edges


def make_matches(clip, edges, outlier_frac=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for e_idx, (i, j) in enumerate(edges):
        pi = clip.pix[i].copy(); pj = clip.pix[j].copy()
        if outlier_frac > 0:
            n_out = int(round(outlier_frac * clip.P))
            for k in rng.choice(clip.P, size=n_out, replace=False):
                kk = int(rng.integers(clip.P))
                pj[k] = clip.pix[j, kk]              # mismatch
        out[e_idx] = Matches(pi, pj, np.ones(clip.P))
    return out


def perturb(poses, rng, rot=0.0, trans=0.0):
    p = poses.copy(); acc_R = np.eye(3); acc_t = np.zeros(3)
    for v in range(len(p.s)):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-12
        acc_R = S.so3_exp(ax * rng.uniform(0, rot)) @ acc_R
        acc_t = acc_t + rng.normal(scale=trans, size=3)
        p.R[v] = acc_R @ p.R[v]; p.t[v] = p.t[v] + acc_t
    return p


def downstream_auroc(clip, edges, matches, poses0, cfg, reduce="median",
                     which="harmonic"):
    res = S.solve_harmonic(clip.pointmaps, edges, matches, poses0, cfg)
    energy = res.eps_k if which == "harmonic" else res.raw_k
    emaps, _ = S.splat_to_pixels(energy, res.index, clip.shapes, reduce=reduce)
    return E.map_auroc(emaps, {v: clip.gt_motion[v] for v in range(clip.n_views)},
                       {v: clip.valid[v] for v in range(clip.n_views)})


def main():
    cfg = S.SolveCfg(n_iters=2, n_irls=3)
    print("Real-data downstream on synthetic cache (interface check):\n")

    # 1) clean GT poses
    clip = make_synth_clip(seed=0)
    edges = view_graph(clip.n_views)
    matches = make_matches(clip, edges)
    au_h = downstream_auroc(clip, edges, matches, clip.poses_gt, cfg, which="harmonic")
    au_r = downstream_auroc(clip, edges, matches, clip.poses_gt, cfg, which="raw")
    print(f"  clean GT poses     : harmonic AUROC={au_h:.3f}  raw AUROC={au_r:.3f}")

    # 2) accumulating camera drift (the headline regime)
    p_drift = perturb(clip.poses_gt, np.random.default_rng(3), rot=0.20, trans=0.30)
    au_h = downstream_auroc(clip, edges, matches, p_drift, cfg, which="harmonic")
    au_r = downstream_auroc(clip, edges, matches, p_drift, cfg, which="raw")
    print(f"  + camera drift     : harmonic AUROC={au_h:.3f}  raw AUROC={au_r:.3f}")

    # 3) correspondence outliers: median vs mean splat readout
    m_out = make_matches(clip, edges, outlier_frac=0.15, seed=1)
    au_med = downstream_auroc(clip, edges, m_out, p_drift, cfg, reduce="median")
    au_mean = downstream_auroc(clip, edges, m_out, p_drift, cfg, reduce="mean")
    print(f"  + 15% mismatches   : median-splat AUROC={au_med:.3f}  "
          f"mean-splat AUROC={au_mean:.3f}")

    ok = (downstream_auroc(clip, edges, matches, clip.poses_gt, cfg) > 0.95)
    print("\n  PASS" if ok else "\n  FAIL", "- downstream pipeline reproduces PoC localization")


if __name__ == "__main__":
    main()
