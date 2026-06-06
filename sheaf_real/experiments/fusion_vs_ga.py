"""
The decision experiment (new planning agent): does ROBUST sheaf-Laplacian fusion
reduce STATIC-region reconstruction contamination vs standard global alignment?

(a) standard GA  : DUSt3R PointCloudOptimizer (confidence-weighted; movers
    contribute to the loss and can distort static depth/poses).
(b) robust GA    : same objective, but the per-pixel weights are reweighted by a
    Huber factor on the GA's own residual ||proj_pts3d - aligned_pred|| (IRLS) —
    i.e. the sheaf Dirichlet energy with harmonic-projection robustness, which
    downweights cross-frame-inconsistent (moving / mispredicted) pixels.

Metric: depth AbsRel vs Sintel GT (per-view median scale-aligned on static
pixels), split into STATIC / DYNAMIC / ALL; plus camera pose error (ATE, rot)
vs GT. Claim: (b) lowers STATIC AbsRel at equal-or-better pose/all cost.

CAVEAT (our own §exp-realism): residual-based downweighting cannot distinguish
movers from structured static prediction error; a null/negative result is
informative.

    conda run -n sheaf python experiments/fusion_vs_ga.py
"""
import os
import sys
import gc
import csv
import numpy as np

_DD = os.path.join(os.path.dirname(__file__), "..", "..", "DDUSt3R")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _DD)
import backbone as B   # noqa  (sets up dust3r path)
import sintel as SI
from pipeline import align_to_grid

STATIC = ["alley_1", "cave_2", "market_2", "market_5", "ambush_4", "temple_3"]
DYN = ["ambush_2", "cave_4", "shaman_2"]      # dynamic-heavy: thesis should bite hardest
SCENES = STATIC + DYN
NFRAMES, NITER, TAU = 5, 300, 3.0
ROOT = "../data"


def run_ga(output, device, niter, robust=False, huber_k=2.0):
    """Standard or IRLS-robust global alignment. Returns the scene."""
    import torch
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.geometry import geotrf
    scene = global_aligner(output, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)
    if not robust:
        return scene
    # one IRLS round: downweight high-residual (inconsistent) pixels
    with torch.no_grad():
        pw = scene.get_pw_poses(); ad = scene.get_adaptors().unsqueeze(1)
        proj = scene.get_pts3d(raw=True)
        ai = geotrf(pw, ad * scene._stacked_pred_i)
        aj = geotrf(pw, ad * scene._stacked_pred_j)
        ri = (proj[scene._ei] - ai).norm(dim=-1)        # (E, area)
        rj = (proj[scene._ej] - aj).norm(dim=-1)
        sig = torch.median(torch.cat([ri[ri > 0], rj[rj > 0]])) + 1e-6
        c = huber_k * sig
        hi = torch.clamp(c / (ri + 1e-9), max=1.0)
        hj = torch.clamp(c / (rj + 1e-9), max=1.0)
        scene._weight_i = scene._weight_i * hi
        scene._weight_j = scene._weight_j * hj
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)
    return scene


def extract(scene):
    import numpy as np
    poses = scene.get_im_poses().detach().cpu().numpy()       # cams2world (N,4,4)
    depths = [np.asarray(d.detach().cpu()) for d in scene.get_depthmaps()]
    return poses, depths


def gt_depth_pose(scene_name, n, shape):
    H, W = shape
    depths, poses = [], []
    for i in range(n):
        d = SI.read_dpt(os.path.join(ROOT, "training", "depth", scene_name,
                                     f"frame_{i+1:04d}.dpt"))
        depths.append(align_to_grid(d.astype(np.float32), nearest=False))
        M, Ncam = SI.read_cam(os.path.join(ROOT, "training", "camdata_left",
                                           scene_name, f"frame_{i+1:04d}.cam"))
        R, t = Ncam[:3, :3], Ncam[:3, 3]                     # world->cam
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ t
        poses.append(c2w)
    return depths, np.array(poses)


def absrel(est, gt, mask):
    """Affine-invariant depth AbsRel: scale+shift align est to GT on the masked
    pixels by least squares (standard monocular-depth eval), then |.|/gt."""
    m = mask & (gt > 1e-3) & (gt < 100) & np.isfinite(est) & (est > 1e-6)
    if m.sum() < 50:
        return float("nan")
    e, g = est[m], gt[m]
    A = np.stack([e, np.ones_like(e)], 1)
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]              # scale+shift to GT
    aligned = np.clip(s * e + t, 1e-3, None)
    return float(np.mean(np.abs(aligned - g) / g))


def pose_err(est, gt):
    ce, cg = est[:, :3, 3], gt[:, :3, 3]
    mu_e, mu_g = ce.mean(0), cg.mean(0)
    Xe, Xg = ce - mu_e, cg - mu_g
    U, _, Vt = np.linalg.svd(Xg.T @ Xe)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = (np.trace(np.diag(_ := np.linalg.svd(Xg.T @ Xe)[1]) @ W)) / ((Xe**2).sum() + 1e-9)
    ce_a = (s * (R @ ce.T)).T + (mu_g - s * R @ mu_e)
    ate = float(np.sqrt(np.mean(np.sum((ce_a - cg)**2, 1))))
    rots = []
    for i in range(len(est)):
        dR = (R @ est[i, :3, :3]).T @ gt[i, :3, :3]
        rots.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return ate, float(np.mean(rots))


def main():
    import torch
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    model = B.load_backbone("dust3r")
    rows = []
    for sc in SCENES:
        paths = [SI.frame_paths(ROOT, sc, "clean")[k] for k in range(NFRAMES)]
        imgs = load_images(paths, size=512, verbose=False)
        pairs = make_pairs(imgs, scene_graph="complete", symmetrize=True)
        output = inference(pairs, model, "cuda", batch_size=1, verbose=False)
        sa = run_ga(output, "cuda", NITER, robust=False)
        pa, da = extract(sa)
        sb = run_ga(output, "cuda", NITER, robust=True)
        pb, db = extract(sb)
        shape = da[0].shape
        gtd, gtp = gt_depth_pose(sc, NFRAMES, shape)
        # GT motion masks per frame (static = ~dynamic & valid)
        stat, dyn, valid = {}, {}, {}
        for i in range(NFRAMES - 1):
            m, _, v = SI.motion_gt(ROOT, sc, i + 1, i + 2, tau=TAU)
            dyn[i] = align_to_grid(m, nearest=True)
            valid[i] = align_to_grid(v, nearest=True)
            stat[i] = (~dyn[i]) & valid[i]
        # AbsRel pooled over frames with masks
        def pooled(depthset, region):
            errs = []
            for i in region:
                e = absrel(depthset[i], gtd[i], region[i])
                if not np.isnan(e):
                    errs.append(e)
            return float(np.mean(errs)) if errs else float("nan")
        rec = dict(
            scene=sc,
            absrel_static_a=pooled(da, {i: stat[i] for i in stat}),
            absrel_static_b=pooled(db, {i: stat[i] for i in stat}),
            absrel_dyn_a=pooled(da, {i: dyn[i] & valid[i] for i in dyn}),
            absrel_dyn_b=pooled(db, {i: dyn[i] & valid[i] for i in dyn}),
            absrel_all_a=pooled(da, {i: valid[i] for i in valid}),
            absrel_all_b=pooled(db, {i: valid[i] for i in valid}),
            )   # pose-vs-GT (ATE/rot) parked: Sintel/DUSt3R camera-convention TODO
        rows.append(rec)
        print(f"  {sc:<12} static a={rec['absrel_static_a']:.3f} b={rec['absrel_static_b']:.3f}"
              f" | dyn a={rec['absrel_dyn_a']:.3f} b={rec['absrel_dyn_b']:.3f}"
              f" | all a={rec['absrel_all_a']:.3f} b={rec['absrel_all_b']:.3f}")
        del output, sa, sb; gc.collect(); torch.cuda.empty_cache()

    os.makedirs("../results", exist_ok=True)
    with open("../results/fusion_vs_ga.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    def grp(names, title):
        print(f"  [{title}] mean AbsRel (a=standard GA, b=robust; lower better):")
        sub = [r for r in rows if r["scene"] in names]
        for k in ["absrel_static", "absrel_dyn", "absrel_all"]:
            a = np.nanmean([r[k + "_a"] for r in sub]); b = np.nanmean([r[k + "_b"] for r in sub])
            print(f"    {k:<14} a={a:.3f}  b={b:.3f}  (Δ={b-a:+.3f})")
    print()
    grp(STATIC, "static-dominant"); grp(DYN, "dynamic-heavy")
    print("  -> results/fusion_vs_ga.csv  (pose-vs-GT parked: convention TODO)")


if __name__ == "__main__":
    main()
