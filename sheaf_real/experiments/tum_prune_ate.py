"""
Level 3 -- the paper-maker: does pruning the DUSt3R scene graph by sheaf
harmonic edge-leverage beat pruning by DUSt3R confidence, measured by camera
trajectory error after the SAME global alignment?

Design. Keep the (good) consecutive-frame backbone always. From the pool of all
non-backbone pairs, ADD K loop-closure edges chosen by:
  - confidence : highest DUSt3R pairwise confidence   (the DUSt3R-style heuristic)
  - leverage   : lowest SO(3) harmonic leverage        (ours: most loop-consistent)
  - oracle     : lowest GT relative-rotation error      (upper bound)
  - random     : control
Run DUSt3R global alignment on each pruned graph (reusing one inference), then
ATE (camera-center RMSE after Sim(3) alignment) + mean rotation error vs TUM GT.
Win = lower ATE than confidence at equal K, or equal ATE at fewer edges.

    conda run -n sheaf python experiments/tum_prune_ate.py
"""
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backbone as B   # noqa: sets DDUSt3R path
import tum as TUM
import pose_sheaf as PS
from tum_pose_error import umeyama, per_pair_predictions, geo_deg

ROOT = "../data/tum/rgbd_dataset_freiburg1_room"
NKEY = 30
STRIDE = 2
NITER = 300
KS = [5, 10, 20, 40]


def rel_rotations(D, n, stride):
    """R_ij (frame j -> i) and per-edge confidence for every unordered pair."""
    s_ = slice(None, None, stride)
    Rmeas, conf = {}, {}
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in D or (b, a) not in D:
                continue
            Pb_in_a = D[(a, b)]["pj"][s_, s_].reshape(-1, 3)
            Pb_in_b = D[(b, a)]["pi"][s_, s_].reshape(-1, 3)
            ca = D[(a, b)]["cj"][s_, s_].reshape(-1)
            cb = D[(b, a)]["ci"][s_, s_].reshape(-1)
            w = np.minimum(ca, cb)
            M, _ = umeyama(Pb_in_b, Pb_in_a, w)
            Rmeas[(a, b)] = M[:3, :3]
            conf[(a, b)] = float(w.mean())
    return Rmeas, conf


def filter_output(output, keep):
    """Subset the inference output to ordered pairs in `keep` (a set)."""
    import torch
    i1 = [int(x) for x in output["view1"]["idx"]]
    i2 = [int(x) for x in output["view2"]["idx"]]
    idx = [k for k in range(len(i1)) if (i1[k], i2[k]) in keep]

    def sub(d):
        o = {}
        for key, val in d.items():
            if torch.is_tensor(val):
                o[key] = val[idx]
            elif isinstance(val, (list, tuple)):
                o[key] = [val[k] for k in idx]
            else:
                o[key] = val
        return o
    return {"view1": sub(output["view1"]), "view2": sub(output["view2"]),
            "pred1": sub(output["pred1"]), "pred2": sub(output["pred2"])}


def run_ga(output, edges_u, device, niter):
    """edges_u: set of unordered (a,b). Run GA on those (both directions)."""
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    keep = set()
    for (a, b) in edges_u:
        keep.add((a, b)); keep.add((b, a))
    import gc
    import torch
    sub = filter_output(output, keep)
    scene = global_aligner(sub, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)
    poses = scene.get_im_poses().detach().cpu().numpy()
    del scene, sub
    gc.collect(); torch.cuda.empty_cache()
    return poses


def ate_rot(est, gt):
    """Sim(3)-align estimated cam centers to GT, return (ATE rmse m, mean rot deg)."""
    ce, cg = est[:, :3, 3], gt[:, :3, 3]
    mu_e, mu_g = ce.mean(0), cg.mean(0)
    Xe, Xg = ce - mu_e, cg - mu_g
    U, Dg, Vt = np.linalg.svd(Xg.T @ Xe)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = (Dg * np.diag(W)).sum() / ((Xe**2).sum() + 1e-12)
    ce_a = (s * (R @ ce.T)).T + (mu_g - s * R @ mu_e)
    ate = float(np.sqrt(np.mean(np.sum((ce_a - cg)**2, 1))))
    rots = [geo_deg(R @ est[i, :3, :3], gt[i, :3, :3]) for i in range(len(est))]
    return ate, float(np.mean(rots))


def main():
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    paths, gtpose, stamps = TUM.load(ROOT, n_key=NKEY)
    n = len(paths)
    print(f"{n} TUM keyframes over {stamps[-1]-stamps[0]:.1f}s")

    model = B.load_backbone("dust3r")
    imgs = load_images(paths, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", symmetrize=True)
    output = inference(pairs, model, "cuda", batch_size=1, verbose=False)
    D = per_pair_predictions(output)
    Rmeas, conf = rel_rotations(D, n, STRIDE)

    backbone = [(k, k + 1) for k in range(n - 1)]
    pool = [e for e in Rmeas if e not in set(backbone)]

    # sheaf leverage on the FULL graph (backbone + pool = complete)
    alledges = backbone + pool
    Rlist = [Rmeas[e] for e in alledges]
    wlist = [conf[e] for e in alledges]
    _, _, lev = PS.so3_solve(n, alledges, Rlist, weights=wlist)
    lev_pool = {e: lev[alledges.index(e)] for e in pool}

    # GT relative-rotation error per pool edge (for oracle + labels)
    gt_err = {}
    for (a, b) in pool:
        GTr = TUM.rel_pose(gtpose[a], gtpose[b])[:3, :3]
        gt_err[(a, b)] = geo_deg(Rmeas[(a, b)], GTr)

    # selection rankings (best-first)
    rank = {
        "confidence": sorted(pool, key=lambda e: -conf[e]),
        "leverage":   sorted(pool, key=lambda e: lev_pool[e]),
        "oracle":     sorted(pool, key=lambda e: gt_err[e]),
        "random":     list(np.array(pool)[np.argsort([hash(e) & 0xffff for e in pool])]),
    }
    rank["random"] = [tuple(int(x) for x in e) for e in rank["random"]]

    rows = []
    # baselines
    a0, r0 = ate_rot(run_ga(output, set(backbone), "cuda", NITER), gtpose)
    print(f"\nbackbone only ({len(backbone)} edges):           ATE {a0*100:5.1f} cm  rot {r0:4.1f} deg")
    rows.append(dict(method="backbone", K=0, edges=len(backbone), ate_cm=a0*100, rot=r0))
    try:   # complete graph is the full pairwise set -> may OOM on a small GPU
        ac, rc = ate_rot(run_ga(output, set(backbone + pool), "cuda", NITER), gtpose)
        print(f"complete graph ({len(backbone)+len(pool)} edges): ATE {ac*100:5.1f} cm  rot {rc:4.1f} deg")
        rows.append(dict(method="complete", K=len(pool), edges=len(backbone)+len(pool), ate_cm=ac*100, rot=rc))
    except Exception as e:
        import torch
        torch.cuda.empty_cache()
        print(f"complete graph: skipped ({type(e).__name__})")

    print(f"\n{'method':<11}" + "".join(f"  K={k:<3}" for k in KS))
    for method in ["confidence", "leverage", "oracle", "random"]:
        cells = []
        for K in KS:
            sel = rank[method][:K]
            ate, rot = ate_rot(run_ga(output, set(backbone + sel), "cuda", NITER), gtpose)
            rows.append(dict(method=method, K=K, edges=len(backbone)+K, ate_cm=ate*100, rot=rot))
            cells.append(f"{ate*100:5.1f}")
        print(f"{method:<11}" + "".join(f"  {c:>5}" for c in cells) + "   (ATE cm)")

    # how good are the edges each method picks (mean GT rot err of top-20)?
    print("\nmean GT rot-err of top-20 selected loop edges:")
    for method in ["confidence", "leverage", "oracle"]:
        sel = rank[method][:20]
        print(f"  {method:<11} {np.mean([gt_err[e] for e in sel]):5.1f} deg")

    os.makedirs("../results", exist_ok=True)
    with open("../results/tum_prune_ate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n-> results/tum_prune_ate.csv")


if __name__ == "__main__":
    main()
