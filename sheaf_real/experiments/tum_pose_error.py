"""
Level 2, step 1 -- the section-8 falsifier (NO sheaf yet).

Run DUSt3R on a TUM loop, recover each pair's RELATIVE pose independently (Sim(3)
Procrustes on the two pixel-aligned pointmap predictions of the shared image),
compare to GT relative pose, and ask the decisive question:

  Is DUSt3R's per-edge relative-pose error SPARSE (a few bad edges -> cycle
  structure can exploit it, green light for the sheaf) or DENSE (every edge noisy
  -> nothing for loop structure to do, the direction is finished)?

Also (a) validates the pose convention -- high-overlap pairs must have small error,
which is the previously-parked convention bug under a clean GT; (b) checks whether
error correlates with baseline (wide-baseline/low-overlap = expected-bad edges).

    conda run -n sheaf python experiments/tum_pose_error.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backbone as B   # noqa: sets DDUSt3R path
import tum as TUM

ROOT = "../data/tum/rgbd_dataset_freiburg1_room"
NKEY = 30
STRIDE = 2     # pixel subsample for Procrustes speed


def umeyama(X, Y, w):
    """Weighted Sim(3): s,R,t with Y ~= s R X + t. X,Y (N,3), w (N,)."""
    w = w / (w.sum() + 1e-12)
    mx = (w[:, None] * X).sum(0)
    my = (w[:, None] * Y).sum(0)
    Xc, Yc = X - mx, Y - my
    Sig = (w[:, None] * Yc).T @ Xc
    U, D, Vt = np.linalg.svd(Sig)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (w * (Xc**2).sum(1)).sum()
    s = float((D * np.diag(S)).sum() / (var + 1e-12))
    t = my - s * R @ mx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, s


def geo_deg(Ra, Rb):
    c = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def transl_angle(ta, tb):
    na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    c = np.clip(ta @ tb / (na * nb), -1, 1)
    return np.degrees(np.arccos(c))


def per_pair_predictions(output):
    """Map (i,j) -> dict with pts3d (view1 in frame i) and pts3d_in_other (view2
    in frame i) and conf maps, for every ordered pair in the batch."""
    import torch  # noqa
    v1, v2 = output["view1"], output["view2"]
    p1, p2 = output["pred1"], output["pred2"]
    idx1 = [int(x) for x in v1["idx"]]
    idx2 = [int(x) for x in v2["idx"]]
    D = {}
    for k in range(len(idx1)):
        i, j = idx1[k], idx2[k]
        D[(i, j)] = dict(
            pi=np.asarray(p1["pts3d"][k].detach().cpu()),               # i in frame i
            pj=np.asarray(p2["pts3d_in_other_view"][k].detach().cpu()),  # j in frame i
            ci=np.asarray(p1["conf"][k].detach().cpu()),
            cj=np.asarray(p2["conf"][k].detach().cpu()),
        )
    return D


def main():
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    paths, poses, stamps = TUM.load(ROOT, n_key=NKEY)
    n = len(paths)
    print(f"{n} TUM keyframes over {stamps[-1]-stamps[0]:.1f}s")

    model = B.load_backbone("dust3r")
    imgs = load_images(paths, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", symmetrize=True)
    output = inference(pairs, model, "cuda", batch_size=1, verbose=False)
    D = per_pair_predictions(output)

    s_ = slice(None, None, STRIDE)
    rows = []
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in D or (b, a) not in D:
                continue
            # relative pose frame b -> frame a, from pixel-aligned points of img b
            Pb_in_a = D[(a, b)]["pj"][s_, s_].reshape(-1, 3)   # b's pts in frame a
            Pb_in_b = D[(b, a)]["pi"][s_, s_].reshape(-1, 3)   # b's pts in frame b
            ca = D[(a, b)]["cj"][s_, s_].reshape(-1)
            cb = D[(b, a)]["ci"][s_, s_].reshape(-1)
            w = np.minimum(ca, cb)
            M, scl = umeyama(Pb_in_b, Pb_in_a, w)              # b -> a  (= g_a^{-1} g_b)
            GT = TUM.rel_pose(poses[a], poses[b])              # b -> a, GT
            r_err = geo_deg(M[:3, :3], GT[:3, :3])
            t_err = transl_angle(M[:3, 3], GT[:3, 3])
            baseline = np.linalg.norm(poses[a][:3, 3] - poses[b][:3, 3])
            gt_rot = geo_deg(np.eye(3), GT[:3, :3])
            rows.append(dict(a=a, b=b, r_err=r_err, t_err=t_err,
                             baseline=baseline, gt_rot=gt_rot,
                             meanconf=float(w.mean())))

    re = np.array([r["r_err"] for r in rows])
    te = np.array([r["t_err"] for r in rows])
    bl = np.array([r["baseline"] for r in rows])
    print(f"\n{len(rows)} pairs. Relative-pose error vs GT:")
    print(f"  rotation deg : median {np.median(re):.1f}  mean {re.mean():.1f}  "
          f"p90 {np.percentile(re,90):.1f}  max {re.max():.1f}")
    print(f"  transl  deg  : median {np.nanmedian(te):.1f}  "
          f"mean {np.nanmean(te):.1f}  max {np.nanmax(te):.1f}")

    # --- sparse vs dense: how concentrated is the total rotation error? ---
    order = np.argsort(re)[::-1]
    cum = np.cumsum(re[order]) / re.sum()
    k10 = max(1, len(re) // 10)
    print(f"\nError concentration (rotation):")
    print(f"  top 10% of edges ({k10}) hold {cum[k10-1]*100:.0f}% of total error")
    print(f"  kurtosis (Fisher) = {kurtosis(re):.2f}   (0=Gaussian; >>0 = heavy tail/sparse)")
    print(f"  max/median ratio  = {re.max()/np.median(re):.1f}")
    # --- expected-bad = wide baseline? correlation ---
    print(f"\n  corr(rot_err, baseline)   = {np.corrcoef(re, bl)[0,1]:+.2f}")
    print(f"  corr(rot_err, GT rotation)= "
          f"{np.corrcoef(re, [r['gt_rot'] for r in rows])[0,1]:+.2f}")
    print(f"  corr(rot_err, -meanconf)  = "
          f"{np.corrcoef(re, [-r['meanconf'] for r in rows])[0,1]:+.2f}")

    # --- SLAM-native view: is there a good backbone + SPARSE bad loop edges? ---
    gap = np.array([abs(r["b"] - r["a"]) for r in rows])
    adj = re[gap == 1]
    lr = re[gap >= 3]
    print(f"\nSLAM topology breakdown:")
    print(f"  adjacent edges (gap=1, 'odometry backbone'): n={len(adj)}  "
          f"rot_err median {np.median(adj):.1f}  p90 {np.percentile(adj,90):.1f}  "
          f"frac<10deg {np.mean(adj<10)*100:.0f}%")
    print(f"  long-range edges (gap>=3, 'loop candidates'): n={len(lr)}  "
          f"rot_err median {np.median(lr):.1f}  frac<10deg {np.mean(lr<10)*100:.0f}%")
    if len(lr):
        o = np.argsort(lr)[::-1]
        cl = np.cumsum(lr[o]) / lr.sum()
        kk = max(1, len(lr) // 10)
        print(f"  among long-range edges: top 10% hold {cl[kk-1]*100:.0f}% of error, "
              f"kurtosis {kurtosis(lr):.2f}")
    # true revisits: temporally far but spatially near -> the real loop closures
    print(f"\nTrue loop-closure edges (gap>=5 AND GT baseline<0.5m):")
    rev = [r for r in rows if abs(r["b"]-r["a"]) >= 5 and r["baseline"] < 0.5]
    rev.sort(key=lambda r: r["r_err"])
    for r in rev[:12]:
        print(f"  ({r['a']:2d},{r['b']:2d}) gap={abs(r['b']-r['a']):2d} "
              f"baseline={r['baseline']:.2f}m rot_err={r['r_err']:5.1f} deg "
              f"conf={r['meanconf']:.1f}")
    if rev:
        rr = np.array([r["r_err"] for r in rev])
        print(f"  -> {len(rev)} revisit edges: rot_err median {np.median(rr):.1f}, "
              f"frac usable (<15deg) {np.mean(rr<15)*100:.0f}%")

    # sanity / convention check: the 5 highest-overlap (smallest baseline) pairs
    lowbl = np.argsort(bl)[:5]
    print(f"\nConvention sanity -- 5 highest-overlap pairs (small baseline):")
    for k in lowbl:
        r = rows[k]
        print(f"  ({r['a']:2d},{r['b']:2d}) baseline={r['baseline']:.2f}m "
              f"rot_err={r['r_err']:5.1f} deg  transl_err={r['t_err']:5.1f} deg")

    os.makedirs("../results", exist_ok=True)
    np.save("../results/tum_pose_err.npy",
            np.array([(r["a"], r["b"], r["r_err"], r["t_err"], r["baseline"],
                       r["gt_rot"], r["meanconf"]) for r in rows]))
    print("\n-> results/tum_pose_err.npy")


def kurtosis(x):
    x = x - x.mean()
    m2 = (x**2).mean()
    m4 = (x**4).mean()
    return m4 / (m2**2 + 1e-12) - 3.0


if __name__ == "__main__":
    main()
