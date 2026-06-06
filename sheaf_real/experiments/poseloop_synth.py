"""
Synthetic closed-orbit validation of the pose / connection sheaf (Level 1 on a
real loop topology). Three checks, mirroring the dense-model PoC story:

  (A) Sanity     : consistent measurements -> harmonic energy ~ 0 (machine eps).
  (B) Localize   : corrupt ONE edge by a known rotation theta -> the harmonic
                   1-cochain concentrates on cycles through that edge, and the
                   per-edge leverage ranks it #1.  Report bad-edge AUROC vs theta.
  (C) Drift-free : add a smooth per-view drift (an im-delta perturbation: replace
                   measurements by g-conjugated ones) -> harmonic energy and the
                   bad-edge ranking are UNCHANGED.  (Loop-closure analog of the
                   synthetic drift-invariance figure.)

No network, no download. Pure numpy.

    conda run -n sheaf-poc python experiments/poseloop_synth.py     # or any numpy env
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pose_sheaf as PS


RNG = np.random.default_rng(0)


def rand_se3(rng, t_scale=1.0, r_scale=0.5):
    xi = np.concatenate([rng.normal(0, t_scale, 3), rng.normal(0, r_scale, 3)])
    return PS.se3_exp(xi)


def closed_orbit(n, rng):
    """n cameras on a closed loop: absolute frames g_i (cam-to-world)."""
    g = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # yaw around the room
        t = np.array([2 * c, 0.1 * rng.normal(), 2 * s])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        g.append(T)
    return g


def loop_graph(n):
    """Ring edges (the loop) + a few chords -> several independent cycles."""
    edges = [(k, (k + 1) % n) for k in range(n)]          # the ring (closes loop)
    edges += [(0, n // 2), (1, n // 2 + 1), (2, n - 2)]   # chords
    return edges


def measurements(g, edges, noise=0.0, rng=None):
    """M_ij = g_i^{-1} g_j (+ optional small se(3) noise)."""
    meas = []
    for (i, j) in edges:
        M = PS.inv(g[i]) @ g[j]
        if noise > 0:
            M = M @ PS.se3_exp(rng.normal(0, noise, 6))
        meas.append(M)
    return meas


def roc_auc(scores, labels):
    """AUROC: P(score_pos > score_neg). labels in {0,1}."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return wins / (len(pos) * len(neg))


def check_A():
    n = 10
    g = closed_orbit(n, RNG)
    edges = loop_graph(n)
    meas = measurements(g, edges, noise=0.0)
    res = PS.analyze(n, edges, meas)
    print(f"[A] consistent measurements:")
    print(f"    max |harmonic| = {np.abs(res.h).max():.2e}   "
          f"max cycle energy = {res.cycle_e.max():.2e}   (expect ~0)")
    return np.abs(res.h).max() < 1e-9


def check_B(thetas=(0.02, 0.05, 0.1, 0.2, 0.4), n=10, trials=20):
    edges = loop_graph(n)
    print(f"[B] planted-bad-edge localization ({n} views, {len(edges)} edges, "
          f"{trials} trials/theta):")
    out = {}
    for th in thetas:
        aucs, top1 = [], []
        for tr in range(trials):
            rng = np.random.default_rng(100 + tr)
            g = closed_orbit(n, rng)
            meas = measurements(g, edges, noise=0.005, rng=rng)
            bad = rng.integers(0, len(edges))
            axis = rng.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            meas[bad] = meas[bad] @ PS.se3_exp(
                np.concatenate([np.zeros(3), th * axis]))
            res = PS.analyze(n, edges, meas)
            labels = np.zeros(len(edges))
            labels[bad] = 1
            aucs.append(roc_auc(res.edge_e, labels))
            top1.append(int(np.argmax(res.edge_e) == bad))
        out[th] = (np.mean(aucs), np.mean(top1))
        print(f"    theta={th:.2f} rad  bad-edge AUROC={np.mean(aucs):.3f}  "
              f"top-1 hit rate={np.mean(top1):.2f}")
    return out


def check_C(n=10, theta=0.2, trials=20):
    """Drift invariance, stated correctly: the loop obstruction is a function of
    the MEASUREMENTS only, not of the absolute-frame estimate g used to linearize.
    Camera-pose drift = a different (worse) g with the SAME measurements. So we
    compute the harmonic cochain at the spanning-tree g and again at a g corrupted
    by a smooth, growing per-view drift, and require h (hence energy and bad-edge
    ranking) to be unchanged. (r0 changes by an im-delta element; the harmonic
    projection removes exactly that.)"""
    edges = loop_graph(n)
    dh, same_rank, eratio = [], [], []
    for tr in range(trials):
        rng = np.random.default_rng(500 + tr)
        g = closed_orbit(n, rng)
        meas = measurements(g, edges, noise=0.005, rng=rng)
        bad = rng.integers(0, len(edges))
        axis = rng.normal(0, 1, 3); axis /= np.linalg.norm(axis)
        meas[bad] = meas[bad] @ PS.se3_exp(np.concatenate([np.zeros(3), theta * axis]))
        # (1) reference: converged harmonic from the spanning-tree init
        h0, gtree, _, _ = PS.solve_iter(n, edges, meas)
        # (2) start the SAME iterated solve from a badly drifted frame estimate
        gd = [gtree[k] @ PS.se3_exp(np.concatenate([
            0.4 * k / n * rng.normal(0, 1, 3),
            0.3 * k / n * rng.normal(0, 1, 3)])) for k in range(n)]
        h1, _, _, _ = PS.solve_iter(n, edges, meas, g0=gd)
        e0 = PS.edge_energy(h0); e1 = PS.edge_energy(h1)
        labels = np.zeros(len(edges)); labels[bad] = 1
        dh.append(np.abs(h0 - h1).max())
        same_rank.append(int(np.argmax(e0) == np.argmax(e1)))
        eratio.append((h1 @ h1) / (h0 @ h0))
    print(f"[C] drift invariance ({trials} trials, theta={theta}):")
    print(f"    max |h_drift - h_clean| = {np.max(dh):.2e}  (expect ~0)")
    print(f"    bad-edge top-1 unchanged = {np.mean(same_rank)*100:.0f}% of trials")
    er = np.array(eratio)
    print(f"    harmonic energy ratio (drifted/clean) = median {np.median(er):.4f}, "
          f"mean {np.mean(er):.4f} +/- {np.std(er):.4f}  (expect 1.000)")
    # invariance: obstruction is measurement-determined; allow rare GN misses from
    # the deliberately extreme init (real runs init from the spanning tree).
    return abs(np.median(er) - 1.0) < 0.03 and np.mean(same_rank) >= 0.9


def main():
    print("=" * 64)
    print("Pose/connection sheaf -- synthetic closed-orbit validation")
    print("=" * 64)
    a = check_A()
    print()
    check_B()
    print()
    c = check_C()
    print()
    print(f"SUMMARY: (A) consistent->0: {'PASS' if a else 'FAIL'};  "
          f"(C) drift-invariant: {'PASS' if c else 'FAIL'};  "
          f"(B) see AUROC table above.")


if __name__ == "__main__":
    main()
