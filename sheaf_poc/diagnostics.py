"""
Supplementary diagnostics behind the main experiment's design choices and the
honest failure boundaries. Run after run_experiment.py:

    conda run -n sheaf-poc python diagnostics.py

Three things this records:

  1. The Sim(3) global-scale collapse. The coordinate residual w_i - w_j is not
     scale-invariant, so the uniform log-scale direction can shrink the whole
     reconstruction to a point and cancel EVERY residual -- obstruction
     included. Fixing the scale gauge restores the obstruction. (Motivates the
     fix_scale_gauge anchor in build_sheaf, and is a modeling subtlety the paper
     should state alongside Remark 1.)

  2. The linearization-breakdown boundary. The harmonic localization is
     invariant to pose drift up to large magnitudes, breaking only at extreme
     drift -- the regime the project notes flagged ("large motion is where the
     linearization is worst"). Here it is, quantified.

  3. The static-dominance precondition. With too few static points the gauge is
     unpinned and the projection explains the object away; localization
     collapses past static == dynamic. This is the generative assumption a
     localization theorem would have to make explicit.
"""

import numpy as np
import sheaf_poc as sp
from run_experiment import perturb_poses, run_case


def _avg(scene, edges, rot, trans, noise, nseed=5, seed0=2000):
    raws, harms = [], []
    for s in range(nseed):
        Rp, tp = perturb_poses(scene, np.random.default_rng(seed0 + s),
                               rot_mag=rot, trans_mag=trans, accumulate=True)
        res = run_case(scene, edges, Rp, tp, noise_std=noise,
                       use_scale=True, seed=s)
        raws.append(res["auroc_raw"]); harms.append(res["auroc_harm"])
    return np.mean(raws), np.mean(harms)


def diag_scale_collapse():
    print("=" * 70)
    print("1. Sim(3) global-scale collapse (clean GT poses, no noise)")
    print("=" * 70)
    scene = sp.make_scene(np.random.default_rng(42), n_views=14,
                          n_static=90, n_dynamic=24,
                          object_speed=0.08, object_spin=0.05)
    edges = sp.make_view_graph(scene.N)
    lab = scene.is_dynamic
    configs = [("SE(3) (no scale)", False, False),
               ("Sim(3), no gauge fix", True, False),
               ("Sim(3) + scale gauge fix", True, True)]
    for name, use_scale, fix in configs:
        d, r0, inc, dim = sp.build_sheaf(scene, edges, scene.cam_R, scene.cam_t,
                                         use_scale=use_scale, noise_std=0.0,
                                         fix_scale_gauge=fix)
        h, xi = sp.harmonic_projection(d, r0)
        Eh, c = sp.per_point_energy(h, inc, scene.P); Eh /= np.maximum(c, 1)
        resid = np.linalg.norm(h) / np.linalg.norm(r0)
        scale = np.round(xi[6::dim][:4], 3) if use_scale else "n/a"
        print(f"  {name:<26} ||h||/||r0||={resid:.2e}  "
              f"AUROC={sp.auroc(Eh, lab):.3f}  scale_xi(4 views)={scale}")
    print("  -> uniform scale_xi = -1 collapses the world; gauge fix restores it.\n")


def diag_linearization_boundary():
    print("=" * 70)
    print("2. Linearization-breakdown boundary (extreme accumulating drift)")
    print("=" * 70)
    scene = sp.make_scene(np.random.default_rng(42), n_views=14,
                          object_speed=0.08, object_spin=0.05)
    edges = sp.make_view_graph(scene.N)
    print(f"  {'drift(rad/step)':<16}{'AUROC raw':>11}{'AUROC harm':>12}")
    for rot in [0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
        r, h = _avg(scene, edges, rot, 1.5 * rot, 0.01)
        print(f"  {rot:<16.2f}{r:>11.3f}{h:>12.3f}")
    print("  -> harmonic holds to ~0.6 rad/step; breaks only at extreme drift.\n")


def diag_static_precondition():
    print("=" * 70)
    print("3. Static-dominance precondition (drift=0.1, 24 dynamic points)")
    print("=" * 70)
    print(f"  {'n_static':<10}{'static frac':>12}{'AUROC harm':>12}")
    for nstat in [90, 60, 30, 24, 16, 8]:
        sc = sp.make_scene(np.random.default_rng(42), n_views=14, n_static=nstat,
                           n_dynamic=24, object_speed=0.08, object_spin=0.05)
        ed = sp.make_view_graph(sc.N)
        _, h = _avg(sc, ed, 0.1, 0.15, 0.01)
        print(f"  {nstat:<10}{nstat / (nstat + 24):>12.2f}{h:>12.3f}")
    print("  -> collapses once the static background stops dominating.\n")


def diag_noise_floor():
    print("=" * 70)
    print("4. Measurement-noise floor (small object motion, no drift)")
    print("=" * 70)
    scene = sp.make_scene(np.random.default_rng(42), n_views=14,
                          object_speed=0.04, object_spin=0.025)
    edges = sp.make_view_graph(scene.N)
    print(f"  {'noise_std':<12}{'AUROC raw':>11}{'AUROC harm':>12}")
    for noise in [0.0, 0.02, 0.05, 0.1, 0.2]:
        r, h = _avg(scene, edges, 0.0, 0.0, noise)
        print(f"  {noise:<12.2f}{r:>11.3f}{h:>12.3f}")
    print("  -> harmonic cannot remove per-measurement noise (not pose-explainable);\n"
          "     both degrade once noise rivals the object's motion.\n")


if __name__ == "__main__":
    diag_scale_collapse()
    diag_linearization_boundary()
    diag_static_precondition()
    diag_noise_floor()
