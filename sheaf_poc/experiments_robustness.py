"""
Robustness sub-experiments: each of the three "honest boundaries" from the
paper, tested independently in the controlled synthetic setting (and, where
possible, mitigated). Run:

    conda run -n sheaf-poc python experiments_robustness.py

Sub-experiments (each writes its own figure to figures/):
  1. multiple moving objects        -> robust_multi.png      (static-dominance generality)
  2. robust gauge anchoring         -> robust_anchor.png     (push past the 50% cliff)
  3. iterated sheaf Gauss-Newton    -> robust_iterated.png   (controlled linearization)
  4. scale-gauge validation         -> robust_scale.png      (invariance + drift absorption)

The combined "everything at once" experiment is intentionally deferred until
each of these holds on its own.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sheaf_poc as sp
from run_experiment import perturb_poses

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "figure.dpi": 200})


def energies(scene, edges, R_est, t_est, s_est=None, use_scale=True,
             noise=0.0, seed=0, robust=False):
    """Per-point harmonic and raw energies (normalized per incidence count)."""
    rng = np.random.default_rng(seed)
    d, r0, inc, dim = sp.build_sheaf(scene, edges, R_est, t_est,
                                     use_scale=use_scale, noise_std=noise,
                                     rng=rng, fix_scale_gauge=True,
                                     pose_s_est=s_est)
    if robust:
        h, _ = sp.robust_harmonic_projection(d, r0, inc)
    else:
        h, _ = sp.harmonic_projection(d, r0)
    Eh, c = sp.per_point_energy(h, inc, scene.P); Eh /= np.maximum(c, 1)
    Er, c2 = sp.per_point_energy(r0, inc, scene.P); Er /= np.maximum(c2, 1)
    return Eh, Er


def _mean_world(scene):
    return np.array([np.mean([scene.world_point(k, v) for v in range(scene.N)],
                             axis=0) for k in range(scene.P)])


# =====================================================================
# 1. Multiple moving objects
# =====================================================================
def exp_multi_objects():
    print("=" * 64)
    print("1. MULTIPLE MOVING OBJECTS  (do they each localize?)")
    print("=" * 64)
    rng = np.random.default_rng(42)
    n_obj, ppo, n_static = 3, 14, 90
    scene = sp.make_scene_multi(rng, n_views=14, n_static=n_static,
                                n_objects=n_obj, pts_per_object=ppo,
                                object_speed=0.08, object_spin=0.05)
    edges = sp.make_view_graph(scene.N)
    R_est, t_est = perturb_poses(scene, np.random.default_rng(7),
                                 rot_mag=0.10, trans_mag=0.15, accumulate=True)
    Eh, Er = energies(scene, edges, R_est, t_est, seed=2)

    lab = scene.is_dynamic
    print(f"  {n_obj} objects x {ppo} pts = {scene.Md} dynamic vs {n_static} "
          f"static  (dynamic frac {scene.Md/scene.P:.2f})")
    print(f"  overall AUROC harm={sp.auroc(Eh, lab):.3f}  raw={sp.auroc(Er, lab):.3f}")
    for j in range(n_obj):
        sel = (scene.object_id == j) | (~lab)         # this object vs static
        y = scene.object_id[sel] == j
        print(f"    object {j}: AUROC vs static = {sp.auroc(Eh[sel], y):.3f}")

    # also: does it survive as #objects grows (until dynamic stops being minority)?
    print("  scaling #objects (12 pts each, 90 static), mean AUROC over 4 seeds:")
    counts, aurocs, fracs = [], [], []
    for nob in [1, 2, 4, 6, 8, 10]:
        au = []
        for s in range(4):
            sc = sp.make_scene_multi(np.random.default_rng(10 + s), n_views=14,
                                     n_static=90, n_objects=nob, pts_per_object=12,
                                     object_speed=0.08, object_spin=0.05)
            ed = sp.make_view_graph(sc.N)
            Rp, tp = perturb_poses(sc, np.random.default_rng(30 + s),
                                   rot_mag=0.06, trans_mag=0.09, accumulate=True)
            E, _ = energies(sc, ed, Rp, tp, seed=s)
            au.append(sp.auroc(E, sc.is_dynamic))
        counts.append(nob); aurocs.append(float(np.mean(au)))
        fracs.append(nob * 12 / (90 + nob * 12))
    print(f"    #objects {counts}")
    print(f"    AUROC    {[round(a, 3) for a in aurocs]}")
    print(f"    dyn.frac {[round(f, 2) for f in fracs]}")

    # ---- figure ----
    pts = _mean_world(scene)
    a, b = 0, 2
    e = np.sqrt(Eh + 1e-15); e /= e.max() + 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    ax = axes[0]
    sc = ax.scatter(pts[:, a], pts[:, b], c=e, cmap="inferno", vmin=0, vmax=1,
                    s=46, edgecolors="0.3", linewidths=0.3)
    colors = plt.cm.tab10(np.arange(n_obj))
    for j in range(n_obj):
        m = scene.object_id == j
        ax.scatter(pts[m, a], pts[m, b], facecolors="none", edgecolors=colors[j],
                   s=150, linewidths=1.8, label=f"object {j}")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(f"3 objects, all localized (AUROC {sp.auroc(Eh, lab):.2f})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="harmonic energy (norm.)")

    ax = axes[1]
    ax.plot(fracs, aurocs, "o-", color="#258")
    ax.axhline(0.5, ls="--", c="0.4", lw=1, label="chance")
    ax.axvline(0.5, ls=":", c="#c33", lw=1, label="dyn = static")
    ax.set_xlabel("dynamic fraction"); ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Holds while objects stay a minority")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/robust_multi.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/robust_multi.png\n")


# =====================================================================
# 2. Robust gauge anchoring
# =====================================================================
def exp_robust_anchoring():
    print("=" * 64)
    print("2. ROBUST GAUGE ANCHORING  (push past the static=dynamic cliff)")
    print("=" * 64)
    n_dyn = 24
    statics = [120, 90, 60, 42, 36, 30, 24, 18]
    print(f"  {n_dyn} dynamic pts; sweeping #static (mean AUROC over 5 seeds)")
    print(f"  {'static frac':<13}{'standard':>10}{'robust':>10}")
    fracs, std_au, rob_au = [], [], []
    for nst in statics:
        su, ru = [], []
        for s in range(5):
            sc = sp.make_scene(np.random.default_rng(50 + s), n_views=14,
                               n_static=nst, n_dynamic=n_dyn,
                               object_speed=0.08, object_spin=0.05)
            ed = sp.make_view_graph(sc.N)
            Rp, tp = perturb_poses(sc, np.random.default_rng(70 + s),
                                   rot_mag=0.06, trans_mag=0.09, accumulate=True)
            Es, _ = energies(sc, ed, Rp, tp, seed=s, robust=False)
            Er, _ = energies(sc, ed, Rp, tp, seed=s, robust=True)
            su.append(sp.auroc(Es, sc.is_dynamic))
            ru.append(sp.auroc(Er, sc.is_dynamic))
        f = nst / (nst + n_dyn)
        fracs.append(f); std_au.append(np.mean(su)); rob_au.append(np.mean(ru))
        print(f"  {f:<13.2f}{np.mean(su):>10.3f}{np.mean(ru):>10.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.7))
    ax.plot(fracs, std_au, "s-", color="0.5", lw=2, label="standard (least squares)")
    ax.plot(fracs, rob_au, "o-", color="#c0392b", lw=2, label="robust (IRLS)")
    ax.axhline(0.5, ls="--", c="0.4", lw=1)
    ax.axvline(0.5, ls=":", c="#258", lw=1.2, label="static = dynamic")
    ax.text(0.345, 0.47, "chance", fontsize=10, color="0.4")
    ax.set_xlabel("static fraction"); ax.set_ylabel("AUROC: moving vs static")
    ax.set_title("Robust anchoring extends the working regime")
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/robust_anchor.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/robust_anchor.png\n")


# =====================================================================
# 3. Iterated sheaf Gauss-Newton
# =====================================================================
def exp_iterated_gn():
    print("=" * 64)
    print("3. ITERATED GAUSS-NEWTON  (is the linearization controlled?)")
    print("=" * 64)
    rng = np.random.default_rng(42)
    scene = sp.make_scene(rng, n_views=14, n_static=90, n_dynamic=24,
                          object_speed=0.08, object_spin=0.05)
    edges = sp.make_view_graph(scene.N)
    drifts = np.linspace(0.1, 1.0, 10)
    print(f"  large initial drift sweep (mean over 3 seeds):")
    print(f"  {'drift':<8}{'single-shot':>13}{'iterated GN':>13}")
    single, iterd = [], []
    for m in drifts:
        ss, it = [], []
        for s in range(3):
            Rp, tp = perturb_poses(scene, np.random.default_rng(200 + s),
                                   rot_mag=m, trans_mag=1.5 * m, accumulate=True)
            Es, _ = energies(scene, edges, Rp, tp, seed=s, noise=0.0)
            ss.append(sp.auroc(Es, scene.is_dynamic))
            h, inc, est, _ = sp.iterated_gn(scene, edges, Rp, tp, n_iters=12)
            E, c = sp.per_point_energy(h, inc, scene.P); E /= np.maximum(c, 1)
            it.append(sp.auroc(E, scene.is_dynamic))
        single.append(np.mean(ss)); iterd.append(np.mean(it))
        print(f"  {m:<8.2f}{np.mean(ss):>13.3f}{np.mean(it):>13.3f}")

    # convergence trace at one large drift
    Rp, tp = perturb_poses(scene, np.random.default_rng(201),
                           rot_mag=0.6, trans_mag=0.9, accumulate=True)
    _, _, _, hist = sp.iterated_gn(scene, edges, Rp, tp, n_iters=12, record=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ax = axes[0]
    ax.plot(drifts, single, "s-", color="0.5", lw=2, label="single-shot")
    ax.plot(drifts, iterd, "o-", color="#c0392b", lw=2, label="iterated GN")
    ax.axhline(0.5, ls="--", c="0.4", lw=1)
    ax.set_xlabel("initial drift per step (rad)")
    ax.set_ylabel("AUROC: moving vs static")
    ax.set_title("Re-linearizing recovers large pose error")
    ax.set_ylim(0.3, 1.02); ax.legend(fontsize=10)

    ax = axes[1]
    ax.semilogy(range(len(hist)), hist, "o-", color="#258")
    ax.set_xlabel("outer iteration")
    ax.set_ylabel(r"GN step size $\|\xi\|$")
    ax.set_title("Convergence at drift = 0.6 rad/step")
    fig.tight_layout()
    fig.savefig("figures/robust_iterated.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/robust_iterated.png\n")


# =====================================================================
# 4. Scale-gauge validation
# =====================================================================
def exp_scale_gauge():
    print("=" * 64)
    print("4. SCALE-GAUGE VALIDATION  (invariance + scale-drift absorption)")
    print("=" * 64)
    rng = np.random.default_rng(42)
    scene = sp.make_scene(rng, n_views=14, n_static=90, n_dynamic=24,
                          object_speed=0.08, object_spin=0.05)
    edges = sp.make_view_graph(scene.N)
    R_est, t_est = perturb_poses(scene, np.random.default_rng(7),
                                 rot_mag=0.08, trans_mag=0.12, accumulate=True)

    # (a) gauge-invariance under random global similarity
    base, _ = energies(scene, edges, R_est, t_est, seed=2)
    base_au = sp.auroc(base, scene.is_dynamic)
    print(f"  (a) base AUROC = {base_au:.4f}; applying random global similarities:")
    grng = np.random.default_rng(3)
    base_med = np.median(base[scene.is_dynamic]) + 1e-15
    inv_au, energy_ratio, sgs = [], [], []
    for sg in np.exp(np.linspace(np.log(0.25), np.log(4.0), 9)):
        Rg = sp.random_rotation(grng, np.pi)
        tg = grng.normal(scale=2.0, size=3)
        sceneg = sp.transform_scene(scene, Rg, tg, float(sg))
        # transform the pose estimates the same way (left global similarity)
        Rg_est = np.einsum('ij,njk->nik', Rg, R_est)
        tg_est = sg * (t_est @ Rg.T) + tg
        Eg, _ = energies(sceneg, edges, Rg_est, tg_est, seed=2)
        inv_au.append(sp.auroc(Eg, scene.is_dynamic))
        energy_ratio.append(np.median(Eg[scene.is_dynamic]) / base_med)
        sgs.append(float(sg))
    print(f"      AUROC across gauges: {[round(float(a), 4) for a in inv_au]} "
          f"(max dev {max(abs(np.array(inv_au)-base_au)):.2e})")
    print(f"      energy ratio / sg^2: "
          f"{[round(float(r)/(s*s), 3) for r, s in zip(energy_ratio, sgs)]} (expect ~1)")

    # (b) per-view scale drift absorbed (not flagged as dynamic)
    print("  (b) per-view scale drift (no rot/trans drift): raw vs harmonic")
    print(f"      {'scale sd':<12}{'AUROC raw':>11}{'AUROC harm':>12}")
    sds, raw_au, har_au = [], [], []
    for sd in [0.0, 0.05, 0.1, 0.2, 0.3]:
        ra, ha = [], []
        for s in range(5):
            srng = np.random.default_rng(400 + s)
            s_est = np.exp(srng.normal(scale=sd, size=scene.N))
            Eh, Er = energies(scene, edges, scene.cam_R, scene.cam_t,
                              s_est=s_est, seed=s)
            ra.append(sp.auroc(Er, scene.is_dynamic))
            ha.append(sp.auroc(Eh, scene.is_dynamic))
        sds.append(sd); raw_au.append(np.mean(ra)); har_au.append(np.mean(ha))
        print(f"      {sd:<12.2f}{np.mean(ra):>11.3f}{np.mean(ha):>12.3f}")

    sgs = np.array(sgs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ax = axes[0]
    ax.loglog(sgs, energy_ratio, "o", ms=8, color="#c0392b", zorder=3,
              label="measured energy ratio")
    ax.loglog(sgs, sgs**2, "--", c="0.4", lw=1.4, label=r"$s_g^2$ (equivariance)")
    ax.set_xlabel("global scale factor $s_g$")
    ax.set_ylabel("dynamic energy / base")
    ax.set_title(f"Gauge-equivariant; AUROC = {base_au:.3f} for all gauges")
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(sds, raw_au, "s-", color="0.5", lw=2, label="raw residual")
    ax.plot(sds, har_au, "o-", color="#c0392b", lw=2, label="harmonic")
    ax.axhline(0.5, ls="--", c="0.4", lw=1)
    ax.set_xlabel("per-view scale drift (s.d. of log-scale)")
    ax.set_ylabel("AUROC: moving vs static")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Scale drift absorbed as consistent")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/robust_scale.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/robust_scale.png\n")


if __name__ == "__main__":
    exp_multi_objects()
    exp_robust_anchoring()
    exp_iterated_gn()
    exp_scale_gauge()
    print("All robustness sub-experiments complete. Figures in figures/.")
