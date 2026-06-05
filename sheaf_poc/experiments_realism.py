"""
Realism stress-tests ("A block") -- gaps between the clean synthetic PoC and
real D^2USt3R data, probed while we still have ground truth.

  A1. correspondence outliers   -> realism_corr.png
      Real matches are imperfect; a mismatch at a static point looks like an
      inconsistency that is not motion. Does it corrupt localization, and does
      robust IRLS protect against it?
  A2. structured prediction error -> realism_structured.png
      Real error is spatially coherent and view-dependent (depth/occlusion
      boundaries), not i.i.d. A coherent mispredicted static patch is
      cohomologically indistinguishable from motion -- only confidence can
      suppress it.
  A3. hyperparameter sensitivity -> realism_hparams.png
      Show the full solver is not finely tuned.

    conda run -n sheaf-poc python experiments_realism.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sheaf_poc as sp
from run_experiment import perturb_poses

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5,
                     "axes.titleweight": "bold", "figure.dpi": 200})


def _ppe(h, inc, P):
    E, c = sp.per_point_energy(h, inc, P)
    return E / np.maximum(c, 1)


def precision_at_k(E, labels, k):
    """Fraction of the top-k highest-energy points that are truly dynamic."""
    idx = np.argsort(E)[::-1][:k]
    return float(np.asarray(labels, bool)[idx].mean())


def solve(scene, edges, R, t, s=None, corr=None, obs_bias=None,
          robust=False, reduce="mean", noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    d, r0, inc, _ = sp.build_sheaf(scene, edges, R, t, use_scale=True,
                                   fix_scale_gauge=True, pose_s_est=s,
                                   corr=corr, obs_bias=obs_bias,
                                   noise_std=noise, rng=rng)
    h = (sp.robust_harmonic_projection(d, r0, inc)[0] if robust
         else sp.harmonic_projection(d, r0)[0])
    return sp.per_point_energy_robust(h, inc, scene.P, reduce=reduce)


# =====================================================================
# A1. Correspondence outliers
# =====================================================================
def exp_correspondence_outliers():
    print("=" * 64)
    print("A1. CORRESPONDENCE OUTLIERS")
    print("=" * 64)
    # Mismatches are SPORADIC (a point is mis-matched on a few edges), while real
    # motion is CONSISTENT (a moving point is inconsistent on every edge). A
    # summed energy readout accumulates the sporadic hits; a median-over-edges
    # readout rejects them. We compare the naive (mean) readout, the median
    # readout, and median on top of a robust solve.
    fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    keys = ["mean (naive)", "median", "robust+median"]
    AU = {k: [] for k in keys}; PR = {k: [] for k in keys}
    for f in fracs:
        acc = {k: [] for k in keys}; prc = {k: [] for k in keys}
        for sd in range(5):
            scene = sp.make_scene_multi(np.random.default_rng(sd), n_views=14,
                                        n_static=90, n_objects=3,
                                        pts_per_object=12, object_speed=0.08,
                                        object_spin=0.05)
            edges = sp.make_view_graph(scene.N)
            R, t = perturb_poses(scene, np.random.default_rng(100 + sd),
                                 rot_mag=0.10, trans_mag=0.15, accumulate=True)
            corr = sp.make_correspondences(scene, edges, f,
                                           np.random.default_rng(500 + sd))
            lab = scene.is_dynamic; Md = int(lab.sum())
            E = {
                "mean (naive)":  solve(scene, edges, R, t, corr=corr,
                                       reduce="mean", seed=sd),
                "median":        solve(scene, edges, R, t, corr=corr,
                                       reduce="median", seed=sd),
                "robust+median": solve(scene, edges, R, t, corr=corr,
                                       robust=True, reduce="median", seed=sd),
            }
            for k in keys:
                acc[k].append(sp.auroc(E[k], lab))
                prc[k].append(precision_at_k(E[k], lab, Md))
        for k in keys:
            AU[k].append(np.mean(acc[k])); PR[k].append(np.mean(prc[k]))
    print(f"  {'outlier frac':<14}" + "".join(f"{k:>16}" for k in keys))
    print("  AUROC:")
    for i, f in enumerate(fracs):
        print(f"  {f:<14.2f}" + "".join(f"{AU[k][i]:>16.3f}" for k in keys))
    print("  precision@(#dynamic):")
    for i, f in enumerate(fracs):
        print(f"  {f:<14.2f}" + "".join(f"{PR[k][i]:>16.3f}" for k in keys))

    colors = {"mean (naive)": "0.5", "median": "#c0392b", "robust+median": "#2166ac"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for k in keys:
        axes[0].plot(fracs, AU[k], "o-", color=colors[k], lw=2, label=k)
        axes[1].plot(fracs, PR[k], "o-", color=colors[k], lw=2, label=k)
    axes[0].set_ylabel("AUROC: moving vs static")
    axes[0].set_title("Median-over-edges readout rejects mismatches")
    axes[1].set_ylabel("Precision@(#dynamic)")
    axes[1].set_title("Motion is consistent across edges; mismatches are not")
    for ax in axes:
        ax.set_xlabel("correspondence-outlier fraction per edge")
        ax.set_ylim(0.3, 1.03); ax.axhline(0.5, ls="--", c="0.6", lw=1)
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/realism_corr.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/realism_corr.png\n")


# =====================================================================
# A2. Structured (coherent, view-dependent) prediction error
# =====================================================================
def _coherent_patch(scene, size, rng):
    """A spatially-contiguous cluster of static points (nearest neighbours of a
    random static seed)."""
    static_idx = np.where(~scene.is_dynamic)[0]
    pts = scene.static_pts
    seed = static_idx[rng.integers(len(static_idx))]
    d = np.linalg.norm(pts - pts[seed], axis=1)
    return static_idx[np.argsort(d[static_idx])[:size]]


def _structured_bias(scene, patch, sigma, rng):
    """View-dependent bias, coherent over the patch (same direction, per-view
    magnitude). Not a global transform -> not pose-explainable -> shows as H^1."""
    obs = np.zeros((scene.N, scene.P, 3))
    direction = rng.normal(size=3); direction /= np.linalg.norm(direction) + 1e-12
    for v in range(scene.N):
        b = rng.normal(scale=sigma)
        for k in patch:
            obs[v, k] = b * direction
    return obs


def exp_structured_error():
    print("=" * 64)
    print("A2. STRUCTURED PREDICTION ERROR (coherent static patch)")
    print("=" * 64)
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    patch_size = 14
    methods = {"standard": [], "robust": [], "confidence": []}
    falsealarm = {"standard": [], "robust": [], "confidence": []}
    for sg in sigmas:
        acc = {m: [] for m in methods}
        fa = {m: [] for m in methods}
        for sd in range(5):
            scene = sp.make_scene(np.random.default_rng(sd), n_views=14,
                                  n_static=90, n_dynamic=24,
                                  object_speed=0.08, object_spin=0.05)
            edges = sp.make_view_graph(scene.N)
            R, t = perturb_poses(scene, np.random.default_rng(100 + sd),
                                 rot_mag=0.08, trans_mag=0.12, accumulate=True)
            patch = _coherent_patch(scene, patch_size, np.random.default_rng(7 + sd))
            obs = _structured_bias(scene, patch, sg, np.random.default_rng(11 + sd))
            lab = scene.is_dynamic; Md = int(lab.sum())
            patch_mask = np.zeros(scene.P, bool); patch_mask[patch] = True
            clean_static = (~lab) & (~patch_mask)

            E_std = solve(scene, edges, R, t, obs_bias=obs, robust=False, seed=sd)
            E_rob = solve(scene, edges, R, t, obs_bias=obs, robust=True, seed=sd)
            # confidence: the network flags the corrupted region as unreliable;
            # discount its energy at detection time (the nu weights in the paper).
            conf = np.ones(scene.P); conf[patch] = 0.05
            E_conf = E_std * conf

            for m, E in [("standard", E_std), ("robust", E_rob),
                         ("confidence", E_conf)]:
                acc[m].append(precision_at_k(E, lab, Md))
                # false alarm: can the patch be told apart from clean static?
                sel = patch_mask | clean_static
                y = patch_mask[sel]
                fa[m].append(sp.auroc(E[sel], y))
        for m in methods:
            methods[m].append(np.mean(acc[m])); falsealarm[m].append(np.mean(fa[m]))
    print(f"  {'sigma':<8}" + "".join(f"{m+' P@K':>16}" for m in methods))
    for i, sg in enumerate(sigmas):
        print(f"  {sg:<8.2f}" + "".join(f"{methods[m][i]:>16.3f}" for m in methods))
    print("  patch false-alarm AUROC (lower=better, 0.5=indistinguishable):")
    for i, sg in enumerate(sigmas):
        print(f"  {sg:<8.2f}" + "".join(f"{falsealarm[m][i]:>16.3f}" for m in methods))

    colors = {"standard": "0.5", "robust": "#e08214", "confidence": "#2166ac"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for m in methods:
        axes[0].plot(sigmas, methods[m], "o-", color=colors[m], lw=2, label=m)
        axes[1].plot(sigmas, falsealarm[m], "o-", color=colors[m], lw=2, label=m)
    axes[0].set_ylabel("Precision@(#dynamic)")
    axes[0].set_title("Coherent error steals detections")
    axes[1].set_ylabel("patch vs clean-static AUROC")
    axes[1].set_title("Only confidence makes the patch invisible")
    axes[1].axhline(0.5, ls="--", c="0.6", lw=1)
    for ax in axes:
        ax.set_xlabel("structured-error magnitude $\\sigma$")
        ax.set_ylim(0.3, 1.03); ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/realism_structured.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/realism_structured.png\n")


# =====================================================================
# A3. Hyperparameter sensitivity (on the hard combined scene)
# =====================================================================
def _hard(seed):
    scene = sp.make_scene_multi(np.random.default_rng(seed), n_views=16,
                                n_static=46, n_objects=3, pts_per_object=12,
                                object_speed=0.07, object_spin=0.04)
    edges = sp.make_view_graph(scene.N)
    R, t = perturb_poses(scene, np.random.default_rng(seed + 1),
                         rot_mag=0.45, trans_mag=0.63, accumulate=True)
    s = np.exp(np.random.default_rng(seed + 2).normal(scale=0.18, size=scene.N))
    return scene, edges, R, t, s


def _full_auroc(seed, **kw):
    scene, edges, R, t, s = _hard(seed)
    h, inc, _ = sp.robust_iterated_gn(scene, edges, R, t, s0=s, **kw)
    E = _ppe(h, inc, scene.P)
    return sp.auroc(E, scene.is_dynamic)


def exp_hyperparameter_sensitivity():
    print("=" * 64)
    print("A3. HYPERPARAMETER SENSITIVITY (hard combined scene)")
    print("=" * 64)
    grids = {
        "huber_k":      ([1.0, 1.5, 2.0, 3.0, 5.0], dict()),
        "n_irls":       ([2, 4, 6, 10, 15], dict()),
        "n_iters":      ([2, 4, 8, 12, 20], dict()),
        "gauge_weight": ([1e1, 1e2, 1e3, 1e4, 1e5], dict()),
    }
    results = {}
    for knob, (vals, base) in grids.items():
        means, sds = [], []
        for v in vals:
            au = [_full_auroc(10 * sd, **{**base, knob: v}) for sd in range(4)]
            means.append(np.mean(au)); sds.append(np.std(au))
        results[knob] = (vals, means, sds)
        print(f"  {knob:<14}" + "  ".join(f"{v}:{m:.3f}"
              for v, m in zip(vals, means)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, knob in zip(axes.ravel(), grids):
        vals, means, sds = results[knob]
        x = np.array(vals, float)
        logx = knob == "gauge_weight"
        (ax.semilogx if logx else ax.plot)(x, means, "o-", color="#c0392b", lw=2)
        ax.errorbar(x, means, yerr=sds, fmt="none", ecolor="#c0392b", capsize=4)
        ax.set_title(knob); ax.set_ylim(0.8, 1.02)
        ax.set_xlabel(knob); ax.set_ylabel("AUROC")
        ax.axhline(1.0, ls=":", c="0.6", lw=1)
    fig.suptitle("Localization AUROC is flat across solver hyperparameters",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/realism_hparams.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/realism_hparams.png\n")


if __name__ == "__main__":
    exp_correspondence_outliers()
    exp_structured_error()
    exp_hyperparameter_sensitivity()
    print("A-block complete. Figures in figures/.")
