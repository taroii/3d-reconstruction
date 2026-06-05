"""
Publication / teammate-facing figures. Two self-contained visual beats:

  story_localization.png : truth -> raw residual (BA) -> harmonic H^1 (ours),
                           same drifted scene. Shows BA's residual is confused
                           by drift while the cohomological residual is not.
  story_drift.png        : AUROC vs camera drift. Harmonic flat at 1; raw decays
                           below chance. Shows drift-invariance.

Labels are intentionally terse -- prose lives in the paper around the figures.
Run after the env is set up:  conda run -n sheaf-poc python make_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sheaf_poc as sp
from run_experiment import perturb_poses, run_case

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 200,
})

CMAP = "inferno"


def _scene_points(scene):
    """Mean world position of each scene point over frames (for plotting)."""
    pts = np.zeros((scene.P, 3))
    for k in range(scene.P):
        pts[k] = np.mean([scene.world_point(k, v) for v in range(scene.N)], axis=0)
    return pts


def figure_localization(drift=0.30, seed=2):
    """Triptych on one drifted scene: ground truth, raw residual, harmonic."""
    rng = np.random.default_rng(42)
    scene = sp.make_scene(rng, n_views=14, n_static=90, n_dynamic=24,
                          object_speed=0.08, object_spin=0.05)
    edges = sp.make_view_graph(scene.N)
    R_est, t_est = perturb_poses(scene, np.random.default_rng(7),
                                 rot_mag=drift, trans_mag=1.5 * drift,
                                 accumulate=True)
    res = run_case(scene, edges, R_est, t_est, noise_std=0.01,
                   use_scale=True, seed=seed)

    pts = _scene_points(scene)
    a, b = 0, 2                      # x, z projection
    dyn = scene.is_dynamic

    def norm(e):
        e = np.sqrt(e + 1e-15)
        return e / (e.max() + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))

    # (1) ground truth
    ax = axes[0]
    ax.scatter(pts[~dyn, a], pts[~dyn, b], c="0.72", s=55,
               edgecolors="0.4", linewidths=0.4, label="static")
    ax.scatter(pts[dyn, a], pts[dyn, b], c="#e23", s=70,
               edgecolors="k", linewidths=0.5, label="moving object")
    ax.set_title("Ground truth")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # (2) raw residual  (3) harmonic
    for ax, key, title in [(axes[1], "E_raw", "Raw residual (BA)"),
                           (axes[2], "E_harm", r"Harmonic $H^1$ (ours)")]:
        c = norm(res[key])
        sc = ax.scatter(pts[:, a], pts[:, b], c=c, cmap=CMAP, vmin=0, vmax=1,
                        s=58, edgecolors="0.25", linewidths=0.4)
        ax.scatter(pts[dyn, a], pts[dyn, b], facecolors="none",
                   edgecolors="cyan", s=165, linewidths=1.8)
        au = res["auroc_raw"] if key == "E_raw" else res["auroc_harm"]
        ax.set_title(f"{title}   AUROC {au:.2f}")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=[axes[1], axes[2]], fraction=0.03, pad=0.015)
    cb.set_label("obstruction energy (norm.)")
    fig.savefig("figures/story_localization.png", bbox_inches="tight")
    plt.close(fig)
    print(f"story_localization.png  (drift={drift}, "
          f"raw AUROC={res['auroc_raw']:.2f}, harm AUROC={res['auroc_harm']:.2f})")


def figure_drift():
    """AUROC vs accumulating camera drift, mean +- sd over seeds."""
    rng = np.random.default_rng(42)
    scene = sp.make_scene(rng, n_views=14, n_static=90, n_dynamic=24,
                          object_speed=0.08, object_spin=0.05)
    edges = sp.make_view_graph(scene.N)
    mags = np.linspace(0.0, 0.35, 12)
    raw_m, raw_s, har_m, har_s = [], [], [], []
    for m in mags:
        rs, hs = [], []
        for s in range(6):
            Rp, tp = perturb_poses(scene, np.random.default_rng(300 + s),
                                   rot_mag=m, trans_mag=1.5 * m, accumulate=True)
            r = run_case(scene, edges, Rp, tp, noise_std=0.01,
                         use_scale=True, seed=s)
            rs.append(r["auroc_raw"]); hs.append(r["auroc_harm"])
        raw_m.append(np.mean(rs)); raw_s.append(np.std(rs))
        har_m.append(np.mean(hs)); har_s.append(np.std(hs))
    raw_m, raw_s = np.array(raw_m), np.array(raw_s)
    har_m, har_s = np.array(har_m), np.array(har_s)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhspan(0.3, 0.5, color="0.92")
    ax.axhline(0.5, ls="--", c="0.4", lw=1)
    ax.text(0.345, 0.475, "chance", ha="right", va="top", fontsize=10, color="0.4")

    ax.plot(mags, har_m, "o-", color="#c0392b", lw=2.2, ms=5,
            label=r"Harmonic $H^1$ (ours)")
    ax.fill_between(mags, har_m - har_s, har_m + har_s, color="#c0392b", alpha=0.2)
    ax.plot(mags, raw_m, "s-", color="0.45", lw=2.2, ms=5,
            label="Raw residual (BA)")
    ax.fill_between(mags, raw_m - raw_s, raw_m + raw_s, color="0.45", alpha=0.25)

    ax.set_xlabel("camera drift per step (rad)")
    ax.set_ylabel("AUROC: moving vs static")
    ax.set_title("Drift-invariance of the obstruction")
    ax.set_xlim(0, 0.35); ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95)
    fig.savefig("figures/story_drift.png", bbox_inches="tight")
    plt.close(fig)
    print(f"story_drift.png  (raw {raw_m[0]:.2f}->{raw_m[-1]:.2f}, "
          f"harm flat at {har_m.mean():.2f})")


if __name__ == "__main__":
    figure_localization()
    figure_drift()
    print("Wrote figures/story_localization.png and figures/story_drift.png")
