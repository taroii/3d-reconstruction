"""
Combined experiment: every caveat at once.

A single hard scene that stacks all four stressors the sub-experiments isolated
(experiments_robustness.py):
  * multiple moving objects,
  * a reduced static fraction,
  * large accumulating camera drift,
  * per-view scale ambiguity (Sim(3)),
and solves it with the full method -- a robust, iterated, scale-anchored sheaf
Gauss-Newton. Run only after each sub-experiment has passed on its own.

Compares three readings of the same data:
  raw residual (what BA minimizes) | naive single-shot harmonic | full method.

    conda run -n sheaf-poc python combined_experiment.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sheaf_poc as sp
from run_experiment import perturb_poses

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "figure.dpi": 200})

DRIFT, SCALE_SD = 0.45, 0.18
N_STATIC, N_OBJ, PPO, N_VIEWS = 46, 3, 12, 16


def make_hard(seed):
    scene = sp.make_scene_multi(np.random.default_rng(seed), n_views=N_VIEWS,
                                n_static=N_STATIC, n_objects=N_OBJ,
                                pts_per_object=PPO, object_speed=0.07,
                                object_spin=0.04)
    edges = sp.make_view_graph(scene.N)
    R_est, t_est = perturb_poses(scene, np.random.default_rng(seed + 1),
                                 rot_mag=DRIFT, trans_mag=1.4 * DRIFT,
                                 accumulate=True)
    s_est = np.exp(np.random.default_rng(seed + 2).normal(scale=SCALE_SD,
                                                          size=scene.N))
    return scene, edges, R_est, t_est, s_est


def _ppe(h, inc, P):
    E, c = sp.per_point_energy(h, inc, P)
    return E / np.maximum(c, 1)


def evaluate(scene, edges, R_est, t_est, s_est):
    """Return per-point energies for the three readings + the full estimate."""
    d, r0, inc, _ = sp.build_sheaf(scene, edges, R_est, t_est, use_scale=True,
                                   fix_scale_gauge=True, pose_s_est=s_est)
    E_raw = _ppe(r0, inc, scene.P)
    h_naive, _ = sp.harmonic_projection(d, r0)
    E_naive = _ppe(h_naive, inc, scene.P)
    h_full, inc_f, est = sp.robust_iterated_gn(scene, edges, R_est, t_est,
                                               s0=s_est, n_iters=12, n_irls=6)
    E_full = _ppe(h_full, inc_f, scene.P)
    return E_raw, E_naive, E_full


def main():
    scene, edges, R_est, t_est, s_est = make_hard(7)
    lab = scene.is_dynamic
    print(f"Combined scene: {scene.N} views, {N_STATIC} static + {scene.Md} "
          f"dynamic in {N_OBJ} objects (static frac {N_STATIC/scene.P:.2f}).")
    print(f"Confounds stacked: accumulating drift {DRIFT} rad/step, "
          f"per-view scale s.d. {SCALE_SD}, Sim(3) scale ambiguity.\n")

    # headline instance (for the spatial figure)
    E_raw, E_naive, E_full = evaluate(scene, edges, R_est, t_est, s_est)

    # AUROC averaged over seeds for the bar chart
    rows = {"raw residual\n(BA objective)": [], "naive harmonic\n(single-shot)": [],
            "full method\n(robust+iterated)": []}
    for sd in range(6):
        sc, ed, R, t, s = make_hard(100 + 10 * sd)
        er, en, ef = evaluate(sc, ed, R, t, s)
        rows["raw residual\n(BA objective)"].append(sp.auroc(er, sc.is_dynamic))
        rows["naive harmonic\n(single-shot)"].append(sp.auroc(en, sc.is_dynamic))
        rows["full method\n(robust+iterated)"].append(sp.auroc(ef, sc.is_dynamic))

    print(f"{'method':<34}{'AUROC (mean +- sd, 6 seeds)':>28}")
    print("-" * 62)
    means = {}
    for name, vals in rows.items():
        v = np.array(vals); means[name] = v.mean()
        print(f"{name.replace(chr(10), ' '):<34}{v.mean():>18.3f} +- {v.std():.3f}")
    print("\nper-object AUROC (full method, headline instance):")
    for j in range(N_OBJ):
        sel = (scene.object_id == j) | (~lab)
        print(f"    object {j}: {sp.auroc(E_full[sel], scene.object_id[sel] == j):.3f}")

    # ---------------- figure ----------------
    pts = np.array([np.mean([scene.world_point(k, v) for v in range(scene.N)],
                            axis=0) for k in range(scene.P)])
    a, b = 0, 2
    e = np.sqrt(E_full + 1e-15); e /= e.max() + 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    sc_ = ax.scatter(pts[:, a], pts[:, b], c=e, cmap="inferno", vmin=0, vmax=1,
                     s=46, edgecolors="0.3", linewidths=0.3)
    colors = plt.cm.tab10(np.arange(N_OBJ))
    for j in range(N_OBJ):
        m = scene.object_id == j
        ax.scatter(pts[m, a], pts[m, b], facecolors="none", edgecolors=colors[j],
                   s=150, linewidths=1.8, label=f"object {j}")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(f"Full method recovers all objects (AUROC {sp.auroc(E_full, lab):.2f})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.colorbar(sc_, ax=ax, fraction=0.046, label="harmonic energy (norm.)")

    ax = axes[1]
    names = list(rows.keys())
    vals = [np.mean(rows[n]) for n in names]
    errs = [np.std(rows[n]) for n in names]
    bars = ax.bar(range(len(names)), vals, yerr=errs, capsize=5,
                  color=["0.6", "0.6", "#c0392b"])
    ax.axhline(0.5, ls="--", c="0.4", lw=1)
    ax.text(2.4, 0.52, "chance", fontsize=10, color="0.4", ha="right")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("AUROC: moving vs static")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("All confounds stacked, same data")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/combined.png", bbox_inches="tight")
    plt.close(fig)
    print("\n  -> figures/combined.png")


if __name__ == "__main__":
    main()
