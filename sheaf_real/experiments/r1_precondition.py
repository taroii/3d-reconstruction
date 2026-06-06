"""Static-dominance precondition on real data: harmonic AUROC vs dynamic
fraction across Sintel scenes (reads results/r1_sintel.csv). The synthetic
A-block predicted collapse as dynamic stops being a minority; this checks it on
real scenes. Pure CPU/CSV."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("../results/r1_sintel.csv")))
df = np.array([float(r["dynfrac"]) for r in rows])
h0 = np.array([float(r["h0"]) for r in rows])
names = [r["scene"] for r in rows]
rcorr = np.corrcoef(df, h0)[0, 1]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(df, h0, s=40, c="#c0392b", zorder=3)
for x, y, n in zip(df, h0, names):
    ax.annotate(n, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0.5, ls="--", c="0.5", lw=1, label="chance")
ax.axvline(0.5, ls=":", c="#258", lw=1, label="dynamic = static")
ax.set_xlabel("dynamic fraction (GT, scored pixels)")
ax.set_ylabel("harmonic AUROC @clean")
ax.set_title(f"Static-dominance precondition on Sintel (Pearson r={rcorr:.2f})")
ax.set_ylim(0.15, 1.02); ax.legend()
fig.tight_layout()
fig.savefig("../results/r1_precondition.png", dpi=150, bbox_inches="tight")
print(f"n={len(rows)} scenes, Pearson r(dynfrac, AUROC)={rcorr:.3f}")
lowdyn = h0[df < 0.35]; highdyn = h0[df >= 0.35]
print(f"  static-dominant (dynfrac<0.35, n={len(lowdyn)}): AUROC {lowdyn.mean():.3f} ± {lowdyn.std():.3f}")
print(f"  dynamic-heavy   (dynfrac>=0.35, n={len(highdyn)}): AUROC {highdyn.mean():.3f} ± {highdyn.std():.3f}")
print("  -> results/r1_precondition.png")
