# Archive

Intermediate / superseded artifacts, kept for provenance but out of the way.
Local only (gitignored). Nothing here is needed to reproduce current results.

- **`sheaf-era-results/`** — figures/CSVs/logs from the abandoned sheaf (harmonic
  H¹) idea. Predates the spectral band-routing pivot. Code that produced them was
  in `sheaf_real/` (now removed; recoverable from git history before the pivot).

- **`spectral-phase1/`** — the first `N_φ` run, trained on **Sintel only**:
  - `nphi_train.log` — 30-epoch log. Overfits: train MAE →13°, held-out val
    plateaus at **best 37.1° (epoch 2)**, drifting to ~39°. Documents the
    Sintel-only generalization ceiling that motivated pulling in more datasets.
  - `nphi_d2ust3r_sintel-only.pth` — the best (epoch-2) head checkpoint. Kept as
    a baseline to compare the multi-dataset retrain against.
