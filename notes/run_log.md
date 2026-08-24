# Premise-check run log

Durable record of what was actually run, on what data, and what came out. Numbers
here are copied from files in `results/premise_check/`, not from recollection —
if this file and those files disagree, the files win.

Protocol: `notes/premise_check.md`. Server: `bruinml`, repo at
`~/taro/3d-reconstruction`, data under `data/` (symlinks into `/mnt/data/premise`).

---

## Run 1 — 2026-08-24

**Pre-registered** 2026-08-24 (`results/premise_check/decision.md`, thresholds
locked before any number existed; `analyze.py` refuses to run if the live
constants no longer match the recorded ones).

**Config** (`results/premise_check/results.json` → `config`):
datasets `middlebury, infinigen, ibims`; streams `vggt_point, vggt_depth,
pi3_local`; GT control `gt_raw`; `w=3`, `delta_min=0.05`, `align=median`;
full pre-registered sweep η∈{0.02,0.05,0.10}, τ∈{1,2,3}, β∈{0.1,0.2,0.3}.

**Scenes measured:** 129 total — middlebury 20, infinigen 9, ibims 100.
By tier: **A = 29, B = 100.** No views skipped.

### Outcome: GO

| stream | FP_model | FP_model (Tier A) | FP_GT | diff | ratio | 95% CI | p (Holm) | sweep |
|---|---|---|---|---|---|---|---|---|
| vggt_point | 0.2184 | 0.2308 | 0.0282 | +19.02 pp | 7.75× | [+16.96, +21.07] | 1.6e-21 | 27/27 stable |
| vggt_depth | 0.2779 | 0.2815 | 0.0282 | +24.98 pp | 9.86× | [+22.92, +26.99] | 5.8e-22 | 27/27 stable |
| pi3_local  | 0.2696 | 0.2478 | 0.0282 | +24.14 pp | 9.57× | [+21.99, +26.32] | 5.8e-22 | 27/27 stable |

Rank-biserial effect size 0.974–0.993, i.e. the difference holds in nearly every
individual scene, not merely on average.

### Robustness checks (`src/diagnostics.py`)

Both were prompted by eyeballing the Sec. 8 figures, and both **failed to break
the result**:

1. **Alignment** (`diag_alignment.csv`). A figure showed the prediction sitting
   ~0.2 m behind GT on a far surface, which a scale-only fit would leave in place
   and which would push a whole surface into the void. Refitting with scale+shift
   does **not** collapse FP — it is unchanged or slightly higher (middlebury
   vggt_point 0.254 → 0.275; ibims 0.239 → 0.230; infinigen 0.185 → 0.185).
2. **Boundary density** (`diag_density.csv`). A second figure's boundary set was
   dominated by a crate's wire mesh, raising the worry that we measure
   unresolvable texture rather than smeared silhouettes. FP is **flat across
   density and highest on the sparsest, cleanest silhouettes** — vggt_point
   sparse 0.243 / moderate 0.209 / dense 0.177 / saturated 0.220. Same pattern for
   all three streams. 42% of `B_eval` is silhouette-like.

### Known caveats on this run

- **Tier A is only 29 scenes**, because Infinigen arrived incomplete (24 of 40
  tarballs; six seed/camera pairs had depth without images). Backfill was running
  when this was written. The headline is therefore dominated by iBims-1 (Tier B,
  real laser GT). `FP_model_tierA` is reported separately and GO was decided on it.
- **FP_GT = 2.8%, not ≈0.** The protocol expects ≈0 on synthetic by construction.
  Most of the control's mass comes from the Tier B real data. A Tier-A-only
  control breakdown has not been produced yet and should be before this is written up.
- Middlebury contributes 20 scenes, not 23 — three zips did not extract.

---

## Measurability gate (Sec. 4)

`results/premise_check/gate.csv`, 8 scenes × 3 views per dataset, 2026-08-24:

| dataset | tier | frac_boundary_valid | masked | verdict |
|---|---|---|---|---|
| middlebury | A | 0.979 | 0.000 | pass |
| pointodyssey | A | 0.962 | 0.000 | pass |
| sintel | A | 0.938 | 0.004 | pass |
| eth3d | B | 0.928 | 0.000 | pass (only 1266 boundary px — sparse GT) |
| spring | A | 0.920 | 0.000 | pass |
| bonn | C | 0.879 | 0.000 | pass |
| tartanair | A | 0.826 | 0.000 | pass |
| infinigen | A | 0.821 | 0.000 | pass |
| ibims | B | 0.765 | 0.047 | pass |
| **nyuv2** | **C** | **0.518** | 0.000 | **borderline — see below** |

**NYUv2 sits on the threshold and its verdict flips with the sample:** 0.482 and
0.487 (6–10 scenes) versus 0.518 (8 scenes) — fail, fail, pass. It must not be
reported as decisively either until run on a much larger sample.

**Bonn passes, contrary to the prediction in `notes/server_requirements.md`** that
it should fail the gate through aggressive boundary masking. Note the gate
*structurally cannot* detect Bonn's kind of masking: Bonn has no separate mask
file, its zeros ARE the deletion, so the evidence is destroyed rather than
flagged and `masked=0.000` is true by construction. Bonn therefore does **not**
validate the gate the way that document intended; a different statistic (e.g.
depth coverage in a band around detected edges) would be needed.

---

## Contamination matrix (Sec. 4)

`results/premise_check/contamination.csv`. Tier A and clean for every Tier 1
model: **pointodyssey, spring, sintel, middlebury, infinigen**. "unknown" counts
as contaminated. The `vggt?` / `pi3?` cells are read from secondary sources and
**still need confirming against each paper's training-data table** before the
headline rests on them.
