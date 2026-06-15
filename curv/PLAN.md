# Curvature-Inclusive D²USt3R — build plan

Augment D²USt3R training with a discrete-curvature signal computed for free from
the GT pointmap `X̄` (GT depth + intrinsics + poses). One added term, no
redesign: keep `L_static + L_dyn` and the frozen-encoder finetune protocol.

Goal question: does curvature supervision improve depth (AbsRel ↓, δ₁ ↑),
**especially at object boundaries and in dynamic regions**, over the D²USt3R
baseline?

## Pieces in this package

| file | role |
|---|---|
| `curvature.py` | GT curvature from depth+K (Gaussian angle-deficit / signed mean), discontinuity + interior validity, scale normalization, signed-log compression, grid alignment, viz |
| `curvature_head.py` | `CurvatureHead` (frozen encoder + DPT head → curvature+conf) and `curvature_conf_loss` (Huber, conf-weighted) |
| `train_curv.py` | standalone frozen-encoder learnability prototype for the head |
| `sanity_curvature.py` | synthetic-surface unit checks + real-frame curvature dump |
| `datasets.py` | TartanAir / PointOdyssey / Sintel(eval-only) frame sources + `load_depth_K` |
| `backbone.py` | load the frozen D²USt3R checkpoint via the bundled `dust3r` pkg |

The **GT curvature is deterministic and cheap**, so it is computed on-the-fly and
cached (`../cache/curv_targets/<mode>/...`). Caching is an optimization, not a
requirement.

## Phases

### Phase 0 — curvature correctness (local-testable)  ✅ scaffolded
`python sanity_curvature.py` — synthetic plane/bump/saddle sign+magnitude checks.
On the server, `--dataset sintel ...` to eyeball a real curvature map (should
fire on edges/corners, be quiet on flat regions, and be empty across depth
cliffs). **Gate:** synthetic checks PASS; real dumps look geometrically sane.

### Phase 1 — Option 2: curvature-weighted loss (cheapest prototype, do first)
No new head, no new params. Reweight D²USt3R's existing per-pixel static/dynamic
losses by GT curvature:

    w(u,v) = 1 + γ · |K_gt(u,v)|          # upweight geometrically complex regions
    L_static, L_dyn  ←  w · L_static, w · L_dyn

This lives in the **DDUSt3R training loop** (the pair loss), not here — it needs
`K_gt` per training pair, which we compute with `curvature.curvature_from_depth_K`
on the same normalized pointmap the loss uses. Half-day change; immediate signal
on whether curvature info helps at all. Sweep γ.

### Phase 1.5 — head learnability gate (this package, standalone)
`python train_curv.py --overfit` then a small `--datasets tartanair` run. Proves
the frozen encoder's features carry recoverable curvature before we pay for the
full-loop integration. **Gate:** overfit drives MAE down; val MAE on held-out
TartanAir envs is well below a predict-zero baseline; qualitative maps fire on
edges.

### Phase 2 — Option 1: curvature head in D²USt3R (first reportable result)
Add a DPT head predicting `Ĉ ∈ W×H×1` to D²USt3R alongside its pointmap +
dynamic-mask heads (mirrors the dynamic-mask head, BCE → Huber). Total:

    L_total = L_static + L_dyn + λ · L_curv

Port `CurvatureHead`/`curvature_conf_loss` into the DDUSt3R model + train loop;
freeze encoder, finetune decoder + DPT heads from the D²USt3R checkpoint. Sweep λ.

### Phase 3 — eval / ablation
Baseline vs +weighted-loss (Phase 1) vs +head (Phase 2). Report per-dataset
(Bonn, TUM-Dynamics, Sintel, KITTI), **all-pixels and dynamic-region**, plus
boundary-restricted metrics. Qualitative: the curvature head should visibly fire
on edges/corners.

### Option 3 (deferred) — curvature-aware dynamic masking
Use predicted-pointmap curvature discontinuities to refine `M_dyn` at inference.
Only if Phases 1–2 show curvature carries real geometric information.

## Decisions to lock during the work
- **Estimator:** start with **mean** curvature (numerically stabler); ablate
  Gaussian (cleaner Theorema-Egregium story) later. `--mode` switches.
- **Target space:** signed-log compressed + Huber (heavy tails). `--no-compress`
  to ablate.
- **Scale:** compute `K_gt` on the **normalized** pointmap (matches D²USt3R's
  `z, z̄` normalization) so the target scale is consistent sample-to-sample.

## Gotchas (encoded in `curvature.py`, restate before each phase)
1. Depth cliffs blow up `|K|` → excluded via `depth_discontinuity_mask` +
   interior mask (the #1 issue).
2. Validity = interior pixels with a fully-valid neighborhood, AND’d with
   D²USt3R’s existing `valid_mask`.
3. Curvature is `1/area` (scale-sensitive) → normalize the pointmap first.
4. Heavy-tailed target → signed-log + robust loss, not raw + MSE.

## Note on where training lives
Phases 1–2 ultimately modify the **DDUSt3R** pair-training loop (cloned on the
server, gitignored). This package owns the curvature math, the head module, the
loss, the GT/datasets plumbing, and the standalone learnability gate; the
integration patches DDUSt3R to import from here.
