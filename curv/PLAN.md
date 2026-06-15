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
`python train_curv.py --overfit` then a real run on the full training mix
`--datasets tartanair,pointodyssey,spring`. Proves the frozen encoder's features
carry recoverable curvature before we pay for the full-loop integration.
**Gate:** overfit drives MAE down; val MAE on held-out scenes is well below a
predict-zero baseline; qualitative maps fire on edges.

Training mix = **TartanAir + PointOdyssey + Spring** (3 of D²USt3R's 5 datasets;
see Reviewer-proofing). Sintel stays eval-only.

### Phase 2 — Option 1: curvature head in D²USt3R (first reportable result)
Add a DPT head predicting `Ĉ ∈ W×H×1` to D²USt3R alongside its pointmap +
dynamic-mask heads (mirrors the dynamic-mask head, BCE → Huber). Total:

    L_total = L_static + L_dyn + λ · L_curv

Port `CurvatureHead`/`curvature_conf_loss` into the DDUSt3R model + train loop;
freeze encoder, finetune decoder + DPT heads from the D²USt3R checkpoint. Sweep λ.

### Phase 3 — eval / ablation
Report deltas (does curvature help?), not a new SOTA data regime. Qualitative:
the curvature head should visibly fire on edges/corners. See **Reviewer-proofing**
for the exact comparison rule and benchmark list.

## Reviewer-proofing (LOCKED — read before running anything)

The contribution is a **delta**, not a dataset. Two rules keep it bulletproof:

1. **Matched-data baseline is the primary comparison.** Always run, on the SAME
   training data, both arms:
   - **Arm A** — fine-tune the D²USt3R checkpoint, *no curvature* (our baseline).
   - **Arm B** — same data, *+curvature* (Option 2 and/or Option 1).
   Headline numbers are **A vs B**. Because the data is identical, no reviewer can
   attack "your training set isn't comprehensive" — the data cancels in the delta.
   The comparison against the *released* D²USt3R checkpoint (trained on all 5 of
   its datasets) is **confounded by data**; report it only as a secondary,
   clearly-labeled line, never the headline.

2. **Train narrow, eval broad.** D²USt3R itself trains on 5 synthetic datasets
   (Table 1: BlinkVision Outdoor, BlinkVision Indoor, Spring, PointOdyssey,
   TartanAir). We train on **TartanAir + PointOdyssey + Spring** — 3 of those 5,
   so we are *in-distribution*, not ad-hoc. The comprehensiveness burden is on
   EVAL, which is cheap to satisfy: evaluate **zero-shot** (no training on them)
   on D²USt3R's own benchmarks — **Sintel, Bonn, TUM-Dynamics, KITTI** (+ ScanNet
   for the static table). Report **all-pixels and dynamic-region**, plus
   boundary-restricted metrics (curvature's expected win is at edges/dynamics).

3. **Forgetting guard.** Fine-tuning the released checkpoint on a 3-dataset subset
   can mildly forget the BlinkVision distribution. Arm A controls for this in the
   delta, but keep the finetune **short / low-LR** (we only add a term) and lead
   with deltas, not absolutes.

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
