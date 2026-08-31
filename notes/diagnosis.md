# Phase 1 — Diagnosis

Follows `PREMISE_CHECK.md` (verdict: GO) and `CONTAMINATION_MATRIX.md`. Goal: make progress on **why** feed-forward pointmap models place points in the void at occlusion boundaries.

**Standing rule: no fix is attempted in this phase.** A remedy for an unknown cause is not a contribution.

---

## 0. Epistemic conventions

Every substantive claim below carries a label. Preserve these labels in notes, code comments, and any draft text. Do not promote a claim to a stronger label without the evidence that warrants it.

| Label | Meaning |
|---|---|
| **[PROVEN]** | Follows mathematically from stated premises. Check the derivation; do not take it on trust. |
| **[MEASURED]** | Observed in our own runs, under stated conditions. Holds for those conditions only. |
| **[REPORTED]** | Stated in the literature. We have not verified it. |
| **[ASSUMED]** | A modelling assumption we rely on and have not validated. |
| **[UNTESTED]** | A hypothesis with no evidence yet. One of an open set. |
| **[CONFOUNDED]** | A comparison that cannot isolate a cause, however suggestive. |

**The candidate-cause list is open.** Nothing here should be read as claiming the explanations considered are the only possible ones, or that any one of them is privileged going in.

---

## 1. What we currently have

### 1.1 Established results

**[PROVEN] (a) — the position term's pointwise minimizer lies on a surface, not between surfaces.**
DUSt3R and VGGT use the *unsquared* norm (VGGT: `‖Σᴰ ⊙ (D̂ − D)‖ − α log Σᴰ`, plus a separate gradient term). For a target distribution supported on candidate depths along a pixel's ray with weights `w_k`, the minimizer of `Σ_k w_k ‖x − d_k‖` is at one of the `d_k`.
*Condition*: requires **[ASSUMED]** that the candidate targets at a pixel are collinear (they lie along that pixel's ray). This holds by construction for a pointmap but must be stated — for non-collinear atoms the geometric median can be strictly interior, so collinearity is load-bearing, not decorative.

**[PROVEN, with a caveat] (b) — confidence weighting does not introduce averaging.**
Minimizing `C·ℓ − α log C` over `C > 0` gives `C* = α/ℓ`; substituting back yields `α log ℓ + const`, whose influence `α/ℓ` decreases in the residual.
*Caveat*: the resulting objective is unbounded below as `ℓ → 0`, so it is degenerate as written. Real implementations clamp or parameterize `C`, and practical behaviour depends on that clamp. Do not present `α log ℓ` as the operative loss without checking what each codebase actually does. **[UNTESTED]** whether the clamped version behaves as the analysis suggests.

**[PROVEN] (c) — the pointwise analysis does not apply to the full objective.**
VGGT adds `‖Σ ⊙ (∇D̂ − ∇D)‖` on depth and pointmaps; π³ adds a normal loss at `λ_normal = 1.0`. Both couple neighbouring pixels, so the objective is not separable across pixels and (a) and (b) do not constrain the full-objective minimizer.

**[PROVEN] (d) — a scope limit on all of the above.**
(a) and (b) describe a *pointwise Bayes-optimal predictor*. A network has shared parameters and cannot choose pixels independently. These results describe what the loss prefers, not what a trained network realizes.

### 1.2 What follows, and what does not

**[PROVEN]** The position term alone does not prefer void-landing predictions, under the collinearity condition.

**[REPORTED]** Recent work attributes flying pixels to models predicting intermediate average depth "to minimize regression loss."

**What we may say:** there is a tension between the reported explanation and result (a). **What we may not yet say:** that the reported explanation is wrong. It may hold for the models it was stated about; it may refer to the full objective rather than the position term; or our collinearity or Bayes-optimality conditions may fail in practice.

### 1.3 Candidate mechanisms — all [UNTESTED], none privileged

An open list, in no order of preference:
- Auxiliary coupling terms (VGGT's gradient term, π³'s normal loss) perturb the minimizer off the surfaces.
- Ground-truth targets are themselves averaged at boundary pixels (finite pixel footprint; interpolation in the GT pipeline).
- The predictor's smoothness or capacity prevents realizing a step regardless of objective.
- Confidence clamping alters the effective loss away from the analysis in (b).
- Resolution and upsampling: predictions at 518px upsampled to GT resolution may be displaced a pixel or two without the model having invented geometry.
- Optimization: the trained network may not reach the objective's minimizer.
- Something not on this list.

---

## 2. Phase 1A — Generality and calibration (inference only, ~1 week)

### A1 — Add an architecturally independent family

**[MEASURED]** π³ initializes its encoder and alternating-attention module from pretrained VGGT and keeps the encoder frozen. The three streams measured so far therefore share a visual frontend.

Add **DUSt3R** (`naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`) and **MASt3R** (`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`). Pairwise, feed-forward only, **no global alignment**. Same pipeline, same pre-registered threshold grid.

**What this establishes:** whether the artifact appears outside the VGGT-derived family. That is a real and useful generality result.

**[CONFOUNDED] — read before interpreting any difference.** VGGT states it applies a gradient-based term "differently from DUSt3R," so DUSt3R plausibly lacks that term. It is tempting to treat a DUSt3R/VGGT rate difference as evidence about the gradient term. **It is not.** The two differ in backbone, pretraining, training data, pairwise vs. multi-view operation, resolution, and output heads. Any difference is consistent with many causes. Record the comparison; do not attribute it. Controlled attribution happens in 1B.

### A2 — Fine-structure recalibration

**[MEASURED]** Infinigen's GT control is 0.025 with control pixels spread across the void (median t ≈ 0.48). **[UNTESTED]** the run log's attribution of this to genuine third surfaces in foliage and thin structure — profile inspection is suggestive, not conclusive.

Produce three variants of every rate: (1) all boundary pixels; (2) `K_p = 2` only with support estimated at a finer jump threshold; (3) silhouette-only (low local boundary density). Report the `K_p` histogram.

**Gate:** if the recalibrated model-minus-GT difference falls below the pre-registered 5-point margin on the clean sets, stop and re-plan before 1B.

---

## 3. Phase 1B — Controlled attribution on synthetic data (~1–2 weeks)

The only genuinely controlled experiment in this phase, and the only place causal claims can originate.

### 3.1 Why the decomposition is defensible

Anything determining a learned predictor is in exactly one of: the **target** (what the loss compares against), the **objective** (the loss form), or the **predictor and its optimization** (architecture, capacity, training procedure). This is a partition by construction, not a hypothesis — the third bucket is defined as a residual: everything not in the first two.

**[ASSUMED]** That the partition is *useful*, i.e. that effects localize rather than smearing across all three. **What the residual bucket cannot do** is name a mechanism; if it dominates, we know only that the cause is not the target or the objective as implemented here.

### 3.2 Construction

Synthetic depth maps with an occlusion boundary at a known location: near and far surfaces, fronto-parallel *and* slanted, gap Δ swept over several magnitudes, and `K ≥ 3` cases included so the two-surface restriction is itself a variable. Render high-resolution, then downsample two ways — nearest-neighbour (clean GT) and area-average (mixed GT) — making "dirty target" a controlled factor rather than a nuisance.

### 3.3 Factorial design

Fully crossed, ≥5 seeds per cell:

| Factor | Levels |
|---|---|
| **Objective** | position only; + gradient term; + normal loss; + both; confidence weighting on/off (with the clamp used in the real codebases, and without) |
| **Predictor** | free per-pixel parameters (no network); small CNN, limited receptive field; larger CNN |
| **Target** | clean GT (nearest); mixed GT (area-average) |

The **free per-pixel** level has no shared parameters, so its optimum is the pointwise minimizer. It measures what the loss alone prefers, with the predictor factor removed by construction.

### 3.4 Pre-registered predictions (record before running)

- Position-only + free params + clean GT → `FP ≈ 0`. **This is a falsifier for (a).** If it fails, the derivation or the collinearity condition is wrong and Section 1.1 must be rewritten.
- Adding gradient and/or normal terms at published weights → **[UNTESTED]**, direction and magnitude both unknown. Do not assume an increase. A null result is informative and must be reported as such.
- Mixed GT → **[UNTESTED]**, plausibly increases `FP` independent of objective.
- Smoother predictor → **[UNTESTED]**, plausibly increases `FP` independent of objective.

### 3.5 Analysis and its limits

Report a variance decomposition across the three factors with effect sizes and confidence intervals.

**Scope statement to carry into any write-up:** results describe *this synthetic construction, these objective implementations, and these small predictors*. **[ASSUMED]** that they transfer to billion-parameter models trained on real data at scale — an extrapolation, not a measurement. 1C is the partial and confounded check on it.

---

## 4. Phase 1C — Backbone ablations (finetuning; conditional)

Run only if 1B produces a non-trivial objective effect.

- Finetune from released checkpoints with terms removed or reweighted: `λ_normal ∈ {0, 0.5, 1.0}` for π³; gradient term on/off for VGGT.
- **Matched-budget control is mandatory**: identical data, schedule, seeds, and step count across arms, including an unmodified-loss arm (the γ = 0 control structure from the ACCV paper).
- **[CONFOUNDED]** Every arm starts from a checkpoint pretrained under the original objective, so a short finetune measures the *marginal* effect of changing a term late in training, not the effect of training with it from the start. State this; do not report a weak effect as a null.
- Report boundary and aggregate metrics side by side.

---

## 5. Deliverables

Write to `results/phase1/`:

1. `1a_generality.csv` — `FP_β` per model per dataset across the threshold grid, with a confound note attached to any cross-model comparison.
2. `1a_calibration.csv` — the three rate variants plus the `K_p` histogram.
3. `1b_factorial.csv` — full factorial with seeds.
4. `1b_decomposition.md` — variance decomposition, effect sizes, and an explicit scope statement.
5. `figs/` — one-dimensional depth profiles across the boundary per objective variant. Inspect before trusting numbers.
6. `decision.md` — which branch of `OUTCOME_TREE.md` we landed on, with labels attached to each supporting claim. Note that the tree's branches are the outcomes we anticipated, not a proof that no other outcome is possible.

---

## 6. Non-goals

Do not: build a refiner; modify a loss with intent to improve rather than to diagnose; tune thresholds to strengthen an effect; add models without re-checking the contamination matrix; or upgrade a claim's label without the evidence for it.