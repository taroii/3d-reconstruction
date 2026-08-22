# Premise Check: Do feed-forward pointmap models actually produce flying pixels?

**Status:** pre-registered feasibility protocol. Fill in results; do not edit the decision rule after seeing them.
**Owner:** _______  **Started:** _______  **Target completion:** 1 week (hard cap: 2)
**Compute:** inference only. No training. No finetuning. If you find yourself training something, you have left the scope of this document.

---

## 1. The one question

> Do VGGT and π³ place significantly more points in the empty space between surfaces at occlusion boundaries than the ground truth itself does?

Everything downstream — the diagnosis, the loss analysis, the refiner — is worthless if the answer is no. This document answers only that question.

**Why the GT comparison is the whole point.** Prior work reports that NYU-v2 ground truth *itself* contains flying-point artifacts, and that datasets like Bonn aggressively mask GT along object boundaries. So "the model has flying pixels" is not a finding; the model may be faithfully reproducing contaminated supervision, or the benchmark may have deleted the very pixels in question. The finding is **model FP rate minus GT FP rate**, measured on data where GT boundaries are trustworthy.

---

## 2. Decision rule (fixed in advance — do not modify after seeing results)

Let `FP_model` and `FP_GT` be flying-pixel rates (Sec. 5), aggregated per scene, compared pairwise across scenes.

| Outcome | Condition | Action |
|---|---|---|
| **GO** | `FP_model − FP_GT ≥ 5` percentage points **and** `FP_model ≥ 2 × FP_GT`, with 95% bootstrap CI on the paired difference excluding 0, **for at least one model on the synthetic set**, and the sign is stable across the full (η, τ, β) sweep in Sec. 7 | Premise holds. Proceed to the decomposition study. |
| **WEAK** | CI excludes 0 but effect is below the GO thresholds, or the effect flips sign anywhere in the sweep | Premise is real but small or threshold-dependent. Do **not** build a diagnosis paper on it. Reconsider scope; a fix-only paper with modest claims may still be viable. |
| **NO-GO** | CI includes 0, **or** `FP_GT` is itself ≥ 15% (GT too contaminated to measure against), **or** the measurability gate in Sec. 4 fails on every dataset | Premise does not survive. Stop. Do not proceed to diagnosis or loss work. Report the null internally and pick a different problem. |

**Pre-commitment:** write the date and the above thresholds into the results file *before* running anything. If a result lands near a boundary, it is WEAK — do not adjust thresholds to reach GO.

---

## 3. Models to evaluate

Evaluate all three prediction streams; they are not equivalent and may disagree.

1. **VGGT — pointmap branch** (direct pointmap output).
2. **VGGT — depth branch** (depth head; unproject with predicted camera).
3. **π³ — local pointmap** (per-view, in that view's own camera frame).

Use released checkpoints, feed-forward only. **No bundle adjustment, no global alignment, no post-processing, no filtering.** Any cleanup step invalidates the measurement — we are measuring the raw model output.

> **Frame note (important).** π³ predicts pointmaps in each view's *own* camera frame, VGGT in a reference frame. To avoid all cross-view alignment issues, **the entire measurement is done per-view, along the camera ray, in that view's own frame.** Convert every prediction to a per-view depth map and never compare across views in this study. This sidesteps pose error entirely, which is a different failure mode we are not testing.

---

## 4. Data requirements and the measurability gate

### Requirements
A dataset is usable only if it provides, per view: RGB, **dense GT depth with sharp, unmasked boundaries**, and a validity mask.

### Tiers
- **Tier A (required — the primary result): synthetic, exact boundaries.** Hypersim is the default choice; any renderer-produced set with exact depth works. This is the only tier where GT boundaries are trustworthy enough to measure against.
- **Tier B (secondary — generalization only): real captured data.** ScanNet++ / ETH3D / whatever is on the server. Expect degraded GT at boundaries; report but do not base the decision on it.

**Use what is already on the server.** Record exactly which datasets, splits, and versions were used in the results file. If no Tier A data is present, acquiring a small Hypersim subset (~50 scenes is plenty) is the first task.

### Measurability gate (run this FIRST — it is cheap and can end the study)
For each candidate dataset, compute:

- `frac_boundary_valid` = fraction of detected boundary pixels that have valid GT depth on **both** sides.
- `frac_boundary_masked` = fraction of boundary pixels removed by the dataset's own validity mask.

**Gate:** if `frac_boundary_valid < 0.5` for a dataset, that dataset **cannot** answer the question — the boundaries have been deleted. Drop it and note it. If every available dataset fails this gate, the outcome is NO-GO on measurability grounds, and that is itself a useful finding (it means the community cannot currently measure this).

---

## 5. Definitions to implement

Match the notation in the paper's Preliminaries exactly.

**Boundary set.** For view `i` with GT depth `D̄`, pixel `p` is a boundary pixel if the relative depth jump to any 4-neighbour exceeds `η`:

```
rel_jump(p) = max over q in N(p) of  |D̄(p) − D̄(q)| / min(D̄(p), D̄(q))
B = { p : rel_jump(p) > η }
```

**Boundary band.** `B_τ` = dilation of `B` by `τ` pixels. **Interior** `I` = valid pixels minus `B_τ`.

**Depth support.** For `p ∈ B`, collect GT depths in a `(2w+1)²` window around `p`, discard invalid, and cluster into distinct surfaces by sorting and splitting wherever consecutive sorted depths differ by more than `η` relatively. Result: `S(p) = {d⁽¹⁾ < … < d⁽ᴷ⁾}`, using each cluster's median as its depth. Record `K_p`.

**Scope for this study:** analyse only pixels with `K_p = 2`. Report the histogram of `K_p` so coverage is explicit; `K_p ≥ 3` is a separate stratum, not part of the go/no-go.

**Void and flying pixel.** With `Δ = d⁽²⁾ − d⁽¹⁾` and margin `β ∈ (0, 0.5)`:

```
void(p)  = [ d⁽¹⁾ + β·Δ , d⁽²⁾ − β·Δ ]
is_FP(p) = depth_pred(p) ∈ void(p)
```

Predictions in front of `d⁽¹⁾` or behind `d⁽²⁾` are **not** flying pixels — count them separately as `outside_rate`.

**Flying-pixel rate.** `FP = |{p ∈ B_eval : is_FP(p)}| / |B_eval|`, where `B_eval` is the set of `K_p = 2` boundary pixels passing all validity checks.

### Core pseudocode
```
for each scene:
  for each view i:
    D_gt   = ground-truth depth
    D_pred = per-view predicted depth (from pointmap z, or depth head)

    # --- scale alignment: CRITICAL, see Sec. 6 ---
    s = align_scale(D_pred, D_gt, mask = I)      # interior ONLY
    D_pred = s * D_pred

    B = boundary_set(D_gt, eta)
    for p in B:
      S = depth_support(D_gt, p, window=w, eta=eta)
      if len(S) != 2: continue
      if not valid_both_sides(p): continue
      d1, d2 = S; delta = d2 - d1
      if delta < delta_min: continue            # reject trivial jumps
      record( is_FP = (d1 + beta*delta) <= D_pred(p) <= (d2 - beta*delta) )
  FP_scene = mean(records)
```

### The GT control
Run the **identical** pipeline with `D_pred := D̄_raw`, where `D̄_raw` is the dataset's own raw/unprocessed depth (pre-cleanup if available), while `S(p)` is computed from the cleaned GT. For synthetic data where raw and clean are identical, `FP_GT ≈ 0` by construction — that is the expected and desired outcome, and it is what makes Tier A the primary tier. For real data, `FP_GT > 0` measures sensor/interpolation contamination.

> If `FP_GT ≈ 0` on synthetic data, that is not a bug — it is the control working. It means any nonzero `FP_model` is attributable to the model.

---

## 6. Confounds that must be controlled

These are the ways this measurement goes wrong. Handle each explicitly.

1. **Scale ambiguity.** Predictions are scale-free. Align using median-ratio or least-squares scaling — **computed on interior pixels only (`I`), never on the boundary band.** Aligning on boundary pixels would let the thing being measured contaminate the alignment. Report results under both alignment variants as a robustness check.
2. **Resolution mismatch.** Models run at fixed input sizes (e.g. VGGT resizes to 518×518 with padding). Evaluate at **GT resolution** by upsampling predictions with nearest-neighbour — *never* bilinear, which manufactures flying pixels at exactly the boundaries under study. Record which resolution each model ran at.
3. **Padding.** Exclude padded image borders from all statistics.
4. **Sky / infinite depth.** Exclude pixels whose GT depth is invalid, zero, or beyond the dataset's max range; these produce spurious enormous `Δ`.
5. **Trivial jumps.** Enforce a minimum absolute gap `delta_min` so that noise-level depth steps are not counted as boundaries.
6. **Occluded far surface.** The far surface must be genuinely visible in the same view's local window; if the window contains no valid far cluster, skip the pixel.
7. **Statistical unit.** Pixels within a scene are heavily correlated. **The unit of analysis is the scene**, not the pixel. Compute `FP_scene`, then aggregate across scenes.

---

## 7. Statistical analysis

- **Pairing.** For each scene, compute `FP_model` and `FP_GT` on the identical pixel set. Analyse the paired difference.
- **Test.** Wilcoxon signed-rank across scenes (do not assume normality). Report the paired-bootstrap 95% CI on the mean difference (≥10,000 resamples over scenes).
- **Effect size.** Report the absolute difference in percentage points, the ratio, and a rank-based effect size. **Effect size governs the decision, not the p-value** — with enough scenes, trivial differences become significant.
- **Multiple models.** Three prediction streams × datasets. Report all; do not cherry-pick. Apply Holm correction across streams.
- **Threshold sensitivity (mandatory).** Sweep `η ∈ {0.02, 0.05, 0.10}`, `τ ∈ {1, 2, 3}`, `β ∈ {0.1, 0.2, 0.3}`. Report FP rate across the full grid. **If the sign of the effect flips anywhere in this grid, the outcome is WEAK regardless of the headline number.** This is the main defence against fooling ourselves with a lucky threshold.

---

## 8. Deliverables

Write to `results/premise_check/`:

1. `decision.md` — pre-registration date, thresholds copied from Sec. 2 *before* running, final outcome (GO / WEAK / NO-GO), one paragraph of justification.
2. `results.json` — per scene, per model, per threshold setting: `FP`, `outside_rate`, `|B_eval|`, `K_p` histogram, `frac_boundary_valid`.
3. `table_main.csv` — rows = model stream, columns = `FP_model`, `FP_GT`, difference, ratio, CI, p-value.
4. `sensitivity.csv` — the full (η, τ, β) grid.
5. `figs/` — for 5 scenes: RGB, GT depth, predicted depth, boundary mask, FP mask overlay, and an unprojected point-cloud render at a silhouette. **Inspect these by eye before trusting any number.** If the FP masks do not visually correspond to smeared silhouettes, the implementation is wrong.
6. `gate.csv` — measurability gate results per dataset.

---

## 9. Interpretation guide

| Result | Meaning |
|---|---|
| `FP_model` high, `FP_GT ≈ 0` (synthetic) | Premise holds. The model is inventing points in empty space. **GO.** |
| `FP_model ≈ FP_GT`, both high (real data) | The model is reproducing contaminated supervision. Not a model failure — a data failure. Re-check on Tier A before concluding anything. |
| `FP_model` low everywhere | Premise fails. Boundary error may exist but is not of the flying-pixel type. **NO-GO** for this framing; the artifact may instead be blur or incompleteness, which is a different (and possibly still interesting) problem. |
| Streams disagree (e.g. VGGT-p high, VGGT-d low) | Genuinely informative: localizes the artifact to a specific output head, which sharpens the later diagnosis. |
| Gate fails everywhere | Cannot be measured with available data. **NO-GO** on measurability. |

---

## 10. Explicit non-goals

Do **not**, in this study: change any loss; finetune or train anything; implement a refiner; test *why* flying pixels occur; compare against other methods; or optimize any threshold to improve the outcome.

This document answers one question. If the answer is NO-GO, the correct next action is to stop and reconsider the project — not to search for a framing that rescues it.

---

## 11. Time budget

| Day | Task |
|---|---|
| 1 | Inventory server data; run the measurability gate (Sec. 4). Kill unusable datasets. |
| 2 | Implement boundary/support/FP metric (Sec. 5); validate on hand-built synthetic cases with known answers. |
| 3 | Run VGGT (both branches) and π³ inference; store per-view depths. |
| 4 | Compute FP rates + GT control; produce `figs/` and **eyeball them**. |
| 5 | Sensitivity sweep + statistics; fill in `decision.md`. |
| 6–7 | Buffer. |

If day 5 arrives without a number, stop and report the blocker rather than extending.