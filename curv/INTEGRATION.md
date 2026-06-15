# Integrating curvature into the DDUSt3R training loop

The curvature math/head/loss/data live in `curv/`. Phases 1–2 wire them into the
**DDUSt3R** pair-training loop, which is cloned on the server (gitignored) and
not visible offline — so this is a precise checklist, with the symbols to confirm
against the real source marked `‹CONFIRM›`. Everything `curv/`-side is done.

## Key fact that simplifies everything

GT curvature is a deterministic function of the **GT pointmap** `X̄` (a `H×W×3`
array already in every D²USt3R training batch as the supervision target). So we do
NOT need depth+K inside the loop — call the pointmap estimators directly:

```python
import curvature as CV                       # curv/ on sys.path
K_gt = CV.mean_curvature_from_points(P_gt)   # P_gt: GT pointmap (H,W,3), numpy
# (or CV.gaussian_curvature_from_points). Normalize P_gt the SAME way D²USt3R
# normalizes pointmaps (its z,z̄ factor) before this call — see Gotcha #3.
```

Recommended route: **precompute** `K_gt` (+ validity) per training frame offline
and inject it into the batch via the dataset — same caching `train_curv.py`
already does (`../cache/curv_targets/...`). This keeps the loop numpy-free and the
two options share one cache.

---

## Option 2 — curvature-weighted loss (do first; ~half a day)

No new params. Reweight the existing per-pixel static/dynamic loss:

```python
import curvature as CV
w = CV.curvature_weight(K_gt, gamma)         # 1 + gamma*|K_gt|; works on torch tensors
# multiply the PER-PIXEL loss BEFORE the valid-mask reduction:
per_pixel_loss = per_pixel_loss * w
```

Checklist:
1. `‹CONFIRM›` the loss class in `DDUSt3R/dust3r/losses.py` — D²USt3R extends
   DUSt3R's `Regr3D` / `ConfLoss` with the static/dynamic split. Find where the
   per-pixel pointmap residual is computed *before* it is averaged over the valid
   mask. That is the multiply point.
2. `‹CONFIRM›` the batch keys for the GT pointmap + valid mask (DUSt3R uses
   `gt1['pts3d']`, `gt1['valid_mask']`, view 2 analogous). Add a `curv_gt` (and
   reuse `valid_mask`) field, populated from the cache by the dataset/collate.
3. Add `gamma` to the training args; **sweep γ ∈ {0.5, 1, 2, 4}**.
4. Run **Arm A (γ=0, baseline) and Arm B (γ>0)** on the same data (see
   PLAN Reviewer-proofing). Headline = A vs B.

`curv/`-side: `curvature.curvature_weight` is ready. Only the precompute hook +
the one-line multiply are DDUSt3R-side.

---

## Option 1 — curvature head (first reportable result)

Add a DPT head predicting `Ĉ ∈ H×W×1` alongside D²USt3R's pointmap +
dynamic-mask heads, and add `λ·L_curv` to the objective.

```
L_total = L_static + L_dyn + λ · L_curv,   L_curv = curvature_conf_loss(Ĉ, ω, K_gt, valid)
```

Note: `curv/curvature_head.py::CurvatureHead` is the **encoder-only** standalone
gate (mirrors N_φ). The *in-model* head should instead hang off the **decoder**
tokens, exactly like D²USt3R's dynamic-mask head — reuse that head's construction
and swap its output to 2 channels (curvature + conf) and its BCE for
`curvature_conf_loss`. The loss function ports verbatim.

Checklist:
1. `‹CONFIRM›` the model class (`AsymmetricCroCo3DStereo` in
   `dust3r/model.py`) and how the **dynamic-mask head** is built/registered
   (`downstream_head*`, `head_type`, `PixelwiseTaskWithDPT` call). Clone that
   path for a `curv_head` with `num_channels=2`.
2. `‹CONFIRM›` the forward returns dict and the criterion call site; add the
   curvature output and `λ·L_curv` (import `curvature_conf_loss` from
   `curv/curvature_head.py`).
3. Freeze encoder, finetune decoder + all DPT heads from the D²USt3R checkpoint
   (unchanged protocol; the new head's weights init fresh). **Sweep λ.**
4. Again run **Arm A vs Arm B** on matched data.

---

## Path note

DDUSt3R training scripts run from the `DDUSt3R/` clone. Put `curv/` on the path
(`sys.path.insert(0, '../curv')` or `PYTHONPATH`) so `import curvature` /
`import curvature_head` resolve. `curv/backbone.py` already adds `../DDUSt3R` to
the path for the reverse direction.

## Decisions inherited from PLAN
- estimator: **mean** first, ablate gaussian; target: **signed-log + Huber**;
  scale: compute `K_gt` on the **normalized** pointmap.
- always report the **matched-data baseline (Arm A)** as the headline; eval
  zero-shot on Sintel / Bonn / TUM-Dynamics / KITTI.
