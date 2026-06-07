# Implementation Plan — Spectral Band-Routing Prior (step 0 → go/no-go)

Concrete build plan for the new idea (paper/template.tex), grounded in the code
actually present in this repo. Scope: get from zero to the **per-band
convergence go/no-go gate** (notes/1.md §1) — the decisive experiment. Stop and
reframe if the low band doesn't separate.

The guiding fork (notes/1.md §0): `L_geo` lives at **test-time inside the
alignment optimizer** (option a). No backbone retraining for the optimizer-side
work; the *only* training is the `N_φ` normal head.

---

## 0. Foundation already in place

**Reusable plumbing** (`spectral/`, moved from the dead sheaf line):
- `backbone.py` — `load_backbone({dust3r,monst3r,d2ust3r})`, `run_clip()`,
  `extract()`. `run_clip` already does inference + global alignment and returns
  the **live optimizer object** (`out["scene"]`). `_backproject(depth, focal)`
  is a pose-free camera-frame pointmap (the shape `L_geo` constrains).
- `sintel.py` — `read_dpt/read_flo/read_cam`, `cam_induced_flow`, `motion_gt`
  (the `(1−M_dyn)` GT gate). GT depth + intrinsics here = free `N_φ` labels.
- `eval.py` — AUROC / IoU / F1 (region diagnostics). For depth metrics use the
  repo's `DDUSt3R/dust3r/depth_eval.py::depth_evaluation` (AbsRel, δ₁/₂/₃; has
  both affine-invariant **and** scale-aware paths → covers Claim 4).
- `datatypes.py`, `correspondence.py`, `match_mast3r.py`, `tum.py`.

**Models** (`DDUSt3R/checkpoints/`, `mast3r/checkpoints/`): D²USt3R
(`ddust3r.pth`), DUSt3R, MonST3R, MASt3R-metric — all present.

**The aligner we hook into** = `DDUSt3R/dust3r/cloud_opt/optimizer.py::PointCloudOptimizer`
(MonST3R fork; has flow + temporal + depth-prior terms = the paper's `L_align`).
Key anchors:
- `forward_batchify(epoch)` (optimizer.py:760) returns `(loss, flow_loss)` — the
  single place to add `+ w_geo·L_geo + w_dc·L_dc`.
- camera-frame (pose-free) pointmap = `_fast_depthmap_to_pts3d(depth, self._grid,
  focals, pp)` (optimizer.py:1030) — the line *before* `geotrf(im_poses, …)` in
  `depth_to_pts3d` (optimizer.py:686). Normals from this don't touch pose.
- `self.dynamic_masks` (list of (H,W) bool) — reuse as the `(1−M_dyn)` gate,
  exactly as `forward_batchify` gates the flow term (optimizer.py:793).
- world-frame pts `self.get_pts3d(raw=True)` — input to the graph Laplacian `L`.
- loop: `base_opt.py::global_alignment_loop → global_alignment_iter` (468/548),
  calls `net(epoch=cur_iter)`. Per-band logging hooks here.
- `N_φ` head: build with `dust3r/heads/dpt_head.py::create_dpt_head` on the
  frozen encoder (`model._encode_image` → `model._downstream_head`).

**⚠️ Dependency to verify first:** all of the above needs the heavy GPU `sheaf`
conda env (torch+cuda, RAFT, SAM2). Memory flags it as "pending env." **Verify
the env loads a backbone + runs `run_clip` on one clip before anything else** —
this is the gate before step 0.

---

## Step 0 — Sintel GT normals (`spectral/normals.py`, pure NumPy/torch)

Supervision for `N_φ`, derived from data we already have. No network.

1. `normals_from_depth(depth, K) -> (H,W,3)`: back-project pixels to camera-frame
   points `P = depth · K⁻¹ [u,v,1]ᵀ` (reuse `sintel.read_cam` for `K`); normal =
   normalized cross product of `∂P/∂x × ∂P/∂y` (central differences). Orient
   toward camera (`n·(−P) > 0`).
2. `normal_validity(depth, K)`: invalidate occlusion/invalid pixels (Sintel
   `occlusions/`, `invalid/`) and large depth-discontinuity pixels (finite-diff
   guard) — these corrupt the cross product.
3. Sanity: render a normal map for `alley_1/frame_0001`, eyeball it (walls vs
   ground distinct). Cache GT normals per scene under `cache/<scene>/normals/`.

**Caveat to surface, not hide:** normals are computed on the **full-res** Sintel
grid; the optimizer lives on the resized/cropped (~208×512) grid. Reuse
`pipeline.align_to_grid`'s `crop_img` path (it was in the old pipeline; replicate
that resize+crop) so GT normals land on the pointmap grid. Bilinear-resample the
vectors then **renormalize**.

---

## Step 1a — Train PoC `N_φ` (`spectral/normal_head.py`, `spectral/train_nphi.py`)

The one model we train. Frozen encoder, small DPT head, Sintel-only PoC.

- **Arch:** `create_dpt_head`-style head, `num_channels = 3 (normal) + 1 (conf)`,
  on frozen DUSt3R encoder features (`model._encode_image`). Freeze everything
  but the head (mirrors D²USt3R's "decoder+head only" finding, §5.4).
- **Supervision:** angular loss `1 − ⟨N̂, N⟩` on valid pixels + a confidence term
  for `ω̂` (e.g. `ω̂` regresses the angular agreement / uncertainty; let
  textureless pixels keep a normal but honest low `ω̂`).
- **Data:** Sintel 23 scenes, held-out split by scene (not frame) to measure
  generalization. **Known limit:** paper trained on BlinkVision/PointOdyssey/
  TartanAir/Spring — none here. This `N_φ` is a PoC; gate generalization on the
  held-out angular error before trusting it downstream (notes/1.md §3 floor).
- **Output gate:** held-out mean angular error. If it's bad, nothing downstream
  is interpretable — fix `N_φ` before touching the optimizer.

---

## Step 1b — `L_geo` in the optimizer (`spectral/band_optimizer.py`)

Subclass, don't fork: `class BandPCOptimizer(PointCloudOptimizer)`.

1. Attach predictions: `set_normal_prior(Nhat_list, omega_list)` → register as
   non-grad buffers on the optimizer grid (same shape as `_grid`).
2. Reconstruction normals **pose-free**: in an override of `forward_batchify`,
   after `proj_pts3d` get camera-frame points via
   `_fast_depthmap_to_pts3d(self.get_depthmaps(raw=True), self._grid,
   self.get_focals(), self.get_principal_points())`; reshape to (N,H,W,3);
   `N_n = 𝔫(·)` by the same central-diff cross product as step 0 (share code).
3. First-order term (template Eq. geo, first summand):
   `L_geo = Σ ω̂·(1−M_dyn)·(1 − ⟨N_n, N̂_n⟩)`, gated by `self.dynamic_masks`
   (reuse the flow-term gating at optimizer.py:793).
4. Add `+ self.w_geo * L_geo` to `loss` in `forward_batchify`'s return.
5. Curvature term (`η`, optional, second summand) deferred — needs the sparse
   `Lχ` from step 1c; ramp in **after** flow warm-up (`flow_loss_start_epoch`),
   per the template's stability schedule.

`L_dc` (scale anchor): `w_dc·(log median_{n,p} D_n − log s₀)²` over
`self.get_depthmaps()`; `s₀` from a GT-calibrated frame (Sintel metric depth).
One line in the same return. (Off for the go/no-go; on for Claim 4.)

Driver `spectral/run_band.py`: mirror `backbone.run_clip` but instantiate
`BandPCOptimizer` instead of `global_aligner(...)`, call `set_normal_prior`, run
a copy of `global_alignment_loop` that records per-iteration residuals (step 1c).
Run **with and without** `w_geo` (toggle = `w_geo=0`) on the same scene/seed.

---

## Step 1c — Per-band residual diagnostic (`spectral/spectral_diag.py`) — the money plot

This is what separates "prior helps" from "prior helps via the spectral
mechanism." Built once, used at chosen iterations (not every step).

1. **Graph `L`** on current `get_pts3d(raw=True)`: kNN (k≈8–16, e.g.
   `sklearn`/`faiss` or torch cdist on the ~10⁵ pts), `W_ij =
   exp(−‖xᵢ−xⱼ‖²/ε)`, `ε` = median squared kNN distance. Sparse `L = D − W`.
   **Never** eigendecompose for the inner loop — only `Lχ` matvecs.
2. **`h_lo(L)`** via few-term Chebyshev (Hann^½ profile); `h_hi = I − h_lo`.
   Needs `λ_max` estimate (a few power iterations).
3. **Residual `r_t`** = per-pixel align term inside `forward` (the `li+lj`
   contributions, optimizer.py:771). Expose it as a per-pixel field (small refactor
   of the dist call, or recompute from `proj_pts3d` vs `aligned_pred`).
4. At logged iterations record `‖h_lo(L)·r_t‖²` and `‖h_hi(L)·r_t‖²`, **with vs
   without `L_geo`**, on ~5 scenes. Plot both bands vs `t` → `results/band_convergence.png`.

**Go:** low-band residual decays faster with the prior, high-band ~unchanged.
**No-go / reframe:** prior accelerates both bands equally (it helps, but not via
the claimed mechanism).

Cross-check (notes/1.md §4.3b), cheap: per-frame `e_n = D_pred − D_gt` (affine-
aligned via `depth_eval`), fit a low-degree 2D polynomial (the literal "bow");
`L_geo` should shrink the low-order fit, leave the residual ~flat.

---

## Execution order & gates

1. **Env smoke test** — load D²USt3R, `run_clip` on one 5-frame clip. ⛔ gate.
2. **Step 0** — GT normals + grid alignment + eyeball sanity.
3. **Step 1a** — train `N_φ`, held-out angular error. ⛔ gate (bad normals ⇒ stop).
4. **Step 1b+1c** — `L_geo` toggle + per-band curves on ~5 scenes. ⛔ **GO/NO-GO**.
5. Only if GO: textureless-region table (Claim 1), K1/K2 controls, `L_dc` metric
   table (Claim 4), preconditioner (Claim 5), curvature rung (Claim 6),
   long-video drift. (notes/1.md §5, §9.)

## Risks / honest caveats
- **Env not yet verified** — top risk; everything is gated on it.
- **`N_φ` train data is Sintel-only** — PoC-grade; generalization unproven.
- **Basis/target coupling** — `L` and `N_n` move with the iterate. Mitigations
  (template §4.6): `N̂` is fixed (stationary attractor); recompute only cheap
  local `N_n`, `Lχ`; refresh `h_lo(L)` every few iters; ramp `η`/precond after
  warm-up. Watch for oscillation on high-motion scenes; log iteration counts.
- **Grid mismatch** (full-res GT vs optimizer grid) — handle once in step 0,
  reuse everywhere.
