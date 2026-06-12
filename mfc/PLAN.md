# PLAN — Multi-Frame Consistency as a Training Objective (paper/new.tex)

Branch: `multiframe`. Goal: fine-tune D²USt3R on N-frame tuples with a
sheaf-Hodge consistency loss — static structure becomes multi-frame-consistent
by construction, dynamic structure stays free — and beat D²USt3R(+global
alignment) on its own multi-frame benchmarks. Target: **CVPR 2027** (Nov 2026
deadline); reassess ICLR 2027 (late Sept) only if the full run lands early.

Design principle this time: **one full-scale training pipeline, few gates.**
Exactly two go/no-go gates (G1, G2 below), then commit to the run. The
"cheap/efficient test-time" pattern is what killed the detector line; the
contribution here lives in training, at D²USt3R's own scale.

---

## 0. Decisions locked in (these close most of new.tex's TODOs)

| # | Decision | Rationale / closes |
|---|---|---|
| D1 | **L_cons penalizes the harmonic part `h` on static pixels**, not the full residual `r`. Full-residual variant is demoted to ablation arm E2(b'). | Pose-robust by construction: camera-explainable error is forgiven, motion-like error on static pixels is punished. Makes the Hodge split do work *in the loss*, not just in diagnostics. Closes the Eq.-(cons) wrinkle. |
| D2 | **Pose-model stalks**: per-view ξ ∈ 𝔰𝔢(3)⊕ℝ (7 dof/view). Harmonic projection = residual minus best per-view Sim(3) correction; normal equations are 7N×7N (N≤5 ⇒ ≤35×35) per tuple — trivially differentiable (`torch.linalg.solve` autograd; no implicit-function machinery needed). | Closes the coordinate-vs-pose-model TODO and the "is the solve too expensive?" risk in one stroke. |
| D3 | Per-edge relative transforms from **confidence-weighted Procrustes** on the pairwise pointmaps (port of `pose_sheaf.py` / the old TUM pipeline code). | Already validated in the sheaf-era work (2–3° median on real data). |
| D4 | **Backbone**: D²USt3R checkpoint, frozen encoder, fine-tune decoder+heads (their §5.4 protocol). Tuples N=5, view graph = temporal chain + 2 loop edges, strides 1–9 (their sampling). | Matches the paper we extend; isolates the objective as the only change. |
| D5 | **Dynamic pixels are ungated** (left free), not positively supervised by chained flow. | Closes the "ungated vs chained-L_dyn" TODO. Chaining is a stronger variant — note as future work, don't build it now. |
| D6 | **Training data, full scale**: PointOdyssey (primary dynamic) + TartanAir (static anchor) + Spring (small, GT flow+masks). BlinkVision if download cooperates — otherwise proceed without; D²USt3R's MonST3R* comparison shows the recipe tolerates dataset deltas if we retrain the baseline equivalently. **Sintel is never trained on** (eval only). | Full-scale robustness bar; server already holds PointOdyssey/TartanAir/Sintel. |
| D7 | **Eval = D²USt3R's protocol verbatim** (`DDUSt3R/dust3r/depth_eval.py`): multi-frame depth (TUM-Dynamics, Bonn, Sintel, KITTI; All + Dynamic), single-frame depth, camera pose (Sintel, TUM, ScanNet). | Tables are directly comparable to the setting paper. |
| D8 | Anchor: static-tuple harmonic→0 (TartanAir tuples) + median-log-depth scale gauge, per new.tex Eq. (anchor). λ_c warm-up over the first epochs (backbone's structured error is harmonic-like at init; don't fight it from step 0). | Non-degeneracy without extra moving parts. |

---

## 1. Phase 1 — `mfc/hodge.py`: differentiable harmonic projection (gate G1)

The **only** isolate-first step. Port `sheaf.py::build_coboundary/solve_harmonic`
to torch with D2's pose-model stalks:

- `harmonic_residual(world_pts_i, world_pts_j, conf, edges) -> h, xi_star`
  — batched over tuples, differentiable w.r.t. the pointmaps.
- Reuse `synth.py` (already on this branch) as the test harness:
  1. **Correctness**: localization AUROC on the synthetic moving-object clips
     matches the NumPy implementation (≈1.0 with drift, per template.tex §3).
  2. **Gradients**: finite-difference check on a tiny clip; confirm pushing
     `h→0` on static points by gradient descent on the *pointmaps* recovers
     consistent geometry without scale collapse (the gauge term active).
  3. **Speed**: fwd+bwd at training shape (N=5, ~210×510 px, conf-top-k
     correspondences per edge) — overhead must be **<20% of a backbone
     forward**. With D2's 35×35 solve this should be comfortable.

**G1 gate (timebox ~1 week):** all three pass → proceed. Speed fails → subsample
correspondences per edge (the solve only needs enough rows to determine 35
dof; the *loss* can still be dense once ξ\* is fixed).

## 2. Phase 2 — `mfc/tuples.py`: tuple pipeline

Extend `datasets.py` (frame enumeration for Sintel/TartanAir/PointOdyssey
already exists) from frames to **tuples**:

- Sample N=5 windows with stride ∈ {1..9}; emit images, GT depth/K/pose.
- **M_dyn** per D²USt3R Eq. 5 (‖f_cam − f‖ > τ): GT flow where it exists
  (TartanAir, Spring, Sintel-debug), off-the-shelf RAFT/SEA-RAFT for
  PointOdyssey — same recipe as the paper. Occlusion mask via fwd–bwd check.
- Cache masks under `cache/mfc_tuples/`; one visual check sheet per dataset
  (masks overlaid on RGB) before training — not an experiment, just eyeballs.
- Add Spring loader (~30 lines, layout similar to Sintel). BlinkVision loader
  only if D6 says go.

## 3. Phase 3 — `mfc/train_mfc.py`: training (gate G2, then the full run)

Fork DDUSt3R's training entry (`launch.py` path) minimally:

- Batch = tuple. Run the pair decoder on the E edges of the tuple graph
  (chain+loops: 6 pairs for N=5), keep D²USt3R's per-pair L_static+L_dyn on
  each edge **unchanged**, add `λ_c·L_cons` (D1: harmonic on static pixels)
  + `λ_a·L_anchor` (D8).
- Optimizer/schedule: D²USt3R's (AdamW, 5e-5, ~50 epochs, 20k tuples/epoch
  equivalent, grad accumulation to fit the server GPU).
- λ_c, λ_a: set once by matching gradient magnitudes to the per-pair loss on
  a few batches (no sweep); λ_c linear warm-up over epochs 1–5.

**G2 gate — one short run** (~3 epochs, capped data): losses decrease, no
scale collapse (depth-median statistics flat), harmonic energy on a held-out
static clip trending down, val depth not degraded vs the checkpoint. This is
a smoke test, not an experiment. Fix what it surfaces, then launch:

**The full run.** One training job at full scale. While it runs: build the
eval harness (Phase 4 plumbing) and write related work (Phase 5).

## 4. Phase 4 — Evaluation and the four experiments

Training jobs total: **1 full + 2 short-schedule ablations** (ablations at
~15 epochs / half data — standard practice, state it in the paper).

| Exp | Question | Runs needed | Falsifier (kept from new.tex) |
|---|---|---|---|
| **E1** headline | Beats D²USt3R(+GA) on multi-frame depth/pose? Also report **GA-iteration count to a fixed tolerance** with our pairs vs theirs — "predictions that glue with fewer optimizer steps" is a concrete, quotable win. | full run only | no static-region gain ⇒ the optimizer was already enough |
| **E2** the split | (a) baseline, (b') naive full-residual L_cons, (c) harmonic L_cons. | 1 short run (b') | (b')≈(c) on dynamic regions ⇒ story collapses to "multi-frame helps" |
| **E3** non-degeneracy | Anchor off. Report harmonic energy on held-out static scenes for both. | 1 short run | removing anchor doesn't hurt ⇒ gameability concern was misplaced |
| **E4** scaling | Depth vs N ∈ {2,3,5,8} and stride 1–9 at **eval time** (one trained model, re-tupled). | none | gap to baseline flat/shrinking in N |

**E5** (free figure, no run): harmonic vs raw residual under injected drift on
real eval clips — the mechanism check; reuse the drift-injection code pattern
from the sheaf era.

## 5. Phase 5 — Writing and positioning (starts week 1, parallel)

- **Lit pass now, not at writing time**: VGGT, CUT3R, MUSt3R, Fast3R, Spann3R,
  π³, and dynamic-scene multi-frame work (St4RTrack, Dynamic Point Maps,
  MegaSAM). Positioning to establish in new.tex's intro: those models get
  multi-view consistency from *architecture*; none supplies a
  static/dynamic-aware consistency *training signal*; ours is in principle
  backbone-agnostic. If E1 lands early, a small "attach the loss to a
  multi-view backbone" experiment is the strongest possible future-proofing —
  optional, decide in September.
- Update new.tex as decisions/results land (D1–D8 already close its TODOs);
  keep the falsifier framing — it reads as rigor.

---

## Timeline (one person, server compute)

| When | What |
|---|---|
| → end June | G1 (hodge.py) ✓, lit pass ✓, tuples.py started |
| July | tuples.py + check sheets ✓, train_mfc.py, **G2 short run** |
| August | **full run** + E1 eval harness; first headline numbers |
| September | E2/E3 short runs, E4/E5, decide on the multi-view-backbone extra |
| October | full tables, paper writing, internal review pass |
| November | CVPR 2027 deadline |

## Risks (watch, don't pre-solve)

- **Bad Procrustes edges early in training** destabilize ξ\* → confidence-weight
  edge rows (D3 already does) and clamp; λ_c warm-up (D8) covers init.
- **M_dyn quality on PointOdyssey** (no GT flow) → same off-the-shelf recipe as
  the paper; the check sheets are the control point.
- **Harmonic-on-static may under-penalize per-pixel error** (it forgives
  anything a per-view Sim(3) explains — by design). The per-pair L_static still
  supervises absolute geometry, so this should be complementary, but G2's
  val-depth check is the early warning.
- **Server contention / one-GPU reality**: the full run may need grad
  accumulation and ~2× wall-clock vs the paper's 4×RTX6000. Budgeted in August.
- **Train/test overlap**: TartanAir is train-only (backbone has seen it — fine,
  so did D²USt3R's); Sintel/Bonn/TUM/KITTI/ScanNet eval-only. Verify once in
  Phase 2.
