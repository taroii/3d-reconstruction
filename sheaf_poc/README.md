# Sheaf Reconstruction — proof of concept

A cheap falsification test for the load-bearing conjecture of the
*Sheaf Reconstruction* paper (`../paper/template.tex`):

> the first cohomology localizes the obstruction to consistency, and in dynamic
> scenes its mass concentrates on moving and occluded regions.

No existing theory gives this; it is the claim the paper would spend months
building theory around. Before that investment, this PoC tests it numerically on
a controlled synthetic scene — **a few seconds of CPU, no training, no GPU.**

## TL;DR — what we found

**SUPPORTED → GO.** On a static-background scene with one known rigid moving
object, the numerically-computed harmonic `H¹` representative lands squarely on
the moving object (**AUROC ≈ 1.000**), and it stays correct under camera/pose
drift that degrades the raw per-point residual to **AUROC ≈ 0.62** (and below
chance across seeds). That gap is the entire thesis made concrete: bundle
adjustment *minimizes* the residual and throws away what's left as noise; the
sheaf view *keeps* it as a cohomology class and reads it as scene dynamics. The
two coincide only when poses are perfect — exactly the case the paper argues is
not the interesting one.

We also mapped the honest boundaries (`diagnostics.py`):

| Finding | Result | Why it matters for the paper |
|---|---|---|
| **Sim(3) scale collapse** | the uniform log-scale gauge shrinks the world to a point and cancels *all* residual (AUROC → 0) | a real modeling subtlety: the coordinate residual is not scale-invariant, so the global scale gauge **must be anchored**. State this alongside Remark 1. |
| **Linearization breakdown** | harmonic localization is invariant to pose drift up to ≈0.6 rad/step; breaks only at extreme drift | the "large motion breaks linearization" worry attaches to **pose-correction** magnitude, not object motion (object motion enters as exact residual). The valid regime is wide. |
| **Static-dominance precondition** | localization collapses once static points stop dominating (sharp transition at static = dynamic) | this **is** the generative assumption a localization theorem needs: *static scene + rigid camera + moving objects*, with separation controlled by the static fraction. |
| **Noise floor** | harmonic cannot remove per-measurement noise (it isn't pose-explainable) | the method buys invariance to **drift**, not to **noise**; both fail once noise rivals the motion. |

**Headline figures** (`python make_figures.py`), the two beats of the story:

- `figures/story_localization.png` — *truth → raw residual (BA) → harmonic H¹
  (ours)* on one drifted scene. BA's residual is a confused wash (its brightest
  points are static); ours lights up exactly the moving object.
- `figures/story_drift.png` — AUROC vs camera drift: harmonic flat at 1.0,
  raw decaying through chance.

**Supplementary diagnostics** (`python run_experiment.py`): `sweep.png`,
`spatial.png`, `localization.png` (per-point histograms), `precondition.png`
(the static-fraction phase transition).

## What the experiment actually builds

The **linearized reconstruction sheaf** in the pose model (paper §2.3, Remark 1):

- **Vertices = views**; stalk = the tangent of that view's world placement,
  `𝔰𝔢(3)` (dim 6) or `𝔰𝔦𝔪(3)` (dim 7).
- **Edges = related view pairs**, with the edge stalk decomposed **per
  co-visible scene point** (one `ℝ³` block each) — this per-point decomposition
  is what lets `H¹` localize *spatially within the scene*, not merely per-view.
- **Coboundary** `δ`: each endpoint proposes a world position for a shared point
  through its current pose estimate; the base residual is their disagreement,
  and a pose correction perturbs it via the left-perturbation Jacobian.
- **Sheaf Laplacian** `L = δᵀδ` (which *is* the bundle-adjustment information
  matrix). The **harmonic `H¹` representative** is the part of the residual no
  pose correction can cancel — computed by projecting the residual onto
  `ker(δᵀ)` with a sparse least-squares solve (`scipy.sparse.linalg.lsqr`).

Per-point harmonic energy is compared against the raw residual energy; ground
truth tells us which points are dynamic, so we score localization with AUROC.
The decisive manipulation is **pose drift**: it confounds the raw residual
(static geometry now disagrees too) but is, by construction, exactly what the
harmonic projection quotients away.

See the module docstring in `sheaf_poc.py` for the full construction and the
sign/convention details.

## Reproduce the environment

This PoC needs only numpy / scipy / matplotlib — **no torch, no CUDA.** It is a
separate, deliberately lightweight env from the heavy `ddust3r` inference env
(see *Next step*).

```bash
conda env create -f environment.yml
conda activate sheaf-poc
```

Python 3.11, pinned in `environment.yml`. The env was created and the results
below were produced on Windows 11 with conda 25.x.

## Run

```bash
# headline figures for slides / paper (story_localization.png, story_drift.png)
python make_figures.py

# main experiment: prints the results table + sweeps, writes figures/
python run_experiment.py

# supplementary diagnostics: scale collapse, failure boundaries
python diagnostics.py
```

Everything is seeded (`numpy.random.default_rng`) and deterministic. Total
runtime is a few seconds.

## Files

| File | Role |
|---|---|
| `sheaf_poc.py` | core library: SE(3)/Sim(3) helpers, synthetic scene, sheaf assembly, harmonic projection, metrics |
| `run_experiment.py` | the experiment: cases A (clean) / B (drift) / C (drift sweep) / D (static-fraction sweep), figures, verdict |
| `make_figures.py` | the two headline story figures (localization triptych, drift-invariance curve) |
| `diagnostics.py` | supplementary: Sim(3) scale-collapse demo and the three failure boundaries |
| `environment.yml` | conda spec for the `sheaf-poc` env |
| `figures/` | generated PNGs (regenerated on every run) |

## Interpreting the verdict honestly

This is a **synthetic, in-the-clean** test: the residuals come from ground-truth
geometry, not from a real network, so it isolates *the sheaf machinery* from
*the quality of the pairwise predictions*. It cannot tell us whether D²USt3R's
real per-pixel residuals are clean enough — only that **if** they are, the
cohomological reading provably localizes dynamics and is drift-invariant in a
way the raw residual is not. That is precisely the load-bearing question the
project notes wanted answered cheaply before committing to the nonabelian bridge
and the localization theorem. It passes.

What it does **not** yet establish (the real next steps):

1. **Real predictions.** Replace the synthetic residual with D²USt3R pairwise
   pointmaps on a real toy clip with one moving object (next section).
2. **Occlusion.** We tested *motion*; the paper also claims *occluded* regions.
   Occlusion is missing-data, not inconsistency, and needs its own test.
3. **The theorem.** The static-dominance precondition (case D) is the
   hypothesis; a separation bound as a function of static fraction / motion /
   noise is the thing to prove.

## Next step — real D²USt3R predictions

The heavier model lives in `../DDUSt3R` and needs its **own** env (pytorch 2.5.1
+ CUDA 12.1, see `../DDUSt3R/README.md`):

```bash
conda create -n ddust3r python=3.11 cmake=3.14.0
conda activate ddust3r
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ../DDUSt3R/requirements.txt
# checkpoint: see ../DDUSt3R/README.md (Google Drive link)
```

The bridge to this PoC: run D²USt3R on view pairs to get per-pixel pointmaps
`X^{n,m}`, push them through current pose estimates to form the per-point edge
residuals `w_i − w_j`, and feed those into `sheaf_poc.build_sheaf` /
`harmonic_projection` in place of the synthetic residuals. The sheaf side of the
code is prediction-source-agnostic by design.
