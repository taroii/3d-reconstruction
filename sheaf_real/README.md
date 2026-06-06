# Sheaf Reconstruction — real-data phase

Ports the synthetic PoC (`../sheaf_poc/`) to **real frozen-backbone pointmap
predictions**, following `../notes/plan.md`. Headline claim (plan §0): from
frozen, rigid-aligning DUSt3R pairwise pointmaps — no flow, no dynamic
supervision, no retraining — the harmonic `H¹` recovers a dense moving-region
mask that (i) agrees with motion GT and (ii) is invariant to camera-pose error
that defeats the raw bundle-adjustment residual.

## Architecture: the cache boundary

Everything that touches the network sits **above** a cache; everything below is
pure NumPy/SciPy and iterates in seconds.

```
frames ──[backbone.py]──▶ cache/PairPred ──[correspondence.py]──▶ cache/Matches
                                                       │
                                                       ▼
                          [sheaf.py]  build_coboundary → solve_harmonic
                                      → splat_to_pixels → render_mask
                                                       │
                                                       ▼
                                [eval.py]  AUROC / IoU / F1   [baselines.py]
```

The network runs **once per clip** to fill `cache/`; all sheaf experiments read
the cache.

## Status

| Module | State |
|---|---|
| `sheaf.py` | **done** — Sim(3) coboundary `B(w)=[I\|−[w]ₓ\|w]`, weighted-LS harmonic, IRLS, outer Gauss–Newton, scale-gauge anchor, splat (median/mean/sum), mask (percentile/Otsu). Plan §4. |
| `eval.py` | **done** — AUROC (rank-sum), IoU/F1, pooled `map_auroc`. Plan §8. |
| `datatypes.py` | **done** — `PairPred`, `Matches` (+ npz IO). |
| `synth.py` | **done** — synthetic cache + end-to-end test; reproduces PoC localization through the real interface (**PASS**). |
| `backbone.py` | **validated** — DUSt3R loads + runs on GPU; `run_clip` → per-view camera-frame pointmaps, conf, Sim(3) poses, focals (smoke test `smoke_backbone.py` PASSES). |
| `correspondence.py` | **skeleton** — `CrossAttnCorr` (primary), `MASt3RCorr` (fallback); next to implement, validate on real frames. |
| `baselines.py`, `experiments/` | pending the above. |

`synth.py` is the contract: it builds pointmaps + pixel correspondences in the
exact format `backbone.py`/`correspondence.py` will emit, so the moment those are
filled the whole pipeline runs unchanged.

## Run the downstream test (no network, no GPU)

```
conda activate sheaf-poc        # numpy/scipy only
python synth.py
```

## Locked design decisions (from plan §1 — do not deviate)

- **Sheaf backbone is DUSt3R/MonST3R, NOT D²USt3R** — the `H¹` signal needs a
  rigid-aligning backbone so motion shows up as inconsistency. D²USt3R is a
  *validation instrument* (its SDAP should *suppress* the dynamic `H¹`).
- **Correspondence is appearance/cross-attention (or MASt3R), never optical
  flow, never 3D-NN for the headline** — that is the "no flow" claim.
- **Clips have ≥3 views and ≥1 loop edge** — drift-invariance only exists with
  cycles; two-view harmonic ≈ epipolar residual.
- **Median-over-edges readout + robust solve** for correspondence noise; **
  confidence weighting is required** (structured error mimics motion). Both
  carried over from the realism stress-tests (`../sheaf_poc/experiments_realism.py`).

## Heavy environment (network side, pending)

```
conda create -n sheaf python=3.11 cmake=3.14.0 && conda activate sheaf
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ../DDUSt3R/requirements.txt
```
Weights → `../DDUSt3R/checkpoints/`: DUSt3R (primary), MonST3R (secondary),
D²USt3R (validation only, Google Drive `1dUy03ohGK2jbzhRLN4HYfDkJIpfGr5lP`).

**Full env setup + extras** are documented in `requirements-sheaf-env.txt`
(beyond DDUSt3R/requirements.txt we add `evo`, `imageio`, `scikit-image`). The
bundled code hard-imports RAFT and SAM2 at load; both are flow/segmentation
tools we never use, so they are **stubbed** (`../DDUSt3R/third_party/raft.py`,
`../DDUSt3R/sam2/build_sam.py`) — flow-free by design, `flow_loss_weight` stays
0, so neither is ever called.

**Data still needed** (the `extras` zip has only flow/occlusion, no RGB/depth):
`MPI-Sintel-training_images.zip` (RGB clean+final — required to run anything)
and the Sintel depth+camera release (`depth/`, `camdata_left/` — for motion GT).

## Week-one go/no-go (plan §10)

On one Sintel clip, compute the `H¹(DUSt3R)` vs `H¹(D²USt3R-SDAP)` differential.
**GO** if SDAP suppresses the dynamic `H¹` (interpretation validated); rethink if
not. Everything downstream of the cache is already in place for this.
