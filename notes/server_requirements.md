# Server Requirements — Premise Check

Companion to `PREMISE_CHECK.md`. Everything here is **inference-only**; no training resources required at this stage.

Two findings up front that shape this whole list:

> **⚠️ Finding 1 — VGGT-1B benchmark contamination.** The VGGT authors have publicly stated that an issue "may have caused benchmark contamination in an ancestor checkpoint of the released 1B model," that reported numbers "may be inflated," and that users should not rely on that model for benchmark assessment pending their investigation. This does not stop us using VGGT-1B, but it means **any dataset in VGGT's training mix gives an optimistic reading**, and our headline result must come from held-out data.
>
> **⚠️ Finding 2 — Hypersim is probably in VGGT's training set.** VGGT's training mix is large and (to my recollection) includes Hypersim, Virtual KITTI, Replica, ScanNet, and Kubric among others. `PREMISE_CHECK.md` named Hypersim as the primary Tier A source; that is now **conditional on verification** (Sec. 4 below). We need at least one clean, exact-boundary dataset that no evaluated model trained on.

---

## 1. Models

Run every model in **feed-forward mode only** — no bundle adjustment, no global alignment, no filtering.

### Tier 1 — required (the paper's subjects)

| Model | Source | Checkpoint | Outputs used | Notes |
|---|---|---|---|---|
| **VGGT-1B** | `github.com/facebookresearch/vggt` | `facebook/VGGT-1B` (HF) | pointmap branch **and** depth branch (evaluate separately) | Non-commercial license. See contamination warning. `VGGT-1B-Commercial` exists but needs an application form — not needed for research. |
| **π³ (Pi3)** | `github.com/yyfz/Pi3` | `yyfz233/Pi3` (HF) | local pointmap → per-view depth | Original ICLR 2026 model. |
| **π³X (Pi3X)** | same repo | `yyfz233/Pi3X` (HF) | same | Newer variant the authors now recommend. Include both — a difference between them is informative. |

### Tier 2 — strongly recommended (lineage + contrast)

| Model | Checkpoint | Why it matters |
|---|---|---|
| **DUSt3R** | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` | The origin of the loss our theory analyses. If flying pixels are loss-induced, DUSt3R is where the argument applies most directly. |
| **MASt3R** | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` | Same backbone family with an added matching head — tests whether the extra head changes boundary behaviour. |
| **VGGT-Omega** | `github.com/facebookresearch/vggt-omega` | Newer VGGT (CVPR 2026 Oral). Reviewers will ask whether the failure persists in the current generation. |
| **MoGe-2** | Microsoft release (verify exact HF id) | Monocular pointmap explicitly targeting *sharp detail*. The closest thing to an existing "solution" — calibrates how low FP can go. |

### Tier 3 — calibration references (monocular; not our subjects)

Needed for one reason: **without a low-FP reference we cannot tell whether a measured rate is high.** A 15% FP rate means nothing in isolation.

| Model | Checkpoint | Role |
|---|---|---|
| **Depth Pro** | `apple/DepthPro` | Sharp-boundary discriminative reference. |
| **Depth Anything V2** | `depth-anything/Depth-Anything-V2-Large` | Standard discriminative baseline. |
| **Marigold** | `prs-eth/marigold-*` (verify id) | Generative reference — literature claims generative models avoid flying pixels; this tests that claim in our own pipeline. |

> Checkpoint ids for Tier 2/3 are from memory and **must be verified** before download. Tier 1 ids are confirmed.

---

## 2. Datasets

### Tier A — exact-boundary synthetic (primary result comes from here)

| Dataset | Approx. size | Multi-view? | Contamination risk | Notes |
|---|---|---|---|---|
| **Hypersim** | ~1.9 TB full; **50–100 scene subset ≈ 30–60 GB is sufficient** | Yes | **HIGH — likely in VGGT training** | Precedent: used by Pixel-Perfect Depth for edge-aware point-cloud eval precisely because its GT point clouds are high quality. ⚠️ **Gotcha: Hypersim `depth_meters` is Euclidean distance-to-camera, not planar z-depth. Convert before use or every boundary measurement is wrong.** |
| **Middlebury 2014** | ~5 GB | Stereo pairs | LOW | Real but structured-light GT, extremely sharp boundaries. Small, clean, likely held out. Good clean-reference candidate. |
| **Infinigen / other recent procedural** | varies | Yes | LOW (too new for older training mixes) | Best bet for a genuinely held-out exact-boundary set. Verify availability. |
| **TartanAir** | ~3 TB full; subset fine | Yes | MEDIUM | Outdoor/indoor synthetic, exact depth. |

### Tier B — real, high-quality (generalization, secondary)

| Dataset | Approx. size | Notes |
|---|---|---|
| **iBims-1** | ~1 GB | **High priority despite being small.** Purpose-built for boundary evaluation with depth-boundary annotations; it is the source of the DBE metrics in our Preliminaries. Mostly single-image — run models in single-view mode (VGGT explicitly supports one to hundreds of views). Almost certainly held out. |
| **ETH3D** | ~100 GB (high-res MVS) | Laser GT, multi-view, high resolution. |
| **DTU** | ~200 GB (MVS subset smaller) | Structured-light GT, 49–64 views/scene, controlled conditions. Used in published VGGT uncertainty analysis. |
| **ScanNet++** | application required; subset fine | Laser-scan GT, better boundaries than ScanNet. |

### Tier C — negative controls (small, cheap, prove the gate works)

| Dataset | Why include |
|---|---|
| **NYUv2** (labeled subset ~3 GB) | Published finding: its GT *itself* contains flying-point artifacts. Should show high `FP_GT` — validates our GT-audit machinery. |
| **Bonn RGB-D** (~20 GB) | Published finding: aggressive GT masking along object boundaries. Should **fail** the measurability gate — validates the gate. |

If these two behave as the literature predicts, our pipeline is trustworthy. If they don't, the implementation is wrong. This is cheap insurance.

---

## 3. Storage and compute

**Storage:** ~500 GB comfortable; ~150 GB minimum with aggressive subsetting. Budget an extra ~200 GB for cached per-view depth predictions (3+ models × several datasets).

**GPU:**
- Minimum: one 24 GB card (3090/4090/A5000). Fine for ≤ ~16 views/scene at 518×518.
- Comfortable: 40–80 GB (A100/H100). Global attention cost grows quadratically with total tokens, so view count is the binding constraint, not model size.
- CPU/RAM: 32 GB+; Hypersim HDF5 and high-res MVS decoding are RAM-hungry.

**Time:** inference is fast (VGGT reconstructs a scene in under a second); wall-clock will be dominated by data download, decoding, and metric computation — not model forward passes.

---

## 4. Contamination matrix — do this before anything else

Build a table: **rows = models, columns = datasets, cells = in-training-set? (yes / no / unknown)**, filled in from each paper's training-data table (VGGT §, π³ §, DUSt3R §, MoGe §).

Rules that follow from it:
- The **headline result must come from a dataset marked "no" for every Tier 1 model.**
- Datasets marked "yes" for any model are reported as a **separate stratum**, explicitly labelled as likely optimistic.
- "Unknown" counts as "yes" for safety.

If no exact-boundary dataset is clean for all Tier 1 models, the fallback is to **render a small held-out set ourselves** (Blender/Kubric-style, a few hundred images with two-surface boundaries and exact depth). That is a day of work and it guarantees a clean primary measurement — worth it if the matrix comes back bad.

---

## 5. Environment

- Separate virtual environments per model repo. DUSt3R/MASt3R, VGGT, and π³ have incompatible pinned dependencies; do not try to unify them.
- CUDA 12.x, PyTorch matching each repo's requirement. VGGT uses bf16 on Ampere+ (compute capability ≥ 8.0), fp16 otherwise.
- HuggingFace token configured (some checkpoints are gated).
- `h5py` for Hypersim, `OpenEXR` for several synthetic sets.
- Pin and record every version in `env/` — this study must be reproducible when we write it up.

---

## 6. Minimum viable configuration

If resources are tight, this is the smallest set that still yields a defensible go/no-go:

1. **Models:** VGGT-1B (both branches) + π³. *(Tier 1 only.)*
2. **Clean primary set:** one exact-boundary dataset verified held out — Middlebury 2014, Infinigen, or self-rendered.
3. **Boundary benchmark:** iBims-1 (tiny, purpose-built, almost certainly held out).
4. **Negative controls:** NYUv2 + Bonn (small, validate the pipeline).
5. **One calibration reference:** Depth Pro or MoGe-2.

Total ≈ 50–80 GB and a single 24 GB GPU. Everything else strengthens the result but does not change the decision.

---

## 7. Inventory checklist

Fill this in and send it back; I'll adapt the protocol to what actually exists.

```
GPUs (count, model, VRAM):
Free disk space:
Datasets already present (name, subset, size):
  [ ] Hypersim            [ ] ScanNet / ScanNet++     [ ] ETH3D
  [ ] NYUv2               [ ] KITTI                   [ ] DTU
  [ ] TartanAir           [ ] Replica                 [ ] Middlebury
  [ ] iBims-1             [ ] Bonn                    [ ] other: ______
Checkpoints already downloaded:
  [ ] VGGT-1B   [ ] Pi3   [ ] Pi3X   [ ] DUSt3R   [ ] MASt3R
  [ ] MoGe/MoGe-2   [ ] Depth Pro   [ ] Depth Anything V2   [ ] Marigold
Internet access from compute nodes (for HF downloads)? y/n
Shared or exclusive GPU access?
```