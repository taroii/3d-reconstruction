# Contamination Matrix — verified from primary sources

Resolves the §07 "UNVERIFIED · HIGHEST PRIORITY" item in the premise-check report. Training sets below are quoted from each paper's own training-data section (PDFs in the project), not secondary sources.

---

## 1. Training sets, as stated by the papers

**DUSt3R** (§4, "Training data") — mixture of **eight** datasets:
Habitat, MegaDepth, ARKitScenes, Static Scenes 3D, BlendedMVS, ScanNet++, CO3D-v2, Waymo.

**MASt3R** (§4.1, "Training data") — mixture of **14** datasets:
Habitat, ARKitScenes, BlendedMVS, MegaDepth, Static Scenes 3D, ScanNet++, CO3D-v2, Waymo, Map-free, WildRGB, VirtualKitti, Unreal4K, TartanAir, + an internal dataset.

**VGGT** (§3.3 "Training Data") — "a large and diverse collection":
Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, **HyperSim**, Mapillary, Habitat, Replica, MVS-Synth, **PointOdyssey**, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin, + a synthetic dataset of artist-created assets similar to Objaverse.

**π³** (§3.4 "Model Training") — "a large-scale aggregation of **15** diverse datasets":
GTA-SfM, CO3D, WildRGB-D, Habitat, ARKitScenes, **TartanAir**, ScanNet, ScanNet++, BlendedMVG, MatrixCity, MegaDepth, **Hypersim**, Taskonomy, Mid-Air, + an internal dynamic-scene dataset.

---

## 2. The matrix

| Dataset (study tier) | DUSt3R | MASt3R | VGGT | π³ | Verdict for the headline |
|---|:--:|:--:|:--:|:--:|---|
| **middlebury** (A) | no | no | no | no | **CLEAN** |
| **infinigen** (A) | no | no | no | no | **CLEAN** |
| **ibims** (B) | no | no | no | no | **CLEAN** |
| **eth3d** (B) | no | no | no* | no* | **CLEAN** (*evaluation-only in both papers) |
| **sintel** (A) | no | no | no* | no* | **CLEAN** (*evaluation-only) |
| **spring** (A) | no | no | no | no | **CLEAN** (newer than all four; low risk) |
| **hypersim** (stratum) | no | no | **YES** | **YES** | **CONTAMINATED — both Tier 1 models** |
| **tartanair** (A) | no | **YES** | no | **YES** | **CONTAMINATED for π³** |
| **pointodyssey** (A) | no | no | **YES** | inherited | **CONTAMINATED for VGGT** |
| **nyuv2** (C) | no | no | no | no | clean, but fails the measurability gate |
| **bonn** (C) | no | no | no | no | clean, but gate is blind to its masking |

**Bottom line: the headline holds.** Middlebury and Infinigen — the two datasets carrying the near-zero control and the primary result — appear in none of the four training mixes. The strongest single fact in the report (Middlebury GT control 0.00028 vs. models 0.25–0.32) rests on genuinely held-out data.

---

## 3. Correction 1 — the Hypersim interpretation in §03 does not hold

The report argues:

> "VGGT scores markedly better on Hypersim than anywhere else; π³ does not move. … That is the pattern contamination produces: the model that likely trained on this data looks better on it, while the model that did not is unaffected."

**π³ trains on Hypersim.** It is named explicitly in π³'s 15-dataset list. So the premise that π³ "did not" train on it is false, and the inference built on it collapses: *both* Tier 1 models trained on Hypersim, yet only VGGT improved there.

That does not resurrect contamination as the explanation — it removes the evidence for it. Remaining candidates for VGGT's ~40% drop on Hypersim:
- **Domain** — the alternative the report already flagged (photorealistic indoor vs. Middlebury tabletop). Now the more economical reading, not the less.
- **Differential exposure** — both trained on Hypersim, but sampling weights and epoch counts differ; VGGT may simply have seen far more of it.
- **π³'s frozen encoder** — π³ initializes from pretrained VGGT and *keeps the encoder frozen* during training, so its capacity to fit any single dataset is constrained relative to VGGT's.

**Action:** strike the contamination reading of the Hypersim pattern from any write-up. The stratum separation is still correct practice, but it must now be justified procedurally (Hypersim is in both training mixes) rather than by the observed VGGT/π³ gap.

---

## 4. Correction 2 — the three streams are not independent witnesses

π³'s own training section states that its final model is **not trained from scratch**: it initializes the encoder and the alternating-attention module from the pretrained VGGT model, and **the encoder stays frozen**.

Consequences the report does not currently account for:

1. **Shared encoder.** `vggt_point`, `vggt_depth`, and `pi3_local` all run on a VGGT-trained frozen encoder. "All three streams clear every threshold" is weaker corroboration than three independent models agreeing — it is closer to two heads plus a third model sharing the same visual frontend. If the artifact originates in encoder features, all three would show it for one common reason.
2. **Transitive contamination.** π³ inherits VGGT's data exposure through that frozen encoder. Any dataset in VGGT's mix (PointOdyssey, Replica, Kubric, DL3DV, Mapillary, Aria, MVS-Synth) is partially contaminated for π³ as well, even when absent from π³'s own list.

**Action:** state the shared-encoder dependency explicitly wherever stream agreement is used as evidence. The cleanest fix is to add a genuinely independent model — **DUSt3R or MASt3R** (different backbone lineage, CroCo-pretrained, no VGGT weights, and both are clean on Middlebury/Infinigen/iBims/ETH3D). If the effect reproduces there, the corroboration argument becomes real.

---

## 5. Two Tier A datasets are misfiled

- **TartanAir** is in π³'s *and* MASt3R's training mixes → move out of clean Tier A; report as a contaminated stratum for those models.
- **PointOdyssey** is in VGGT's training mix → same, for VGGT (and π³ transitively).

Neither currently carries the headline, so this is bookkeeping rather than a threat — but a reviewer checking the training tables will catch it.

---

## 6. Residual unknowns (cannot be closed from the papers)

- **Internal datasets.** MASt3R and π³ each list "an internal dataset" with no contents disclosed. Irreducible. State it as a limitation rather than claiming a fully verified clean set.
- **Pi3X and VGGT-Omega.** Newer checkpoints; training data not verified here. If either is added to the study, re-verify before letting it near the headline.
- **Sub-scene exclusions.** VGGT notes it excluded specific ScanNet scenes to keep ScanNet-1500 fair, and that DUSt3R/MASt3R trained on most of MegaDepth excluding scenes 0015 and 0022 — evidence that these groups track contamination at scene granularity. Our matrix is at dataset granularity, which is coarser but sufficient here since our clean sets appear in no mix at all.

---

## 7. Net effect on the project

| Report claim | Status |
|---|---|
| Headline (Middlebury/Infinigen clean, control ≈ 0) | **Confirmed** |
| Hypersim = high contamination risk, keep separate | **Confirmed** (both Tier 1 models) |
| Hypersim VGGT/π³ gap indicates contamination | **Refuted** — π³ also trains on Hypersim |
| Three streams agreeing = strong corroboration | **Weakened** — shared frozen VGGT encoder |
| TartanAir, PointOdyssey as clean Tier A | **Refuted** — reclassify |

The go/no-go verdict is unaffected: **GO stands.** What changes is one interpretive claim, the strength of the multi-stream argument, and two dataset classifications. Adding DUSt3R or MASt3R as an architecturally independent stream is now the highest-value next experiment.