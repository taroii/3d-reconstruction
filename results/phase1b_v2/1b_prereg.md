# Phase 1B v2 — pre-registration

<!-- PREREG_1B_V2 {"alpha": 0.05, "code_md5": {"factorial.py": "3fac32629d15c82b888f347e758ddf37", "fpmetrics.py": "1844934794ae9d3b1e761200b1538270"}, "collapse_std": 0.02, "control": ["C:squared/free", "control", {"predictor": "free", "target": "clean"}, "objective", "position_squared", "position"], "delta": 0.05, "failed_rmse": 0.4, "falsifier_max": 0.05, "jitter_px": 1.0, "lr": {"cnn_large": 0.01, "cnn_small": 0.01, "free": 0.05}, "max_excluded": 0.2, "metric": {"beta": 0.2, "delta_min": "0.05*gap", "eta": 0.05, "tau": 2, "w": 3}, "min_pair_frac": 0.5, "n_boot": 20000, "n_real": 25, "n_repro": 6, "objectives": {"position": {"use_conf": false, "w_grad": 0.0, "w_normal": 0.0}, "position+both": {"use_conf": false, "w_grad": 1.0, "w_normal": 1.0}, "position+conf": {"floor": true, "use_conf": true, "w_grad": 0.0, "w_normal": 0.0}, "position+conf_noclip": {"floor": false, "use_conf": true, "w_grad": 0.0, "w_normal": 0.0}, "position+grad": {"use_conf": false, "w_grad": 1.0, "w_normal": 0.0}, "position+normal": {"use_conf": false, "w_grad": 0.0, "w_normal": 1.0}, "position_squared": {"squared": true, "use_conf": false, "w_grad": 0.0, "w_normal": 0.0}}, "opt_gap_frac": 0.1, "predictors": ["free", "cnn_small", "cnn_large"], "primary": [["O:grad/free/clean", "objective", {"predictor": "free", "target": "clean"}, "objective", "position+grad", "position"], ["O:grad/cnn_small/clean", "objective", {"predictor": "cnn_small", "target": "clean"}, "objective", "position+grad", "position"], ["O:grad/cnn_large/clean", "objective", {"predictor": "cnn_large", "target": "clean"}, "objective", "position+grad", "position"], ["O:normal/free/clean", "objective", {"predictor": "free", "target": "clean"}, "objective", "position+normal", "position"], ["O:normal/cnn_small/clean", "objective", {"predictor": "cnn_small", "target": "clean"}, "objective", "position+normal", "position"], ["O:normal/cnn_large/clean", "objective", {"predictor": "cnn_large", "target": "clean"}, "objective", "position+normal", "position"], ["O:conf/free/clean", "objective", {"predictor": "free", "target": "clean"}, "objective", "position+conf", "position"], ["O:conf/cnn_small/clean", "objective", {"predictor": "cnn_small", "target": "clean"}, "objective", "position+conf", "position"], ["O:conf/cnn_large/clean", "objective", {"predictor": "cnn_large", "target": "clean"}, "objective", "position+conf", "position"], ["O:conf_noclip/free/clean", "objective", {"predictor": "free", "target": "clean"}, "objective", "position+conf_noclip", "position"], ["O:conf_noclip/cnn_small/clean", "objective", {"predictor": "cnn_small", "target": "clean"}, "objective", "position+conf_noclip", "position"], ["O:conf_noclip/cnn_large/clean", "objective", {"predictor": "cnn_large", "target": "clean"}, "objective", "position+conf_noclip", "position"], ["T:mixed/free", "target", {"objective": "position", "predictor": "free"}, "target", "mixed", "clean"], ["T:mixed/cnn_small", "target", {"objective": "position", "predictor": "cnn_small"}, "target", "mixed", "clean"], ["T:mixed/cnn_large", "target", {"objective": "position", "predictor": "cnn_large"}, "target", "mixed", "clean"], ["P:cnn_small", "predictor", {"objective": "position", "target": "clean"}, "predictor", "cnn_small", "free"], ["P:cnn_large", "predictor", {"objective": "position", "target": "clean"}, "predictor", "cnn_large", "free"]], "scenes": {"fronto_d0.2": {"boundary": 0.4, "gap": 0.2, "h": 48, "k": 2, "mid_frac": 0.1, "near": 2.0, "px_offset": 0.5, "slant_far": 0.0, "slant_near": 0.0, "tilt": 0.137, "w": 64}, "fronto_d0.5": {"boundary": 0.4, "gap": 0.5, "h": 48, "k": 2, "mid_frac": 0.1, "near": 2.0, "px_offset": 0.5, "slant_far": 0.0, "slant_near": 0.0, "tilt": 0.137, "w": 64}, "fronto_d1.0": {"boundary": 0.4, "gap": 1.0, "h": 48, "k": 2, "mid_frac": 0.1, "near": 2.0, "px_offset": 0.5, "slant_far": 0.0, "slant_near": 0.0, "tilt": 0.137, "w": 64}, "fronto_d2.0": {"boundary": 0.4, "gap": 2.0, "h": 48, "k": 2, "mid_frac": 0.1, "near": 2.0, "px_offset": 0.5, "slant_far": 0.0, "slant_near": 0.0, "tilt": 0.137, "w": 64}, "fronto_d3.0": {"boundary": 0.4, "gap": 3.0, "h": 48, "k": 2, "mid_frac": 0.1, "near": 2.0, "px_offset": 0.5, "slant_far": 0.0, "slant_near": 0.0, "tilt": 0.137, "w": 64}, "k3_d2.0": {"boundary": 0.4, "gap": 2.0, "h": 48, "k": 3, "mid_frac": 0.12, "near": 2.0, "px_offset": 0.5, "slant_far": 0.0, "slant_near": 0.0, "tilt": 0.137, "w": 64}, "slanted_d1.0": {"boundary": 0.4, "gap": 1.0, "h": 48, "k": 2, "mid_frac": 0.1, "near": 2.0, "px_offset": 0.5, "slant_far": 0.5, "slant_near": 0.5, "tilt": 0.137, "w": 64}}, "schedule": "adam+cosine-to-zero", "secondary": ["O:both/free/clean", "O:both/cnn_small/clean", "O:both/cnn_large/clean", "O:grad/free/mixed", "O:grad/cnn_small/mixed", "O:grad/cnn_large/mixed", "O:normal/free/mixed", "O:normal/cnn_small/mixed", "O:normal/cnn_large/mixed", "O:conf/free/mixed", "O:conf/cnn_small/mixed", "O:conf/cnn_large/mixed", "O:conf_noclip/free/mixed", "O:conf_noclip/cnn_small/mixed", "O:conf_noclip/cnn_large/mixed"], "seeds": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009], "steps": 1500, "super": 8, "tag": "PREREG_1B_V2", "targets": ["clean", "mixed"], "unsettled_frac": 0.05} -->

Recorded 2026-09-16 14:41:33, BEFORE the run. `--run` parses the marker above
and refuses to start if any design constant differs from it, or if a single byte of
`src/factorial.py` or `src/fpmetrics.py` has changed. Pilot: `1b_pilot.md` (FP never
computed). Commit this file together with the code it names.

## Why this is a second run

The first run (`results/phase1/`) is not used for any claim. Its cells scored two
distinct boundary pixels each, a collapsed CNN scored as maximal void-landing,
most CNN fits never left initialization, and its verdict came from variance shares
whose size depended on which levels were included. Each is fixed in the code
(see the `factorial.py` docstring) and covered by a self-test.

## Design

- scenes (7): fronto_d0.2, fronto_d0.5, fronto_d1.0, fronto_d2.0, fronto_d3.0, slanted_d1.0, k3_d2.0 — boundary at 0.4 of the width, tilted
  0.137 px per row, so a row-invariant scene cannot quantize FP.
- objectives (7): position, position+grad, position+normal, position+both, position+conf, position+conf_noclip, position_squared
- predictors: free (lr 0.05), cnn_small (lr 0.01), cnn_large (lr 0.01)
- targets: clean (nearest-neighbour), mixed (area-average)
- seeds: 1000–1009 (fresh; v1 used 0–4, the stopped
  confirmation run 5–14, the pilot 900–901)
- 2940 cells, 1500 Adam steps with cosine decay to zero, 25 jittered
  realizations each
- metric: η=0.05, τ=2, β=0.2, identical to Phase 0

## Exclusions (fixed now, all blind to boundary behaviour)

A cell is excluded from every contrast if its prediction is **collapsed** (spatial std
< 0.02·gap, or non-finite) or **failed** (interior RMSE > 0.4·gap, or
non-finite). Both are computed away from the boundary band. A predictor with more than 20% of its cells excluded has all of
its contrasts marked NOT EVALUATED. A contrast needs at least 50% of its
70 (scene, seed) pairs.

## Gates (checked before any branch)

1. **Falsifier.** position + free + clean, over every fitted cell (the optimizer gap
   is NOT an exclusion — on clean targets it would remove exactly the cells with
   FP > 0): mean FP ≤ 0.05. On a breach, if any of those fits is further
   than 0.1·gap from the closed-form weighted median the optimizer failed
   and there is NO VERDICT; if all are at the closed form, the run reports
   FALSIFIED and §1.1 is rewritten before anything else. Note what this can test:
   on clean targets the closed form is on a surface by construction, so the gate
   checks that the fit reaches it and the metric scores it correctly. Claim (a)
   itself is checked in closed form by the self-test.
2. **Positive control.** `C:squared/free` (squared − unsquared norm) must be PRESENT+.
   Otherwise the apparatus cannot detect void-landing and there is NO VERDICT.
3. **Reproducibility.** 6 cells (two per predictor) are re-fitted after the run
   and must match their recorded FP, final loss and interior RMSE exactly.

## Primary contrasts (family of 17, Bonferroni)

Effect = mean over (scene, seed) pairs of FP_A − FP_B. CI: percentile bootstrap,
20000 resamples within scene, two-sided level α/m = 0.05/17.

| id | bucket | effect | held fixed |
|---|---|---|---|
| `O:grad/free/clean` | objective | `position+grad` − `position` | predictor=free, target=clean |
| `O:grad/cnn_small/clean` | objective | `position+grad` − `position` | predictor=cnn_small, target=clean |
| `O:grad/cnn_large/clean` | objective | `position+grad` − `position` | predictor=cnn_large, target=clean |
| `O:normal/free/clean` | objective | `position+normal` − `position` | predictor=free, target=clean |
| `O:normal/cnn_small/clean` | objective | `position+normal` − `position` | predictor=cnn_small, target=clean |
| `O:normal/cnn_large/clean` | objective | `position+normal` − `position` | predictor=cnn_large, target=clean |
| `O:conf/free/clean` | objective | `position+conf` − `position` | predictor=free, target=clean |
| `O:conf/cnn_small/clean` | objective | `position+conf` − `position` | predictor=cnn_small, target=clean |
| `O:conf/cnn_large/clean` | objective | `position+conf` − `position` | predictor=cnn_large, target=clean |
| `O:conf_noclip/free/clean` | objective | `position+conf_noclip` − `position` | predictor=free, target=clean |
| `O:conf_noclip/cnn_small/clean` | objective | `position+conf_noclip` − `position` | predictor=cnn_small, target=clean |
| `O:conf_noclip/cnn_large/clean` | objective | `position+conf_noclip` − `position` | predictor=cnn_large, target=clean |
| `T:mixed/free` | target | `mixed` − `clean` | predictor=free, objective=position |
| `T:mixed/cnn_small` | target | `mixed` − `clean` | predictor=cnn_small, objective=position |
| `T:mixed/cnn_large` | target | `mixed` − `clean` | predictor=cnn_large, objective=position |
| `P:cnn_small` | predictor | `cnn_small` − `free` | objective=position, target=clean |
| `P:cnn_large` | predictor | `cnn_large` − `free` | objective=position, target=clean |

Each contrast is classified with δ = 0.05 (Phase 0's 5-point margin):

| status | condition |
|---|---|
| PRESENT+ | CI lower bound > 0 **and** effect ≥ δ |
| PRESENT− | CI upper bound < 0 **and** effect ≤ −δ |
| NEGLIGIBLE | CI inside [−δ, δ] |
| INCONCLUSIVE | anything else |

## Decision rule

The target contrasts are expected to be PRESENT+ nearly by construction: a free
predictor fitted to blended targets reproduces the blend, which sits in the void.
They size the target effect; they do not discover it. So attribution is decided
first on CLEAN targets, where only the objective and the predictor can act.

A bucket (objective: the 12 `O:` contrasts; predictor: the 2 `P:` contrasts) is
**resolved** when every contrast in it is PRESENT+, PRESENT− or NEGLIGIBLE. Only
PRESENT+ implicates a bucket as a cause; PRESENT− is a definite reading that the
term does not cause void-landing, and is listed separately.

| outcome-tree branch | condition |
|---|---|
| MIXED | an objective contrast AND a predictor contrast are PRESENT+ |
| OBJECTIVE | an objective contrast is PRESENT+ and the predictor bucket is resolved with none PRESENT+ |
| RESIDUAL | a predictor contrast is PRESENT+ and the objective bucket is resolved with none PRESENT+ |
| TARGET | both buckets resolved with nothing PRESENT+, and a target contrast is PRESENT+ |
| NULL | both buckets resolved with nothing PRESENT+, and no target contrast PRESENT+ |
| NO VERDICT | anything else (one bucket PRESENT+ with the other unresolved is reported as that effect plus "branch undetermined"), or a gate failed |

Target contrasts never promote a clean-target result to MIXED: since they are
PRESENT+ nearly by construction, that would make OBJECTIVE and RESIDUAL
unreachable. **Phase 1C** runs only for the objective terms (gradient, normal) with a
PRESENT+ contrast on at least one predictor.

## Stated before seeing anything

- Any branch involving TARGET is conditional on the real TRAINING ground truth being
  blended at boundaries. 1B cannot measure that; the Phase 0 GT controls on the
  training sets (Hypersim, TartanAir, PointOdyssey) are the available evidence.
- A PREDICTOR effect compares a CNN that sees an image with a free map that does
  not, under a different learning rate. It is "smoothness or optimization",
  not smoothness alone.
- Not predicted: that the objective dominates. The §1.3 list is open.

## Reported but not used by the rule

Secondary contrasts at unadjusted α = 0.05 (the `both` level; every objective
contrast under mixed targets); a sensitivity re-analysis dropping cells whose
prediction still moved more than 0.05·gap over the last 10% of steps;
per-scene effects; exclusion counts by predictor and objective; the blocked
variance decomposition.
