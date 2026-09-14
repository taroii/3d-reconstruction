# Phase 1C — pre-registered arms (notes/diagnosis.md §4)

Recorded 2026-09-04, BEFORE any finetuning. `--run` refuses to start without this
file, so the arms cannot be chosen after seeing a result.

## Gate

§4 is conditional on 1B producing a non-trivial objective effect. **[MEASURED]**
`results/phase1/1b_decomposition.md`: objective eta^2 = 0.066, position = 0.311
vs position+grad = 0.490 and position+normal = 0.388. Gate met.

## Arms

| arm | family | term | weight | role |
|---|---|---|---|---|
| pi3_normal_1.0 | pi3 | lambda_normal | 1.0 | UNMODIFIED-LOSS CONTROL |
| pi3_normal_0.5 | pi3 | lambda_normal | 0.5 | ablation |
| pi3_normal_0.0 | pi3 | lambda_normal | 0.0 | ablation |
| pi3_step0 | pi3 | lambda_normal | 1.0 | released ckpt, 0 steps (ANCHOR) |
| vggt_grad_on | vggt | w_grad | 1.0 | UNMODIFIED-LOSS CONTROL |
| vggt_grad_off | vggt | w_grad | 0.0 | ablation |
| vggt_step0 | vggt | w_grad | 1.0 | released ckpt, 0 steps (ANCHOR) |

## Matched budget (mandatory)

Identical across every non-anchor arm of a family, and asserted by
`check_matched_budget` before the first step:

- data: 4800 samples, order drawn once with data_seed=20260904, the
  SAME ordered list for every arm (verified by SHA over the sequence)
- schedule: 1200 optimizer steps x 4 samples, lr 1e-05, warmup 50,
  AdamW wd 0.01, grad clip 1.0
- seeds: init_seed=7 at every arm start
- scope: {'pi3': 'point+conf', 'vggt': 'heads'} — frozen backbone (see the feasibility verdict)
- resolution: train 392 px, eval 518 px

## Predictions

| # | Prediction | Label going in |
|---|---|---|
| 1 | removing the coupling term (lambda_normal=0 / grad off) changes boundary FP relative to the matched-budget control | **[UNTESTED]** — direction and magnitude both unknown. 1B found position+grad and position+normal ABOVE position-only on synthetic data, but that is a different predictor, a different construction and a from-scratch fit. Do not assume the sign carries. |
| 2 | aggregate metrics (AbsRel, delta<1.25) move less than boundary FP | **[UNTESTED]**, plausible. If aggregates move as much, the arms differ in overall fit quality and the boundary contrast is not attributable to boundaries. |
| 3 | the step-0 anchor differs from the matched-budget control | **[UNTESTED]** — this is the size of the effect of finetuning AT ALL, and it bounds what an arm difference can be read to mean. |

Explicitly NOT predicted: that any arm difference will be large, or that a null
result would show the term does not matter. Per §4 and the feasibility verdict, a
weak effect is the expected outcome of changing a term late with a frozen
backbone, whether or not the term matters at scale.
