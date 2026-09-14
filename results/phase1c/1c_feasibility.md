# Phase 1C — feasibility verdict

Recorded 2026-09-04. One shared NVIDIA RTX 4000 Ada, 19.56 GiB usable, **3.5 GiB working budget** (observed free: 0.9-6.0 GiB).

## Full finetuning does not fit — [PROVEN] from the parameter counts

- **pi3**: 958.7 M params (958.7 M excluding the unused track head) x 16 B/param for fp32 AdamW = **14.3 GiB** of static state, before a single stored activation. That is 4.1x the working budget, and leaves 5.3 GiB for activations even on a card with nobody else on it.
- **vggt**: 1256.5 M params (1190.6 M excluding the unused track head) x 16 B/param for fp32 AdamW = **17.7 GiB** of static state, before a single stored activation. That is 5.1x the working budget, and leaves 1.8 GiB for activations even on a card with nobody else on it.

## What was measured to fit — [MEASURED] 2026-09-04

Frozen weights bf16, trainable modules fp32, AdamW, forward + backward + step on random inputs, peak `torch.cuda.max_memory_allocated`.

| family | scope | px | views | trainable | peak GiB | s/step |
|---|---|---|---|---|---|---|
| pi3 | point+conf | 392 | 1 | 133.1 M | 3.06 | 0.51 |
| pi3 | point | 392 | 1 | 66.7 M | 2.94 | 0.38 |
| pi3 | point | 392 | 2 | 66.7 M | 2.96 | 0.35 |
| pi3 | point | 518 | 1 | 66.7 M | 2.96 | 0.56 |
| vggt | heads | 392 | 1 | 65.3 M | 3.19 | 0.61 |
| vggt | heads | 392 | 2 | 65.3 M | 3.52 | 0.79 |
| vggt | heads+last4 | 392 | 1 | 115.7 M | 3.28 | 0.64 |
| vggt | heads | 518 | 1 | 65.3 M | 3.63 | 0.75 |
| vggt | heads | 518 | 2 | 65.3 M | OOM | — |

## The card is shared

**[MEASURED]** 2026-09-04: a neighbour's job held 14.0-16.0 GiB throughout, leaving 3.5-6.0 GiB and falling. One probe was OOM-killed when free memory reached 0.9 GiB mid-run. The working budget is ~3.5 GiB and is not guaranteed; `--run` re-checks it before every arm and aborts rather than competing. No other user's process is ever killed.

## What the restricted experiment can and cannot answer

**CAN**: whether the coupling term, applied late, changes where the READOUT places points at occlusion boundaries, against a matched-budget control. A positive result is a real transfer test of the 1B objective effect to a billion-parameter model on real data.

**CANNOT**: (i) whether a model trained from scratch without the term behaves differently; (ii) whether the effect lives in the backbone features — those are frozen at weights learned under the original objective; (iii) anything about the term's effect on multi-view aggregation. A NULL result is therefore close to uninformative.

