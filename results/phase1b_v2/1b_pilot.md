# Phase 1B v2 — pilot (CNN learning rates)

2026-09-14 22:04:13 · 672 fits · seeds [900, 901] (disjoint from the main run) · FP was never computed.

Criterion: exclusion (collapsed or failed) at most 10%, then the lowest median interior RMSE as a fraction of the gap. Interior means away from the boundary band, so nothing here sees boundary behaviour.

| predictor | lr | fits | excluded | median interior RMSE / gap | median settle / gap | eligible |
|---|---|---|---|---|---|---|
| cnn_small | 0.01 | 84 | 0.0% | 0.0000 | 0.0013 | True |
| cnn_small | 0.003 | 84 | 0.0% | 0.0001 | 0.0008 | True |
| cnn_small | 0.001 | 84 | 0.0% | 0.0002 | 0.0006 | True |
| cnn_small | 0.0003 | 84 | 0.0% | 0.0003 | 0.0004 | True |
| cnn_large | 0.01 | 84 | 8.3% | 0.0003 | 0.0017 | True |
| cnn_large | 0.003 | 84 | 0.0% | 0.0004 | 0.0011 | True |
| cnn_large | 0.001 | 84 | 0.0% | 0.0005 | 0.0013 | True |
| cnn_large | 0.0003 | 84 | 0.0% | 0.0008 | 0.0012 | True |

**PASSED** — chosen: cnn_small=0.01, cnn_large=0.01
