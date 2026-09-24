# Phase 1B v2 — effect sizes

Generated 2026-09-17 10:37:48 · 2940/2940 cells · device cuda

## Exclusions

| predictor | excluded | status |
|---|---|---|
| free | 0.0% | ok |
| cnn_small | 3.9% | ok |
| cnn_large | 15.0% | ok |

By predictor and objective: cnn_large/position 14, cnn_large/position+both 20, cnn_large/position+conf 33, cnn_large/position+conf_noclip 37, cnn_large/position+grad 15, cnn_large/position+normal 16, cnn_large/position_squared 12, cnn_small/position 5, cnn_small/position+both 10, cnn_small/position+conf 6, cnn_small/position+conf_noclip 7, cnn_small/position+grad 7, cnn_small/position_squared 3

## Gates

- falsifier (position/free/clean): FP = 0.0000 over 70 cells (max 0.05); 0 fits further than 0.1·gap from the closed form
- positive control: PRESENT+ (+0.293 [+0.285, +0.300])
- reproducibility: fronto_d3.0/position_squared/free/clean/1004 same, fronto_d0.5/position+grad/free/clean/1002 same, fronto_d1.0/position+both/cnn_small/mixed/1000 same, fronto_d0.5/position+conf_noclip/cnn_small/clean/1000 same, fronto_d3.0/position+conf_noclip/cnn_large/clean/1009 same, fronto_d3.0/position+normal/cnn_large/clean/1002 same

## Primary contrasts (simultaneous 95% CIs, α/m = 0.0029)

| contrast | FP_A | FP_B | effect | CI | pairs | status |
|---|---|---|---|---|---|---|
| `O:grad/free/clean` | 0.033 | 0.000 | **+0.033** | [+0.025, +0.041] | 70/70 | NEGLIGIBLE |
| `O:grad/cnn_small/clean` | 0.046 | 0.080 | **-0.034** | [-0.045, -0.023] | 65/70 | NEGLIGIBLE |
| `O:grad/cnn_large/clean` | 0.021 | 0.084 | **-0.063** | [-0.096, -0.042] | 60/70 | PRESENT- |
| `O:normal/free/clean` | 0.087 | 0.000 | **+0.087** | [+0.078, +0.096] | 70/70 | PRESENT+ |
| `O:normal/cnn_small/clean` | 0.072 | 0.080 | **-0.008** | [-0.017, -0.001] | 68/70 | NEGLIGIBLE |
| `O:normal/cnn_large/clean` | 0.060 | 0.086 | **-0.026** | [-0.058, -0.004] | 60/70 | INCONCLUSIVE |
| `O:conf/free/clean` | 0.000 | 0.000 | **+0.000** | [+0.000, +0.000] | 70/70 | NEGLIGIBLE |
| `O:conf/cnn_small/clean` | 0.105 | 0.081 | **+0.024** | [+0.004, +0.051] | 66/70 | INCONCLUSIVE |
| `O:conf/cnn_large/clean` | 0.114 | 0.083 | **+0.031** | [-0.008, +0.069] | 53/70 | INCONCLUSIVE |
| `O:conf_noclip/free/clean` | 0.000 | 0.000 | **+0.000** | [+0.000, +0.000] | 70/70 | NEGLIGIBLE |
| `O:conf_noclip/cnn_small/clean` | 0.083 | 0.081 | **+0.002** | [-0.009, +0.014] | 66/70 | NEGLIGIBLE |
| `O:conf_noclip/cnn_large/clean` | 0.092 | 0.085 | **+0.007** | [-0.024, +0.042] | 52/70 | NEGLIGIBLE |
| `T:mixed/free` | 0.295 | 0.000 | **+0.295** | [+0.293, +0.297] | 70/70 | PRESENT+ |
| `T:mixed/cnn_small` | 0.285 | 0.080 | **+0.204** | [+0.194, +0.214] | 66/70 | PRESENT+ |
| `T:mixed/cnn_large` | 0.288 | 0.085 | **+0.204** | [+0.170, +0.225] | 61/70 | PRESENT+ |
| `P:cnn_small` | 0.080 | 0.000 | **+0.080** | [+0.069, +0.095] | 68/70 | PRESENT+ |
| `P:cnn_large` | 0.084 | 0.000 | **+0.084** | [+0.066, +0.114] | 64/70 | PRESENT+ |

### Sensitivity: unsettled cells dropped

No primary status changes.

### Per-scene effects for PRESENT contrasts

| contrast | fronto_d0.2 | fronto_d0.5 | fronto_d1.0 | fronto_d2.0 | fronto_d3.0 | slanted_d1.0 | k3_d2.0 |
|---|---|---|---|---|---|---|---|
| `O:grad/cnn_large/clean` | -0.055 | -0.047 | -0.056 | -0.091 | -0.057 | -0.037 | -0.098 |
| `O:normal/free/clean` | +0.058 | +0.089 | +0.102 | +0.085 | +0.054 | +0.111 | +0.108 |
| `T:mixed/free` | +0.294 | +0.294 | +0.294 | +0.294 | +0.294 | +0.294 | +0.302 |
| `T:mixed/cnn_small` | +0.212 | +0.206 | +0.212 | +0.205 | +0.219 | +0.230 | +0.148 |
| `T:mixed/cnn_large` | +0.210 | +0.220 | +0.228 | +0.178 | +0.216 | +0.236 | +0.149 |
| `P:cnn_small` | +0.063 | +0.074 | +0.066 | +0.072 | +0.061 | +0.061 | +0.163 |
| `P:cnn_large` | +0.083 | +0.066 | +0.072 | +0.101 | +0.068 | +0.055 | +0.138 |

## Secondary contrasts (unadjusted 95% CIs, not used by the rule)

| contrast | FP_A | FP_B | effect | CI | pairs | status |
|---|---|---|---|---|---|---|
| `O:both/free/clean` | 0.060 | 0.000 | **+0.060** | [+0.053, +0.066] | 70/70 | PRESENT+ |
| `O:both/cnn_small/clean` | 0.043 | 0.080 | **-0.037** | [-0.044, -0.029] | 64/70 | NEGLIGIBLE |
| `O:both/cnn_large/clean` | 0.023 | 0.086 | **-0.063** | [-0.084, -0.047] | 57/70 | PRESENT- |
| `O:grad/free/mixed` | 0.315 | 0.295 | **+0.020** | [+0.017, +0.023] | 70/70 | NEGLIGIBLE |
| `O:grad/cnn_small/mixed` | 0.286 | 0.285 | **+0.001** | [-0.007, +0.006] | 67/70 | NEGLIGIBLE |
| `O:grad/cnn_large/mixed` | 0.280 | 0.289 | **-0.009** | [-0.016, -0.002] | 54/70 | NEGLIGIBLE |
| `O:normal/free/mixed` | 0.306 | 0.295 | **+0.011** | [+0.009, +0.014] | 70/70 | NEGLIGIBLE |
| `O:normal/cnn_small/mixed` | 0.284 | 0.285 | **-0.001** | [-0.005, +0.002] | 67/70 | NEGLIGIBLE |
| `O:normal/cnn_large/mixed` | 0.290 | 0.290 | **+0.001** | [-0.010, +0.015] | 57/70 | NEGLIGIBLE |
| `O:conf/free/mixed` | 0.295 | 0.295 | **+0.000** | [+0.000, +0.000] | 70/70 | NEGLIGIBLE |
| `O:conf/cnn_small/mixed` | 0.287 | 0.285 | **+0.003** | [-0.000, +0.006] | 66/70 | NEGLIGIBLE |
| `O:conf/cnn_large/mixed` | 0.306 | 0.288 | **+0.019** | [+0.005, +0.033] | 49/70 | NEGLIGIBLE |
| `O:conf_noclip/free/mixed` | 0.295 | 0.295 | **+0.000** | [+0.000, +0.000] | 70/70 | NEGLIGIBLE |
| `O:conf_noclip/cnn_small/mixed` | 0.281 | 0.285 | **-0.004** | [-0.011, +0.001] | 67/70 | NEGLIGIBLE |
| `O:conf_noclip/cnn_large/mixed` | 0.295 | 0.289 | **+0.006** | [-0.004, +0.020] | 45/70 | NEGLIGIBLE |

## Blocked variance decomposition (description only)

Scene blocked; position_squared and excluded cells removed. Replicate noise 0.064 of total, scene and its interactions 0.034. Unbalanced after exclusions, so approximate.

| bucket | share of systematic variance |
|---|---|
| objective | 0.017 |
| predictor | 0.020 |
| target | 0.969 |

## Scope

**[MEASURED]** for this synthetic construction, these objective implementations and these small predictors. **[ASSUMED]** that any of it transfers to billion-parameter models trained at scale; Phase 1C is the partial, confounded check.
