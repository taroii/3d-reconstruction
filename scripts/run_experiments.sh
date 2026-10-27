#!/usr/bin/env bash
# =============================================================================
# The pre-registered arm matrix: 6 arms x 5 seeds = 30 runs, start to finish.
#
# Arms factor as (signal) x (weight form), because that is the shape of the
# reviewers' objections rather than an arbitrary list:
#
#   none:clamp:0          baseline. Reproduces VGGT's published objective --
#                         verified numerically against upstream, not asserted.
#   umbrella:clamp:1      the ACCV method exactly as published.
#   grad:clamp:1          the published first-order control. The estimator
#                         diagnostic puts its pixel ranking at Spearman
#                         rho >= 0.996 with umbrella on smooth scenes, which
#                         predicts these two arms score alike.
#   umbrella:unitmean:1   mean-preserving. Isolates the reweighting from the
#                         confidence-coefficient confound: under `clamp`,
#                         E[w] = 1.52 at lam=1, which divides the effective
#                         alpha from 0.2 to 0.10 and could produce the paper's
#                         monotone trend on its own.
#   umbrella:binary:1     top-decile indicator at the clamp ceiling. Asks
#                         whether the signal's MAGNITUDE carries anything past
#                         the identity of the pixels it selects.
#   cotangent:clamp:1     area-normalized Meyer et al. estimator -- the only
#                         one of the three that actually measures mean
#                         curvature. The "is anything second-order doing work"
#                         arm nobody has run.
#
# Runs are SEQUENTIAL: one GPU, shared with other people. A run is skipped only
# when it left a `done.json` completion marker recording the step count we are
# asking for, so the matrix is resumable without redoing finished work AND
# without mistaking an interrupted run for a finished one -- `checkpoint.pt`
# alone cannot tell the two apart, because it is written periodically mid-run.
#
#   scripts/run_experiments.sh                 the whole matrix
#   scripts/run_experiments.sh --seeds 0       one seed, all arms
#   scripts/run_experiments.sh --arms none:clamp:0,umbrella:clamp:1
#   STEPS=500 scripts/run_experiments.sh --seeds 0   a fast pilot
# =============================================================================
set -uo pipefail
PROJ="${PROJ:-$HOME/taro/vggt-curv}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${CONDA_ROOT:-$HOME/miniconda3}/envs/${ENV_NAME:-vggt-curv}/bin/python"
RUNS="$PROJ/runs"; LOGS="$PROJ/logs"
mkdir -p "$RUNS" "$LOGS"

ARMS="none:clamp:0 umbrella:clamp:1 grad:clamp:1 umbrella:unitmean:1 umbrella:binary:1 cotangent:clamp:1"
SEEDS="0 1 2 3 4"
STEPS="${STEPS:-6000}"

while [ $# -gt 0 ]; do
  case "$1" in
    --arms)  ARMS=$(echo "$2" | tr ',' ' '); shift 2 ;;
    --seeds) SEEDS=$(echo "$2" | tr ',' ' '); shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

n_total=$(( $(echo $ARMS | wc -w) * $(echo $SEEDS | wc -w) ))
printf '%s\n' "matrix: $(echo $ARMS | wc -w) arms x $(echo $SEEDS | wc -w) seeds = $n_total runs, $STEPS steps each"
printf '%s\n' "estimated: $(echo "$n_total $STEPS" | awk '{printf "%.1f", $1*$2*1.24/3600}') GPU-hours at 1.24 s/step"

# Fail fast rather than 20 hours in: every module's self-test must pass before
# a single run starts.
echo; echo "=== preflight ==="
for m in estimator_diag weighted_loss datasets train; do
  printf '%-18s ' "$m"
  if PROJ="$PROJ" "$PY" "$REPO/src/$m.py" --selftest >/dev/null 2>&1; then
    echo ok
  else
    echo FAILED; echo "preflight failed for $m; not starting the matrix" >&2; exit 1
  fi
done

done_n=0; fail_n=0; skip_n=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    name="$(echo "$arm" | tr ':' '_')_seed${seed}"
    out="$RUNS/$name"
    # Resume on the COMPLETION MARKER, never on checkpoint.pt -- that file is
    # written periodically mid-run too, so a run killed by an OOM or a reboot
    # leaves one behind and would be skipped here as finished. done.json is
    # written once, last, and records the step count actually reached; a run
    # whose marker disagrees with the step count we are asking for is re-run.
    if [ -f "$out/done.json" ] \
       && grep -q "\"step\": $STEPS," "$out/done.json" 2>/dev/null; then
      echo "skip  $name (complete at $STEPS steps)"; skip_n=$((skip_n+1)); continue
    fi
    if [ -e "$out/checkpoint.pt" ] && [ ! -f "$out/done.json" ]; then
      echo "redo  $name (partial checkpoint, no completion marker)"
    fi
    echo; echo "########## $name  $(date -Is) ##########"
    t0=$SECONDS
    if PROJ="$PROJ" "$PY" "$REPO/src/train.py" \
         --arm "$arm" --seed "$seed" --steps "$STEPS" --out "$out" \
         > "$LOGS/run_$name.log" 2>&1; then
      echo "ok    $name  $(( (SECONDS-t0)/60 )) min"; done_n=$((done_n+1))
    else
      echo "FAIL  $name  -- see $LOGS/run_$name.log" >&2
      tail -5 "$LOGS/run_$name.log" >&2
      fail_n=$((fail_n+1))
    fi
  done
done

echo; echo "########## MATRIX DONE $(date -Is) ##########"
echo "completed $done_n   skipped $skip_n   failed $fail_n"
[ "$fail_n" -eq 0 ] || exit 1
