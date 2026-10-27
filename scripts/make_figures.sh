#!/usr/bin/env bash
# =============================================================================
# Every figure and table in the paper, from result files, in one command.
#
# Three stages, each skippable and each resumable:
#
#   diag   the estimator diagnostic -> results/estimator_diag/{report.md,
#          diag.csv,diag.json}. No GPU, no data, ~1 min: analytic surfaces
#          with closed-form curvature.
#   eval   scores the base checkpoint and every COMPLETE run in $RUNS,
#          appending to results/eval.csv. Skips arm/seed pairs already in the
#          CSV, so adding one new arm costs one evaluation rather than all of
#          them.
#   figs   results/* -> paper/figures/*.{tex,pdf,png}
#
#   scripts/make_figures.sh              all three
#   scripts/make_figures.sh figs         re-draw only, no GPU needed
#   scripts/make_figures.sh eval figs    after the matrix finishes
#   FORCE=1 scripts/make_figures.sh eval re-score everything from scratch
#
# Nothing here recomputes a metric. If a figure and a table disagree, one of
# them has a bug; neither is a judgement call.
# =============================================================================
set -uo pipefail
PROJ="${PROJ:-$HOME/taro/vggt-curv}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${CONDA_ROOT:-$HOME/miniconda3}/envs/${ENV_NAME:-vggt-curv}/bin/python"
RUNS="${RUNS:-$PROJ/runs}"
RES="$REPO/results"
FIGS="$REPO/paper/figures"
EVAL="$RES/eval.csv"
LIMIT="${LIMIT:-150}"
DATASETS="${DATASETS:-ibims1,nyuv2,bonn}"
mkdir -p "$RES" "$FIGS"

STAGES="${*:-diag eval figs}"
has() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
say() { printf '\n========== %s ==========\n' "$1"; }

# -- self-tests first. A figure built by broken code is worse than no figure,
# -- because it looks finished.
say "selftests"
for m in estimator_diag evaluate figures; do
  if PROJ="$PROJ" "$PY" "$REPO/src/$m.py" --selftest >/dev/null 2>&1; then
    printf '  %-18s ok\n' "$m"
  else
    printf '  %-18s FAILED\n' "$m"
    PROJ="$PROJ" "$PY" "$REPO/src/$m.py" --selftest
    exit 1
  fi
done

if has diag; then
  say "diagnostic"
  PROJ="$PROJ" "$PY" "$REPO/src/estimator_diag.py" \
       --analysis all --out "$RES/estimator_diag" || exit 1
fi

if has eval; then
  say "evaluation"
  [ -n "${FORCE:-}" ] && rm -f "$EVAL"
  # Already-scored arm/seed pairs, so re-running this stage after adding one
  # arm does not re-score the other twenty-nine.
  seen() {
    [ -f "$EVAL" ] || return 1
    awk -F, -v a="$1" -v s="$2" 'NR>1 && $1==a && $2==s {found=1} END{exit !found}' "$EVAL"
  }
  if seen base -1; then
    echo "skip  base (already scored)"
  else
    PROJ="$PROJ" "$PY" "$REPO/src/evaluate.py" --ckpt base \
         --datasets "$DATASETS" --limit "$LIMIT" --out "$RES" || exit 1
  fi
  n_sk=0; n_ev=0; n_part=0
  for d in "$RUNS"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    # Only complete runs. A directory holding a mid-run periodic checkpoint
    # would otherwise be scored as a finished arm.
    if [ ! -f "$d/done.json" ]; then
      echo "skip  $name (no completion marker -- not finished)"
      n_part=$((n_part+1)); continue
    fi
    arm=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['arm'])" \
          "$d/done.json" 2>/dev/null) || continue
    seed=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['seed'])" \
           "$d/done.json" 2>/dev/null) || continue
    if seen "$arm" "$seed"; then
      echo "skip  $name (already scored)"; n_sk=$((n_sk+1)); continue
    fi
    echo "eval  $name  ($arm seed $seed)"
    PROJ="$PROJ" "$PY" "$REPO/src/evaluate.py" --ckpt "$d" \
         --datasets "$DATASETS" --limit "$LIMIT" --out "$RES" || exit 1
    n_ev=$((n_ev+1))
  done
  echo "evaluated $n_ev   already scored $n_sk   unfinished $n_part"
fi

if has figs; then
  say "figures"
  # Written as a real if: `[ -f x ] && cmd || echo` prints the "missing file"
  # message when the COMMAND fails too, which would hide a broken table behind
  # a message saying there was no data.
  if [ -f "$EVAL" ]; then
    PROJ="$PROJ" "$PY" "$REPO/src/figures.py" table_results \
         --eval "$EVAL" --out "$FIGS" || exit 1
  else
    echo "  no $EVAL yet -- skipping results table"
  fi
  if [ -f "$RES/estimator_diag/diag.json" ]; then
    PROJ="$PROJ" "$PY" "$REPO/src/figures.py" fig_diagnostic \
         --diag "$RES/estimator_diag/diag.json" --out "$FIGS" || exit 1
    PROJ="$PROJ" "$PY" "$REPO/src/figures.py" table_diagnostic \
         --diag "$RES/estimator_diag/diag.json" --out "$FIGS" || exit 1
  else
    echo "  no diag.json yet -- run the diag stage"
  fi
  say "written"
  ls -la "$FIGS"
fi
