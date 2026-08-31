#!/usr/bin/env bash
# Phase 1 — Diagnosis (notes/diagnosis.md).
#
# STANDING RULE FROM THE PLAN: no fix is attempted in this phase. Nothing here
# modifies a loss with intent to improve, tunes a threshold to strengthen an
# effect, or adds a model without the contamination matrix being re-checked.
#
# Stages (default: all):
#   1a-generality    add DUSt3R + MASt3R, same pipeline, same threshold grid
#   1a-calibration   three rate variants + the K_p histogram, with the §2 gate
#   1b-factorial     the controlled synthetic experiment
#
#   bash scripts/phase1.sh                     # everything
#   bash scripts/phase1.sh 1b-factorial        # just the factorial
#   SEEDS=8 bash scripts/phase1.sh 1b-factorial
set -u
cd "$(dirname "$0")/.."

# A stage that fails must not let the pipeline report success. Without this the
# script printed "wrote .../1a_generality.csv" for a file that did not exist and
# exited 0 with an empty deliverables directory.
FAILED=0
must () {                      # must <label> <command...>
  local label="$1"; shift
  local rc=0
  # Capture the status immediately. `if "$@"; then ...; fi` followed by
  # `local rc=$?` reads the status of the `if`, not of the command -- which made
  # `must` always return 0, so a failed stage still looked like a success and the
  # stages after it ran anyway.
  "$@" || rc=$?
  [ "$rc" -eq 0 ] && return 0
  say "!! FAILED (exit $rc): $label"
  FAILED=$((FAILED+1))
  return "$rc"
}
export PREMISE_DATA_ROOT=${PREMISE_DATA_ROOT:-data}

OUT=${OUT:-results/phase1}
CACHE=${CACHE:-results/premise_check}          # Phase 0's prediction cache
VG=${VG:-~/anaconda3/envs/premise-vggt/bin/python}
P3=${P3:-~/anaconda3/envs/premise-pi3/bin/python}
DS3=${DS3:-~/anaconda3/envs/premise-dust3r/bin/python}
PY=${PY:-~/anaconda3/envs/3d-recon/bin/python}

# Clean-for-every-model sets carry the result; the contaminated ones are named
# separately so a cross-model number can never quietly pool them.
CLEAN=${CLEAN:-middlebury,infinigen,ibims,eth3d}
STRATA=${STRATA:-hypersim}
SCENES=${SCENES:-100}
VIEWS=${VIEWS:-4}
SEEDS=${SEEDS:-5}
STAGES=${*:-1a-generality 1a-calibration 1b-factorial}
# The streams Phase 1 measures. Named once: the recalibration must be given the
# SAME set, because it now drops any view where one of them is missing.
ALL_STREAMS=${ALL_STREAMS:-dust3r_point,mast3r_point,vggt_point,vggt_depth,pi3_local}

say () { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$OUT"

# Reuse Phase 0's predictions rather than recomputing 150+ views per stream.
# HARDLINKS individual files, not symlinked directories: the Phase 0 predictions
# live in TWO caches (the main run and the Hypersim stratum), and a symlinked
# directory can only point at one of them, so the other stratum's views would be
# silently recomputed. View keys are dataset-prefixed, so the two caches never
# collide. Hardlinks cost no disk and leave the originals untouched.
CACHES=${CACHES:-results/premise_check results/premise_check_hypersim}
link_cache () {
  local n=0
  for c in $CACHES; do
    [ -d "$c/preds" ] || continue
    for d in "$c"/preds/*; do
      [ -d "$d" ] || continue
      local s; s=$(basename "$d")
      mkdir -p "$OUT/preds/$s"
      for f in "$d"/*.npz; do
        [ -e "$f" ] || continue
        local b; b=$(basename "$f")
        [ -e "$OUT/preds/$s/$b" ] && continue
        ln "$f" "$OUT/preds/$s/$b" 2>/dev/null || cp "$f" "$OUT/preds/$s/$b"
        n=$((n+1))
      done
    done
  done
  say "reused $n cached predictions from: $CACHES"
}

for st in $STAGES; do
  echo; say "=========== $st ==========="
  case "$st" in

  1a-generality)
    link_cache
    [ -f "$OUT/decision.md" ] || $VG src/analyze.py --preregister --out "$OUT"
    say "contamination matrix (re-checked before adding models, per §6)"
    $PY src/data.py --matrix --out "$OUT"

    must "dust3r inference" $DS3 src/infer.py --streams dust3r_point \
         --datasets "$CLEAN,$STRATA" --scenes $SCENES --views $VIEWS --out "$OUT"
    must "mast3r inference" $DS3 src/infer.py --streams mast3r_point \
         --datasets "$CLEAN,$STRATA" --scenes $SCENES --views $VIEWS --out "$OUT"
    must "vggt inference" $VG src/infer.py --streams vggt_point,vggt_depth \
        --datasets "$CLEAN,$STRATA" --scenes $SCENES --views $VIEWS --out "$OUT"
    must "pi3 inference" $P3 src/infer.py --streams pi3_local \
        --datasets "$CLEAN,$STRATA" --scenes $SCENES --views $VIEWS --out "$OUT"

    if must "measure" $VG src/measure.py --streams "$ALL_STREAMS" \
         --datasets "$CLEAN" --scenes $SCENES --views $VIEWS --out "$OUT" \
         --delta-min 0.05; then
      must "analyze" $VG src/analyze.py --decide --out "$OUT"
      if cp -f "$OUT/table_main.csv" "$OUT/1a_generality.csv" 2>/dev/null; then
        say "wrote $OUT/1a_generality.csv"
      else
        say "!! $OUT/table_main.csv absent - 1a_generality.csv NOT written"
        FAILED=$((FAILED+1))
      fi
    fi
    echo
    echo "  [CONFOUNDED] Any DUSt3R-vs-VGGT difference is consistent with many"
    echo "  causes: backbone, pretraining, training data, pairwise vs multi-view,"
    echo "  resolution, output heads. Record it; do not attribute it. Controlled"
    echo "  attribution happens in 1B."
    ;;

  1a-calibration)
    # 1a-calibration reads the prediction cache, so it needs the links even when
    # run on its own (the documented `bash scripts/phase1.sh 1a-calibration`).
    link_cache
    must "recalibration" $PY src/diagnostics.py --recalibrate --streams "$ALL_STREAMS" \
        --datasets "$CLEAN,$STRATA" --scenes $SCENES --views 2 --out "$OUT"
    ;;

  1b-factorial)
    # §3.4: the predictions are recorded before the run, and --run refuses
    # to start without them.
    [ -f "$OUT/1b_prereg.md" ] || $VG src/factorial.py --preregister --seeds "$SEEDS" --out "$OUT"
    must "factorial" $VG src/factorial.py --run --seeds "$SEEDS" --out "$OUT"
    ;;

  *) echo "unknown stage: $st" >&2; exit 2 ;;
  esac
done

echo
if [ "$FAILED" -gt 0 ]; then
  say "=========== $FAILED STAGE(S) FAILED - deliverables are incomplete ==========="
else
  say "=========== all stages succeeded ==========="
fi
say "deliverables in $OUT:"
ls -la "$OUT" 2>/dev/null | grep -vE '^total|preds' | sed 's/^/  /'
echo
echo "  decision.md selects a Test 2 branch of notes/outcome_tree.md from the"
echo "  variance decomposition, with §0 labels attached. Note the tree's RESIDUAL"
echo "  branch (predictor smoothness) is NOT the decomposition's unexplained"
echo "  bucket (interactions/scene/seed); the two are reported separately."
exit $(( FAILED > 0 ? 1 : 0 ))
