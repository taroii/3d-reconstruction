#!/usr/bin/env bash
# The premise check end to end (notes/premise_check.md). Inference only -- if
# this script ever starts training something, it has left the protocol's scope.
#
# Stages (default: all four, in order):
#   infer    feed-forward per-view depth for every stream -> prediction cache
#   measure  boundary/support/FP over the (eta,tau,beta) grid -> results.json
#   analyze  stats + decision rule -> decision.md, table_main.csv, sensitivity.csv
#   figs     per-scene panels + silhouette point clouds -> figs/
#
#   bash scripts/premise.sh                      # everything
#   bash scripts/premise.sh measure analyze      # re-run analysis only
#   SCENES=40 VIEWS=8 bash scripts/premise.sh
#
# Run `bash scripts/gate.sh` and `python src/analyze.py --preregister` FIRST:
# the thresholds must be committed before any number exists.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-results/premise_check}
# Default to the Tier A sets that the contamination matrix reports clean for every
# Tier 1 model. Add hypersim/eth3d/nyuv2/bonn explicitly -- they are reported as a
# separate, labelled stratum, never folded into the headline.
DATASETS=${DATASETS:-middlebury,infinigen,sintel,spring}
STREAMS=${STREAMS:-vggt_point,vggt_depth,pi3_local}
SCENES=${SCENES:-20}
VIEWS=${VIEWS:-8}
# Sec. 6.5: minimum absolute depth gap (metres) so noise-level steps are not
# counted as occlusion boundaries. Datasets differ in scale -- revisit per dataset.
DELTA_MIN=${DELTA_MIN:-0.05}
STAGES=${*:-infer measure analyze figs}

if [ ! -f "$OUT/decision.md" ]; then
  echo "!! $OUT/decision.md not found."
  echo "!! Pre-register before measuring:  $PY src/analyze.py --preregister --out $OUT"
  exit 1
fi

for st in $STAGES; do
  echo; echo "=========== $st ==========="
  case "$st" in
    infer)   "$PY" src/infer.py   --streams "$STREAMS" --datasets "$DATASETS" \
                                  --scenes "$SCENES" --views "$VIEWS" --out "$OUT" ;;
    measure) "$PY" src/measure.py --streams "$STREAMS" --datasets "$DATASETS" \
                                  --scenes "$SCENES" --views "$VIEWS" --out "$OUT" \
                                  --delta-min "$DELTA_MIN" ;;
    analyze) "$PY" src/analyze.py --decide --out "$OUT" ;;
    figs)    "$PY" src/analyze.py --figs   --out "$OUT" ;;
    *) echo "unknown stage: $st" >&2; exit 2 ;;
  esac
done

echo; echo "Deliverables in $OUT/ (Sec. 8). Look at figs/ by eye before trusting"
echo "any number: if the FP masks do not sit on smeared silhouettes, it is wrong."
