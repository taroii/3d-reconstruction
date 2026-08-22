#!/usr/bin/env bash
# Day 1 (notes/premise_check.md Sec. 4 + Sec. 11): validate the metric on cases
# with known answers, inventory what is actually on disk, then run the
# measurability gate. This is cheap and it can END THE STUDY -- a dataset whose
# boundaries have been masked away cannot answer the question, and if every
# dataset fails, the outcome is NO-GO on measurability grounds.
#
#   bash scripts/gate.sh
#   PY=~/anaconda3/envs/3d-recon/bin/python DATASETS=tartanair bash scripts/gate.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-results/premise_check}
DATASETS=${DATASETS:-tartanair,pointodyssey,spring,sintel,hypersim,middlebury,infinigen,eth3d,nyuv2,bonn}
SCENES=${SCENES:-8}
VIEWS=${VIEWS:-4}

echo "=== 1/4  metric self-test (hand-built cases with known answers) ==="
"$PY" src/fpmetrics.py --selftest

echo; echo "=== 2/4  contamination matrix (Sec. 4: do this before anything else) ==="
"$PY" src/data.py --matrix --out "$OUT"

echo; echo "=== 3/4  data inventory + loader smoke test ==="
"$PY" src/data.py --list --verify --datasets "$DATASETS" --out "$OUT" || \
  echo "  !! a loader failed to decode -- fix before measuring"

echo; echo "=== 4/4  measurability gate -> $OUT/gate.csv ==="
rc=0
"$PY" src/data.py --gate --datasets "$DATASETS" --scenes "$SCENES" \
      --views "$VIEWS" --out "$OUT" || rc=$?

echo; echo "Gate done. Any dataset with frac_boundary_valid < 0.5 is unusable and"
echo "must be dropped from the study (record it -- that is a result, not a bug)."
exit $rc
