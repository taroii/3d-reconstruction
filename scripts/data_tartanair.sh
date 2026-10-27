#!/usr/bin/env bash
# TartanAir -- the training source.
#
# The bundled `download_training.py` has no per-environment flag: it fetches
# every environment, which is hundreds of GB and impossible on this machine's
# home uplink. The HuggingFace mirror (theairlabcmu/tartanair) exposes the same
# paths file-by-file, so we select exactly the zips we want and nothing else.
#
# Easy trajectories, left camera only: image_left.zip + depth_left.zip per
# environment. Right camera, flow and segmentation are never downloaded.
#
# The five environments are chosen to span the two confounds the estimator
# diagnostic identified, not for variety alone:
#   westerndesert   open outdoor, large within-image depth range  -> the Z^2 bias
#   office          indoor planes seen at many slants             -> the slant bias
#   carwelding      industrial interior, dense fine structure
#   japanesealley   narrow urban, strong occlusion boundaries
#   seasidetown     outdoor urban, mixed depth and structure
# Total ~17.5 GB, about 2.5 h at the ~2 MB/s this link sustains.
set -uo pipefail
source "$(dirname "$0")/data_common.sh"
DEST="$DATA/tartanair"
is_done "$DEST" && { echo "tartanair already complete"; exit 0; }
need_space 60
mkdir -p "$DEST"

ENVS="${TARTANAIR_ENVS:-westerndesert office carwelding japanesealley seasidetown}"

failed=0
for e in $ENVS; do
  if is_done "$DEST/$e"; then echo "$e already done"; continue; fi
  say "$e"

  # --- fetch just this environment's two zips (resumable) -------------------
  if ! "$PY" - "$DEST" "$e" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
dest, env = sys.argv[1], sys.argv[2]
snapshot_download(
    "theairlabcmu/tartanair", repo_type="dataset", local_dir=dest,
    allow_patterns=[f"{env}/Easy/image_left.zip", f"{env}/Easy/depth_left.zip"],
    max_workers=2,          # the link collapses under more concurrency
)
print("fetched", env)
PYEOF
  then
    echo "WARN: $e download failed, continuing with the rest" >&2
    failed=$((failed+1)); continue
  fi

  # --- extract --------------------------------------------------------------
  ok=1
  for z in "$DEST/$e/Easy/image_left.zip" "$DEST/$e/Easy/depth_left.zip"; do
    [ -f "$z" ] || { echo "WARN: missing $z" >&2; ok=0; break; }
    unzip -q -o "$z" -d "$DEST" || { echo "WARN: unzip failed for $z" >&2; ok=0; break; }
  done

  # Only call an environment done once extraction produced the depth arrays the
  # loader needs. A directory existing proves nothing about whether an
  # interrupted extraction finished, and marking on that alone is how a
  # half-populated dataset gets silently blessed as complete.
  if [ "$ok" -eq 1 ] && find "$DEST/$e" -name '*_left_depth.npy' -print -quit | grep -q .; then
    rm -f "$DEST/$e/Easy/image_left.zip" "$DEST/$e/Easy/depth_left.zip"
    mark_done "$DEST/$e"
    printf '%-18s done  %s\n' "$e" "$(du -sh "$DEST/$e" | cut -f1)"
  else
    echo "WARN: $e produced no depth files; leaving archives in place" >&2
    failed=$((failed+1))
  fi
  sleep 20                      # be polite to the shared uplink between envs
done

say "layout"
for e in $ENVS; do
  [ -d "$DEST/$e" ] && printf '%-18s %s\n' "$e" "$(du -sh "$DEST/$e" | cut -f1)"
done
du -sh "$DEST"

if [ "$failed" -eq 0 ]; then
  mark_done "$DEST"
  echo "tartanair complete: $(echo $ENVS | wc -w) environments"
else
  echo "tartanair INCOMPLETE: $failed environment(s) failed; not marking done" >&2
  exit 1
fi
