#!/usr/bin/env bash
# ETH3D high-res multi-view -- the REAL-DATA zero-shot evaluation set.
#
# Chosen deliberately over Sintel. The ACCV reviewers' standing complaints were
# that the evidence rested on a single synthetic zero-shot benchmark and that
# the method "appears to be mainly applicable to training with synthetic
# datasets" (Reviewer Xcc4). ETH3D answers both: laser-scanned ground truth on
# real scenes, with camera poses, so depth AND pose can be reported from one
# download. Sintel's depth archive is also no longer served by MPI.
#
#   undistorted  5.5 GB  images + calibration + poses
#   scan_eval    1.8 GB  laser ground truth
set -uo pipefail
source "$(dirname "$0")/data_common.sh"
DEST="$DATA/eth3d"
is_done "$DEST" && { echo "eth3d already complete"; exit 0; }
need_space 30
mkdir -p "$DEST/archives"

BASE="https://www.eth3d.net/data"
ARCHIVES="multi_view_training_dslr_undistorted.7z multi_view_training_dslr_scan_eval.7z"

for a in $ARCHIVES; do
  if is_done "$DEST/archives/$a.d"; then echo "$a already fetched"; continue; fi
  say "$a"
  # -c resumes; -P picks the directory. `wget -c -O file` is the combination
  # GNU wget warns against, because -O truncates on some paths.
  wget -4 -c -P "$DEST/archives" "$BASE/$a" || { echo "ERROR: $a failed" >&2; exit 1; }
  mkdir -p "$DEST/archives/$a.d" && mark_done "$DEST/archives/$a.d"
  sleep 15                        # be polite to the shared uplink
done

# 7z extraction: prefer the system binary, fall back to pure-python py7zr since
# we have no root on this machine to apt-install p7zip.
say "extracting"
if command -v 7z >/dev/null 2>&1; then
  for a in $ARCHIVES; do 7z x -y -o"$DEST" "$DEST/archives/$a" >/dev/null || exit 1; done
elif command -v 7za >/dev/null 2>&1; then
  for a in $ARCHIVES; do 7za x -y -o"$DEST" "$DEST/archives/$a" >/dev/null || exit 1; done
else
  "$PY" -c 'import py7zr' 2>/dev/null || "$PIP" install py7zr || {
    echo "ERROR: no 7z and py7zr unavailable" >&2; exit 1; }
  for a in $ARCHIVES; do
    "$PY" - "$DEST/archives/$a" "$DEST" <<'PYEOF'
import sys, py7zr
with py7zr.SevenZipFile(sys.argv[1], "r") as z:
    z.extractall(path=sys.argv[2])
print("extracted", sys.argv[1])
PYEOF
  done
fi

say "layout"
n=$(find "$DEST" -maxdepth 1 -mindepth 1 -type d ! -name archives | wc -l)
echo "$n scene directories"
[ "$n" -ge 10 ] || { echo "ERROR: expected >=10 ETH3D scenes, found $n" >&2; exit 1; }
du -sh "$DEST"
mark_done "$DEST"
