#!/usr/bin/env bash
# Download the LEFT-VIEW slice of the Spring training set (the only part Option 2
# needs: RGB + disparity + forward/backward flow + camera data) from DaRUS
# (CC BY 4.0, no login). ~118 GB total; the two flow archives are ~94 GB of it.
#
# Resumable (wget -c) and disk-safe: each archive is extracted then its zip is
# deleted before the next download, so peak extra disk ~= one zip + its extract.
#
#   nohup bash curv/server/download_spring.sh > spring_dl.log 2>&1 &
#   tail -f spring_dl.log
#
# After it finishes, smoke-test the readers (depth AND flow) before any training:
#   cd curv && python sanity_curvature.py --dataset spring --pick 0
set -eu
ROOT=data/spring
BASE=https://darus.uni-stuttgart.de/api/access/datafile
mkdir -p "$ROOT"
cd "$ROOT"

# id : human name (order: small/cheap first so failures surface early)
FILES=(
  "198954 train_cam_data.zip"
  "198961 train_disp1_left.zip"
  "199097 train_frame_left.zip"
  "199011 train_flow_FW_left.zip"
  "199006 train_flow_BW_left.zip"
)

for entry in "${FILES[@]}"; do
  id="${entry%% *}"; name="${entry#* }"
  # skip if this component already looks extracted (any matching dir under train/)
  comp="${name#train_}"; comp="${comp%.zip}"           # e.g. flow_FW_left
  if ls -d train/*/"$comp" >/dev/null 2>&1; then
    echo "=== SKIP $name (already extracted) ==="; continue
  fi
  echo "=== DOWNLOAD $name (id $id) ==="
  # curl, NOT wget: DaRUS 303-redirects to a presigned S3 URL whose AWS
  # signature contains %2F-encoded slashes; wget re-decodes them on the
  # follow-up GET and S3 returns 403. curl -L preserves the URL verbatim.
  # -C - resumes (S3 honours Range); each retry re-fetches a fresh signed URL.
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$name" "$BASE/$id"
  echo "=== UNZIP $name ==="
  unzip -q -o "$name" && rm -f "$name"
done

echo "=== RESULT TREE (one sequence) ==="
seq=$(ls train | head -1)
echo "seq: $seq"; ls "train/$seq"
echo "ALL SPRING DONE"
