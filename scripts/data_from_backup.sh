#!/usr/bin/env bash
# Stage evaluation datasets from the recovered pre-wipe drive.
#
# ~/server_backup is /dev/sda1 (label `bigdata`), a 7200rpm HDD mounted
# read-only. These sets are copied onto the NVMe rather than read in place for
# three reasons: the mount is not guaranteed to survive a reboot, read-only
# means we cannot write derived caches beside the data, and spinning rust is a
# poor place to put a dataloader's random reads.
#
# Nothing here is downloaded -- it is all already on the machine. See
# decision_log.md D41-D43.
set -uo pipefail
source "$(dirname "$0")/data_common.sh"
BACKUP="${BACKUP:-$HOME/server_backup}"

if ! findmnt -n "$BACKUP" >/dev/null 2>&1; then
  echo "ERROR: $BACKUP is not mounted. Mount /dev/disk/by-uuid/" \
       "1787f295-fd6e-41dc-aea3-49ff6f086cad there first." >&2
  exit 1
fi

# name:source:approx_GB
SETS="
ibims1:$BACKUP/premise/ibims1:0.5
nyuv2:$BACKUP/premise/nyuv2:3
bonn:$BACKUP/premise/bonn:32
"

failed=0
for entry in $SETS; do
  name="${entry%%:*}"; rest="${entry#*:}"
  src="${rest%%:*}"; gb="${rest##*:}"
  dest="$DATA/$name"
  if is_done "$dest"; then echo "$name already staged"; continue; fi
  if [ ! -d "$src" ]; then
    echo "WARN: $src missing, skipping $name" >&2; failed=$((failed+1)); continue
  fi
  need_space "$gb"
  say "$name  (~${gb} GB)"
  mkdir -p "$dest"
  # -a preserves times so a partial copy is visible; --delete is deliberately
  # NOT used, because the source is someone else's recovered data and this
  # script must never be able to act on it.
  if rsync -a --info=progress2 "$src"/ "$dest"/; then
    mark_done "$dest"
    printf '%-10s staged  %s\n' "$name" "$(du -sh "$dest" | cut -f1)"
  else
    echo "WARN: rsync failed for $name" >&2; failed=$((failed+1))
  fi
done

say "staged datasets"
for d in ibims1 nyuv2 bonn; do
  [ -d "$DATA/$d" ] && printf '%-10s %s\n' "$d" "$(du -sh "$DATA/$d" | cut -f1)"
done
[ "$failed" -eq 0 ] || { echo "$failed set(s) failed" >&2; exit 1; }
