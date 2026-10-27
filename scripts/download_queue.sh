#!/usr/bin/env bash
# =============================================================================
# Sequential download queue.
#
# The server is on a home connection: ~2 MB/s to HuggingFace, ~4 MB/s to MPI,
# and it collapses to ~70 KB/s when several transfers compete. Everything here
# therefore runs STRICTLY ONE AT A TIME, in priority order, with a short pause
# between jobs so other users of the machine are not starved.
#
# Each job is idempotent, so the queue can be killed and restarted at any point
# and will resume where it stopped rather than re-fetching what is already on
# disk.
#
#   scripts/download_queue.sh            run the whole queue
#   scripts/download_queue.sh weights    run one job
# =============================================================================
set -uo pipefail                 # NOT -e: one failed job must not kill the queue
source "$(dirname "$0")/data_common.sh"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
PAUSE="${PAUSE:-60}"

run_job() {
  local name="$1"; shift
  printf '\n\n########## %s  (start %s) ##########\n' "$name" "$(date -Is)"
  local t0=$SECONDS
  "$@"
  local rc=$? dt=$((SECONDS - t0))
  printf '########## %s  rc=%d  %dm%ds ##########\n' "$name" "$rc" $((dt/60)) $((dt%60))
  return $rc
}

job_weights()   { bash "$SCRIPTS/setup.sh" weights; }
job_tartanair() { bash "$SCRIPTS/data_tartanair.sh"; }
job_eth3d()     { bash "$SCRIPTS/data_eth3d.sh"; }

# Priority order. Weights first: nothing -- not even the memory probe -- can run
# without them, and they are only ~5 GB. Training data next, evaluation last.
QUEUE="${*:-weights tartanair eth3d}"
FAILED=""

for j in $QUEUE; do
  case "$j" in
    weights)   run_job weights   job_weights   || FAILED="$FAILED weights" ;;
    tartanair) run_job tartanair job_tartanair || FAILED="$FAILED tartanair" ;;
    eth3d)     run_job eth3d     job_eth3d     || FAILED="$FAILED eth3d" ;;
    *) echo "unknown job: $j" >&2; FAILED="$FAILED $j" ;;
  esac
  # Pause between jobs, but not after the last one.
  [ "$j" = "${QUEUE##* }" ] || { echo "--- pausing ${PAUSE}s ---"; sleep "$PAUSE"; }
done

printf '\n\n########## QUEUE DONE %s ##########\n' "$(date -Is)"
# A queue that runs for hours must not make you scroll back through its output
# to find out whether anything failed.
if [ -n "$FAILED" ]; then
  echo "FAILED JOBS:$FAILED" >&2
else
  echo "all jobs succeeded"
fi
bash "$SCRIPTS/setup.sh" verify
