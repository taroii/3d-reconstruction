# Shared settings for the dataset download scripts. Sourced, not executed.
# Every script here must be safe to re-run: they are long, they will be
# interrupted, and re-downloading 185 GB because a script was not resumable is
# the kind of mistake that costs a day.
PROJ="${PROJ:-$HOME/taro/vggt-curv}"
ENV_NAME="${ENV_NAME:-vggt-curv}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
PIP="$CONDA_ROOT/envs/$ENV_NAME/bin/pip"
DATA="$PROJ/data"
LOGS="$PROJ/logs"
mkdir -p "$DATA" "$LOGS"

say()  { printf '\n=== %s ===\n' "$*"; }
done_marker() { echo "$1/.complete"; }
is_done() { [ -f "$(done_marker "$1")" ]; }
mark_done() { date -Is > "$(done_marker "$1")"; }
need_space() {   # need_space <gb> -- refuse to start a download that cannot finish
  local need=$1 avail
  avail=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc '0-9')
  if [ "$avail" -lt "$need" ]; then
    echo "ERROR: need ~${need} GB free, have ${avail} GB" >&2; exit 1
  fi
  echo "disk check ok: ${avail} GB free, need ~${need} GB"
}
