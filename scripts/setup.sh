#!/usr/bin/env bash
# =============================================================================
# Server setup for the VGGT boundary-reweighting study.
#
# Everything this project owns lives under one directory ($PROJ) so it can be
# deleted in one command and never collides with other users of this shared
# machine. Nothing is installed into a shared conda base.
#
#   PROJ/
#   |-- 3d-reconstruction/   this repo (branch: vggt)
#   |-- third_party/vggt/    upstream VGGT clone (not vendored into our repo)
#   |-- ckpt/                model weights
#   |-- data/                datasets (hundreds of GB)
#   `-- runs/                training outputs; only summaries go back to the repo
#
# Stages are independent and idempotent -- rerunning a finished stage is a no-op,
# and a stage that died halfway can simply be rerun.
#
#   scripts/setup.sh env        conda environment + pytorch
#   scripts/setup.sh code       clone VGGT
#   scripts/setup.sh weights    download the VGGT checkpoint
#   scripts/setup.sh data       download datasets (very long; see notes below)
#   scripts/setup.sh verify     report what is present and working
#   scripts/setup.sh all        every stage in order
# =============================================================================
set -euo pipefail

PROJ="${PROJ:-$HOME/taro/vggt-curv}"
ENV_NAME="${ENV_NAME:-vggt-curv}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
PY_VER="${PY_VER:-3.11}"
# cu124 wheels run fine on newer drivers; the box reports driver 595.99 / CUDA 13.2.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

say() { printf '\n=== %s ===\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- stages ------------------------------------------------------------------

stage_env() {
  say "conda environment: $ENV_NAME (python $PY_VER)"
  if [ -x "$PY" ]; then
    echo "already exists: $ENV_PREFIX"
  else
    "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" "python=$PY_VER"
  fi

  say "pytorch"
  # Skip the ~2.5 GB download when a CUDA-enabled torch is already importable.
  if "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "torch already installed with CUDA: $("$PY" -c 'import torch; print(torch.__version__)')"
  else
    "$PIP" install --index-url "$TORCH_INDEX" torch torchvision
  fi

  say "python dependencies"
  "$PIP" install \
    numpy scipy matplotlib pillow tqdm \
    einops safetensors huggingface_hub \
    opencv-python-headless h5py imageio \
    trimesh scikit-image pandas

  echo "environment ready: $ENV_PREFIX"
}

stage_code() {
  say "upstream VGGT"
  mkdir -p "$PROJ/third_party"
  if [ -d "$PROJ/third_party/vggt/.git" ]; then
    echo "already cloned"
  else
    # Shallow: this is a home connection at ~2 MB/s, and we need one commit's
    # worth of code, not the full history.
    git clone --depth 1 https://github.com/facebookresearch/vggt.git "$PROJ/third_party/vggt"
  fi
  # Record the commit so the baseline is reproducible. A shallow clone cannot
  # check out an arbitrary older hash, so this records rather than rewinds; if
  # the recorded hash ever stops matching, that is a signal to re-clone, not
  # something to paper over.
  HEAD_NOW=$(git -C "$PROJ/third_party/vggt" rev-parse HEAD)
  if [ -f "$PROJ/third_party/vggt.commit" ]; then
    WANT=$(cat "$PROJ/third_party/vggt.commit")
    [ "$WANT" = "$HEAD_NOW" ] || echo "WARNING: vggt is at $HEAD_NOW, pinned to $WANT" >&2
  else
    echo "$HEAD_NOW" > "$PROJ/third_party/vggt.commit"
  fi
  echo "vggt at $(cat "$PROJ/third_party/vggt.commit")"
}

stage_weights() {
  say "VGGT checkpoint"
  mkdir -p "$PROJ/ckpt"
  "$PY" - "$PROJ/ckpt" <<'PYEOF'
import sys, pathlib
from huggingface_hub import hf_hub_download
dest = pathlib.Path(sys.argv[1])
p = hf_hub_download("facebook/VGGT-1B", "model.safetensors", local_dir=str(dest))
print("checkpoint:", p, pathlib.Path(p).stat().st_size / 1e9, "GB")
PYEOF
}

stage_data() {
  say "datasets"
  mkdir -p "$PROJ/data"
  cat <<'NOTE'
Datasets are downloaded by their own scripts so each can be run, resumed and
timed independently -- `setup.sh data` only dispatches to them:

  scripts/data_sintel.sh        ~6 GB    zero-shot eval
  scripts/data_tartanair.sh     ~60 GB   training (stop after ~8 environments)
  scripts/data_pointodyssey.sh  ~185 GB  training

Run them individually inside tmux. They are not chained here because a failure
200 GB into a chain is worse than one that stops at its own stage.
NOTE
}

stage_verify() {
  say "verify"
  printf 'PROJ          %s\n' "$PROJ"
  printf 'env           %s\n' "$([ -x "$PY" ] && echo present || echo MISSING)"
  if [ -x "$PY" ]; then
    "$PY" - <<'PYEOF'
import importlib, torch
print(f"torch         {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu           {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
for m in ("numpy", "scipy", "matplotlib", "einops", "huggingface_hub", "cv2", "h5py"):
    try:
        importlib.import_module(m)
        print(f"{m:<14}ok")
    except Exception as e:
        print(f"{m:<14}MISSING ({e})")
PYEOF
  fi
  printf 'vggt          %s\n' "$([ -d "$PROJ/third_party/vggt/.git" ] && echo present || echo MISSING)"
  printf 'checkpoint    %s\n' "$([ -f "$PROJ/ckpt/model.safetensors" ] && du -h "$PROJ/ckpt/model.safetensors" | cut -f1 || echo MISSING)"
  for d in sintel tartanair pointodyssey; do
    printf 'data/%-9s %s\n' "$d" "$([ -d "$PROJ/data/$d" ] && du -sh "$PROJ/data/$d" 2>/dev/null | cut -f1 || echo MISSING)"
  done
  printf 'disk free     %s\n' "$(df -h "$PROJ" | tail -1 | awk '{print $4}')"
}

# --- dispatch ----------------------------------------------------------------

mkdir -p "$PROJ"
case "${1:-all}" in
  env)     stage_env ;;
  code)    stage_code ;;
  weights) stage_weights ;;
  data)    stage_data ;;
  verify)  stage_verify ;;
  all)     stage_env; stage_code; stage_weights; stage_data; stage_verify ;;
  *)       echo "usage: $0 {env|code|weights|data|verify|all}" >&2; exit 2 ;;
esac
