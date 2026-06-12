# Server setup — Multi-frame consistency training (branch `multiframe`)

End-to-end guide to prepare a fresh Linux + NVIDIA GPU server for the
**multi-frame consistency fine-tune** of D²USt3R (paper/new.tex; plan in
`mfc/PLAN.md`): tuple data on **PointOdyssey + TartanAir (+ Spring)**, the
D²USt3R checkpoint, and the `dust3r` package the training fork builds on.
Sections 1–5 (env, clones, checkpoint, data) are ready to run now; §6 (smoke
tests / training entry) fills in as `mfc/hodge.py` → `mfc/tuples.py` →
`mfc/train_mfc.py` land (PLAN Phases 1–3).

**Hardware:** 1 NVIDIA GPU, the bigger the better — the full run fine-tunes the
D²USt3R decoder+heads on 5-frame tuples (the paper used 4×RTX6000 @ bs 4/GPU;
on one card expect gradient accumulation and ~2× wall-clock). CUDA 12.1 driver.
Disk: ~6 GB code+checkpoint+Sintel, **plus** PointOdyssey (~185 GB) and a
TartanAir slice (tens of GB).

Final layout (everything under one parent dir; **run from `mfc/`**):

```
3d-reconstruction/              # this repo (branch: multiframe)
├── mfc/                        # our code — RUN EVERYTHING FROM HERE
├── DDUSt3R/                    # cloned separately (dust3r pkg + ckpt + train code)
│   ├── checkpoints/ddust3r.pth #   downloaded
│   ├── third_party/raft.py     #   stub (installed)
│   └── sam2/build_sam.py       #   stub (installed)
├── data/
│   ├── training/{clean,depth,camdata_left,invalid,flow,occlusions}/<scene>/  # Sintel (EVAL ONLY)
│   ├── tartanair/<env>/<Easy|Hard>/P0xx/{image_left,depth_left,pose_left.txt}/
│   ├── pointodyssey/{train,val}/<seq>/{rgbs,depths,annot.npz}
│   └── spring/...              # small; loader added in PLAN Phase 2
├── cache/                      # created at runtime (tuple/mask cache)
└── results/                    # created at runtime (logs/checkpoints)
```

---

## 0. (Local, once) push the branch

From your local machine:

```bash
cd /c/Users/Polar/Documents/3d-reconstruction
git push -u origin multiframe
```

`data/`, `cache/`, `results/`, `DDUSt3R/`, `mast3r/`, `*.pth`, `*.zip` are
gitignored — downloaded on the server, not cloned.

---

## 1. Clone the repos

```bash
git clone -b multiframe https://github.com/taroii/3d-reconstruction.git
cd 3d-reconstruction
git clone https://github.com/cvlab-kaist/DDUSt3R.git
git -C DDUSt3R checkout c900005e48c0f5de2ac6df965100e6bd7d3dd5f1   # pin known-good
```

(`croco` is vendored inside DDUSt3R — no `--recursive`. `mast3r` is only needed
if we revive correspondence experiments; skip.)

## 2. Python environment + dependencies

```bash
conda create -n mfc python=3.11 cmake=3.14.0 -y
conda activate mfc
# torch FIRST, CUDA-matched (don't let requirements pull a CPU build):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r mfc/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 3. Install the RAFT/SAM2 stubs

The bundled optimizer hard-imports RAFT and SAM2 at load time; a fresh DDUSt3R
clone lacks these modules:

```bash
python mfc/server/install_stubs.py
```

**Note (PLAN Phase 2):** unlike the spectral line, we DO need a real flow model
for PointOdyssey dynamic masks (no GT flow there). When `tuples.py` lands, a
real RAFT/SEA-RAFT checkpoint replaces the stub for mask *precomputation* only;
training itself never calls flow.

## 4. Download the D²USt3R checkpoint

```bash
mkdir -p DDUSt3R/checkpoints
gdown 1dUy03ohGK2jbzhRLN4HYfDkJIpfGr5lP -O DDUSt3R/checkpoints/ddust3r.pth   # ~4.4 GB
```

The only checkpoint training needs (we fine-tune decoder+heads from it).
DUSt3R/MonST3R baselines come later for eval tables.

---

## 5. Datasets

Per frame the tuple pipeline needs **RGB + GT z-depth + intrinsics + camera
pose**, and flow/masks where the dataset provides them (`mfc/datasets.py::CFG`
holds the roots; `mfc/tuples.py` will consume them). Priorities follow
`mfc/PLAN.md` D6: **PointOdyssey and TartanAir are required** (primary dynamic
supervision and the static anchor, respectively); Spring is small and worth
adding; Sintel is **eval-only** — never sampled for training.

### 5a. Sintel (small; eval + debugging)

As before, plus **`flow/` and `occlusions/`** (both in
`MPI-Sintel-complete.zip`) — needed for eval-time dynamic masks and the E5
diagnostic. If pushing your local copy (PowerShell, scp fallback if rsync's
bundled ssh balks at the cloudflared ProxyCommand):

```powershell
ssh bruinml 'mkdir -p /home/<user>/.../3d-reconstruction/data/training'
cd C:\Users\<you>\...\3d-reconstruction\data\training
scp -r clean depth camdata_left invalid flow occlusions bruinml:/home/<user>/.../3d-reconstruction/data/training/
```

Or re-download on the server (http://sintel.is.tue.mpg.de):
`MPI-Sintel-depth-training-20150305.zip` → `training/{depth,camdata_left}/`;
`MPI-Sintel-complete.zip` → `training/{clean,invalid,flow,occlusions}/`.

### 5b. TartanAir (required — the static anchor)

Tools + download script: <https://github.com/castacks/tartanair_tools>. We need
**left RGB + left depth + poses** (pose files ride along in the trajectory
zips; verify `pose_left.txt` is present after unzip — it's what makes static
tuples supervisable). GT flow exists for TartanAir if we want it for masks,
but static scenes ⇒ M_dyn ≡ 0, so it's not needed.

```bash
git clone https://github.com/castacks/tartanair_tools.git
pip install boto3 colorama
cd tartanair_tools
python download_training.py \
  --output-dir /ABS/PATH/3d-reconstruction/data/tartanair \
  --rgb --depth --only-left --unzip              # add --huggingface if S3 stalls
cd ..
# tree:  data/tartanair/<env>/<Easy|Hard>/P0xx/{image_left,depth_left}/ + pose_left.txt
```

⚠️ All environments = hundreds of GB. Each env unzips independently —
**Ctrl-C after ~6–8 diverse environments**; the loader uses what's on disk.

Loader facts (`datasets.py`): depth is `*_left_depth.npy` float32 z-depth in
meters; intrinsics fixed `fx=fy=320, cx=320, cy=240`; val = ~10% of
trajectories by deterministic hash.

### 5c. PointOdyssey (required — the primary dynamic source)

~185 GB. Hosted on HuggingFace `aharley/pointodyssey` (`train.tar.gz.part{aa..ad}`
~135 GB, `val.tar.gz` 20 GB). Headless/resumable:

```bash
mkdir -p data/pointodyssey && cd data/pointodyssey
huggingface-cli download aharley/pointodyssey --repo-type dataset \
  --include "val.tar.gz" "train.tar.gz.part*" --local-dir .
cat train.tar.gz.part?? > train.tar.gz && rm train.tar.gz.part??
tar xzf val.tar.gz && tar xzf train.tar.gz
cd ../..
# layout: data/pointodyssey/{train,val}/<seq>/{rgbs/, depths/, annot.npz}
```

Loader facts: depth 16-bit PNG, meters = png/65535×1000 (z-depth); intrinsics
AND extrinsics from `annot.npz` per frame. No GT flow → dynamic masks via
off-the-shelf RAFT/SEA-RAFT at tuple-cache time (PLAN Phase 2), the same
recipe D²USt3R itself used for this dataset.

### 5d. Spring (small, optional-but-cheap)

GT flow + dynamic content, ~6k frames (https://spring-benchmark.org). Loader
added in PLAN Phase 2; grab it when the Phase-2 work starts.

---

## 6. Smoke tests + training — filled in by PLAN Phases 1–3

The old `N_φ` sections don't apply on this branch. As the modules land, this
section gains:

1. **G1** (Phase 1): `python -m pytest mfc/test_hodge.py` — differentiable
   harmonic projection vs `synth.py` ground truth + speed benchmark.
2. **Phase 2**: `python tuples.py --dataset <d> --check-sheet` → eyeball masks
   under `results/tuple_sheets/`.
3. **G2 + full run** (Phase 3): `python train_mfc.py ...` — short-run gate,
   then the full fine-tune. Recipe per `mfc/PLAN.md` §3.

Until then, the only meaningful server check is: env builds (§2), checkpoint
loads, and `python -c "import backbone as B; B.load_backbone('d2ust3r')"`
succeeds from `mfc/`.
