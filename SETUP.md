# Server setup — Spectral band-routing prior (`N_φ` training)

End-to-end guide to run **`N_φ` training** (the DPT normal head on the frozen
D²USt3R encoder) on a fresh Linux + NVIDIA GPU server, on **Sintel + TartanAir +
PointOdyssey** (multi-dataset, to get a normal prior that generalizes — Sintel
alone overfits; see §9). The repo also holds the alignment-side code for the
later go/no-go, but this guide targets the training run.

**Hardware:** 1 NVIDIA GPU ≥12 GB (trains at `--bs 4` on a 12 GB card; raise on
bigger). CUDA 12.1 driver. Disk: ~6 GB code+checkpoint+Sintel, **plus** whatever
slice of TartanAir / PointOdyssey you pull (tens–hundreds of GB if greedy — start
small, §5).

Final layout (everything under one parent dir; **run from `spectral/`**):

```
3d-reconstruction/              # this repo (branch: new-idea)
├── spectral/                   # our code — RUN EVERYTHING FROM HERE
├── DDUSt3R/                    # cloned separately (dust3r pkg + encoder)
│   ├── checkpoints/ddust3r.pth #   downloaded
│   ├── third_party/raft.py     #   stub (installed)
│   └── sam2/build_sam.py       #   stub (installed)
├── data/
│   ├── training/{clean,depth,camdata_left,invalid}/<scene>/      # Sintel
│   ├── tartanair/<env>/<Easy|Hard>/P0xx/{image_left,depth_left}/ # TartanAir
│   └── pointodyssey/{train,val}/<seq>/{rgbs,depths,annot.npz}    # PointOdyssey
├── cache/                      # created at runtime (GT-normals cache)
└── results/                   # created at runtime (logs/checkpoints)
```

---

## 0. (Local, once) push the branch

The work is on `new-idea` and is **not committed/pushed yet**. From your local
machine:

```bash
cd /c/Users/Polar/Documents/3d-reconstruction
git add spectral SETUP.md
git add -u                       # stage the sheaf_real->spectral moves & deletions
git commit -m "Spectral band-routing: multi-dataset N_phi training + scaffolding"
git push -u origin new-idea
```

`data/`, `cache/`, `results/`, `archive/`, `DDUSt3R/`, `mast3r/`, `*.pth`, `*.zip`
are gitignored — downloaded on the server, not cloned.

---

## 1. Clone the repos

```bash
git clone -b new-idea https://github.com/taroii/3d-reconstruction.git
cd 3d-reconstruction
git clone https://github.com/cvlab-kaist/DDUSt3R.git
git -C DDUSt3R checkout c900005e48c0f5de2ac6df965100e6bd7d3dd5f1   # pin known-good
```

(`croco` is vendored inside DDUSt3R — no `--recursive`. `mast3r` is only for the
later correspondence experiments; skip for training.)

---

## 2. Python environment + dependencies

```bash
conda create -n spectral python=3.11 cmake=3.14.0 -y
conda activate spectral
# torch FIRST, CUDA-matched (don't let requirements pull a CPU build):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r spectral/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

---

## 3. Install the RAFT/SAM2 stubs

The bundled optimizer hard-imports RAFT and SAM2 at load time but never calls
them (we keep `flow_loss_weight=0`); a fresh DDUSt3R clone lacks these modules:

```bash
python spectral/server/install_stubs.py
```

---

## 4. Download the D²USt3R checkpoint

```bash
mkdir -p DDUSt3R/checkpoints
gdown 1dUy03ohGK2jbzhRLN4HYfDkJIpfGr5lP -O DDUSt3R/checkpoints/ddust3r.pth   # ~4.4 GB
```

The **only** checkpoint training needs (we train a head on the frozen D²USt3R
encoder). DUSt3R/MonST3R/MASt3R are only for the later baselines (appendix).

---

## 5. Datasets

`N_φ` needs, per frame, only **RGB + GT z-depth + camera intrinsics** — it derives
normals itself (`datasets.py` → `normals.normals_on_grid_from_depth_K`). Dataset
roots live in `spectral/datasets.py::CFG`; the on-disk layouts below are what the
loaders expect. **You don't need all of each dataset** — training caps frames per
dataset with `--max-per-dataset`, so a modest slice is enough.

### 5a. Sintel (small, ~2.5 GB needed)

```bash
# easiest: copy what you already have locally
rsync -av /c/Users/Polar/Documents/3d-reconstruction/data/training/{clean,depth,camdata_left,invalid} \
      user@server:/path/to/3d-reconstruction/data/training/
# or re-download (http://sintel.is.tue.mpg.de):
#   MPI-Sintel-depth-training-20150305.zip  -> training/{depth,camdata_left}/
#   MPI-Sintel-complete.zip                 -> training/{clean,invalid,...}/
```

### 5b. TartanAir

Tools + download script: <https://github.com/castacks/tartanair_tools>. We need
only the **left RGB + left depth** modalities. Download a handful of environments
to start (each env is several GB):

```bash
mkdir -p data/tartanair
# via tartanair_tools (downloads per-environment zips):
python tartanair_tools/download_training.py --output-dir data/tartanair \
    --rgb --depth --only-left
# unzip so the tree is:  data/tartanair/<env>/<Easy|Hard>/P0xx/{image_left,depth_left}/
```

Loader facts (already handled in `datasets.py`): depth is `*_left_depth.npy`
float32 **z-depth in meters**; intrinsics are fixed `fx=fy=320, cx=320, cy=240`;
val = ~10% of trajectories held out by a deterministic hash.

### 5c. PointOdyssey

Site / download: <https://pointodyssey.com> (or repo
<https://github.com/y-zheng18/point_odyssey>). Download **both `train/` and
`val/`** (val is the held-out split):

```bash
mkdir -p data/pointodyssey
# unzip so the tree is:
#   data/pointodyssey/{train,val}/<seq>/{rgbs/, depths/, annot.npz}
```

Loader facts: depth is 16-bit PNG, **meters = png/65535×1000** (z-depth);
intrinsics from `annot.npz['intrinsics'][frame_idx]`.

---

## 6. Verify the setup (smoke tests)

Run **from inside `spectral/`**. Do the per-dataset normals check for **each
dataset you downloaded** — it confirms the loader, the depth/intrinsics decode,
and that GT normals land on the backbone grid:

```bash
cd spectral
# (a) backbone loads + a 3-frame alignment runs (~7 s)
python -c "import backbone as B, sintel as SI; m=B.load_backbone('d2ust3r'); \
  p=SI.frame_paths('../data','alley_1','clean')[:3]; o=B.run_clip(m,p,niter=30); \
  print('run_clip OK', o['localpts'][0].shape)"

# (b) per-dataset GT-normals + grid-match (expect 'grid-match ... OK', sane valid%)
python sanity_normals.py --dataset sintel       --index 0
python sanity_normals.py --dataset tartanair    --index 0      # if downloaded
python sanity_normals.py --dataset pointodyssey --index 0      # if downloaded
#   -> writes results/normals_sanity/<dataset>_*.png; eyeball that walls/ground
#      are flat-colored and curved surfaces show smooth gradients.

# (c) the head learns (overfit one batch: ~54° -> ~13°)
python train_nphi.py --overfit
```

If a dataset's check prints **MISMATCH** or "no frames found", fix its `CFG` root
/ directory layout before training (the message tells you which path it checked).

---

## 7. Run training

```bash
cd spectral
mkdir -p ../results
python -u train_nphi.py \
    --datasets sintel,tartanair,pointodyssey \
    --max-per-dataset 6000 \
    --epochs 30 --bs 4 \
    2>&1 | tee ../results/nphi_train.log
```

- **First epoch is slow**: it computes + caches GT normals for every selected
  frame into `../cache/nphi_normals/<dataset>/<scene>/<idx>.npz`. Later epochs are
  fast.
- `--max-per-dataset` evenly subsamples each set (balances huge synthetic data
  against small Sintel). Tune it to your disk/time budget; `None` uses all.
- Val MAE is reported **overall and per-dataset** each epoch; best checkpoint →
  `spectral/checkpoints/nphi_sintel_tartanair_pointodyssey.pth`.
- **Flags:** `--bs`, `--lr` (1e-4), `--epochs`, `--alpha` (conf-loss weight 0.2),
  `--datasets`, `--max-per-dataset`, `--seed`.

**Baseline to beat:** the Sintel-only run (`archive/spectral-phase1/`) plateaus at
**val MAE ≈ 37°**. Watch whether the multi-dataset per-dataset val MAEs come in
materially lower and, crucially, whether the held-out numbers stop drifting up.

---

## 8. ⚠️ Why multi-dataset (the Sintel-only ceiling)

The first run trained on **Sintel only (23 scenes)** and overfit: train MAE → 16°
but held-out val plateaued at **~37° within a couple of epochs**, then drifted up.
That's too narrow a distribution. TartanAir + PointOdyssey add large, diverse
synthetic scenes with the same `depth + K → normals` recipe, which is the fix.

A held-out normal error in the **low-to-mid 20s°** would make the downstream
go/no-go interpretable. If val MAE is still ~35°+ after adding both datasets,
revisit: more data variety, augmentation (color/flip/crop), stronger weight decay,
or a lighter head (`NormalHead(... feature_dim=...)`). Always report `N_φ`'s
held-out angular error next to any downstream result.

---

## Appendix — extras for the later alignment experiments (not needed to train)

```bash
git clone https://github.com/naver/mast3r.git     # correspondence caching
# + DUSt3R / MonST3R / MASt3R checkpoints into the respective checkpoints/ dirs.
```

Filenames the loader expects (`spectral/backbone.py::CKPT`):
`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`,
`MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth`, `ddust3r.pth`.
