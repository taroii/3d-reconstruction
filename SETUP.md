# Server setup — Spectral band-routing prior (`N_φ` training)

End-to-end guide to run **`N_φ` training** (the DPT normal head on the frozen
D²USt3R encoder) on a fresh Linux + NVIDIA GPU server. The repo also contains the
alignment-side code (`band_optimizer.py`, `spectral_diag.py`) for the later
go/no-go experiment, but this guide targets the one thing that needs a real
training run.

**Hardware:** 1 NVIDIA GPU, ≥12 GB (trains at batch 4 on a 12 GB RTX 4070; raise
`--bs` on bigger cards). CUDA 12.1-capable driver. ~20 GB disk (checkpoint ~4.4 GB
+ Sintel ~10 GB + normals cache).

The whole project lives under one parent dir. Final layout:

```
3d-reconstruction/              # this repo (branch: new-idea)
├── spectral/                   # our code — RUN EVERYTHING FROM HERE
├── DDUSt3R/                    # cloned separately (bundled dust3r pkg + encoder)
│   ├── checkpoints/ddust3r.pth #   downloaded
│   ├── third_party/raft.py     #   stub (installed)
│   └── sam2/build_sam.py       #   stub (installed)
├── data/training/{clean,depth,camdata_left,invalid}/<scene>/   # Sintel
├── cache/                      # created at runtime (GT-normals cache)
└── results/                    # created at runtime (logs/checkpoints)
```

---

## 0. (Local, once) push the branch

The work is on the `new-idea` branch and is **not committed/pushed yet**. From
your local machine:

```bash
cd /c/Users/Polar/Documents/3d-reconstruction
git add spectral SETUP.md
git add -u                       # stage the sheaf_real->spectral moves & deletions
git commit -m "Spectral band-routing: N_phi training + alignment scaffolding"
git push -u origin new-idea
```

`data/`, `cache/`, `results/`, `DDUSt3R/`, `mast3r/`, and `*.pth` are gitignored —
they are downloaded on the server below, not cloned.

---

## 1. Clone the repos

```bash
git clone -b new-idea https://github.com/taroii/3d-reconstruction.git
cd 3d-reconstruction

# bundled dust3r/croco package + the D²USt3R encoder (pin the known-good commit)
git clone https://github.com/cvlab-kaist/DDUSt3R.git
git -C DDUSt3R checkout c900005e48c0f5de2ac6df965100e6bd7d3dd5f1
```

(`croco` is vendored inside DDUSt3R — no `--recursive` needed. `mast3r` is only
for the later correspondence experiments; skip it for training.)

---

## 2. Python environment + dependencies

```bash
conda create -n spectral python=3.11 cmake=3.14.0 -y
conda activate spectral

# torch FIRST, CUDA-matched (do not let requirements pull a CPU build):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r spectral/requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.5.1+cu121  NVIDIA ...
```

---

## 3. Install the RAFT/SAM2 stubs

The bundled optimizer hard-imports RAFT and SAM2 at load time but never calls
them (we keep `flow_loss_weight=0`). A fresh DDUSt3R clone lacks these modules,
so install the flow-free stubs:

```bash
python spectral/server/install_stubs.py        # writes DDUSt3R/{third_party,sam2}/...
```

---

## 4. Download the D²USt3R checkpoint

```bash
mkdir -p DDUSt3R/checkpoints
gdown 1dUy03ohGK2jbzhRLN4HYfDkJIpfGr5lP -O DDUSt3R/checkpoints/ddust3r.pth
# ~4.4 GB. (Source: DDUSt3R repo README Google Drive link.)
```

> This is the **only** checkpoint training needs (we train a head on the frozen
> D²USt3R encoder). DUSt3R / MonST3R / MASt3R checkpoints are only for the later
> alignment experiments — see the appendix.

---

## 5. Download Sintel data

Training needs, per scene: **RGB** (`clean`), **GT depth** (`depth`),
**intrinsics** (`camdata_left`), and optionally **validity** (`invalid`).

**Fastest, most reliable — copy what you already have locally** (you downloaded
these once already):

```bash
# from your local machine:
rsync -av /c/Users/Polar/Documents/3d-reconstruction/data/training/{clean,depth,camdata_left,invalid} \
      user@server:/path/to/3d-reconstruction/data/training/
```

**Or re-download from the official Sintel site:**

```bash
mkdir -p data && cd data
# depth + camera intrinsics  (sintel-depth: http://sintel.is.tue.mpg.de/depth)
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip
unzip -q MPI-Sintel-depth-training-20150305.zip      # -> training/{depth,camdata_left}/
# RGB (clean/final) + invalid/occlusions  (http://sintel.is.tue.mpg.de/)
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip -q MPI-Sintel-complete.zip                     # -> training/{clean,final,invalid,...}/
cd ..
```

If a `wget` URL has moved, grab the same-named zips from the Sintel download
pages and unzip both into `data/` so they merge under `data/training/`.

**Verify the layout** (paths the code reads):

```bash
ls data/training/clean/alley_1/frame_0001.png \
   data/training/depth/alley_1/frame_0001.dpt \
   data/training/camdata_left/alley_1/frame_0001.cam
```

---

## 6. Verify the setup (smoke tests)

Run everything **from inside `spectral/`** (the code uses paths relative to it):

```bash
cd spectral

# (a) backbone loads + a 3-frame alignment runs (~7 s)
python -c "import backbone as B, sintel as SI; m=B.load_backbone('d2ust3r'); \
  p=SI.frame_paths('../data','alley_1','clean')[:3]; o=B.run_clip(m,p,niter=30); \
  print('run_clip OK', o['localpts'][0].shape)"

# (b) GT normals derive + land on the backbone grid (writes results/normals_sanity/)
python sanity_normals.py --scene alley_1 --idx 1      # expect 'grid-match ... OK'

# (c) the head learns (overfit one batch: ~53° -> ~12°)
python train_nphi.py --overfit
```

If all three pass, the server is ready.

---

## 7. Run training

```bash
cd spectral
mkdir -p ../results
python -u train_nphi.py --epochs 30 --bs 4 2>&1 | tee ../results/nphi_train.log
```

- **First epoch is slow**: it computes and caches GT normals for all ~1064 frames
  into `../cache/<scene>/normals/`. Later epochs are much faster.
- Held-out split is **by scene** (`ambush_6, cave_4, market_5, temple_3` are val).
- Best checkpoint (lowest val MAE) is saved to
  `spectral/checkpoints/nphi_d2ust3r.pth`.

**Useful flags:** `--bs` (raise on bigger GPUs), `--lr` (default 1e-4),
`--epochs`, `--alpha` (confidence-loss weight, default 0.2),
`--limit N` (cap frames/scene for a quick dry run).

---

## 8. ⚠️ Known issue: `N_φ` overfits on Sintel-only data

In the local run, **train MAE falls to ~16° but held-out val MAE plateaus at
~37–40°** (best ~37° early, then drifts up — classic overfitting to the 19 train
scenes). The paper trained on four datasets (BlinkVision, PointOdyssey, TartanAir,
Spring); **we only have Sintel (23 scenes)**, which is too narrow.

A ~37° held-out normal error is the §3 "sanity floor" and is **borderline for the
downstream go/no-go to be interpretable**. Before trusting `N_φ`, consider:

- **More data** (the real fix): add some of the paper's training datasets.
- **Augmentation** (color/flip/crop) and stronger **weight decay / dropout** in
  the head.
- **Early stopping** — the best val is reached within the first few epochs.
- A **lighter head** (less capacity to memorize) — see `NormalHead(... feature_dim=)`.

Report `N_φ`'s held-out angular error alongside any downstream result.

---

## Appendix — extras for the later alignment experiments (not needed to train)

```bash
# correspondence caching (match_mast3r.py) uses a separate mast3r clone:
git clone https://github.com/naver/mast3r.git
# + its checkpoint (MASt3R metric) and the DUSt3R/MonST3R checkpoints for the
#   baselines, into DDUSt3R/checkpoints/ and mast3r/checkpoints/.
```

Filenames the loader expects (`spectral/backbone.py::CKPT`):
`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`,
`MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth`, `ddust3r.pth`.
