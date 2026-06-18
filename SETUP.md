# Server setup — Curvature-Inclusive D²USt3R (branch `curvature`)

End-to-end guide to prepare a fresh Linux + NVIDIA GPU server for the
**curvature-augmented fine-tune** of D²USt3R (plan in `curv/PLAN.md`): a discrete
per-pixel curvature signal computed for free from the GT pointmap, added to
D²USt3R training as a loss reweighting (Option 2) and/or a new DPT head
(Option 1). Training data: **PointOdyssey + TartanAir**; Sintel is **eval-only**.
Sections 1–5 (env, clones, checkpoint, data) run now; §6 (smoke tests / training)
follows `curv/PLAN.md` Phases 0–2.

**Hardware:** 1 NVIDIA GPU, the bigger the better — we fine-tune the D²USt3R
decoder + DPT heads from its checkpoint. CUDA 12.1 driver. Disk: ~6 GB
code+checkpoint+Sintel, **plus** PointOdyssey (~185 GB) and a TartanAir slice
(tens of GB).

Final layout (everything under one parent dir; **run from `curv/`**):

```
3d-reconstruction/              # this repo (branch: curvature)
├── curv/                       # our code — RUN EVERYTHING FROM HERE
├── DDUSt3R/                    # cloned separately (dust3r pkg + ckpt + train code)
│   ├── checkpoints/ddust3r.pth #   downloaded
│   ├── third_party/raft.py     #   stub (installed)
│   └── sam2/build_sam.py       #   stub (installed)
├── data/
│   ├── training/{clean,depth,camdata_left,invalid,flow,occlusions}/<scene>/  # Sintel (EVAL ONLY)
│   ├── tartanair/<env>/<Easy|Hard>/P0xx/{image_left,depth_left,pose_left.txt}/
│   ├── pointodyssey/{train,val}/<seq>/{rgbs,depths,annot.npz}
│   └── spring/train/<seq>/{frame_left,disp1_left,cam_data}/
├── cache/                      # created at runtime (curvature-target cache)
└── results/                    # created at runtime (logs/checkpoints)
```

---

## 0. (Local, once) push the branch

From your local machine:

```bash
cd /c/Users/Polar/Documents/3d-reconstruction
git push -u origin curvature
```

`data/`, `cache/`, `results/`, `DDUSt3R/`, `mast3r/`, `*.pth`, `*.zip` are
gitignored — downloaded on the server, not cloned.

---

## 1. Clone the repos

```bash
git clone -b curvature https://github.com/taroii/3d-reconstruction.git
cd 3d-reconstruction
git clone https://github.com/cvlab-kaist/DDUSt3R.git
git -C DDUSt3R checkout c900005e48c0f5de2ac6df965100e6bd7d3dd5f1   # pin known-good
```

(`croco` is vendored inside DDUSt3R — no `--recursive`. `mast3r` is not needed.)

## 2. Python environment + dependencies

```bash
conda create -n curv python=3.11 cmake=3.14.0 -y
conda activate curv
# torch FIRST, CUDA-matched (don't let requirements pull a CPU build):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r curv/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 3. Install the RAFT/SAM2 stubs

The bundled optimizer hard-imports RAFT and SAM2 at load time; a fresh DDUSt3R
clone lacks these modules:

```bash
python curv/server/install_stubs.py
```

(Curvature training is flow-free — these only satisfy the import.)

## 4. Download the D²USt3R checkpoint

```bash
mkdir -p DDUSt3R/checkpoints
gdown 1dUy03ohGK2jbzhRLN4HYfDkJIpfGr5lP -O DDUSt3R/checkpoints/ddust3r.pth   # ~4.4 GB
```

We fine-tune decoder + heads from this. DUSt3R/MonST3R baselines come later for
eval tables.

---

## 5. Datasets

Per frame curvature needs **GT z-depth + intrinsics** (poses too, for the full
D²USt3R pair loss). Training mix follows `curv/PLAN.md`: **TartanAir +
PointOdyssey + Spring** (3 of D²USt3R's 5 training datasets); Sintel is
**eval-only** — never sampled for training.

### 5a. Sintel (small; eval + debugging)

```powershell
# push your local copy, or re-download on the server (http://sintel.is.tue.mpg.de):
#   MPI-Sintel-depth-training-20150305.zip -> training/{depth,camdata_left}/
#   MPI-Sintel-complete.zip                -> training/{clean,invalid,flow,occlusions}/
```

### 5b. TartanAir (required — the static anchor)

Tools: <https://github.com/castacks/tartanair_tools>. We need **left RGB + left
depth + poses**.

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

⚠️ All environments = hundreds of GB. **Ctrl-C after ~6–8 diverse environments**;
the loader uses what's on disk.

Loader facts (`curv/datasets.py`): depth is `*_left_depth.npy` float32 z-depth in
meters; intrinsics fixed `fx=fy=320, cx=320, cy=240`; val = whole held-out
environments (every 5th in sorted order).

### 5c. PointOdyssey (required — the primary dynamic source)

~185 GB. HuggingFace `aharley/pointodyssey`. Headless/resumable:

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
from `anno.npz['intrinsics'][idx]` (note: PO names it `anno.npz`, not `annot.npz`;
seqs without it are skipped); official `train/` vs `val/` split.

### 5d. Spring (small — ~6k frames, robustness/comprehensiveness)

High-res synthetic stereo (https://spring-benchmark.org; data on DARUS
https://darus.uni-stuttgart.de doi:10.18419/darus-3376). Download the **train**
split (test has no GT) and extract to `data/spring/train/`:

```bash
# (download spring train archive from DARUS, then)
mkdir -p data/spring && tar xf spring_train.tar -C data/spring   # -> data/spring/train/<seq>/
```

Loader facts (`curv/spring.py`): left camera only; disparity `.dsp5` is HDF5
(key `disparity`), GT stored at 2× → subsampled `[::2,::2]` to 1920×1080 with NO
value rescale; depth `Z = fx·B/d`, baseline **B = 0.065 m**; zero disparity = sky
→ dropped; `K` from `cam_data/intrinsics.txt` (rows `fx fy cx cy`). Needs `h5py`
(in `requirements.txt`). val = whole held-out sequences (every 8th, sorted).
**Smoke-test the loader first:** `python sanity_curvature.py --dataset spring`.

---

## 6. Smoke tests + training — `curv/PLAN.md` Phases 0–2

Run from inside `curv/`.

1. **Phase 0** (curvature correctness):
   `python sanity_curvature.py` — synthetic plane/bump/saddle sign checks.
   `python sanity_curvature.py --dataset tartanair` (also `pointodyssey`,
   `spring`) — dump a real curvature map (fires on edges, empty across cliffs).
   This is also the per-dataset loader smoke-test.
2. **Phase 1.5** (head learnability gate):
   `python train_curv.py --overfit` → loss/MAE should drop sharply.
   `python train_curv.py --datasets tartanair,pointodyssey,spring \
       --max-per-dataset 4000 --epochs 20`.
3. **Phase 1 / Phase 2** (the actual augmentation): integrate the curvature
   reweighting (Option 2) and/or `CurvatureHead` + `curvature_conf_loss`
   (Option 1) into the **DDUSt3R** pair-training loop, importing from `curv/`.
   Recipe per `curv/PLAN.md`.

Until the env is built, the minimal server check is: env builds (§2), checkpoint
loads, and `python -c "import backbone as B; B.load_backbone('d2ust3r')"`
succeeds from `curv/`.
