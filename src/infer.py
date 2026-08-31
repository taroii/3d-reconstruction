r"""
Feed-forward inference -> cached per-view depth maps. Three prediction streams
(Sec. 3), all raw model output: no bundle adjustment, no global alignment, no
filtering, no confidence thresholding.

    vggt_point   VGGT pointmap branch  (world_points -> z)
    vggt_depth   VGGT depth branch     (depth head)
    pi3_local    pi3 local pointmap    (per-view own frame -> z)

FRAME HANDLING. Every view is run as its OWN forward pass (sequence length 1).
With a single input view a model's reference frame IS that view's camera frame,
so `world_points` z is already a per-view depth and no predicted pose is ever
applied. This is the cheapest possible way to honour the protocol's frame note:
pose error cannot leak into the measurement because pose is never used.

RESOLUTION / PADDING (Sec. 6.2-6.3). Images are resized so the long side is 518
and padded to a multiple of 14; the pad is recorded and CROPPED OUT of the
prediction before caching, so padded borders never enter any statistic. Caching
is at model resolution; the nearest-neighbour upsample to GT resolution happens
once, in `load_pred`. Bilinear is never used anywhere -- it manufactures exactly
the artifact under study.

Caches to <out>/preds/<stream>/<view key>.npz. Re-running skips existing files.

    python src/infer.py --streams vggt_point,vggt_depth,pi3_local \
        --datasets tartanair,sintel --scenes 20 --views 8
"""
from __future__ import annotations

import os
import glob
import json
import argparse

import numpy as np

import data as D

MODEL_LONG_SIDE = 518
PATCH = 14
# DUSt3R is a different backbone lineage (CroCo ViT-L/16, 512px checkpoint), so it
# needs its own input geometry. Everything else about the measurement is identical.
GEOM = {"vggt": (518, 14), "pi3": (518, 14), "dust3r": (512, 16), "mast3r": (512, 16)}
DUST3R_REPO = os.path.expanduser("~/taro/premise_setup/dust3r")
MAST3R_REPO = os.path.expanduser("~/taro/premise_setup/mast3r")
DUST3R_CKPT = "/mnt/data/premise/models/dust3r/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
MAST3R_CKPT = ("/mnt/data/premise/models/mast3r/"
               "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
_MODELS: dict[str, object] = {}
_FWD_ERRORS: dict[str, int] = {}
_NO_RGB: list[str] = []


# --------------------------------------------------------------------------- #
# preprocessing / geometry bookkeeping
# --------------------------------------------------------------------------- #
# Padding colour per model family. VGGT's released load_and_preprocess_images
# pads WHITE; padding black instead puts a hard synthetic edge at the image
# border, which is the last thing a flying-pixel study wants near its statistics.
PAD_VALUE = {"vggt": 1.0, "pi3": 1.0, "dust3r": 1.0, "mast3r": 1.0}

# Input range each family expects. VGGT and pi3 normalize internally from [0,1].
# The CroCo family does NOT: every released DUSt3R/MASt3R loader applies
# `Normalize((0.5,)*3, (0.5,)*3)`, i.e. maps [0,1] -> [-1,1], and the model has no
# normalization of its own. Feeding [0,1] silently halves the contrast and shifts
# the mean; measured per-scene swings of up to 6 points, which is larger than the
# 5-point pre-registered margin. PAD_VALUE 1.0 maps to +1.0, so white padding
# stays white.
INPUT_RANGE = {"vggt": "01", "pi3": "01", "dust3r": "pm1", "mast3r": "pm1"}


def to_model_range(x, fam):
    return x * 2.0 - 1.0 if INPUT_RANGE.get(fam, "01") == "pm1" else x


def _preprocess(rgb, pad=1.0, long_side=MODEL_LONG_SIDE, patch=PATCH):
    """RGB uint8 (H,W,3) -> float32 (3,Hm,Wm) in [0,1] plus crop metadata.

    Long side -> 518, aspect preserved, then padded up to a multiple of the patch
    size. `meta` records the un-padded region so the prediction is cropped back
    before it is ever measured.

    DELIBERATE DEVIATION from VGGT's released preprocessing: that helper resizes
    WIDTH to 518 and centre-crops the height. Cropping would throw away image
    content, and with it real occlusion boundaries -- silently changing which
    population we sample. Long-side resize keeps the whole frame. The padding
    colour does follow VGGT (white), and the pad is cropped out of the output, so
    nothing padded ever reaches a statistic.
    """
    H, W = rgb.shape[:2]
    s = long_side / max(H, W)
    h, w = max(1, round(H * s)), max(1, round(W * s))
    r = (np.arange(h) + 0.5) * H / h
    c = (np.arange(w) + 0.5) * W / w
    small = rgb[np.clip(r.astype(int), 0, H - 1)][:, np.clip(c.astype(int), 0, W - 1)]
    ph, pw = (-h) % patch, (-w) % patch
    t0, l0 = ph // 2, pw // 2
    canvas = np.full((h + ph, w + pw, 3), float(pad), np.float32)
    canvas[t0:t0 + h, l0:l0 + w] = small.astype(np.float32) / 255.0
    return canvas.transpose(2, 0, 1), dict(top=t0, left=l0, h=h, w=w, src=(H, W))


def _crop(pred, meta):
    """Drop the padded border from a model-resolution map (Sec. 6.3)."""
    return pred[meta["top"]:meta["top"] + meta["h"], meta["left"]:meta["left"] + meta["w"]]


def nn_resize(a, shape):
    """Nearest-neighbour resample. NEVER bilinear: interpolating across an
    occlusion boundary invents points in the void we are trying to measure."""
    H, W = shape
    r = np.clip(((np.arange(H) + 0.5) * a.shape[0] / H).astype(int), 0, a.shape[0] - 1)
    c = np.clip(((np.arange(W) + 0.5) * a.shape[1] / W).astype(int), 0, a.shape[1] - 1)
    return a[r][:, c]


# --------------------------------------------------------------------------- #
# model streams
# --------------------------------------------------------------------------- #
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_vggt():
    """Load VGGT-1B, preferring a local `model.pt`.

    The repo ships the SAME 5 GB checkpoint twice -- `model.pt` and
    `model.safetensors`. `from_pretrained` wants the safetensors copy, so on a
    slow link it blocks for a second full 5 GB download of weights already on
    disk. `model.pt` is a plain state_dict (it is what VGGT's own README loads),
    so we use it when present and only fall back to the hub otherwise.
    """
    if "vggt" not in _MODELS:
        try:
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise SystemExit(
                "VGGT not importable. Install into the run env, e.g.\n"
                "  pip install git+https://github.com/facebookresearch/vggt.git") from e
        import glob as _glob
        import torch
        hits = sorted(_glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/model.pt")))
        if hits:
            print(f"  VGGT: local state_dict {hits[-1]}", flush=True)
            m = VGGT()
            sd = torch.load(hits[-1], map_location="cpu", weights_only=True)
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing:
                raise SystemExit(f"VGGT state_dict is missing {len(missing)} keys "
                                 f"(first: {missing[:3]}) -- refusing to run a "
                                 f"partially initialised model.")
            if unexpected:
                print(f"  VGGT: ignoring {len(unexpected)} unexpected keys", flush=True)
        else:
            m = VGGT.from_pretrained("facebook/VGGT-1B")
        # Cache BEFORE the device move: if .to(cuda) OOMs, an uncached model
        # means the next view re-reads 5 GB from disk, and the whole run
        # degenerates into reloading the checkpoint once per view.
        _MODELS["vggt"] = m.eval()
        _MODELS["vggt"] = m.to(_device()).eval()
    return _MODELS["vggt"]


def _load_pi3():
    if "pi3" not in _MODELS:
        try:
            from pi3.models.pi3 import Pi3
        except ImportError as e:
            raise SystemExit(
                "pi3 not importable. Install into the run env, e.g.\n"
                "  pip install git+https://github.com/yyfz/Pi3.git") from e
        m = Pi3.from_pretrained("yyfz233/Pi3").eval()
        _MODELS["pi3"] = m                      # cache before the device move
        _MODELS["pi3"] = m.to(_device()).eval()
    return _MODELS["pi3"]


def _amp(torch, fam="vggt"):
    """Autocast policy per family.

    VGGT and pi3 ship bf16 inference. DUSt3R/MASt3R do not: their own
    `inference()` defaults to `use_amp=False` and the DPT head re-enters fp32
    explicitly. Measured 5.0 points of FP difference on one scene from autocast
    alone -- not acceptable slack for a thresholded boundary statistic.
    """
    dev = _device()
    if dev != "cuda" or fam in ("dust3r", "mast3r"):
        return torch.autocast("cpu", enabled=False)
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    return torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16)


def _fwd_vggt(x):
    """-> dict with 'point' and 'depth', both (h,w) at model resolution."""
    import torch
    m = _load_vggt()
    with torch.no_grad(), _amp(torch):
        # VGGT.forward promotes a 4-D (S,3,h,w) input to B=1 internally.
        p = m(x[None].to(_device()))                      # -> B=1, S=1
    out = {}
    wp = p.get("world_points")
    if wp is not None:                                    # (1,1,h,w,3), S=1 -> own frame
        out["point"] = wp[0, 0, ..., 2].float().cpu().numpy()
    d = p.get("depth")
    if d is not None:                                     # (1,1,h,w,1)
        out["depth"] = d[0, 0, ..., 0].float().cpu().numpy()
    return out


def _fwd_pi3(x):
    """-> dict with 'local': per-view pointmap z in this view's own frame."""
    import torch
    m = _load_pi3()
    with torch.no_grad(), _amp(torch):
        # Pi3.forward unpacks B, N, _, H, W = imgs.shape, so it needs FIVE dims.
        # x is (3,h,w), so a single [None] would be 4-D and raise.
        p = m(x[None, None].to(_device()))                # (1,N=1,3,h,w)
    lp = p.get("local_points", p.get("local_point"))
    if lp is None:
        raise KeyError(f"pi3 returned no local pointmap; keys={list(p)}")
    return {"local": lp[0, 0, ..., 2].float().cpu().numpy()}


def _load_dust3r():
    """Load upstream DUSt3R (not the D2USt3R fork).

    DUSt3R is the architecturally INDEPENDENT witness: CroCo-pretrained, no VGGT
    weights anywhere in its lineage, and its eight-dataset training mix contains
    none of this study's clean sets. Every other stream here runs on a
    VGGT-trained frozen encoder, so this is the only one that can corroborate
    rather than echo.
    """
    if "dust3r" not in _MODELS:
        import sys
        if DUST3R_REPO not in sys.path:
            sys.path.insert(0, DUST3R_REPO)
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as e:
            raise SystemExit(
                f"dust3r not importable from {DUST3R_REPO}.\n"
                f"  git clone --recursive https://github.com/naver/dust3r.git {DUST3R_REPO}") from e
        import torch
        if not os.path.exists(DUST3R_CKPT):
            raise SystemExit(f"DUSt3R checkpoint missing at {DUST3R_CKPT}")
        m = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_CKPT)
        _MODELS["dust3r"] = m
        _MODELS["dust3r"] = m.to(_device()).eval()
    return _MODELS["dust3r"]


def _load_mast3r():
    """MASt3R: same CroCo lineage as DUSt3R with an added matching head.

    Included because §2 A1 asks for it alongside DUSt3R. Note it is NOT a second
    independent witness -- it shares DUSt3R's backbone and pretraining, so
    DUSt3R+MASt3R agreeing is one lineage, exactly as VGGT+pi3 is another.
    """
    if "mast3r" not in _MODELS:
        import sys
        # Insert in REVERSE priority: the last insert(0) ends up first on the
        # path, and mast3r's `import dust3r.*` must resolve to its own pinned
        # submodule, not the standalone clone.
        for r in (DUST3R_REPO, os.path.join(MAST3R_REPO, "dust3r"), MAST3R_REPO):
            if os.path.isdir(r) and r not in sys.path:
                sys.path.insert(0, r)
        try:
            from mast3r.model import AsymmetricMASt3R
        except ImportError as e:
            raise SystemExit(
                f"mast3r not importable from {MAST3R_REPO}.\n"
                f"  git clone --recursive https://github.com/naver/mast3r.git {MAST3R_REPO}") from e
        import torch
        if not os.path.exists(MAST3R_CKPT):
            raise SystemExit(f"MASt3R checkpoint missing at {MAST3R_CKPT}")
        m = AsymmetricMASt3R.from_pretrained(MAST3R_CKPT)
        _MODELS["mast3r"] = m
        _MODELS["mast3r"] = m.to(_device()).eval()
    return _MODELS["mast3r"]


def _fwd_pairwise(x, loader, fam):
    """Shared forward for the CroCo-lineage pairwise models.

    Both DUSt3R and MASt3R take two views. A single view is fed as the SELF-PAIR
    (I, I) and we read X^{1,1}, the pointmap of view 1 in view 1's OWN camera
    frame -- exactly the per-view quantity this study measures, with no pose and
    no cross-view alignment. Documented deviation: these models were designed for
    genuine stereo pairs, and a degenerate self-pair is not what they were
    trained on.
    """
    import torch
    m = loader()
    dev = _device()
    xb = to_model_range(x[None], fam).to(dev)
    shape = torch.tensor([[x.shape[1], x.shape[2]]], device=dev)
    v1 = dict(img=xb, true_shape=shape, idx=[0], instance=["0"])
    v2 = dict(img=xb, true_shape=shape, idx=[1], instance=["1"])
    with torch.no_grad(), _amp(torch, fam):
        r1, _ = m(v1, v2)
    return {"point": r1["pts3d"][0, ..., 2].float().cpu().numpy()}


STREAMS = {
    "dust3r_point": ("dust3r", "point"),
    "mast3r_point": ("mast3r", "point"),
    "vggt_point": ("vggt", "point"),
    "vggt_depth": ("vggt", "depth"),
    "pi3_local":  ("pi3", "local"),
}


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def pred_path(out, stream, key):
    return os.path.join(out, "preds", stream, key.replace("/", "__") + ".npz")


def load_pred(out, stream, key, gt_shape):
    """Cached model-resolution depth -> GT resolution by nearest neighbour."""
    p = pred_path(out, stream, key)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    z = d["depth"].astype(np.float64)
    if "gt_hw" in d:
        ch, cw = (int(x) for x in d["gt_hw"])          # frame the prediction was made for
        gh, gw = int(gt_shape[0]), int(gt_shape[1])    # frame it is being measured on
        # A pure RESCALE is fine: some datasets ship RGB at a different resolution
        # from their depth (Infinigen renders 1280x720 images against 2560x1440
        # depth), and nearest-neighbour to GT resolution is exactly what Sec. 6.2
        # prescribes. A change of ASPECT is not fine -- that would stretch the
        # prediction off the boundary it is being scored against.
        if ch and cw and gh and gw and abs((cw / ch) - (gw / gh)) > 1e-3:
            raise ValueError(
                f"{p}: cached prediction was made for a {(ch, cw)} frame with aspect "
                f"{cw/ch:.4f}, but it is being measured on {(gh, gw)} with aspect "
                f"{gw/gh:.4f}. Rescaling across a different aspect ratio would "
                f"misregister the prediction against the boundary. Re-run inference "
                f"for this view.")
    return nn_resize(z, gt_shape)


def require_gpu(min_free_gb=6.0):
    """Abort early if the GPU cannot hold the model.

    bruinml is a SHARED box. When a colleague's job holds most of the 20 GB,
    every view OOMs, and without this check the run still walks the whole
    dataset producing nothing. Fail loudly at the start instead.
    """
    import torch
    if not torch.cuda.is_available():
        print("  ! no CUDA device; running on CPU (slow)", flush=True)
        return
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 2**30, total / 2**30
    print(f"  GPU: {free_gb:.1f} GB free of {total_gb:.1f} GB", flush=True)
    if free_gb < min_free_gb:
        import subprocess
        try:
            who = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        except Exception:
            who = "(nvidia-smi unavailable)"
        raise SystemExit(
            f"Only {free_gb:.1f} GB of GPU memory is free; this needs ~{min_free_gb:.0f} GB.\n"
            f"Current GPU processes:\n  {who}\n"
            f"This is a shared machine -- do NOT kill another user's job. Wait for it "
            f"to finish, or run when the GPU is free.")


def run(streams, datasets, n_scenes, n_views, out, overwrite=False):
    import torch
    require_gpu()
    todo = [(ds, s, v) for ds in datasets for s in D.scenes(ds)[:n_scenes]
            for v in D.views(ds, s, n_views)]
    print(f"{len(todo)} views x {len(streams)} streams", flush=True)
    families = {STREAMS[s][0] for s in streams}
    manifest = []

    for i, (ds, sc, v) in enumerate(todo):
        need = [s for s in streams
                if overwrite or not os.path.exists(pred_path(out, s, v.key))]
        if not need:
            continue
        try:
            vd = D.load(v, with_rgb=True)
        except Exception as e:
            print(f"  ! load {v.key}: {type(e).__name__}: {e}", flush=True)
            continue
        if vd.rgb is None:
            # A view with no readable RGB cannot be fed to a model. Skip it
            # instead of dying: a single missing file used to abort the entire
            # inference pass and leave the prediction cache silently partial.
            _NO_RGB.append(v.key)
            continue
        raw, metas = {}, {}
        for fam in families:
            if not any(STREAMS[s][0] == fam for s in need):
                continue
            ls, pt = GEOM.get(fam, (MODEL_LONG_SIDE, PATCH))
            x, meta = _preprocess(vd.rgb, PAD_VALUE.get(fam, 1.0), ls, pt)
            metas[fam] = meta
            xt = torch.from_numpy(x)
            fwd = {"vggt": _fwd_vggt, "pi3": _fwd_pi3,
                   "dust3r": lambda t: _fwd_pairwise(t, _load_dust3r, "dust3r"),
                   "mast3r": lambda t: _fwd_pairwise(t, _load_mast3r, "mast3r")}[fam]
            try:
                raw[fam] = fwd(xt)
            except Exception as e:
                print(f"  ! {fam} {v.key}: {type(e).__name__}: {e}", flush=True)
                _FWD_ERRORS[fam] = _FWD_ERRORS.get(fam, 0) + 1
        if not raw:
            continue
        for s in need:
            fam, field = STREAMS[s]
            if fam not in raw or field not in raw[fam]:
                continue
            z = _crop(raw[fam][field], metas[fam]).astype(np.float32)
            p = pred_path(out, s, v.key)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            # Write to a temp file and rename. The Phase 1 cache is populated by
            # HARDLINKING Phase 0's files, so writing in place would share an
            # inode and silently rewrite the Phase 0 artefact. os.replace swaps
            # the name onto a fresh inode and leaves the original untouched.
            tmp = p + ".tmp"
            np.savez_compressed(tmp, depth=z, model_hw=np.array(z.shape),
                                gt_hw=np.array(metas[fam]["src"]))
            os.replace(tmp if tmp.endswith(".npz") else tmp + ".npz", p)
            manifest.append(dict(stream=s, key=v.key, dataset=ds, scene=sc,
                                 model_hw=list(z.shape), gt_hw=list(metas[fam]["src"])))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)

    os.makedirs(out, exist_ok=True)
    mp = os.path.join(out, "inference_manifest.json")
    old = json.load(open(mp)) if os.path.exists(mp) else []
    json.dump(old + manifest, open(mp, "w"), indent=1)
    print(f"wrote {len(manifest)} predictions; manifest -> {mp}")
    if _NO_RGB:
        print(f"\n!! {len(_NO_RGB)} view(s) had no readable RGB and were skipped, e.g. "
              f"{_NO_RGB[:3]}. measure.py drops any view missing a stream, so these "
              f"are excluded from the study rather than silently half-measured.")
    if _FWD_ERRORS:
        print("\n!! forward-pass failures (these streams produced NO data):")
        for fam, n in _FWD_ERRORS.items():
            print(f"   {fam}: {n} views failed")
        print("   Fix these before measuring -- measure.py drops any view where a "
              "stream is missing, so a broken model silently shrinks the study.")
        # Non-zero exit, or the `|| say "!! ... returned $?"` guards in the runner
        # scripts are inert and a broken model produces a quietly smaller study.
        raise SystemExit(1)


def _main():
    ap = argparse.ArgumentParser(description="feed-forward per-view depth inference")
    ap.add_argument("--streams", default=",".join(STREAMS))
    ap.add_argument("--datasets", default="tartanair,sintel")
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--out", default="results/premise_check")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    bad = [s for s in a.streams.split(",") if s not in STREAMS]
    if bad:
        raise SystemExit(f"unknown streams {bad}; choose from {list(STREAMS)}")
    run(a.streams.split(","), [d for d in a.datasets.split(",") if d],
        a.scenes, a.views, a.out, a.overwrite)


if __name__ == "__main__":
    raise SystemExit(_main())
