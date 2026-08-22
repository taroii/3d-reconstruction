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
_MODELS: dict[str, object] = {}
_FWD_ERRORS: dict[str, int] = {}


# --------------------------------------------------------------------------- #
# preprocessing / geometry bookkeeping
# --------------------------------------------------------------------------- #
# Padding colour per model family. VGGT's released load_and_preprocess_images
# pads WHITE; padding black instead puts a hard synthetic edge at the image
# border, which is the last thing a flying-pixel study wants near its statistics.
PAD_VALUE = {"vggt": 1.0, "pi3": 1.0}


def _preprocess(rgb, pad=1.0):
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
    s = MODEL_LONG_SIDE / max(H, W)
    h, w = max(1, round(H * s)), max(1, round(W * s))
    r = (np.arange(h) + 0.5) * H / h
    c = (np.arange(w) + 0.5) * W / w
    small = rgb[np.clip(r.astype(int), 0, H - 1)][:, np.clip(c.astype(int), 0, W - 1)]
    ph, pw = (-h) % PATCH, (-w) % PATCH
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
    if "vggt" not in _MODELS:
        try:
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise SystemExit(
                "VGGT not importable. Install into the run env, e.g.\n"
                "  pip install git+https://github.com/facebookresearch/vggt.git") from e
        import torch
        m = VGGT.from_pretrained("facebook/VGGT-1B").to(_device()).eval()
        _MODELS["vggt"] = m
    return _MODELS["vggt"]


def _load_pi3():
    if "pi3" not in _MODELS:
        try:
            from pi3.models.pi3 import Pi3
        except ImportError as e:
            raise SystemExit(
                "pi3 not importable. Install into the run env, e.g.\n"
                "  pip install git+https://github.com/yyfz/Pi3.git") from e
        m = Pi3.from_pretrained("yyfz233/Pi3").to(_device()).eval()
        _MODELS["pi3"] = m
    return _MODELS["pi3"]


def _amp(torch):
    dev = _device()
    if dev != "cuda":
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


STREAMS = {
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
    if "gt_hw" in d and tuple(d["gt_hw"]) != tuple(gt_shape):
        raise ValueError(
            f"{p}: cached prediction was made for a {tuple(d['gt_hw'])} GT frame but "
            f"{tuple(gt_shape)} was requested. Resizing across a different aspect "
            f"ratio would misregister the prediction against the boundary. Re-run "
            f"inference for this view.")
    return nn_resize(z, gt_shape)


def run(streams, datasets, n_scenes, n_views, out, overwrite=False):
    import torch
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
        raw = {}
        for fam in families:
            if not any(STREAMS[s][0] == fam for s in need):
                continue
            x, meta = _preprocess(vd.rgb, PAD_VALUE.get(fam, 1.0))
            xt = torch.from_numpy(x)
            try:
                raw[fam] = (_fwd_vggt if fam == "vggt" else _fwd_pi3)(xt)
            except Exception as e:
                print(f"  ! {fam} {v.key}: {type(e).__name__}: {e}", flush=True)
                _FWD_ERRORS[fam] = _FWD_ERRORS.get(fam, 0) + 1
        if not raw:
            continue
        for s in need:
            fam, field = STREAMS[s]
            if fam not in raw or field not in raw[fam]:
                continue
            z = _crop(raw[fam][field], meta).astype(np.float32)
            p = pred_path(out, s, v.key)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez_compressed(p, depth=z, model_hw=np.array(z.shape),
                                gt_hw=np.array(meta["src"]))
            manifest.append(dict(stream=s, key=v.key, dataset=ds, scene=sc,
                                 model_hw=list(z.shape), gt_hw=list(meta["src"])))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)

    os.makedirs(out, exist_ok=True)
    mp = os.path.join(out, "inference_manifest.json")
    old = json.load(open(mp)) if os.path.exists(mp) else []
    json.dump(old + manifest, open(mp, "w"), indent=1)
    print(f"wrote {len(manifest)} predictions; manifest -> {mp}")
    if _FWD_ERRORS:
        print("\n!! forward-pass failures (these streams produced NO data):")
        for fam, n in _FWD_ERRORS.items():
            print(f"   {fam}: {n} views failed")
        print("   Fix these before measuring -- measure.py drops any view where a "
              "stream is missing, so a broken model silently shrinks the study.")


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
    _main()
