"""
Multi-dataset frame sources for N_phi training (Sintel + TartanAir + PointOdyssey).

Every dataset reduces to the SAME contract: a list of `Frame`s, and
`load_depth_K(frame) -> (depth z-depth float32, K 3x3)`. GT normals come from the
agnostic `normals.normals_on_grid_from_depth_K`, so the supervision convention is
identical across datasets. Invalid/sky pixels are passed in as depth=nan and the
core drops them.

Formats (verified):
  sintel       : depth .dpt, K from .cam, invalid/ png -> nan. (data/training/)
  tartanair    : depth_left/*_left_depth.npy float32 meters, fixed K=[320,320,320,240].
                 <env>/<Easy|Hard>/P0xx/{image_left,depth_left}
  pointodyssey : depths/*.png 16-bit, meters = png/65535*1000; K = annot.npz
                 ['intrinsics'][idx]. <split>/<seq>/{rgbs,depths,annot.npz}

Paths are relative to the mfc/ working dir.

NOTE: only the Sintel path is testable locally; TartanAir/PointOdyssey loaders
are written to the documented formats and must be smoke-tested on the server with
`sanity_normals.py --dataset <name> ...` before a full run.
"""
import os
import glob
from dataclasses import dataclass

import numpy as np

import sintel as SI

# per-dataset roots (relative to mfc/) and normal-extraction params
CFG = {
    "sintel": dict(root="../data", rel_thresh=0.05, max_depth=None),
    "tartanair": dict(root="../data/tartanair", rel_thresh=0.05, max_depth=200.0),
    "pointodyssey": dict(root="../data/pointodyssey", rel_thresh=0.05, max_depth=200.0),
}
TARTANAIR_K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])

# explicit Sintel val scenes; TartanAir uses a hash rule; PointOdyssey uses its
# official train/ vs val/ split dirs.
SINTEL_VAL = {"ambush_6", "cave_4", "market_5", "temple_3"}


@dataclass
class Frame:
    dataset: str
    scene: str          # scene / trajectory / sequence id
    idx: int            # frame index within scene (for cache key + PO intrinsics)
    rgb_path: str
    depth_path: str
    cam_path: str       # sintel .cam | PO annot.npz | tartanair ""
    key: str            # cache key "dataset/scene/idx"


# ----------------------------- enumeration ------------------------------------
def _sintel_frames(split):
    root = CFG["sintel"]["root"]
    scenes = sorted(os.path.basename(p) for p in
                    glob.glob(os.path.join(root, "training", "clean", "*")))
    want_val = split == "val"
    out = []
    for s in scenes:
        if (s in SINTEL_VAL) != want_val:
            continue
        for i, rgb in enumerate(SI.frame_paths(root, s, "clean")):
            n = i + 1                                   # GT files are 1-based
            out.append(Frame("sintel", s, n, rgb,
                             SI._gt_path(root, "depth", s, n, "dpt"),
                             SI._gt_path(root, "camdata_left", s, n, "cam"),
                             f"sintel/{s}/{n:06d}"))
    return out


def _tartanair_val_envs(envs):
    """Hold out WHOLE environments so train/val never share an environment -- a
    true cross-environment generalization split (every 5th env in sorted order,
    deterministic, >=1 held out)."""
    return set(sorted(set(envs))[::5])


def _tartanair_frames(split):
    root = CFG["tartanair"]["root"]
    rgbs = sorted(glob.glob(os.path.join(root, "*", "*", "P*", "image_left", "*_left.png")))
    envs = [os.path.relpath(r, root).split(os.sep)[0] for r in rgbs]
    val_envs = _tartanair_val_envs(envs)                # environment-disjoint
    want_val = split == "val"
    if want_val:
        print(f"    tartanair held-out environments: {sorted(val_envs)}", flush=True)
    out = []
    for rgb in rgbs:
        env = os.path.relpath(rgb, root).split(os.sep)[0]
        if (env in val_envs) != want_val:
            continue
        traj = os.path.dirname(os.path.dirname(rgb))    # .../P0xx
        scene = os.path.relpath(traj, root).replace(os.sep, "/")
        base = os.path.basename(rgb).replace("_left.png", "")
        depth = os.path.join(traj, "depth_left", base + "_left_depth.npy")
        if not os.path.exists(depth):       # partial download: skip, don't crash
            continue
        idx = int(base)
        out.append(Frame("tartanair", scene, idx, rgb, depth, "",
                         f"tartanair/{scene.replace('/', '_')}/{idx:06d}"))
    return out


def _pointodyssey_frames(split):
    root = CFG["pointodyssey"]["root"]
    sub = "val" if split == "val" else "train"          # official PO split dirs
    out = []
    for seqdir in sorted(glob.glob(os.path.join(root, sub, "*"))):
        if not os.path.isdir(seqdir):
            continue
        scene = os.path.basename(seqdir)
        annot = os.path.join(seqdir, "annot.npz")
        rgbs = sorted(glob.glob(os.path.join(seqdir, "rgbs", "*.jpg")) +
                      glob.glob(os.path.join(seqdir, "rgbs", "*.png")))
        for rgb in rgbs:
            idx = int("".join(filter(str.isdigit, os.path.basename(rgb))))
            depth = os.path.join(seqdir, "depths", "depth_%05d.png" % idx)
            if not os.path.exists(depth):               # fall back to index match
                ds = sorted(glob.glob(os.path.join(seqdir, "depths", "*.png")))
                depth = ds[idx] if idx < len(ds) else None
            if depth:
                out.append(Frame("pointodyssey", scene, idx, rgb, depth, annot,
                                 f"pointodyssey/{scene}/{idx:06d}"))
    return out


_ENUM = {"sintel": _sintel_frames, "tartanair": _tartanair_frames,
         "pointodyssey": _pointodyssey_frames}


# ------------------------------- loading --------------------------------------
def load_depth_K(fr):
    """Return (depth HxW float32 z-depth, K 3x3). Invalid pixels -> nan."""
    if fr.dataset == "sintel":
        depth = SI.read_dpt(fr.depth_path)
        K, _ = SI.read_cam(fr.cam_path)
        inv = fr.depth_path.replace(os.sep + "depth" + os.sep,
                                    os.sep + "invalid" + os.sep).replace(".dpt", ".png")
        if os.path.exists(inv):
            import imageio.v2 as imageio
            depth = depth.copy()
            depth[imageio.imread(inv) != 0] = np.nan
        return depth.astype(np.float32), K

    if fr.dataset == "tartanair":
        return np.load(fr.depth_path).astype(np.float32), TARTANAIR_K

    if fr.dataset == "pointodyssey":
        import imageio.v2 as imageio
        d16 = imageio.imread(fr.depth_path).astype(np.float32)
        depth = d16 / 65535.0 * 1000.0                  # -> meters (z-depth)
        K = np.load(fr.cam_path)["intrinsics"][fr.idx].astype(np.float64)
        return depth, K

    raise ValueError(f"unknown dataset {fr.dataset}")


# ------------------------------- assembly -------------------------------------
def build_frames(datasets, split, max_per_dataset=None, seed=0):
    """Combined Frame list across `datasets` for the given split. `max_per_dataset`
    evenly subsamples (balances huge synthetic sets against small Sintel)."""
    frames = []
    for name in datasets:
        fr = _ENUM[name](split)
        if max_per_dataset and len(fr) > max_per_dataset:
            stride = len(fr) / max_per_dataset          # even stride across scenes
            fr = [fr[int(i * stride)] for i in range(max_per_dataset)]
        print(f"  [{name}/{split}] {len(fr)} frames", flush=True)
        frames += fr
    return frames
