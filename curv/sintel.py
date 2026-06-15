"""
Sintel data loading + motion ground truth. Pure NumPy.

Used by `datasets.py` (depth/cam readers) and the curvature sanity tooling.
Sintel is EVAL-ONLY for the curvature project (never sampled for training);
the .flo/.dpt/.cam binary formats are the standard MPI-Sintel ones.
"""

import os
import glob
import numpy as np

TAG_FLOAT = 202021.25


def _read_tag(f):
    tag = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert abs(tag - TAG_FLOAT) < 1e-2, f"bad Sintel tag {tag}"


def read_flo(path):
    """MPI-Sintel optical flow -> (H,W,2)."""
    with open(path, "rb") as f:
        _read_tag(f)
        w, h = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=2 * w * h)
    return data.reshape(h, w, 2)


def read_dpt(path):
    """MPI-Sintel depth -> (H,W) metric depth."""
    with open(path, "rb") as f:
        _read_tag(f)
        w, h = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=w * h)
    return data.reshape(h, w)


def read_cam(path):
    """MPI-Sintel camera -> (M intrinsics 3x3, N extrinsics 3x4 world->cam)."""
    with open(path, "rb") as f:
        _read_tag(f)
        M = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
        N = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)
    return M, N


def frame_paths(root, scene, pass_="clean"):
    """Sorted RGB frame paths for a scene (training split)."""
    d = os.path.join(root, "training", pass_, scene)
    return sorted(glob.glob(os.path.join(d, "frame_*.png")))


def _gt_path(root, kind, scene, idx, ext):
    return os.path.join(root, "training", kind, scene, f"frame_{idx:04d}.{ext}")
