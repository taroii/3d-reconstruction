"""
Spring dataset readers (https://spring-benchmark.org). Stereo synthetic data;
we use the LEFT camera only and convert its disparity to z-depth.

Verified format facts (Spring FAQ + cv-stuttgart/spring_utils):
  - layout: <root>/<split>/<seq>/frame_left/frame_left_<NNNN>.png   (1-based 4-digit)
            <root>/<split>/<seq>/disp1_left/disp1_left_<NNNN>.dsp5
            <root>/<split>/<seq>/cam_data/{intrinsics,extrinsics}.txt
  - disparity is HDF5 (.dsp5), dataset key "disparity"; GT is stored at 2x
    (4K) but the VALUES relate to the HD image, so subsample `[::2, ::2]` with
    NO value scaling to align with the 1920x1080 RGB.
  - depth:  Z = fx * B / d,  B = 0.065 m (constant for Spring). d<=0 is sky ->
    infinite depth -> dropped as nan.
  - intrinsics.txt: rows of `fx fy cx cy` (one per frame, or a single shared
    row). For HD images, so no rescaling needed.

NOTE: intrinsics.txt column order / per-frame-vs-shared is the one assumption not
yet confirmed against real files -- smoke-test with `sanity_curvature.py
--dataset spring ...` on the server before a full run.
"""
import os
import glob
import numpy as np

BASELINE = 0.065        # meters, constant for Spring


def read_dsp5(path):
    """Read a Spring .dsp5 disparity map (HDF5 key 'disparity'). Returns the
    raw 2x-resolution array; caller subsamples [::2,::2] to HD."""
    import h5py
    with h5py.File(path, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"{path}: no 'disparity' key")
        return np.asarray(f["disparity"][()])


def read_intrinsics(path, idx):
    """K (3x3) for 1-based frame `idx`. intrinsics.txt rows are `fx fy cx cy`;
    a single row is treated as shared across all frames."""
    arr = np.loadtxt(path)
    row = arr if arr.ndim == 1 else (arr[idx - 1] if idx - 1 < len(arr) else arr[0])
    fx, fy, cx, cy = row[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def disp_to_depth(disp, fx, baseline=BASELINE):
    """Z-depth (H,W) from HD disparity (H,W) + fx. Non-positive disparity (sky)
    and non-finite values -> nan (dropped downstream)."""
    disp = disp.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = fx * baseline / disp
    depth[(disp <= 0) | ~np.isfinite(depth)] = np.nan
    return depth


def frame_paths(root, split, seq):
    """Sorted left-RGB frame paths for one Spring sequence."""
    d = os.path.join(root, split, seq, "frame_left")
    return sorted(glob.glob(os.path.join(d, "frame_left_*.png")))
