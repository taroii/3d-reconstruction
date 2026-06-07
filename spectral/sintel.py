"""
Sintel data loading + motion ground truth (plan S3). Pure NumPy.

Motion GT (uniform rule, plan S3): a pixel is dynamic iff its observed flow
departs from the camera-induced (rigid-scene) flow by more than tau:
    dynamic = || flow_gt - flow_cam_induced || > tau ,
computed from GT depth + camera intrinsics/extrinsics (D2USt3R Eq.5 with GT).
Report a tau-sensitivity curve so the threshold is not a free parameter.

NOTE: validated against the real Sintel files once the archives are extracted;
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


def cam_induced_flow(depth_i, M_i, N_i, M_j, N_j):
    """Rigid-scene (camera-induced) flow from frame i to j, given GT depth at i
    and both cameras. Returns (H,W,2). Extrinsics N=[R|t] map world->camera."""
    H, W = depth_i.shape
    R_i, t_i = N_i[:, :3], N_i[:, 3]
    R_j, t_j = N_j[:, :3], N_j[:, 3]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1)          # (H,W,3)
    Minv = np.linalg.inv(M_i)
    cam_i = (pix @ Minv.T) * depth_i[..., None]                  # cam-i coords
    world = (cam_i - t_i) @ R_i                                  # R_i^{-1}=R_i^T => (X-t)R
    cam_j = world @ R_j.T + t_j
    proj = cam_j @ M_j.T
    uj = proj[..., 0] / np.clip(proj[..., 2], 1e-6, None)
    vj = proj[..., 1] / np.clip(proj[..., 2], 1e-6, None)
    return np.stack([uj - xs, vj - ys], axis=-1)


def motion_gt(root, scene, idx_i, idx_j, tau=3.0, pass_="clean"):
    """Per-pixel dynamic mask for frame i wrt j (uniform rule). Uses GT flow
    (i->i+1 convention; for non-adjacent j this is an approximation), depth, cam.
    Returns (mask HxW bool, residual HxW, valid HxW)."""
    flow = read_flo(_gt_path(root, "flow", scene, idx_i, "flo"))
    depth_i = read_dpt(_gt_path(root, "depth", scene, idx_i, "dpt"))
    M_i, N_i = read_cam(_gt_path(root, "camdata_left", scene, idx_i, "cam"))
    M_j, N_j = read_cam(_gt_path(root, "camdata_left", scene, idx_j, "cam"))
    induced = cam_induced_flow(depth_i, M_i, N_i, M_j, N_j)
    resid = np.linalg.norm(flow - induced, axis=-1)
    valid = np.ones(resid.shape, bool)
    occ_p = _gt_path(root, "occlusions", scene, idx_i, "png")
    if os.path.exists(occ_p):
        import imageio.v2 as imageio
        valid &= imageio.imread(occ_p) == 0                     # 0 = non-occluded
    return resid > tau, resid, valid
