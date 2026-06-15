"""
GT discrete curvature from depth + intrinsics (curvature project, step 0).
Pure NumPy (grid alignment lazily imports the dust3r resize so the core stays
GPU-free).

Curvature is a deterministic per-pixel function of the GT pointmap X-bar built
from GT depth + intrinsics (the same X-bar D2USt3R supervises against). Because
the pointmap is pixel-aligned, a pixel's image neighbors (u +/- 1, v +/- 1) are
its 3D-surface neighbors, so curvature is a local stencil on the point grid.

Two estimators (brief Section "Computing curvature from a pointmap"):
  - Gaussian (angular deficit):  K(u,v) = 2*pi - sum_i theta_i  over the fan of
    triangles around p. Flat -> 0, convex bump -> >0, saddle -> <0. Cleanest
    conceptual story (Theorema Egregium), but noisier numerically.
  - Mean (umbrella Laplace-Beltrami):  H ~= 1/2 ||Delta p||, signed by the local
    normal. Tends to behave better numerically. Default.

Gotchas handled here (see SETUP.md / brief):
  1. Depth discontinuities blow up |K|: `depth_discontinuity_mask` excludes
     silhouette/cliff pixels; folded into the validity mask.
  2. Validity: only interior pixels with a fully-valid neighborhood are kept.
  3. Scale dependence: curvature has units 1/area, so it is scale-sensitive.
     `normalize=True` computes K on the pointmap rescaled by its valid-mean
     norm-to-origin (mirrors D2USt3R's pointmap normalization), so the target
     scale is consistent sample-to-sample.
  4. Heavy tails: `signed_log` compresses the target; pair with a robust loss.

The convention here (camera-frame points P = Z * K^{-1} [u,v,1], z-depth) is the
SAME one used for the predicted pointmap, so GT and prediction are comparable.
"""

import os
import numpy as np

import sintel as SI

# 8-neighbour ring in CCW order (row, col offsets) for the Gaussian angle fan.
_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def backproject(depth, K):
    """Camera-frame points (H,W,3) from z-depth (H,W) + intrinsics K (3x3)."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1)          # (H,W,3)
    rays = pix @ np.linalg.inv(K).T                              # 3rd comp ~ 1
    return rays * depth[..., None]


def _shift(P, dr, dc):
    """Neighbour fetch: result[i,j] = P[i+dr, j+dc]. Wraps at the border (the
    1-px border is invalidated by `_interior` so the wrap never reaches output)."""
    return np.roll(np.roll(P, -dr, axis=0), -dc, axis=1)


def _interior(shape):
    """True on all but the 1-px image border (where the stencil would wrap)."""
    m = np.zeros(shape, bool)
    m[1:-1, 1:-1] = True
    return m


def normals_from_points(P):
    """Unit normals (H,W,3) from a camera-frame point map via central diffs,
    oriented toward the camera (origin). Returns (normals, tangent_valid)."""
    dPdy = np.gradient(P, axis=0)
    dPdx = np.gradient(P, axis=1)
    n = np.cross(dPdx, dPdy)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    tangent_valid = mag[..., 0] > 1e-12
    n = n / np.clip(mag, 1e-12, None)
    flip = np.sum(n * P, axis=-1) > 0    # want n . P < 0  (face the camera)
    n = np.where(flip[..., None], -n, n)
    return n, tangent_valid


def depth_discontinuity_mask(depth, rel_thresh=0.05):
    """True where depth jumps to a 4-neighbor by > rel_thresh * depth (cliff /
    silhouette pixels whose curvature stencil would span a discontinuity)."""
    d = depth
    jumps = [
        np.abs(np.diff(d, axis=1, prepend=d[:, :1])),
        np.abs(np.diff(d, axis=1, append=d[:, -1:])),
        np.abs(np.diff(d, axis=0, prepend=d[:1, :])),
        np.abs(np.diff(d, axis=0, append=d[-1:, :])),
    ]
    jump = np.maximum.reduce(jumps)
    return jump > rel_thresh * np.clip(d, 1e-6, None)


def gaussian_curvature_from_points(P, eps=1e-12):
    """Discrete Gaussian curvature via angular deficit: K = 2*pi - sum theta_i
    over the fan of 8 triangles around each pixel. Integrated form (not divided
    by area); use `normalize` upstream for scale consistency."""
    ring = [_shift(P, dr, dc) - P for dr, dc in _RING]          # 8 x (H,W,3)
    norms = [np.linalg.norm(v, axis=-1) for v in ring]
    theta = np.zeros(P.shape[:2])
    for k in range(8):
        a, b = ring[k], ring[(k + 1) % 8]
        cos = (a * b).sum(-1) / np.clip(norms[k] * norms[(k + 1) % 8], eps, None)
        theta += np.arccos(np.clip(cos, -1.0, 1.0))
    return 2.0 * np.pi - theta


def mean_curvature_from_points(P, normals=None, eps=1e-12):
    """Signed mean curvature H ~= 1/2 ||Delta p|| via the 4-neighbour umbrella
    Laplacian, signed by agreement of the Laplacian with the surface normal
    (convex > 0, concave < 0)."""
    lap = (_shift(P, -1, 0) + _shift(P, 1, 0) +
           _shift(P, 0, -1) + _shift(P, 0, 1)) / 4.0 - P
    mag = np.linalg.norm(lap, axis=-1)
    if normals is None:
        normals, _ = normals_from_points(P)
    sign = np.sign((lap * normals).sum(-1))
    sign[sign == 0] = 1.0
    return 0.5 * mag * sign


def signed_log(K):
    """Heavy-tail-compressed target: sign(K) * log(1 + |K|)."""
    return np.sign(K) * np.log1p(np.abs(K))


def curvature_from_depth_K(depth, K, mode="mean", rel_thresh=0.05,
                           max_depth=None, normalize=True):
    """Dataset-AGNOSTIC core: full-res curvature (H,W) + validity (H,W) from a
    z-depth map and intrinsics. Invalidates: depth discontinuities, non-finite
    depth, the 1-px border, and (optionally) far/sky pixels. Pass invalid pixels
    in as depth=nan and they are dropped here.

    mode: "mean" (default, numerically stabler) or "gaussian" (angle deficit).
    normalize: rescale the pointmap by its valid-mean norm-to-origin before
    computing curvature, so the target scale is consistent across samples."""
    d0 = np.nan_to_num(depth, nan=0.0)
    P = backproject(d0, K)

    valid = (np.isfinite(depth) & ~depth_discontinuity_mask(d0, rel_thresh)
             & _interior(depth.shape))
    if max_depth is not None:
        valid = valid & (depth < max_depth)

    if normalize:
        norm = np.linalg.norm(P, axis=-1)
        scale = float(np.mean(norm[valid])) if valid.any() else 1.0
        P = P / max(scale, 1e-6)

    if mode == "gaussian":
        Kmap = gaussian_curvature_from_points(P)
    elif mode == "mean":
        Kmap = mean_curvature_from_points(P)
    else:
        raise ValueError(f"unknown curvature mode {mode!r}")
    Kmap = np.where(valid, Kmap, 0.0)
    return Kmap.astype(np.float32), valid


def curvature_on_grid_from_depth_K(depth, K, mode="mean", size=512,
                                   rel_thresh=0.05, max_depth=None,
                                   normalize=True, compress=True):
    """Agnostic: GT curvature + validity resampled to the backbone grid. If
    `compress`, returns the signed-log-compressed target (recommended)."""
    Kmap, valid = curvature_from_depth_K(depth, K, mode, rel_thresh,
                                         max_depth, normalize)
    if compress:
        Kmap = signed_log(Kmap)
    K_g = to_grid(Kmap.astype(np.float32), size=size, nearest=False)
    v_g = to_grid(valid, size=size, nearest=True)
    return K_g.astype(np.float32), v_g


# --------------------------------------------------------------------------
# Sintel convenience wrapper (for sanity_curvature.py)
# --------------------------------------------------------------------------
def gt_curvature(root, scene, idx, mode="mean", rel_thresh=0.05, normalize=True):
    """Full-res GT curvature + validity + depth + K for one Sintel frame."""
    depth = SI.read_dpt(SI._gt_path(root, "depth", scene, idx, "dpt"))
    K, _ = SI.read_cam(SI._gt_path(root, "camdata_left", scene, idx, "cam"))
    inv_p = SI._gt_path(root, "invalid", scene, idx, "png")
    if os.path.exists(inv_p):
        import imageio.v2 as imageio
        depth = depth.copy()
        depth[imageio.imread(inv_p) != 0] = np.nan
    Kmap, valid = curvature_from_depth_K(depth, K, mode, rel_thresh,
                                         normalize=normalize)
    return Kmap, valid, depth, K


# --------------------------------------------------------------------------
# Grid alignment: land a full-res field on the backbone's pointmap grid using
# the SAME dust3r crop_img transform that load_images applies to the RGB, so GT
# curvature is pixel-aligned with the predicted depth/pointmaps.
# --------------------------------------------------------------------------
def _crop_img():
    import sys
    _DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
    if _DD not in sys.path:
        sys.path.insert(0, _DD)
    from dust3r.utils.image import crop_img
    return crop_img


def to_grid(field, size=512, nearest=False):
    """Resize+crop a full-res (H,W) or (H,W,C) array onto the (size) pointmap
    grid via crop_img. Floats go through LANCZOS (nearest=False), masks through
    NEAREST (nearest=True)."""
    import PIL.Image
    crop_img = _crop_img()

    if field.dtype == bool:
        pil = PIL.Image.fromarray((field.astype(np.uint8) * 255), mode="L")
        return np.asarray(crop_img(pil, size, nearest=True)) > 127

    if field.ndim == 2:
        pil = PIL.Image.fromarray(field.astype(np.float32), mode="F")
        return np.asarray(crop_img(pil, size, nearest=nearest))

    chans = []
    for c in range(field.shape[-1]):
        pil = PIL.Image.fromarray(field[..., c].astype(np.float32), mode="F")
        chans.append(np.asarray(crop_img(pil, size, nearest=nearest)))
    return np.stack(chans, axis=-1)


def curvature_to_rgb(K, p=98):
    """Diverging colormap encoding of a (signed) curvature map for visualization.
    Blue = concave/negative, red = convex/positive, white ~ flat."""
    import matplotlib.cm as cm
    lim = np.percentile(np.abs(K), p) if np.any(K) else 1.0
    t = np.clip(K / max(lim, 1e-6), -1, 1) * 0.5 + 0.5
    return cm.get_cmap("RdBu_r")(t)[..., :3]
