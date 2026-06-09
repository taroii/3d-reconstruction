"""
Sintel GT surface normals from depth + intrinsics (PLAN step 0). Pure NumPy
(grid alignment lazily imports the dust3r resize so the core stays GPU-free).

Supervision for the appearance->normal prior N_phi: per-pixel camera-frame
surface normals derived from the GT depth + camera we already have, plus a
validity mask (Sintel `invalid`, depth discontinuities, degenerate tangents) so
the cross product is never taken across an edge.

`normals_from_points` is the shared convention used on BOTH sides:
  - GT here (step 0), and
  - the optimizer's reconstruction normals N_n (step 1b),
so keep it stable:
  - camera-frame points  P = Z * K^{-1} [u,v,1]      (Sintel depth is z-depth)
  - n = normalize( dP/dx  x  dP/dy ),  central differences
  - oriented toward the camera at the origin:  n . P < 0
"""

import os
import numpy as np

import sintel as SI


def backproject(depth, K):
    """Camera-frame points (H,W,3) from z-depth (H,W) + intrinsics K (3x3)."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    pix = np.stack([xs, ys, np.ones_like(xs)], axis=-1)          # (H,W,3)
    rays = pix @ np.linalg.inv(K).T                              # 3rd comp ~ 1
    return rays * depth[..., None]


def normals_from_points(P):
    """Unit normals (H,W,3) from a camera-frame point map via central diffs,
    oriented toward the camera (origin). Returns (normals, tangent_valid)."""
    dPdy = np.gradient(P, axis=0)        # one-sided at borders
    dPdx = np.gradient(P, axis=1)
    n = np.cross(dPdx, dPdy)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    tangent_valid = mag[..., 0] > 1e-12
    n = n / np.clip(mag, 1e-12, None)
    flip = np.sum(n * P, axis=-1) > 0    # want n . P < 0  (face the camera)
    n = np.where(flip[..., None], -n, n)
    return n, tangent_valid


def depth_discontinuity_mask(depth, rel_thresh=0.05):
    """True where depth jumps to a 4-neighbor by > rel_thresh * depth (edge
    pixels whose normal would span a discontinuity)."""
    d = depth
    jumps = [
        np.abs(np.diff(d, axis=1, prepend=d[:, :1])),
        np.abs(np.diff(d, axis=1, append=d[:, -1:])),
        np.abs(np.diff(d, axis=0, prepend=d[:1, :])),
        np.abs(np.diff(d, axis=0, append=d[-1:, :])),
    ]
    jump = np.maximum.reduce(jumps)
    return jump > rel_thresh * np.clip(d, 1e-6, None)


def normals_from_depth_K(depth, K, rel_thresh=0.05, max_depth=None):
    """Dataset-AGNOSTIC core: full-res normals + validity from a z-depth map and
    intrinsics. Used for every dataset (Sintel/TartanAir/PointOdyssey) so the
    supervision convention is identical. Invalidates: degenerate tangents,
    non-finite depth, depth discontinuities, and (optionally) far/sky pixels.
    Pass invalid pixels in as depth=nan and they are dropped here."""
    P = backproject(np.nan_to_num(depth, nan=0.0), K)
    n, tangent_valid = normals_from_points(P)
    valid = tangent_valid & np.isfinite(depth) & ~depth_discontinuity_mask(
        np.nan_to_num(depth, nan=0.0), rel_thresh)
    if max_depth is not None:
        valid &= depth < max_depth
    return n, valid


def normals_on_grid_from_depth_K(depth, K, size=512, rel_thresh=0.05, max_depth=None):
    """Agnostic: GT normals + validity resampled to the backbone grid."""
    n, valid = normals_from_depth_K(depth, K, rel_thresh, max_depth)
    n_g = to_grid(n, size=size, nearest=False)
    v_g = to_grid(valid, size=size, nearest=True)
    mag = np.linalg.norm(n_g, axis=-1, keepdims=True)
    v_g &= mag[..., 0] > 1e-6
    n_g = n_g / np.clip(mag, 1e-6, None)
    return n_g, v_g


def gt_normals(root, scene, idx, rel_thresh=0.05):
    """Full-res GT normals + validity for one Sintel frame (1-based idx).
    Thin Sintel wrapper over the agnostic core (kept for sanity_normals)."""
    depth = SI.read_dpt(SI._gt_path(root, "depth", scene, idx, "dpt"))
    K, _ = SI.read_cam(SI._gt_path(root, "camdata_left", scene, idx, "cam"))
    inv_p = SI._gt_path(root, "invalid", scene, idx, "png")
    if os.path.exists(inv_p):
        import imageio.v2 as imageio
        depth = depth.copy()
        depth[imageio.imread(inv_p) != 0] = np.nan      # 255 = invalid -> dropped
    n, valid = normals_from_depth_K(depth, K, rel_thresh)
    return n, valid, depth, K


# --------------------------------------------------------------------------
# Grid alignment: land a full-res field on the backbone's pointmap grid using
# the SAME dust3r crop_img transform that load_images applies to the RGB, so GT
# normals are pixel-aligned with the predicted depth/pointmaps.
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
        out = np.asarray(crop_img(pil, size, nearest=True)) > 127
        return out

    if field.ndim == 2:
        pil = PIL.Image.fromarray(field.astype(np.float32), mode="F")
        return np.asarray(crop_img(pil, size, nearest=nearest))

    chans = []
    for c in range(field.shape[-1]):
        pil = PIL.Image.fromarray(field[..., c].astype(np.float32), mode="F")
        chans.append(np.asarray(crop_img(pil, size, nearest=nearest)))
    return np.stack(chans, axis=-1)


def gt_normals_on_grid(root, scene, idx, size=512, rel_thresh=0.05):
    """GT normals + validity resampled to the backbone grid (renormalized)."""
    n, valid, depth, K = gt_normals(root, scene, idx, rel_thresh)
    n_g = to_grid(n, size=size, nearest=False)
    v_g = to_grid(valid, size=size, nearest=True)
    mag = np.linalg.norm(n_g, axis=-1, keepdims=True)
    v_g &= mag[..., 0] > 1e-6
    n_g = n_g / np.clip(mag, 1e-6, None)
    return n_g, v_g


def normal_to_rgb(n):
    """Encode a unit-normal map as RGB in [0,1] for visualization."""
    return (n * 0.5 + 0.5).clip(0, 1)
