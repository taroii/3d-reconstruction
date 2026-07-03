"""
Shared helpers for the qualitative figures. Model loading, a frame iterator that
runs several checkpoints on the same held-out frames, scale-and-shift depth
alignment (same as eval_depth.py), the boundary (top-decile |K|) mask, and a
z-buffered point-cloud renderer for novel-view pointmap renders. Needs the dust3r
env + a GPU (run after training, when the GPU is free).
"""
import os
import sys
import numpy as np
import torch

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

import curvature as CV

# Sintel is the zero-shot regime where the boundary effect is real; use it for the
# error maps (Figure C). Its GT depth is INVALID on moving objects, so it cannot
# render dynamic content in a pointmap.
SINTEL = ("400 @ SintelDUSt3R(dataset_location='../data/training', dset='clean', "
          "S=2, strides=[7], resolution=(512,224), load_dynamic_mask=False)")

# PointOdyssey val has DENSE GT depth including moving objects, so the pointmap
# renders actually show the dynamic content (in-distribution: qualitative, not a
# boundary-error win).
POINTODYSSEY = ("400 @ PointOdysseyDUSt3R(dset='val', dataset_location='../data/pointodyssey', "
                "S=2, strides=[4], resolution=(512,288))")


def load_models(paths, dev):
    from dust3r.model import load_model
    ms = []
    for p in paths:
        m = load_model(p, dev, verbose=False)
        m.eval()
        ms.append(m)
    return ms


def frame_iter(dataset_str, models, dev, max_frames=0, bs=1):
    """Yield one dict per frame: rgb (H,W,3 in [0,1]), preds (list of H,W,3 pred
    pointmaps, one per model), gt depth, valid mask, K, scene/frame ids."""
    from dust3r.datasets import get_data_loader
    try:
        from dust3r.inference import loss_of_one_batch
    except ImportError:
        from dust3r.training import loss_of_one_batch
    loader = get_data_loader(dataset_str, batch_size=bs, num_workers=4,
                             shuffle=False, drop_last=False)
    d = getattr(loader, "dataset", None)
    if d is not None and hasattr(d, "set_epoch"):
        d.set_epoch(0)
    n = 0
    for batch in loader:
        with torch.no_grad():
            outs = [loss_of_one_batch(batch, m, None, dev, symmetrize_batch=False,
                                      use_amp=True) for m in models]
        v1 = outs[0]["view1"]
        gt = v1["depthmap"].float().cpu().numpy()
        valid = v1["valid_mask"].cpu().numpy().astype(bool)
        K = v1["camera_intrinsics"].float().cpu().numpy()
        img = v1["img"].float().cpu().numpy()                       # (B,3,H,W), in [-1,1]
        labels = v1.get("label", None)
        insts = v1.get("instance", None)
        for b in range(gt.shape[0]):
            preds = [o["pred1"]["pts3d"][b].detach().float().cpu().numpy() for o in outs]
            rgb = np.clip(img[b].transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
            yield dict(
                rgb=rgb, preds=preds, gt=gt[b], valid=valid[b], K=K[b],
                scene=(labels[b] if labels is not None else "scene"),
                frame=("".join(filter(str.isdigit, str(insts[b]))) if insts is not None else str(b)),
            )
            n += 1
            if max_frames and n >= max_frames:
                return


def ssi_align(pred, gt, mask):
    """Least-squares scale+shift of pred onto gt over mask (per-frame ssi)."""
    p, g = pred[mask], gt[mask]
    if p.size < 10:
        return pred
    A = np.stack([p, np.ones_like(p)], axis=1)
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]
    return pred * s + t


def boundary_mask(gt, K, valid, bpct=90.0):
    """Top (100-bpct)% of valid pixels by |GT curvature| (the paper's boundary set)."""
    Kmap, kvalid = CV.curvature_from_depth_K(gt, K, mode="mean", normalize=True)
    vk = valid & kvalid & (gt > 0)
    if vk.sum() < 50:
        return np.zeros_like(valid), Kmap
    thr = np.percentile(np.abs(Kmap[vk]), bpct)
    return vk & (np.abs(Kmap) >= thr), Kmap


def abs_rel(pred_d, gt, mask):
    """Per-pixel AbsRel map (0 outside mask) and its mean over mask."""
    m = mask & (gt > 0) & np.isfinite(pred_d)
    err = np.zeros_like(gt, dtype=np.float64)
    err[m] = np.abs(pred_d[m] - gt[m]) / gt[m]
    mean = float(err[m].mean()) if m.any() else 0.0
    return err, mean, m


def render_points(pts, rgb, valid, yaw=25.0, pitch=12.0, out=420, radius=1,
                  bg=1.0, flip=False):
    """Novel-view render of a colored pointmap by orthographic projection with a
    painter's-algorithm splat. pts:(H,W,3) camera-frame points, rgb:(H,W,3) in
    [0,1], valid:(H,W). Rotate yaw about the vertical axis and pitch about the
    horizontal, then project. A rotated view exposes depth-edge bleeding that the
    frontal view hides."""
    P = pts[valid].astype(np.float64)
    C = rgb[valid]
    if P.shape[0] < 100:
        return np.full((out, out, 3), bg)
    Q = P - np.median(P, axis=0)
    ry, rx = np.deg2rad(yaw), np.deg2rad(pitch)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Qr = Q @ Ry.T @ Rx.T
    x, y, z = Qr[:, 0], Qr[:, 1], Qr[:, 2]
    lo = np.percentile(np.concatenate([x, y]), 1.0)
    hi = np.percentile(np.concatenate([x, y]), 99.0)
    span = max(hi - lo, 1e-6)
    m = int(out * 0.06)
    u = ((x - lo) / span * (out - 2 * m) + m).astype(int)
    v = ((y - lo) / span * (out - 2 * m) + m).astype(int)   # Y is down (OpenCV), no flip
    inb = (u >= 0) & (u < out) & (v >= 0) & (v < out)
    u, v, z, C = u[inb], v[inb], z[inb], C[inb]
    order = np.argsort(-z if flip else z)          # far first, near painted last
    u, v, C = u[order], v[order], C[order]
    img = np.full((out, out, 3), bg)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            img[np.clip(v + dv, 0, out - 1), np.clip(u + du, 0, out - 1)] = C
    return np.clip(img, 0, 1)
