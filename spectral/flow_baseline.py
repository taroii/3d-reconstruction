"""
Optical-flow motion-mask baseline for the flow-vs-sheaf head-to-head (plan §3).
The fair baseline: dense RAFT flow vs the camera-induced (rigid) flow computed
from the SAME DUSt3R geometry Formulation A uses; motion = where they disagree
(D2USt3R Eq.5 form). Only the motion *cue* differs (dense flow vs sheaf H^1);
geometry source, GT, tau, threshold, metrics, scenes are held identical.

RAFT = torchvision raft_large (Sintel-finetuned C_T_SKHT_V2), auto-downloaded;
no external repo. Runs on the (208,512) grid (crop_img, same as the backbone) so
the residual aligns pixel-for-pixel with A's pointmaps/poses.

STAGED: validated once the GPU frees from the R1 batch; the RAFT call and the
DUSt3R-induced-flow geometry are standard but untested here yet.
"""

import os
import sys
import numpy as np

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)


def load_raft(device="cuda", weights=None):
    """RAFT-Large. weights name via arg or $RAFT_WEIGHTS; default C_T_SKHT_V2
    (Sintel-finetuned). Use C_T_V2 (Chairs+Things, NO Sintel) for a bias-free
    flow baseline when evaluating on Sintel."""
    import os
    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    name = weights or os.environ.get("RAFT_WEIGHTS", "C_T_SKHT_V2")
    w = getattr(Raft_Large_Weights, name)
    model = raft_large(weights=w, progress=False).to(device).eval()
    return model


def _load_512(path):
    """Image -> (208,512)-style grid tensor in [-1,1] (RAFT input), via the same
    crop_img the backbone uses, so flow aligns with DUSt3R geometry."""
    import torch
    import PIL.Image
    from dust3r.utils.image import crop_img
    pil = crop_img(PIL.Image.open(path).convert("RGB"), 512)
    arr = np.asarray(pil).astype(np.float32) / 255.0          # (H,W,3) [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]          # (1,3,H,W)
    return t * 2.0 - 1.0, arr.shape[:2]


def predict_flow(model, path_a, path_b, device="cuda"):
    """RAFT flow a->b on the (208,512) grid. Returns (H,W,2)."""
    import torch
    ta, hw = _load_512(path_a); tb, _ = _load_512(path_b)
    with torch.no_grad():
        flows = model(ta.to(device), tb.to(device))           # list, coarse->fine
    return flows[-1][0].permute(1, 2, 0).cpu().numpy()         # (H,W,2)


def induced_flow_dust3r(depth_i, focal_i, focal_j, g_i, g_j):
    """Camera-induced (rigid) flow i->j from DUSt3R depth + cams2world poses.
    g_* are 4x4 cam->world. Principal point at image center. Backproject through
    camera i's focal; reproject through camera j's focal (DUSt3R focals are
    per-view, so the two differ)."""
    H, W = depth_i.shape
    cx, cy = W / 2.0, H / 2.0
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    Xc = (xs - cx) / focal_i * depth_i
    Yc = (ys - cy) / focal_i * depth_i
    cam_i = np.stack([Xc, Yc, depth_i], axis=-1)              # (H,W,3)
    Ri, ti = g_i[:3, :3], g_i[:3, 3]
    Rj, tj = g_j[:3, :3], g_j[:3, 3]
    world = cam_i @ Ri.T + ti                                 # cam_i -> world
    cam_j = (world - tj) @ Rj                                 # world -> cam_j (R^-1=R^T)
    uj = focal_j * cam_j[..., 0] / np.clip(cam_j[..., 2], 1e-6, None) + cx
    vj = focal_j * cam_j[..., 1] / np.clip(cam_j[..., 2], 1e-6, None) + cy
    return np.stack([uj - xs, vj - ys], axis=-1)


def flow_residual_map(model, path_i, path_j, depth_i, focal_i, focal_j, g_i, g_j,
                      device="cuda"):
    """Per-pixel motion cue for the flow baseline: ||RAFT_flow - induced_flow||."""
    pred = predict_flow(model, path_i, path_j, device=device)
    induced = induced_flow_dust3r(depth_i, focal_i, focal_j, g_i, g_j)
    return np.linalg.norm(pred - induced, axis=-1)            # (H,W)
