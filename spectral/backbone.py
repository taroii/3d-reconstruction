"""
Frozen-backbone inference + extraction (plan S2, S5). NETWORK SIDE.

Runs the DDUSt3R repo's DUSt3R/MonST3R/D2USt3R once per clip and extracts, per
view, the pose-independent camera-frame pointmap + confidence + an initial
Sim(3) pose. Everything downstream (sheaf.py) consumes these. Plan S1a: the
sheaf backbone is DUSt3R/MonST3R; D2USt3R is the validation instrument only.

Requires the heavy `sheaf` conda env. Imports the bundled dust3r package.
"""

import os
import sys
import numpy as np

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

CKPT = {
    "dust3r":  "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    "monst3r": "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth",
    "d2ust3r": "ddust3r.pth",
}


def load_backbone(name="dust3r", ckpt_dir=None, device="cuda"):
    """Load a frozen backbone by name. Uses dust3r.model.load_model (strict=False
    key remap), which tolerates the cross-fork checkpoints."""
    from dust3r.model import load_model
    ckpt_dir = ckpt_dir or os.path.join(_DD, "checkpoints")
    path = os.path.join(ckpt_dir, CKPT[name])
    model = load_model(path, device, verbose=False)
    model.eval()
    return model


def run_clip(model, image_paths, device="cuda", size=512, niter=300,
             schedule="cosine", lr=0.01, scene_graph="complete",
             pointcloud=True):
    """Run backbone + global alignment on a clip (ordered list of frame paths).

    niter controls the global-alignment optimization; truncating it is how the
    headline drift experiment induces pose error. Returns the extracted dict
    (see `extract`)."""
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    imgs = load_images(image_paths, size=size, verbose=False)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None,
                       symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    mode = (GlobalAlignerMode.PointCloudOptimizer if pointcloud
            else GlobalAlignerMode.PairViewer)
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    if pointcloud:
        scene.compute_global_alignment(init="mst", niter=niter,
                                       schedule=schedule, lr=lr)
    return extract(scene)


def _backproject(depth, focal):
    """Camera-frame pointmap from depth + focal (principal point at center).
    Pose-independent, so the drift experiment can vary poses freely."""
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    X = (xs - cx) / focal * depth
    Y = (ys - cy) / focal * depth
    return np.stack([X, Y, depth], axis=-1)


def extract(scene):
    """Per-view: camera-frame pointmap, confidence, initial Sim(3) pose, image."""
    import torch  # noqa
    N = len(scene.imgs)
    poses = scene.get_im_poses().detach().cpu().numpy()        # (N,4,4) cams2world
    focals = np.asarray(scene.get_focals().detach().cpu()).reshape(-1)
    depths = [np.asarray(d.detach().cpu()) for d in scene.get_depthmaps()]
    confs = [np.asarray(c.detach().cpu()) for c in scene.get_conf()]
    imgs = [np.asarray(im) for im in scene.imgs]
    localpts = [_backproject(depths[v], float(focals[v])) for v in range(N)]
    return {
        "localpts": localpts,                 # list (H,W,3) camera-frame
        "conf": confs,                        # list (H,W)
        "R": poses[:, :3, :3].copy(),         # (N,3,3)
        "t": poses[:, :3, 3].copy(),          # (N,3)
        "s": np.ones(N),                      # Sim(3) scale (GA folds scale into depth)
        "imgs": imgs, "focals": focals, "depths": depths,
        "shapes": [d.shape for d in depths],
        "scene": scene,
    }
