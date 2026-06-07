"""
MASt3R feature matching -> per-edge correspondences (plan S1b, Formulation A).
NETWORK SIDE. Runs the MASt3R model and writes Matches (pix_i, pix_j, conf) per
edge to the cache.

ISOLATION: MASt3R bundles its own `dust3r` package, which clashes with DDUSt3R's
MonST3R-fork `dust3r`. This module inserts ONLY the mast3r repo on sys.path and
must be run in its own process (never imported alongside backbone.py). The cache
boundary makes this clean: this writes matches/*.npz, the sheaf reads them.

    conda run -n sheaf python match_mast3r.py --clip <clean/scene> --out <dir>
"""

import os
import sys
import argparse
import numpy as np

_M = os.path.join(os.path.dirname(__file__), "..", "mast3r")
sys.path.insert(0, os.path.join(_M, "dust3r"))   # its dust3r + croco
sys.path.insert(0, _M)                            # the mast3r package

CKPT = os.path.join(_M, "checkpoints",
                    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")


def load_matcher(device="cuda"):
    from mast3r.model import AsymmetricMASt3R
    return AsymmetricMASt3R.from_pretrained(CKPT).to(device).eval()


def match_pair(model, img_i_path, img_j_path, device="cuda", size=512,
               subsample=8, border=3):
    """Return (pix_i, pix_j, conf, (H,W)) for one ordered pair.
    pix_* are (M,2) int (row, col) in the size-512 resized frame."""
    import torch
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    images = load_images([img_i_path, img_j_path], size=size, verbose=False)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output["pred1"], output["pred2"]
    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()
    H, W = desc1.shape[:2]
    m0, m1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=subsample,
                                 device=device, dist="dot", block_size=2 ** 13)
    # m0, m1 are (M,2) as (x, y); drop border matches
    def ok(m):
        return ((m[:, 0] >= border) & (m[:, 0] < W - border) &
                (m[:, 1] >= border) & (m[:, 1] < H - border))
    keep = ok(m0) & ok(m1)
    m0, m1 = m0[keep], m1[keep]
    # confidence from desc_conf at matched pixels (geometric mean)
    dc1 = pred1["desc_conf"].squeeze(0).detach().cpu().numpy()
    dc2 = pred2["desc_conf"].squeeze(0).detach().cpu().numpy()
    c = np.sqrt(np.maximum(dc1[m0[:, 1], m0[:, 0]], 1e-6) *
                np.maximum(dc2[m1[:, 1], m1[:, 0]], 1e-6))
    pix_i = np.stack([m0[:, 1], m0[:, 0]], axis=1)   # (row, col)
    pix_j = np.stack([m1[:, 1], m1[:, 0]], axis=1)
    return pix_i.astype(np.int32), pix_j.astype(np.int32), c.astype(np.float32), (H, W)


def match_clip(image_paths, edges, out_dir, device="cuda", size=512,
               max_matches=3000):
    os.makedirs(out_dir, exist_ok=True)
    model = load_matcher(device)
    rng = np.random.default_rng(0)
    meta = {}
    for e_idx, (i, j) in enumerate(edges):
        pix_i, pix_j, conf, hw = match_pair(model, image_paths[i], image_paths[j],
                                            device=device, size=size)
        if len(conf) > max_matches:                  # cap by confidence
            sel = np.argsort(conf)[::-1][:max_matches]
            pix_i, pix_j, conf = pix_i[sel], pix_j[sel], conf[sel]
        np.savez_compressed(os.path.join(out_dir, f"{i}_{j}.npz"),
                            pix_i=pix_i, pix_j=pix_j, conf=conf)
        meta[f"{i}_{j}"] = {"n": int(len(conf)), "hw": list(hw)}
        print(f"  edge {i}->{j}: {len(conf)} matches  (HxW {hw})")
    return meta


def _parse_clip(clip_dir):
    import glob
    return sorted(glob.glob(os.path.join(clip_dir, "frame_*.png")))


def _edges(n):
    edges = [(k, k + 1) for k in range(n - 1)] + [(0, n - 1), (0, n // 2)]
    return [(a, b) for a, b in edges if a != b]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="single dir of frame_*.png")
    ap.add_argument("--out", help="output dir (single-clip mode)")
    ap.add_argument("--scenes", nargs="*", help="batch: scene names under root")
    ap.add_argument("--root", default="../data", help="batch: dataset root")
    ap.add_argument("--out_root", default="../cache", help="batch: cache root")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--nframes", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.scenes:                              # batch mode: one model load
        model = load_matcher(args.device)
        sub = "matches" if args.stride == 1 else f"matches_s{args.stride}"
        for sc in args.scenes:
            clip = os.path.join(args.root, "training", "clean", sc)
            paths = _parse_clip(clip)[::args.stride][:args.nframes]
            out = os.path.join(args.out_root, sc, sub)
            if os.path.exists(os.path.join(out, "0_1.npz")):
                print(f"  {sc}: cached, skip"); continue
            os.makedirs(out, exist_ok=True)
            edges = _edges(len(paths))
            print(f"matching {sc}: {len(paths)} frames, {len(edges)} edges")
            for e_idx, (i, j) in enumerate(edges):
                pi, pj, conf, hw = match_pair(model, paths[i], paths[j],
                                              device=args.device)
                if len(conf) > 3000:
                    sel = np.argsort(conf)[::-1][:3000]
                    pi, pj, conf = pi[sel], pj[sel], conf[sel]
                np.savez_compressed(os.path.join(out, f"{i}_{j}.npz"),
                                    pix_i=pi, pix_j=pj, conf=conf)
    else:                                        # single-clip mode
        paths = _parse_clip(args.clip)
        paths = [paths[k] for k in args.frames] if args.frames else paths[:8]
        edges = _edges(len(paths))
        print(f"matching {len(paths)} frames, {len(edges)} edges -> {args.out}")
        match_clip(paths, edges, args.out, device=args.device)
