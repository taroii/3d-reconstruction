"""
PREMISE TEST for L_cc (the direct one): does the BASE model's PREDICTED curvature
DISAGREE across the two views at DYNAMIC pixels, while AGREEING at STATIC ones?
That cross-view inconsistency is exactly the residual L_cc removes, so this is the
real go/no-go for the Faithful/L_cc build (premise_test.py tested Option 2's
premise; this tests L_cc's own). Inference only -- NO training. Needs Spring (for
dense GT flow) downloaded.

For each Spring pair (i, i+stride):
  - run the base model -> X1 = X_hat^{1,1} (view-1 grid), X2 = X_hat^{2,1}
    (view-2 grid, expressed in view-1 frame)
  - kappa(X1), kappa(X2); warp kappa(X2) into the view-1 grid via GT forward flow
    f, and measure inconsistency  |kappa(X1) - warp(kappa(X2), f)|
  - split pixels by M_dyn (Eq 5: ||f_cam - f_gt|| > tau, f_cam from GT depth+pose)
    minus occlusions (Eq 4, fwd/bwd GT flow), then compare the inconsistency on
    DYNAMIC vs STATIC pixels.

SELF-CHECK built in: on STATIC pixels the two views are rigidly aligned, so their
predicted curvature MUST already agree -> static inconsistency ~ 0. If static is
NOT near zero, the Spring flow/pose conventions (the /2 flow scale, the
world->cam extrinsic assumption) are wrong -- fix those before trusting the
dynamic number. The verdict we want: dynamic inconsistency >> static.

  cd curv
  python premise_cc.py --device cuda --max_pairs 150
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

import spring as SP
import dynamic as DY
from cc_loss import kappa, flow_warp


def _to_grid(arr, H, W, mode):
    """Resize an (H0,W0) or (H0,W0,C) numpy array to (H,W) -> torch (1,C,H,W)."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.ndim == 2:
        t = t[None, None]
    else:
        t = t.permute(2, 0, 1)[None]
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(t, size=(H, W), mode=mode, **kw)


def _relative_pose(E1, E2):
    """cam1->cam2 assuming E = world->cam: X_cam2 = E2 @ inv(E1) @ X_cam1."""
    return E2 @ np.linalg.inv(E1)


def _load_rgb(path, H, W, dev):
    import PIL.Image
    im = PIL.Image.open(path).convert("RGB").resize((W, H), PIL.Image.BILINEAR)
    return torch.from_numpy(np.array(im)).float().permute(2, 0, 1)[None].to(dev) / 255.0


def _grid_flow(root, seq, frame, direction, H, W, s, dev):
    """One consecutive Spring flow (FW: frame->frame+1, BW: frame->frame-1) resized
    to the (H,W) grid and scaled by s. (1,2,H,W)."""
    p = SP.flow_path(root, "train", seq, frame, direction)
    return _to_grid(SP.read_flow_hd(p), H, W, "bilinear").to(dev) * s


def _compose(flows):
    """Compose consecutive flows f_{k->k+1} into f_{0->n}: warp each next flow by
    the running composite and add. A single Spring flow spans ONE frame, so a
    stride-n correspondence needs this chain. flows: list of (1,2,H,W)."""
    comp = flows[0]
    for nxt in flows[1:]:
        w, _ = flow_warp(nxt, comp)
        comp = comp + w
    return comp


def _spring_pairs(root, stride, max_pairs):
    """(rgb_a, rgb_b, seq, idx_a) pairs `stride` frames apart, keeping only those
    with every intermediate FW and BW flow present (needed to compose a correct
    i->i+stride correspondence)."""
    pairs = []
    for seqdir in sorted(glob.glob(os.path.join(root, "train", "*"))):
        if not os.path.isdir(seqdir):
            continue
        seq = os.path.basename(seqdir)
        rgbs = SP.frame_paths(root, "train", seq)
        for a in range(len(rgbs) - stride):
            idx = int("".join(filter(str.isdigit, os.path.basename(rgbs[a])))[-4:])
            fok = all(SP.flow_path(root, "train", seq, idx + k, "FW") for k in range(stride))
            bok = all(SP.flow_path(root, "train", seq, idx + stride - k, "BW") for k in range(stride))
            if fok and bok:
                pairs.append((rgbs[a], rgbs[a + stride], seq, idx))
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../DDUSt3R/checkpoints/ddust3r.pth")
    ap.add_argument("--root", default="../data/spring")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=150)
    ap.add_argument("--tau", type=float, default=1.5, help="M_dyn px threshold (Eq 5)")
    ap.add_argument("--occ", type=float, default=1.5, help="M_occ px threshold (Eq 4)")
    args = ap.parse_args()
    dev = args.device

    from dust3r.model import load_model
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    model = load_model(args.ckpt, dev, verbose=False)
    model.eval()

    pairs = _spring_pairs(args.root, args.stride, args.max_pairs)
    if not pairs:
        raise SystemExit(f"no Spring flow pairs under {args.root}/train -- "
                         "is the flow (.flo5) download present?")
    print(f"{len(pairs)} Spring pairs (stride {args.stride})")

    # accumulators: [sum_inconsistency, n_pixels, sum_|kappa(X1)|]
    dyn = [0.0, 0, 0.0]
    stat = [0.0, 0, 0.0]
    dyn_frac = []
    rgbchk = [0.0, 0]   # warped-view2 vs view1 photometric error (flow self-check)
    rgbbase = [0.0, 0]  # unwarped view2 vs view1

    for rgb_a, rgb_b, seq, idx_a in pairs:
        try:
            imgs = load_images([rgb_a, rgb_b], size=512, verbose=False)
            mp = make_pairs(imgs, scene_graph="complete", prefilter=None,
                            symmetrize=False)
            out = inference(mp, model, dev, batch_size=1, verbose=False)
        except Exception as e:
            print(f"  skip {seq}/{idx_a}: inference failed ({e})")
            continue

        # 'complete' returns both directions; the forward (view0->view1) pair is
        # index 0 -- that's the one our GT forward flow f corresponds to.
        X1 = out["pred1"]["pts3d"][:1].float()                   # X_hat^{1,1}
        p2 = out["pred2"]
        X2 = (p2["pts3d_in_other_view"] if "pts3d_in_other_view" in p2
              else p2["pts3d"])[:1].float()                      # X_hat^{2,1}
        H, W = X1.shape[1], X1.shape[2]

        # --- GT geometry for view 1, resized onto the (H,W) grid ---
        cam = os.path.join(args.root, "train", seq, "cam_data")
        disp_p = os.path.join(args.root, "train", seq, "disp1_left",
                              f"disp1_left_{idx_a:04d}.dsp5")
        disp = SP.read_dsp5(disp_p)[::2, ::2]
        K = SP.read_intrinsics(os.path.join(cam, "intrinsics.txt"), idx_a)
        depth0 = SP.disp_to_depth(disp, float(K[0, 0]))          # native HD z-depth
        H0, W0 = depth0.shape
        s = W / W0                                               # pure-resize factor
        E1 = SP.read_extrinsics(os.path.join(cam, "extrinsics.txt"), idx_a)
        E2 = SP.read_extrinsics(os.path.join(cam, "extrinsics.txt"), idx_a + args.stride)
        T = torch.from_numpy(_relative_pose(E1, E2)).float()[None].to(dev)
        Kg = K.copy(); Kg[:2] *= s
        Kg = torch.from_numpy(Kg).float()[None].to(dev)

        depthg = _to_grid(depth0, H, W, "nearest").to(dev)[:, 0]  # (1,H,W)
        validg = torch.isfinite(depthg) & (depthg > 0)
        depthg = torch.nan_to_num(depthg, nan=0.0)

        # correct i->i+stride correspondence: compose the consecutive GT flows
        f = _compose([_grid_flow(args.root, seq, idx_a + k, "FW", H, W, s, dev)
                      for k in range(args.stride)])
        b = _compose([_grid_flow(args.root, seq, idx_a + args.stride - k, "BW", H, W, s, dev)
                      for k in range(args.stride)])

        # flow self-check: warp view-2 RGB by f; a correctly scaled/directed flow
        # reconstructs view-1, so warped error << unwarped baseline
        rgb1, rgb2 = _load_rgb(rgb_a, H, W, dev), _load_rgb(rgb_b, H, W, dev)
        r2w, rinb = flow_warp(rgb2, f)
        rm = (rinb > 0.5).expand_as(rgb1)
        rgbchk[0] += float((r2w[rm] - rgb1[rm]).abs().sum()); rgbchk[1] += int(rm.sum())
        rgbbase[0] += float((rgb2[rm] - rgb1[rm]).abs().sum()); rgbbase[1] += int(rm.sum())

        f_cam = DY.cam_flow(depthg, Kg, Kg, T)
        Mdyn = DY.dynamic_mask(f_cam, f, tau=args.tau)            # (1,1,H,W)
        Mocc = DY.occlusion_mask(f, b, t=args.occ)

        # --- cross-view curvature inconsistency on the view-1 grid ---
        k1, k2 = kappa(X1), kappa(X2)
        k2w, inb = flow_warp(k2, f)
        incons = (k1 - k2w).abs()                                # (1,1,H,W)

        base = (Mocc < 0.5) & (inb > 0.5) & validg[:, None]
        dset = base & (Mdyn > 0.5)
        sset = base & (Mdyn < 0.5)
        ak1 = k1.abs()
        if dset.sum() > 0:
            dyn[0] += float(incons[dset].sum()); dyn[1] += int(dset.sum())
            dyn[2] += float(ak1[dset].sum())
        if sset.sum() > 0:
            stat[0] += float(incons[sset].sum()); stat[1] += int(sset.sum())
            stat[2] += float(ak1[sset].sum())
        dyn_frac.append(float((Mdyn > 0.5).float().mean()))

    d_mean, d_k = dyn[0] / max(dyn[1], 1), dyn[2] / max(dyn[1], 1)
    s_mean, s_k = stat[0] / max(stat[1], 1), stat[2] / max(stat[1], 1)
    # relative = inconsistency as a fraction of the curvature signal itself
    print(f"\n{'region':9s} {'mean|incons|':>13s} {'mean|kappa|':>12s} "
          f"{'incons/kappa':>13s} {'n_px':>14s}")
    print(f"{'DYNAMIC':9s} {d_mean:13.5f} {d_k:12.5f} "
          f"{d_mean / max(d_k, 1e-9):13.3f} {dyn[1]:14,d}")
    print(f"{'STATIC':9s} {s_mean:13.5f} {s_k:12.5f} "
          f"{s_mean / max(s_k, 1e-9):13.3f} {stat[1]:14,d}")
    print(f"\nmean dynamic-mask fraction = {np.mean(dyn_frac):.3f}  (stride {args.stride})")
    print(f"dynamic / static inconsistency ratio = {d_mean / max(s_mean, 1e-9):.2f}")

    rc = rgbchk[0] / max(rgbchk[1], 1)
    rb = rgbbase[0] / max(rgbbase[1], 1)
    ok = rc < 0.7 * rb
    print(f"\nFLOW SELF-CHECK (RGB warp): warped {rc:.4f} vs unwarped {rb:.4f}  "
          f"-> {'flow aligns, numbers above are trustworthy' if ok else 'FLOW MIS-ALIGNED, DO NOT trust the numbers above'}")
    if not ok:
        print("  (the composed flow does not reconstruct view 1; check the read_flow_hd")
        print("   scale or the FW/BW direction before drawing any L_cc conclusion.)")
    else:
        print("VERDICT for L_cc: only if dynamic >> static AND incons/kappa is small on")
        print("dynamic pixels does L_cc have a clean residual to remove. Note curvature is")
        print("only rigid-invariant, so genuine non-rigid deformation also shows up as")
        print("dynamic inconsistency that L_cc cannot separate from prediction error.")


if __name__ == "__main__":
    main()
