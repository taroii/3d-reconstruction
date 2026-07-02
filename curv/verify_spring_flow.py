"""
Verify the Spring GT optical-flow convention that premise_cc.py relies on, with
NO model and NO GPU (pure image warping on CPU, safe to run alongside training).

premise_cc warps view-2 quantities into view-1 with the forward flow, loaded as
  f = read_flow_hd(fw) [ = flo5[::2,::2] * 0.5 ]  ->  resized to grid  *  (W/W0)
If that scale/direction is correct, warping view-2's RGB by the same f should
reconstruct view-1's RGB: the photometric error drops sharply versus the unwarped
baseline. If the /2 factor or the resize is wrong, warping does not help (or hurts).

The script reports, averaged over pairs, the mean absolute RGB error for:
  - baseline: |rgb1 - rgb2|            (no warp)
  - warp, /2 scale (premise_cc's current convention)
  - warp, no /2   (scale 1.0)
The convention whose warp gives the LOWEST error is the correct one. If neither
warp beats the baseline, the direction or another convention is off, and
premise_cc's numbers are a warp artifact rather than a real measurement.

  cd curv
  python verify_spring_flow.py --max_pairs 30            # numbers only
  python verify_spring_flow.py --max_pairs 30 --save     # + a side-by-side PNG
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

import spring as SP
from cc_loss import flow_warp
import torch.nn.functional as F


def _load_rgb(path, H, W):
    import PIL.Image
    im = PIL.Image.open(path).convert("RGB").resize((W, H), PIL.Image.BILINEAR)
    return torch.from_numpy(np.array(im)).float().permute(2, 0, 1)[None] / 255.0


def _flow_to_grid(flo, H, W):
    """Resize a raw (Hn,Wn,2) flo5 flow to the (H,W) grid (values still in native
    pixels) and apply the resolution scale W/Wn to the vectors. Returns (1,2,H,W),
    the u=1 reference flow. A remaining unit factor is swept separately."""
    t = torch.from_numpy(np.ascontiguousarray(flo)).float().permute(2, 0, 1)[None]
    Wn = flo.shape[1]
    return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False) * (W / Wn)


def _pairs(root, stride, n):
    out = []
    for seqdir in sorted(glob.glob(os.path.join(root, "train", "*"))):
        seq = os.path.basename(seqdir)
        rgbs = SP.frame_paths(root, "train", seq)
        for a in range(len(rgbs) - stride):
            idx = int("".join(filter(str.isdigit, os.path.basename(rgbs[a])))[-4:])
            fw = SP.flow_path(root, "train", seq, idx, "FW")
            if fw:
                out.append((rgbs[a], rgbs[a + stride], fw))
            if n and len(out) >= n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/spring")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=30)
    ap.add_argument("--H", type=int, default=288)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--save", action="store_true", help="dump a side-by-side PNG for pair 0")
    args = ap.parse_args()
    H, W = args.H, args.W

    pairs = _pairs(args.root, args.stride, args.max_pairs)
    if not pairs:
        raise SystemExit(f"no Spring FW-flow pairs under {args.root}/train")
    print(f"{len(pairs)} pairs (stride {args.stride})")

    US = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]     # extra unit factor on top of W/Wn
    acc = {"baseline": [0.0, 0]}
    for u in US:
        acc[u] = [0.0, 0]

    def add(key, err, n):
        acc[key][0] += err * n; acc[key][1] += n

    shape_printed = False
    for k, (p1, p2, fw) in enumerate(pairs):
        rgb1, rgb2 = _load_rgb(p1, H, W), _load_rgb(p2, H, W)
        flo = SP.read_flo5(fw).astype(np.float32)               # RAW (Hn,Wn,2), no subsample
        if not shape_printed:
            print(f"raw .flo5 shape {flo.shape}  (grid {H}x{W}; resolution scale W/Wn = {W/flo.shape[1]:.4f})")
            shape_printed = True
        ref = _flow_to_grid(flo, H, W)                          # u=1 grid flow (1,2,H,W)
        mag = ref.norm(dim=1, keepdim=True)                    # in grid pixels
        hi = (mag > 3.0)                                        # high-flow pixels only
        nhi = int(hi.sum())
        if nhi < 500:
            continue
        rgb1e = rgb1.expand(1, 3, H, W)
        hi3 = hi.expand(1, 3, H, W)
        add("baseline", float((rgb1e[hi3] - rgb2.expand(1, 3, H, W)[hi3]).abs().mean()), nhi)
        best_u, best_warp = None, None
        for u in US:
            warped, inb = flow_warp(rgb2, ref * u)
            m = hi & (inb > 0.5)
            if m.sum() < 100:
                continue
            m3 = m.expand(1, 3, H, W)
            err = float((warped[m3] - rgb1e[m3]).abs().mean())
            add(u, err, int(m.sum()))
        if args.save and k == 0:
            for u in US:
                warped, _ = flow_warp(rgb2, ref * u)
                _dump(rgb1, rgb2, warped, f"u{u}", u)

    b = acc["baseline"][0] / max(acc["baseline"][1], 1)
    print(f"\nHigh-flow-pixel RGB error (|flow| > 3 px):")
    print(f"  {'unwarped baseline':22s} {b:8.4f}")
    errs = {}
    for u in US:
        s, n = acc[u]
        if n:
            errs[u] = s / n
            print(f"  warp, scale = (W/Wn)*{u:<6g} {errs[u]:8.4f}")
    if not errs:
        raise SystemExit("no high-flow pixels found; try --stride 8 or more pairs")
    bu = min(errs, key=errs.get)
    be = errs[bu]
    print(f"\nBest scale factor u = {bu} (grid error {be:.4f} vs baseline {b:.4f}).")
    if be < 0.6 * b:
        note = "aligns" if abs(bu - 1.0) < 1e-6 else f"aligns, but at u={bu}, NOT u=1"
        print(f"  => flow warps correctly at u={bu} ({note}).")
        print(f"     premise_cc uses [::2,::2] then *0.5 then *(W/Wn_hd); express the winning")
        print(f"     u as the equivalent read_flow_hd scale and set spring.read_flow_hd to it,")
        print(f"     then rerun premise_cc for clean numbers.")
    else:
        print("  => even the best scale barely beats the unwarped baseline. The forward-flow")
        print("     warp does not reconstruct view 1, so the flow direction/units are off in a")
        print("     way a single scale does not fix. premise_cc's numbers are not trustworthy;")
        print("     do not rely on them in the paper until the flow loader is corrected.")


def _dump(rgb1, rgb2, warped, key, sc):
    import PIL.Image
    def to_img(t):
        a = torch.nan_to_num(t[0], nan=0.0).clamp(0, 1).permute(1, 2, 0).numpy()
        return (a * 255).astype(np.uint8)
    diff = (rgb1 - warped).abs().clamp(0, 1)
    strip = np.concatenate([to_img(rgb1), to_img(rgb2), to_img(warped), to_img(diff)], axis=1)
    name = f"verify_flow_{key.replace(' ', '_').replace('/', '')}.png"
    PIL.Image.fromarray(strip).save(name)
    print(f"  saved {name}  (view1 | view2 | warp(view2) | |view1-warp|)")


if __name__ == "__main__":
    main()
