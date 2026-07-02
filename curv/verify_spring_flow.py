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
    return torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0


def _flow_to_grid(flo_hd, H, W):
    """Resize an (H0,W0,2) HD flow to the (H,W) grid and scale the vectors by W/W0
    (same as premise_cc). Returns (1,2,H,W)."""
    t = torch.from_numpy(np.ascontiguousarray(flo_hd)).float().permute(2, 0, 1)[None]
    W0 = flo_hd.shape[1]
    return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False) * (W / W0)


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

    acc = {"baseline": [0.0, 0], "warp /2": [0.0, 0], "warp no-/2": [0.0, 0]}

    def add(key, err, n):
        acc[key][0] += err * n; acc[key][1] += n

    for k, (p1, p2, fw) in enumerate(pairs):
        rgb1, rgb2 = _load_rgb(p1, H, W), _load_rgb(p2, H, W)
        flo = SP.read_flo5(fw)[::2, ::2].astype(np.float32)      # HD (H0,W0,2), no scale yet
        # whole-image baseline (no warp)
        add("baseline", float((rgb1 - rgb2).abs().mean()), H * W)
        for key, sc in (("warp /2", 0.5), ("warp no-/2", 1.0)):
            f = _flow_to_grid(flo * sc, H, W)
            warped, inb = flow_warp(rgb2, f)                     # view2 RGB -> view1 grid
            m = inb.expand_as(warped) > 0.5
            if m.any():
                err = float((warped[m] - rgb1.expand_as(warped)[m]).abs().mean())
                add(key, err, int(inb.sum()))
            if args.save and k == 0:
                _dump(rgb1, rgb2, warped, key, sc)

    print(f"\n{'variant':12s} {'mean |RGB err|':>16s}")
    for key in ("baseline", "warp /2", "warp no-/2"):
        s, n = acc[key]
        print(f"{key:12s} {s / max(n,1):16.4f}")
    b = acc["baseline"][0] / max(acc["baseline"][1], 1)
    h = acc["warp /2"][0] / max(acc["warp /2"][1], 1)
    fu = acc["warp no-/2"][0] / max(acc["warp no-/2"][1], 1)
    print("\nReading:")
    print(f"  warp /2 vs baseline:   {h:.4f} vs {b:.4f}  "
          f"({'aligns' if h < 0.7*b else 'does NOT align'})")
    print(f"  warp no-/2 vs baseline:{fu:.4f} vs {b:.4f}  "
          f"({'aligns' if fu < 0.7*b else 'does NOT align'})")
    best = min(("warp /2", h), ("warp no-/2", fu), key=lambda x: x[1])[0]
    if min(h, fu) < 0.7 * b:
        print(f"  => flow convention OK; best is '{best}'. If that is 'warp /2', "
              f"premise_cc's scale is correct and its inconsistency numbers are real.")
    else:
        print("  => NEITHER warp aligns the images. The flow direction/scale is off; "
              "premise_cc's inconsistency is a warp artifact and must be fixed before use.")


def _dump(rgb1, rgb2, warped, key, sc):
    import PIL.Image
    def to_img(t):
        return (t[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    diff = (rgb1 - warped).abs().clamp(0, 1)
    strip = np.concatenate([to_img(rgb1), to_img(rgb2), to_img(warped), to_img(diff)], axis=1)
    name = f"verify_flow_{key.replace(' ', '_').replace('/', '')}.png"
    PIL.Image.fromarray(strip).save(name)
    print(f"  saved {name}  (view1 | view2 | warp(view2) | |view1-warp|)")


if __name__ == "__main__":
    main()
