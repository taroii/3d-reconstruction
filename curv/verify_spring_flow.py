"""
Find the correct Spring flow scale, fast and with NO model (pure image warping on
CPU). Builds the SAME composed multi-frame flow premise_cc uses, then sweeps a
multiplier u around premise_cc's current scale (u=1) and warps view-2's RGB into
view-1. The u that minimizes photometric error on moving pixels is the correct
scale:
  - best at u ~= 1  -> premise_cc's scale is right; the residual error is
    occlusion/disocclusion, not a scale bug, so its numbers can be trusted.
  - best at u ~= 2  -> the flow is 2x too small (drop the 0.5 in read_flow_hd).
  - best at u ~= 0.5-> too large.

  cd curv
  python verify_spring_flow.py --stride 4 --max_pairs 40
"""
import os
import sys
import argparse
import numpy as np
import torch

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

from cc_loss import flow_warp
from premise_cc import _grid_flow, _compose, _load_rgb, _spring_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/spring")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_pairs", type=int, default=40)
    ap.add_argument("--H", type=int, default=288)
    ap.add_argument("--W", type=int, default=512)
    args = ap.parse_args()
    H, W = args.H, args.W
    s = W / 1920.0                                  # HD-px -> grid-px (premise_cc's s)
    dev = "cpu"

    pairs = _spring_pairs(args.root, args.stride, args.max_pairs)
    if not pairs:
        raise SystemExit(f"no Spring pairs (with all intermediate flows) under {args.root}/train")
    print(f"{len(pairs)} pairs (stride {args.stride}); u=1 is premise_cc's current scale")

    US = [0.5, 0.71, 1.0, 1.41, 2.0]
    base = [0.0, 0]
    acc = {u: [0.0, 0] for u in US}

    for rgb_a, rgb_b, seq, idx in pairs:
        f = _compose([_grid_flow(args.root, seq, idx + k, "FW", H, W, s, dev)
                      for k in range(args.stride)])
        hi = (f.norm(dim=1, keepdim=True) > 2.0)     # pixels that actually move
        if hi.sum() < 500:
            continue
        rgb1, rgb2 = _load_rgb(rgb_a, H, W, dev), _load_rgb(rgb_b, H, W, dev)
        hi3 = hi.expand(1, 3, H, W)
        base[0] += float((rgb1[hi3] - rgb2[hi3]).abs().sum()); base[1] += int(hi.sum())
        for u in US:
            warped, inb = flow_warp(rgb2, f * u)
            m = hi & (inb > 0.5)
            m3 = m.expand(1, 3, H, W)
            if m.sum() > 100:
                acc[u][0] += float((warped[m3] - rgb1[m3]).abs().sum()); acc[u][1] += int(m.sum())

    b = base[0] / max(base[1], 1)
    print(f"\nRGB error on moving pixels (|flow| > 2 px):")
    print(f"  {'unwarped baseline':20s} {b:8.4f}")
    errs = {u: acc[u][0] / acc[u][1] for u in US if acc[u][1]}
    for u in US:
        if u in errs:
            print(f"  warp, scale x {u:<5g}    {errs[u]:8.4f}")
    if not errs:
        raise SystemExit("no moving pixels found; try a larger --stride")
    bu = min(errs, key=errs.get)
    print(f"\nbest multiplier u = {bu}  (error {errs[bu]:.4f} vs baseline {b:.4f})")
    if abs(bu - 1.0) < 1e-6:
        print("=> premise_cc's scale is correct. Residual error is occlusion, not a scale")
        print("   bug, so premise_cc's dynamic-vs-static numbers are trustworthy.")
    else:
        print(f"=> premise_cc's flow is off by ~{bu}x. Multiply the read_flow_hd scale by")
        print(f"   {bu} (e.g. 0.5 -> {0.5*bu:g}) and rerun premise_cc.")


if __name__ == "__main__":
    main()
