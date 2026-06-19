"""
Depth eval for the curvature finetune. Reports AbsRel / delta1 on held-out
frames, broken down by region:
  ALL      - every valid pixel
  BOUNDARY - top (100-bpct)% of pixels by |GT curvature| (where the curvature
             hypothesis predicts the gain)
  DYNAMIC  - GT dynamic-mask pixels, if the dataset provides one (e.g. Sintel)

Runs the D2USt3R model pairwise (reusing DDUSt3R's loss_of_one_batch with
criterion=None) and aligns predicted depth to GT per-frame by MEDIAN SCALE
(scale-invariant; identical alignment for every arm -> the A/B delta is fair).
GT curvature for the boundary mask reuses curv/curvature.py.

  cd curv
  # PointOdyssey val (held-out sequences, in-distribution)
  python eval_depth.py --ckpt ../DDUSt3R/results/armA/checkpoint-best.pth \
    --dataset "400 @ PointOdysseyDUSt3R(dset='val', dataset_location='../data/pointodyssey', S=2, strides=[4], resolution=(512,288))"
  # Sintel (zero-shot; carries a dynamic mask)
  python eval_depth.py --ckpt ../DDUSt3R/results/armA/checkpoint-best.pth \
    --dataset "300 @ SintelDUSt3R(dataset_location='../data/training', dset='clean', S=2, strides=[7], resolution=(512,224), load_dynamic_mask=False)"

Run the SAME --dataset for every arm (armA, armB_g1, armB_g4) and compare the
deltas. Also run it once on the base ../DDUSt3R/checkpoints/ddust3r.pth as a
reference point.
"""
import os
import sys
import argparse
import numpy as np
import torch

_DD = os.path.join(os.path.dirname(__file__), "..", "DDUSt3R")
if _DD not in sys.path:
    sys.path.insert(0, _DD)

import curvature as CV


def _metrics(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    if p.size == 0:
        return None
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    return absrel, d1, int(p.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="DUSt3R 'N @ Dataset(...)' string")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--bpct", type=float, default=90.0,
                    help="BOUNDARY = top (100-bpct)%% of pixels by |GT curvature|")
    args = ap.parse_args()
    dev = "cuda"

    from dust3r.model import load_model
    from dust3r.datasets import get_data_loader
    try:
        from dust3r.inference import loss_of_one_batch
    except ImportError:
        from dust3r.training import loss_of_one_batch

    model = load_model(args.ckpt, dev, verbose=False)
    model.eval()
    loader = get_data_loader(args.dataset, batch_size=args.bs, num_workers=4,
                             shuffle=False, drop_last=False)

    agg = {}  # region -> [sum_absrel*n, sum_d1*n, n_px, n_frames]

    def add(region, m):
        if m is None:
            return
        a, d, n = m
        s = agg.setdefault(region, [0.0, 0.0, 0.0, 0.0])
        s[0] += a * n; s[1] += d * n; s[2] += n; s[3] += 1

    for batch in loader:
        with torch.no_grad():
            res = loss_of_one_batch(batch, model, None, dev,
                                    symmetrize_batch=False, use_amp=True)
        pred1, view1 = res["pred1"], res["view1"]
        pred_d = pred1["pts3d"].detach().float().cpu().numpy()[..., 2]   # B,H,W (cam-z)
        gt_d = view1["depthmap"].float().cpu().numpy()                   # B,H,W
        valid = view1["valid_mask"].cpu().numpy().astype(bool)
        K = view1["camera_intrinsics"].float().cpu().numpy()            # B,3,3
        dyn = None
        if "dynamic_mask" in view1:
            dyn = view1["dynamic_mask"].cpu().numpy().astype(bool)

        for b in range(pred_d.shape[0]):
            vd = valid[b] & (gt_d[b] > 0) & np.isfinite(pred_d[b])
            if vd.sum() < 100:
                continue
            scale = np.median(gt_d[b][vd]) / max(np.median(pred_d[b][vd]), 1e-6)
            pa = pred_d[b] * scale                                       # aligned pred depth
            add("ALL", _metrics(pa, gt_d[b], vd))

            # BOUNDARY = high |GT curvature| (computed from GT depth + K)
            Kmap, kvalid = CV.curvature_from_depth_K(gt_d[b], K[b], mode="mean",
                                                     normalize=True)
            vk = vd & kvalid
            if vk.sum() > 50:
                thr = np.percentile(np.abs(Kmap[vk]), args.bpct)
                add("BOUNDARY", _metrics(pa, gt_d[b], vk & (np.abs(Kmap) >= thr)))

            if dyn is not None and (vd & dyn[b]).sum() > 50:
                add("DYNAMIC", _metrics(pa, gt_d[b], vd & dyn[b]))

    name = args.dataset.split("@")[1].split("(")[0].strip()
    print(f"\nckpt    {args.ckpt}")
    print(f"dataset {name}")
    print(f"{'region':9s} {'AbsRel':>8s} {'delta1':>8s} {'frames':>7s}")
    for region in ("ALL", "BOUNDARY", "DYNAMIC"):
        if region in agg:
            s0, s1, npx, nf = agg[region]
            print(f"{region:9s} {s0/npx:8.4f} {s1/npx:8.4f} {int(nf):7d}")


if __name__ == "__main__":
    main()
