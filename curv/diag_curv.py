"""
Diagnose a trained curvature head: is it actually capturing curvature, or just
regressing toward zero? Global MAE is dominated by the ~flat majority of pixels
(where predict-zero is unbeatable), so it can't answer this. We report
curvature-FOCUSED metrics on held-out val frames:

  - correlation(pred, GT) over all valid pixels, and over the high-|K| tail,
  - MAE(model) vs MAE(predict-zero) on the high-|K| tail (the pixels we care
    about). A head that learned edges beats predict-zero HERE even if it loses
    on the global average.

Also dumps side-by-side GT/pred PNGs so you can eyeball whether the prediction
fires on the same edges/corners as GT.

  python diag_curv.py --ckpt checkpoints/curv_tartanair_mean.pth --dataset tartanair --n 60
"""
import argparse
import numpy as np
import torch

import backbone as B
import datasets as DS
import curvature as CV
from curvature_head import CurvatureHead
from train_curv import cached_grid_curvature


def _pearson(sx, sy, sxx, syy, sxy, n):
    if n < 2:
        return float("nan")
    cov = sxy / n - (sx / n) * (sy / n)
    vx = max(sxx / n - (sx / n) ** 2, 0.0)
    vy = max(syy / n - (sy / n) ** 2, 0.0)
    return cov / max((vx * vy) ** 0.5, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="tartanair")
    ap.add_argument("--split", default="val", choices=["train", "val"],
                    help="val = generalization; train = can the head even fit it?")
    ap.add_argument("--n", type=int, default=60, help="frames to score")
    ap.add_argument("--pct", type=float, default=90.0, help="high-|K| tail percentile")
    ap.add_argument("--dump", type=int, default=4, help="GT/pred PNG pairs to write")
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location=dev)
    mode = ck.get("mode", "mean")
    compress = ck.get("compress", True)
    tscale = ck.get("target_scale", 1.0)        # match training standardization
    bb = B.load_backbone(ck.get("backbone", "d2ust3r"))
    net = CurvatureHead(bb).to(dev)
    if ck.get("hooks"):
        net.hooks = ck["hooks"]
    net.dpt.load_state_dict(ck["head"])
    net.eval()
    print(f"loaded {args.ckpt}  mode={mode} compress={compress}", flush=True)

    from dust3r.utils.image import load_images
    import random
    frames = DS.build_frames([args.dataset], args.split)
    random.seed(0); random.shuffle(frames); frames = frames[:args.n]

    # streaming accumulators (global + high-|K| tail + coarse 16x16-pooled)
    G = dict(sx=0., sy=0., sxx=0., syy=0., sxy=0., n=0., mae_m=0., mae_z=0.)
    H = dict(sx=0., sy=0., sxx=0., syy=0., sxy=0., n=0., mae_m=0., mae_z=0.)
    P = dict(sx=0., sy=0., sxx=0., syy=0., sxy=0., n=0., mae_m=0., mae_z=0.)

    def block_acc(D, pred, gt, valid, b=16):
        h, w = pred.shape
        h2, w2 = (h // b) * b, (w // b) * b
        blk = lambda a: a[:h2, :w2].reshape(h2 // b, b, w2 // b, b)
        m = blk(valid.astype(np.float32)); cnt = m.sum(axis=(1, 3))
        ok = cnt > 0.5 * b * b
        pm = (blk(pred) * m).sum(axis=(1, 3))[ok] / cnt[ok]
        qm = (blk(gt) * m).sum(axis=(1, 3))[ok] / cnt[ok]
        D["sx"] += pm.sum(); D["sy"] += qm.sum()
        D["sxx"] += (pm * pm).sum(); D["syy"] += (qm * qm).sum()
        D["sxy"] += (pm * qm).sum(); D["n"] += pm.size

    dumped = 0
    for fr in frames:
        g = load_images([fr.rgb_path], size=512, verbose=False)[0]
        img = g["img"].to(dev)
        ts = torch.tensor(np.asarray(g["true_shape"])).to(dev)
        with torch.no_grad():
            pred, _ = net(img, ts)
        pred = pred[0].cpu().numpy()
        gt, valid = cached_grid_curvature(fr, mode, compress)
        gt = gt / tscale                            # same space as the trained head
        v = valid & np.isfinite(pred)
        p, q = pred[v], gt[v]
        if p.size == 0:
            continue

        def acc(D, pp, qq):
            D["sx"] += pp.sum(); D["sy"] += qq.sum()
            D["sxx"] += (pp * pp).sum(); D["syy"] += (qq * qq).sum()
            D["sxy"] += (pp * qq).sum(); D["n"] += pp.size
            D["mae_m"] += np.abs(pp - qq).sum(); D["mae_z"] += np.abs(qq).sum()

        acc(G, p, q)
        thr = np.percentile(np.abs(q), args.pct)
        hk = np.abs(q) >= thr
        acc(H, p[hk], q[hk])
        block_acc(P, pred, gt, v)

        if dumped < args.dump:
            import imageio.v2 as imageio

            def rgb(a):
                return (CV.curvature_to_rgb(np.where(valid, a, 0.0)) * 255).astype(np.uint8)
            key = fr.key.replace("/", "_")
            imageio.imwrite(f"diag_{key}_gt.png", rgb(gt))
            imageio.imwrite(f"diag_{key}_pred.png", rgb(pred))
            dumped += 1

    print(f"\nsplit={args.split}  frames scored {len(frames)}  valid px {int(G['n'])}")
    print(f"GLOBAL        corr {_pearson(G['sx'],G['sy'],G['sxx'],G['syy'],G['sxy'],G['n']):+.3f}"
          f"   MAE model {G['mae_m']/max(G['n'],1):.4f}   predict-zero {G['mae_z']/max(G['n'],1):.4f}")
    print(f"TOP-{100-args.pct:.0f}% |K|    corr {_pearson(H['sx'],H['sy'],H['sxx'],H['syy'],H['sxy'],H['n']):+.3f}"
          f"   MAE model {H['mae_m']/max(H['n'],1):.4f}   predict-zero {H['mae_z']/max(H['n'],1):.4f}")
    print(f"POOLED 16x16  corr {_pearson(P['sx'],P['sy'],P['sxx'],P['syy'],P['sxy'],P['n']):+.3f}"
          f"   (coarse-scale; absorbs sub-pixel offset / head smoothness)")
    print(f"\nverdict: TOP-tail corr >0 with MAE<predict-zero => captures curvature."
          f" If POOLED >> GLOBAL => coarse signal exists, fine-scale lost.")
    print(f"dumped {dumped} GT/pred PNG pairs (diag_*_gt.png / diag_*_pred.png)")


if __name__ == "__main__":
    main()
