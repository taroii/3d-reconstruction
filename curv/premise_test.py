"""
PREMISE TEST (Task 1): is the BASE model's depth error actually higher on
high-curvature pixels? If not, the curvature-weighted loss (Option 2) has no
headroom to exploit. Inference only -- NO training, no server GPU needed. Runs
on a local GPU (--device cuda) or CPU (--device cpu).

This is exactly eval_depth.py's "BOUNDARY" idea (top-10% |K|) generalized to all
ten deciles, computed PER-PIXEL on the *base* checkpoint we finetune from. We
reuse eval_depth's depth extraction + per-frame median-scale alignment, then:

  - per-pixel AbsRel  r_i = |d_hat_i - d_i| / d_i   (eval_depth reports the frame
    MEAN; here we keep every pixel for stratification)
  - per-pixel delta1  (max(d_hat/d, d/d_hat) < 1.25)
  - GT curvature |K_i| from curvature.curvature_from_depth_K (SAME estimator,
    normalization, and depth-discontinuity/interior validity mask as training/
    eval), restricted to pixels valid under BOTH the depth-eval mask and the
    curvature validity mask.
  - the per-image-normalized signal  S_i = |K_i| / mean_valid(|K|)  -- identical
    to the loss weight w_i = 1 + gamma*min(S_i/S_bar, tau); deciles are taken on
    S so the binning matches the weight exactly.

Pool pixels across all frames, bin into deciles of S, and report per decile:
mean AbsRel, mean delta1, pixel count. Done twice: (a) all valid pixels, and
(b) dynamic-only pixels (view1['dynamic_mask'], if the dataset provides it).
Also: Spearman rho(S_i, r_i) and the top/bottom-decile AbsRel ratio.

Outputs per run:  premise_<dataset>.csv  +  premise_<dataset>.png

  cd curv
  # PointOdyssey-val (in-distribution, held-out sequences)
  python premise_test.py --device cuda \
    --dataset "200 @ PointOdysseyDUSt3R(dset='val', dataset_location='../data/pointodyssey', S=2, strides=[4], resolution=(512,288))"
  # Sintel (zero-shot); request the dynamic mask so the dynamic-only curve fills in
  python premise_test.py --device cuda \
    --dataset "200 @ SintelDUSt3R(dataset_location='../data/training', dset='clean', S=2, strides=[7], resolution=(512,224), load_dynamic_mask=True)"

DECISION (report the numbers, don't just threshold):
  ratio < ~1.2 and |rho| < ~0.1  -> FLAT: Option 2 unmotivated, lean on L_cc.
  ratio ~1.3-1.5+, monotone, rho>0 -> HEADROOM: Option 2 motivated; gains should
                                      concentrate in boundary/dynamic metrics.
The dynamic-only curve is the one that matters most for the story.
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


def _spearman(x, y, rng, cap=3_000_000):
    """Spearman rho on (a subsample of) paired 1-D arrays, no scipy."""
    n = x.size
    if n < 10:
        return float("nan")
    if n > cap:
        sel = rng.choice(n, cap, replace=False)
        x, y = x[sel], y[sel]
    rx = np.argsort(np.argsort(x)).astype(np.float64)   # tie-naive ranks (ok at scale)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _po_dynamic_mask(loc, dset, label, instance, H, W):
    """PointOdyssey dynamic mask for one frame: foreground (non-zero) pixels of
    masks/mask_<num>.png, nearest-resized to the (H,W) grid. PO native 960x540
    has the grid's aspect ratio, so the loader does a pure resize (no crop) and
    this aligns with the depth. Returns bool (H,W), or None if the file is absent."""
    import re
    import PIL.Image
    import imageio.v2 as imageio
    num = re.sub(r"\D", "", instance)                       # 'rgb_01508.jpg' -> '01508'
    path = os.path.join(loc, dset, label, "masks", f"mask_{num}.png")
    if not os.path.exists(path):
        return None
    m = imageio.imread(path)
    fg = (m != 0).any(axis=-1) if m.ndim == 3 else (m != 0)
    pil = PIL.Image.fromarray(fg.astype(np.uint8) * 255, mode="L").resize(
        (W, H), PIL.Image.NEAREST)
    return np.asarray(pil) > 127


def _decile_table(S, r, d1, n_bins=10):
    """Bin pixels by deciles of S; return per-bin (S_lo, S_hi, n, mean_absrel,
    mean_delta1). Edges from the pooled S distribution."""
    edges = np.percentile(S, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(S, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append((edges[b], edges[b + 1], 0, float("nan"), float("nan")))
        else:
            rows.append((edges[b], edges[b + 1], n,
                         float(r[m].mean()), float(d1[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../DDUSt3R/checkpoints/ddust3r.pth",
                    help="BASE checkpoint we finetune from (default = ddust3r.pth)")
    ap.add_argument("--dataset", required=True, help="DUSt3R 'N @ Dataset(...)' string")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()
    dev = args.device
    use_amp = dev == "cuda"

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
    for obj in (getattr(loader, "dataset", None), getattr(loader, "sampler", None),
                getattr(loader, "batch_sampler", None)):
        if obj is not None and hasattr(obj, "set_epoch"):
            obj.set_epoch(0)

    S_all, r_all, d1_all, dyn_all = [], [], [], []
    n_frames = 0
    has_dyn = False

    # PO carries no dynamic_mask in the view; locate its masks/ from the dataset
    # string so the dynamic split is a REAL foreground/background segmentation.
    import re as _re
    _ml = _re.search(r"dataset_location=['\"]([^'\"]+)['\"]", args.dataset)
    _md = _re.search(r"dset=['\"]([^'\"]+)['\"]", args.dataset)
    po_loc = _ml.group(1) if _ml else None
    po_dset = _md.group(1) if _md else None

    for batch in loader:
        with torch.no_grad():
            res = loss_of_one_batch(batch, model, None, dev,
                                    symmetrize_batch=False, use_amp=use_amp)
        pred1, view1 = res["pred1"], res["view1"]
        pred_d = pred1["pts3d"].detach().float().cpu().numpy()[..., 2]
        gt_d = view1["depthmap"].float().cpu().numpy()
        valid = view1["valid_mask"].cpu().numpy().astype(bool)
        K = view1["camera_intrinsics"].float().cpu().numpy()
        dyn = (view1["dynamic_mask"].cpu().numpy().astype(bool)
               if "dynamic_mask" in view1 else None)
        labels = view1.get("label", None)
        instances = view1.get("instance", None)

        for b in range(pred_d.shape[0]):
            vd = valid[b] & (gt_d[b] > 0) & np.isfinite(pred_d[b])
            if vd.sum() < 100:
                continue
            scale = np.median(gt_d[b][vd]) / max(np.median(pred_d[b][vd]), 1e-6)
            pa = pred_d[b] * scale

            Kmap, kvalid = CV.curvature_from_depth_K(gt_d[b], K[b], mode="mean",
                                                     normalize=True)
            vk = vd & kvalid
            if vk.sum() < 100:
                continue
            absK = np.abs(Kmap[vk])
            S = absK / max(absK.mean(), 1e-8)               # per-image normalized signal
            pp, gg = pa[vk], gt_d[b][vk]
            r = np.abs(pp - gg) / gg
            ratio = np.maximum(pp / gg, gg / pp)
            d1 = (ratio < 1.25).astype(np.float32)

            S_all.append(S.astype(np.float32))
            r_all.append(r.astype(np.float32))
            d1_all.append(d1)
            dynb = None
            if po_loc is not None and labels is not None:
                dynb = _po_dynamic_mask(po_loc, po_dset, labels[b], instances[b],
                                        gt_d[b].shape[0], gt_d[b].shape[1])
            if dynb is None and dyn is not None:
                dynb = dyn[b]
            if dynb is not None:
                has_dyn = True
                dyn_all.append(dynb[vk])
            else:
                dyn_all.append(np.zeros(vk.sum(), bool))
            n_frames += 1
        if args.max_frames and n_frames >= args.max_frames:
            break

    if not S_all:
        raise SystemExit("no valid frames -- check the dataset string / paths")

    S = np.concatenate(S_all); r = np.concatenate(r_all)
    d1 = np.concatenate(d1_all); dyn = np.concatenate(dyn_all)
    rng = np.random.default_rng(0)
    name = args.dataset.split("@")[1].split("(")[0].strip()
    tag = ("_" + args.tag) if args.tag else ""

    def block(label, mask):
        Sm, rm, dm = S[mask], r[mask], d1[mask]
        if Sm.size < 100:
            return None, None
        rows = _decile_table(Sm, rm, dm)
        rho = _spearman(Sm, rm, rng)
        # ratio over the lowest vs highest NON-EMPTY decile (curvature is zero-
        # inflated, so the bottom decile can be empty -> would give nan)
        finite = [row[3] for row in rows if row[2] > 0 and np.isfinite(row[3])]
        ratio = (finite[-1] / finite[0]) if len(finite) >= 2 and finite[0] > 0 else float("nan")
        print(f"\n=== {name}  [{label}]  pixels={Sm.size:,}  frames~{n_frames} ===")
        print(f"{'decile':>6} {'S_lo':>7} {'S_hi':>7} {'n_px':>10} {'AbsRel':>8} {'delta1':>8}")
        for i, (lo, hi, n, a, dd) in enumerate(rows, 1):
            print(f"{i:>6} {lo:7.3f} {hi:7.3f} {n:>10,} {a:8.4f} {dd:8.4f}")
        print(f"Spearman rho(S, AbsRel) = {rho:+.3f}    "
              f"top/bottom-decile AbsRel ratio = {ratio:.3f}")
        return rows, (rho, ratio)

    rows_all, stat_all = block("all valid", np.ones(S.size, bool))
    rows_dyn, stat_dyn = None, None
    if has_dyn:
        frac = float(dyn.mean())
        if frac > 0.99 or frac < 0.001:
            print(f"\n[warn] dynamic mask degenerate (foreground frac={frac:.3f}) "
                  f"-- not a meaningful subset, skipping dynamic curve.")
        else:
            print(f"\n[info] dynamic (foreground) fraction = {frac:.3f}")
            rows_dyn, stat_dyn = block("dynamic", dyn)

    # --- CSV ---
    csv = f"premise_{name}{tag}.csv"
    with open(csv, "w") as f:
        f.write("decile,S_lo,S_hi,n_all,absrel_all,delta1_all,"
                "n_dyn,absrel_dyn,delta1_dyn\n")
        for i in range(10):
            la = rows_all[i]
            ld = rows_dyn[i] if rows_dyn else (0, 0, 0, float("nan"), float("nan"))
            f.write(f"{i+1},{la[0]:.5f},{la[1]:.5f},{la[2]},{la[3]:.5f},{la[4]:.5f},"
                    f"{ld[2]},{ld[3]:.5f},{ld[4]:.5f}\n")
    print(f"\nwrote {csv}")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(1, 11)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, [row[3] for row in rows_all], "o-", label="all valid")
        if rows_dyn:
            ax.plot(x, [row[3] for row in rows_dyn], "s--", label="dynamic")
        ax.set_xlabel("curvature decile (low |K| -> high |K|)")
        ax.set_ylabel("mean AbsRel")
        sub = f"rho={stat_all[0]:+.2f}, top/bot={stat_all[1]:.2f}" if stat_all else ""
        ax.set_title(f"{name}: base-model AbsRel vs curvature decile\n{sub}")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        png = f"premise_{name}{tag}.png"
        fig.savefig(png, dpi=130)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
