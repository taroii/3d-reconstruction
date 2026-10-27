"""
Evaluation: score a trained arm on four datasets, three of them real.

Scores a checkpoint produced by `train.py`, or the base VGGT weights as the
reference row, and writes one CSV row per (arm, seed, dataset, region).

------------------------------------------------------------------------------
The boundary metric, and why it is not the ACCV paper's
------------------------------------------------------------------------------
The ACCV submission reported "boundary" metrics over the top decile of pixels
by ground-truth |K| -- the SAME functional that defines the training weight.
Reviewer CYqH called this out precisely:

    "Four of the eight are boundary metrics whose pixel subset is selected by
     the top decile of ground-truth |K_i|, the same functional that defines the
     training weight, so a gain there is close to guaranteed if the mechanism
     does anything. [...] the boundary columns should not be counted as
     corroboration."

That objection is correct and unanswerable on its own terms, so this file does
not use that subset at all. Instead iBims-1 ships INDEPENDENT ground-truth depth
boundary maps, drawn from the dataset's own laser measurements and owing nothing
to our signal. Scoring on `edge` versus `interior` pixels defined by those maps
breaks the circularity completely: the region is chosen by the benchmark, not by
the thing under test.

`src/estimator_diag.py` gives the second reason this matters. On a frame holding
both genuine curvature and a pure depth step, 100% of the step pixels land in
the umbrella signal's top decile against 0% for a real curvature estimator. The
paper's own boundary subset was, to a good approximation, a depth-edge mask --
so selecting on it was selecting on the mechanism.

Regions reported for every dataset:
    all        every valid pixel
    edge       within `EDGE_DILATE` px of a ground-truth depth boundary
    interior   everything else
For iBims-1 the edges come from the dataset. Elsewhere there is no ground-truth
edge map, so edges are derived from the GT DEPTH by a relative-jump test -- a
property of the target, never of a prediction, and never of our weight.

------------------------------------------------------------------------------
Alignment
------------------------------------------------------------------------------
Scale-and-shift invariant, fit per frame by least squares on the valid pixels,
the standard protocol. Identical for every arm, so the A/B delta is fair. Fit on
ALL valid pixels and then scored per region -- fitting per region would let each
region choose a flattering scale and make the regions incomparable.

  python src/evaluate.py --selftest
  python src/evaluate.py --ckpt base --datasets ibims1 --limit 20
  python src/evaluate.py --ckpt runs/umbrella_clamp_1_seed0 --out results/eval
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.environ.get("PROJ", os.path.abspath(os.path.join(_HERE, "..", "..")))
_VGGT = os.path.join(_PROJ, "third_party", "vggt")
for _p in (_VGGT, os.path.join(_VGGT, "training"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datasets as DS                                    # noqa: E402
import train as T                                        # noqa: E402

EDGE_DILATE = 3          # px either side of a boundary counted as "edge"
EDGE_REL_JUMP = 0.05     # relative depth jump defining a boundary
DELTA = 1.25


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def align_scale_shift(pred, gt, mask):
    """Least-squares (s, t) minimising ||s*pred + t - gt|| over `mask`.

    Returned unapplied to invalid pixels. If the system is singular -- a
    constant prediction, or too few pixels -- returns None so the caller can
    drop the frame rather than score an arbitrary fit.
    """
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    if p.size < 100:
        return None
    # A constant prediction makes the design matrix rank-deficient, and lstsq
    # returns a minimum-norm solution rather than failing -- so a COLLAPSED
    # model would be "aligned" to the mean of the ground truth and then scored,
    # which flatters it enormously. Require real variation in the prediction.
    spread = float(np.std(p))
    if spread < 1e-6 * max(abs(float(np.mean(p))), 1.0):
        return None
    A = np.stack([p, np.ones_like(p)], 1)
    try:
        sol, *_ = np.linalg.lstsq(A, g, rcond=None)
    except np.linalg.LinAlgError:
        return None
    s, t = float(sol[0]), float(sol[1])
    if not np.isfinite(s) or not np.isfinite(t) or s <= 0:
        return None
    return s, t


def gt_edges(depth, valid, rel=EDGE_REL_JUMP):
    """Depth boundaries from the GROUND TRUTH, by relative jump.

    Computed from the target only. Nothing here touches a prediction or the
    training signal, so the region cannot be selected by the thing under test.
    """
    d = np.where(valid, depth, np.nan)
    e = np.zeros(d.shape, bool)
    for dr, dc in ((0, 1), (1, 0)):
        a = d[:-1 if dr else None, :-1 if dc else None]
        b = d[dr:, dc:]
        with np.errstate(invalid="ignore", divide="ignore"):
            jump = np.abs(a - b) / np.minimum(a, b)
        hit = np.nan_to_num(jump) > rel
        e[:-1 if dr else None, :-1 if dc else None] |= hit
        e[dr:, dc:] |= hit
    return e & valid


def _dilate(m, k):
    out = m.copy()
    for _ in range(k):
        g = out.astype(np.uint8)
        nb = np.zeros_like(g)
        nb[1:, :] |= g[:-1, :]; nb[:-1, :] |= g[1:, :]
        nb[:, 1:] |= g[:, :-1]; nb[:, :-1] |= g[:, 1:]
        out = out | (nb > 0)
    return out


def depth_metrics(pred, gt, mask):
    """AbsRel and delta<1.25 over `mask`. None if the region is too small."""
    if mask.sum() < 50:
        return None
    p, g = pred[mask], gt[mask]
    ok = np.isfinite(p) & np.isfinite(g) & (g > 0) & (p > 0)
    if ok.sum() < 50:
        return None
    p, g = p[ok], g[ok]
    ratio = np.maximum(p / g, g / p)
    return {"absrel": float(np.mean(np.abs(p - g) / g)),
            "delta1": float(np.mean(ratio < DELTA)),
            "n": int(ok.sum())}


def score_frame(pred_depth, view, edges=None):
    """Per-region metrics for one frame, under one shared scale-shift fit."""
    gt, valid = view.depth, view.valid
    valid = valid & np.isfinite(pred_depth) & (pred_depth > 0)
    fit = align_scale_shift(pred_depth, gt, valid)
    if fit is None:
        return None
    s, t = fit
    aligned = s * pred_depth + t

    if edges is None:
        edges = gt_edges(gt, valid)
    band = _dilate(edges, EDGE_DILATE) & valid
    regions = {"all": valid, "edge": band, "interior": valid & ~band}
    out = {}
    for name, m in regions.items():
        r = depth_metrics(aligned, gt, m)
        if r is not None:
            out[name] = r
    return out


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #

def load_arm(ckpt, device):
    """Base VGGT, optionally with a trained arm's heads loaded over it."""
    model = T.load_model(device)
    cfg = {"arm": "base", "seed": -1, "step": 0}
    if ckpt and ckpt != "base":
        path = ckpt if ckpt.endswith(".pt") else os.path.join(ckpt, "checkpoint.pt")
        blob = torch.load(path, map_location=device, weights_only=False)
        missing = model.load_state_dict(blob["heads"], strict=False)
        # strict=False is needed because we only save the two trained heads;
        # but anything UNEXPECTED means the checkpoint does not match the model.
        if missing.unexpected_keys:
            raise RuntimeError(f"checkpoint has unexpected keys: "
                               f"{missing.unexpected_keys[:5]}")
        cfg = dict(blob.get("cfg", {}), step=blob.get("step", 0))
        # A run directory with no completion marker holds a mid-run periodic
        # save (see train.py). Scoring one as a finished arm is how a
        # half-trained model gets into a results table, so it is marked here
        # and carried into the CSV rather than being quietly averaged in.
        cfg["complete"] = 1
        if not ckpt.endswith(".pt"):
            if not os.path.exists(os.path.join(ckpt, "done.json")):
                cfg["complete"] = 0
                print(f"  !! INCOMPLETE: {ckpt} has no done.json -- these are "
                      f"mid-run weights at step {cfg['step']}, not a finished arm")
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_depth(model, views, device):
    """Predicted depth per view. Images only -- no poses, no targets.

    Deliberately NOT `train.build_batch`: that assembles supervision targets and
    needs camera poses, which iBims-1 and NYUv2 do not have. Inference needs
    pixels. The resize is `train._resize_view`, the same transform training
    used, so the model sees the distribution it was trained on.
    """
    vs = [T._resize_view(v) for v in views]
    imgs = np.stack([v.rgb for v in vs]).astype(np.float32) / 255.0
    images = torch.from_numpy(imgs).permute(0, 3, 1, 2)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, psi = model.aggregator(images)
    # The aggregator returns None for uncached layers so that layer indices stay
    # stable; mapping .float() over the list without that guard crashes. This
    # mirrors train.forward exactly, which matters beyond the crash: evaluation
    # must put the heads in the same numeric regime training did.
    tokens = [t if t is None else t.detach() for t in tokens]
    with torch.autocast("cuda", enabled=False):
        d, _ = model.depth_head(tokens, images=images, patch_start_idx=psi)
    return d[0, :, :, :, 0].float().cpu().numpy(), vs


def _spread(items, limit):
    """Evenly spaced deterministic subsample -- NOT a prefix.

    `items[:limit]` reads as harmless and is not. iBims-1's names sort by scene
    category and NYUv2's test index is ordered by scene, so a prefix evaluates
    two or three room types and reports the number as the dataset's. Spreading
    across the whole index keeps a 25-frame spot check representative of the
    full set.

    Deterministic on purpose: every arm must be scored on the SAME frames, or
    the arm-to-arm difference is partly a difference in which frames were
    drawn, which at these effect sizes would swamp the effect.
    """
    items = list(items)
    if not limit or limit >= len(items):
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, limit).round().astype(int))
    return [items[i] for i in idx]


def eval_dataset(model, name, device, limit=None, seq_limit=4):
    """Per-frame region metrics for one dataset.

    Every set is evaluated SINGLE-VIEW. VGGT accepts S=1, and it keeps the
    protocol identical across datasets: iBims-1 and NYUv2 have no second view
    to offer, so giving Bonn or TartanAir several would score them on a
    different task.
    """
    rows = []
    if name == "ibims1":
        ds = DS.IBims1()
        names = ds.names_with_edges()          # only the 85 with GT edge maps
        names = _spread(names, limit)
        for n in names:
            v = ds.view(ds.names.index(n))
            vv = T._resize_view(v)
            edges = v.meta.get("edges")
            if edges is not None:
                import cv2
                e = cv2.resize(edges.astype(np.uint8),
                               (vv.depth.shape[1], vv.depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                e = None
            pd, _ = predict_depth(model, [v], device)
            r = score_frame(pd[0], vv, edges=e)
            if r:
                rows.append((n, r))
    elif name == "nyuv2":
        ds = DS.NYUv2()
        n_tot = len(ds)
        idxs = _spread(range(n_tot), limit)
        for i in idxs:
            v = ds.view(i)
            pd, _ = predict_depth(model, [v], device)
            r = score_frame(pd[0], T._resize_view(v))
            if r:
                rows.append((str(i), r))
    elif name == "bonn":
        ds = DS.Bonn()
        # Budget per sequence must divide by the sequences actually VISITED,
        # not by every sequence Bonn has: dividing by all of them and then
        # iterating a 4-sequence subset delivered a small fraction of the
        # requested frames (--limit 2 gave 4, --limit 150 gave ~24).
        seqs = _spread(ds.seqs, seq_limit)
        per = max(1, (limit or 200) // max(len(seqs), 1))
        for sq in seqs:
            pr = ds.pairs(sq)
            for k in np.linspace(0, len(pr) - 1, min(per, len(pr))).astype(int):
                v = ds.view(sq, int(k))
                pd, _ = predict_depth(model, [v], device)
                r = score_frame(pd[0], T._resize_view(v))
                if r:
                    rows.append((f"{sq}:{k}", r))
    else:
        raise ValueError(f"no evaluator for {name!r}")
    return rows


def aggregate(rows):
    """Mean per region, weighted by nothing -- each FRAME counts once.

    Pixel-weighting would let a few large-valid-area frames dominate, and the
    per-frame mean is what the depth literature reports.
    """
    out = {}
    for region in ("all", "edge", "interior"):
        vals = [r[region] for _, r in rows if region in r]
        if not vals:
            continue
        out[region] = {
            "absrel": float(np.mean([v["absrel"] for v in vals])),
            "delta1": float(np.mean([v["delta1"] for v in vals])),
            "frames": len(vals),
            "px": int(sum(v["n"] for v in vals)),
        }
    return out


# --------------------------------------------------------------------------- #

def _selftest():
    # -- alignment recovers a known scale and shift exactly ------------------
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 10, (40, 50))
    m = np.ones_like(gt, bool)
    pred = (gt - 0.7) / 2.5
    s, t = align_scale_shift(pred, gt, m)
    assert abs(s - 2.5) < 1e-6 and abs(t - 0.7) < 1e-6, f"align gave s={s}, t={t}"
    r = depth_metrics(s * pred + t, gt, m)
    assert r["absrel"] < 1e-9, f"perfect prediction scored absrel {r['absrel']}"

    # -- a singular fit must be refused, not scored --------------------------
    assert align_scale_shift(np.ones((40, 50)), gt, m) is None, \
        "a constant prediction must not produce a fit"
    assert align_scale_shift(gt, gt, np.zeros_like(m)) is None, \
        "an empty mask must not produce a fit"

    # -- subsampling must SPAN the index, not take a prefix ------------------
    # The failure this guards against is silent: a prefix returns the right
    # NUMBER of frames and the wrong frames, so every downstream metric looks
    # well-formed. Checked against a labelled index standing in for iBims-1's
    # category-sorted names: a prefix would draw every sample from category 0.
    cats = [f"cat{i // 10}_{i:03d}" for i in range(85)]
    sub = _spread(cats, 25)
    assert len(sub) == 25, f"spread returned {len(sub)} of 25"
    assert len(set(c.split('_')[0] for c in sub)) == 9, \
        f"spread covered {len(set(c.split('_')[0] for c in sub))} of 9 categories"
    assert sub != cats[:25], "spread degenerated into a prefix"
    assert sub[0] == cats[0] and sub[-1] == cats[-1], "spread misses the ends"
    assert _spread(cats, None) == cats and _spread(cats, 999) == cats, \
        "no limit must return everything"
    assert _spread(cats, 25) == _spread(cats, 25), "spread is not deterministic"

    # -- the edge detector must find a step and ignore a ramp ----------------
    step = np.ones((40, 50)); step[:, 25:] = 2.0
    e = gt_edges(step, np.ones_like(step, bool))
    assert e[:, 24].all() and e[:, 25].all(), "step edge not detected"
    assert not e[:, :20].any(), "edge detector fired on flat region"
    ramp = np.tile(np.linspace(1.0, 1.4, 50), (40, 1))   # 0.8% per px, under 5%
    assert not gt_edges(ramp, np.ones_like(ramp, bool)).any(), \
        "edge detector fired on a smooth ramp"

    # -- regions must partition the valid set --------------------------------
    v = DS.View(rgb=np.zeros((40, 50, 3), np.uint8), depth=step.astype(np.float32),
                K=np.eye(3), c2w=None)
    out = score_frame(step.astype(np.float32) * 3.0, v)
    assert set(out) >= {"all", "edge", "interior"}, f"missing regions: {set(out)}"
    assert out["edge"]["n"] + out["interior"]["n"] == out["all"]["n"], \
        "edge and interior do not partition all"

    # -- the boundary region must NOT come from our own signal ---------------
    import inspect
    src = inspect.getsource(gt_edges) + inspect.getsource(score_frame)
    for forbidden in ("umbrella", "cotangent", "compute_weight", "SIGNALS"):
        assert forbidden not in src, (
            f"the boundary region references {forbidden!r}; selecting the "
            f"evaluation region with the training signal is exactly the "
            f"circularity the ACCV reviewers rejected")
    print("selftest: metrics ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="base", help="'base', a run dir, or a .pt")
    ap.add_argument("--datasets", default="ibims1,nyuv2,bonn")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_arm(a.ckpt, device)
    arm, seed = cfg.get("arm", "base"), cfg.get("seed", -1)
    print(f"arm={arm} seed={seed} step={cfg.get('step', 0)}")

    rows = []
    for name in a.datasets.split(","):
        res = eval_dataset(model, name, device, limit=a.limit)
        agg = aggregate(res)
        for region, m in agg.items():
            rows.append({"arm": arm, "seed": seed, "step": cfg.get("step", 0),
                         "complete": cfg.get("complete", 1),
                         "dataset": name, "region": region, **m})
            print(f"  {name:<9} {region:<9} absrel {m['absrel']:.4f}  "
                  f"delta1 {m['delta1']:.4f}  ({m['frames']} frames)")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, "eval.csv")
        new = not os.path.exists(p)
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"appended {len(rows)} rows to {p}")


if __name__ == "__main__":
    main()
