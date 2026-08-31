r"""
The measurement (Sec. 5 pseudocode): per view, align on the interior, classify
every K_p == 2 boundary pixel, pool per scene, over the full threshold grid.

The scene is the unit of analysis (Sec. 6.7), so a scene's FP is pooled over all
of its evaluated pixels (sum n_fp / sum n_eval), never a mean of per-view rates,
which would over-weight views that happen to contain few boundary pixels.

Streams measured: whatever `infer.py` cached, plus the mandatory GT control
`gt_raw` -- the identical pipeline with D_pred := the dataset's own raw depth,
while the support S(p) still comes from the cleaned GT. Every dataset here is
synthetic, so raw == clean and FP_gt_raw ~= 0 is the control WORKING: it is what
makes any nonzero model FP attributable to the model.

    python src/measure.py --datasets tartanair,sintel --scenes 20 --views 8
"""
from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

import data as D
import infer as INF
import fpmetrics as FM

GT_CONTROL = "gt_raw"


def measure(datasets, streams, n_scenes, n_views, out, *, w=3, delta_min=0.0,
            align="median", grid=True):
    if delta_min <= 0:
        print("  ! delta_min=0: Sec. 6.5 requires a minimum absolute depth gap so "
              "noise-level steps are not counted as boundaries. Pass --delta-min.",
              flush=True)
    etas = FM.ETA_GRID if grid else (FM.ETA0,)
    taus = FM.TAU_GRID if grid else (FM.TAU0,)
    betas = FM.BETA_GRID if grid else (FM.BETA0,)
    scenes_out, skipped = [], defaultdict(int)

    for ds in datasets:
        for sc in D.scenes(ds)[:n_scenes]:
            acc = defaultdict(lambda: dict(n_fp=0, n_eval=0, n_outside=0,
                                           n_boundary=0, n_nonfinite=0))
            khist, fbv_num, fbv_den, used = defaultdict(int), 0.0, 0, 0
            for v in D.views(ds, sc, n_views):
                try:
                    vd = D.load(v)
                except Exception as e:
                    skipped[f"load:{type(e).__name__}"] += 1
                    continue
                gt_shape = vd.depth.shape
                preds = {GT_CONTROL: vd.depth_raw.astype(np.float64)}
                miss = []
                for s in streams:
                    z = INF.load_pred(out, s, v.key, gt_shape)
                    if z is None:
                        miss.append(s)
                    else:
                        preds[s] = z
                if miss:
                    # Use a view only when EVERY stream has a prediction for it.
                    # Otherwise a stream absent from the cache on some views would
                    # be pooled over a different view set than the GT control and
                    # the other streams, which silently rescales its scene FP.
                    for s in miss:
                        skipped[f"missing:{s}"] += 1
                    skipped["view_dropped_incomplete"] += 1
                    continue
                recs = FM.fp_view_grid(vd.depth, vd.valid, preds, etas=etas, taus=taus,
                                       betas=betas, w=w, delta_min=delta_min, align=align)
                for r in recs:
                    a = acc[(r["stream"], r["eta"], r["tau"], r["beta"])]
                    a["n_fp"] += r["n_fp"]
                    a["n_eval"] += r["n_eval"]
                    a["n_outside"] += r["n_outside"]
                    a["n_boundary"] += r["n_boundary"]
                    a["n_nonfinite"] += r.get("n_nonfinite", 0)
                centre = [r for r in recs if r["eta"] == FM.ETA0 and r["tau"] == FM.TAU0]
                if centre:
                    for k, n in centre[0]["K_hist"].items():
                        khist[int(k)] += int(n)
                    # pixel-weighted, not a mean of per-view rates: Sec. 4 thresholds
                    # this at 0.5, and an unweighted mean lets a view with a handful
                    # of boundary pixels flip the gate verdict for a whole scene.
                    f, nb = centre[0]["frac_boundary_valid"], centre[0]["n_boundary"]
                    if np.isfinite(f) and nb:
                        fbv_num += f * nb
                        fbv_den += nb
                used += 1

            if not used:
                continue
            cells = [dict(stream=st, eta=e, tau=t, beta=b,
                          FP=(a["n_fp"] / a["n_eval"]) if a["n_eval"] else float("nan"),
                          outside_rate=(a["n_outside"] / a["n_eval"]) if a["n_eval"] else float("nan"),
                          n_eval=a["n_eval"], n_fp=a["n_fp"], n_boundary=a["n_boundary"],
                          n_nonfinite=a["n_nonfinite"])
                     for (st, e, t, b), a in sorted(acc.items(), key=lambda kv: str(kv[0]))]
            scenes_out.append(dict(dataset=ds, scene=sc, tier=D.SPECS[ds]["tier"],
                                   n_views=used, K_hist=dict(sorted(khist.items())),
                                   frac_boundary_valid=(fbv_num / fbv_den) if fbv_den
                                                       else float("nan"),
                                   cells=cells))
            print(f"  {ds}/{sc}: {used} views, "
                  f"{max((c['n_eval'] for c in cells), default=0)} eval px", flush=True)

    os.makedirs(out, exist_ok=True)
    res = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               config=dict(datasets=datasets, streams=streams, gt_control=GT_CONTROL,
                           n_scenes=n_scenes, n_views=n_views, w=w, delta_min=delta_min,
                           align=align, etas=list(etas), taus=list(taus), betas=list(betas),
                           centre=dict(eta=FM.ETA0, tau=FM.TAU0, beta=FM.BETA0)),
               skipped=dict(skipped), scenes=scenes_out)
    p = os.path.join(out, "results.json")
    # json.dump emits bare NaN by default, which is not valid JSON and cannot be
    # read by anything but Python. Non-finite values become null.
    def _clean(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer, np.floating)):
            return _clean(o.item())
        return o
    json.dump(_clean(res), open(p, "w"), indent=1, allow_nan=False)
    print(f"\n{len(scenes_out)} scenes -> {p}")
    if skipped:
        print("skipped:", dict(skipped))
    if not scenes_out:
        print("\nNothing measured. Run src/infer.py first to populate the prediction cache.")
    return res


def _main():
    ap = argparse.ArgumentParser(description="flying-pixel measurement over scenes")
    ap.add_argument("--datasets", default="tartanair,sintel")
    ap.add_argument("--streams", default=",".join(INF.STREAMS))
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--out", default="results/premise_check")
    ap.add_argument("--w", type=int, default=3, help="depth-support window radius")
    ap.add_argument("--delta-min", type=float, default=0.0,
                    help="minimum absolute depth gap, rejects trivial jumps (Sec. 6.5)")
    ap.add_argument("--align", default="median", choices=("median", "lstsq"))
    ap.add_argument("--no-grid", action="store_true", help="centre thresholds only")
    a = ap.parse_args()
    res = measure([d for d in a.datasets.split(",") if d],
                  [s for s in a.streams.split(",") if s], a.scenes, a.views, a.out,
                  w=a.w, delta_min=a.delta_min, align=a.align, grid=not a.no_grid)
    # A run that measured nothing must not look like a success to the caller.
    return 0 if res["scenes"] else 4


if __name__ == "__main__":
    raise SystemExit(_main())
