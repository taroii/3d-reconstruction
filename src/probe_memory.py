"""
How much of VGGT can we actually train on one 20 GB card, and how fast?

This decides the experiment budget, so it runs before any training code is
written.  Nothing here needs a dataset: images are random tensors in [0, 1],
which is all that memory and throughput depend on.

The question is not academic.  VGGT is ~1.26B parameters; mixed-precision AdamW
needs roughly 16 bytes per parameter for weights, gradients and optimizer state,
which is ~19 GB before a single activation.  A full fine-tune cannot fit, and
there are public reports of it failing even on 48 GB.  The ACCV recipe already
pointed at the way out -- it froze the encoder and trained the decoder and DPT
heads -- so the VGGT analogue is to freeze DINOv2 and the 24-layer alternating
aggregator and train only the prediction heads.  Then the aggregator runs under
`no_grad`, its activations are never retained, and gradients flow through the
heads alone.

Modes probed:

  heads_only   Aggregator under no_grad, tokens detached, heads trainable.
               The candidate recipe.
  full         Everything trainable.  Expected to OOM; measured anyway, because
               "we tried and it does not fit" is a sentence the paper may need.

For each mode the probe sweeps the number of input views and the image
resolution, since both drive activation memory, and reports peak VRAM, forward
and backward time, and a projected wall-clock cost per training run.

`aggregator_fwd_s` is reported separately on purpose.  In heads-only training
the aggregator is a fixed function of the input, so its output can in principle
be precomputed once and cached, and the fraction of step time it accounts for is
exactly the speed-up that would buy.

Shared-machine etiquette: this allocates most of a 20 GB GPU that other people
use.  `--max-foreign-mib` refuses to start if someone else already holds more
than that much VRAM, and it is checked again between configurations.

  python src/probe_memory.py --list
  python src/probe_memory.py --out results/probe
"""
import argparse
import csv
import json
import os
import gc
import subprocess
import sys
import time

import torch

# The upstream clone lives beside the repo, not inside it (see scripts/setup.sh).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.environ.get("PROJ", os.path.abspath(os.path.join(_HERE, "..", "..")))
_VGGT = os.path.join(_PROJ, "third_party", "vggt")
if _VGGT not in sys.path:
    sys.path.insert(0, _VGGT)

CKPT = os.path.join(_PROJ, "ckpt", "model.safetensors")

# Sweep points.  518 is VGGT's native training resolution; 392 and 336 are the
# next two multiples of the 14-pixel patch size and are the realistic fallbacks
# if 518 does not fit.
VIEWS = [2, 4, 8]
RESOLUTIONS = [518, 392]
MODES = ["heads_only", "full"]


# --------------------------------------------------------------------------- #
# GPU courtesy
# --------------------------------------------------------------------------- #

def foreign_vram_mib():
    """VRAM held by processes that are not this one.

    This is a shared machine and the probe is about to allocate most of a 20 GB
    card.  Walking over somebody's running job to measure our own throughput
    would be a bad trade.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
    except Exception:
        return 0, []
    mine, others, total = os.getpid(), [], 0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, mem = [x.strip() for x in line.split(",")[:2]]
        if int(pid) != mine:
            total += int(mem)
            others.append((int(pid), int(mem)))
    return total, others


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def load_model(device, enable_track=False):
    """VGGT with the local checkpoint.

    The track head is left out by default: it is supervised by a separate
    CoTracker-style objective, is down-weighted to 0.05 in VGGT's own loss, and
    is irrelevant to a study about the geometry regression terms.  Carrying it
    would inflate every memory number here with something we will never train.
    """
    from vggt.models.vggt import VGGT
    model = VGGT(enable_camera=True, enable_point=True, enable_depth=True,
                 enable_track=enable_track)
    if os.path.exists(CKPT):
        from safetensors.torch import load_file
        sd = load_file(CKPT)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("track_head")]
        unexpected = [k for k in unexpected if not k.startswith("track_head")]
        if missing or unexpected:
            print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    else:
        print(f"  WARNING: no checkpoint at {CKPT}; random weights "
              f"(memory and timing are still valid)")
    return model.to(device)


def set_trainable(model, mode):
    """Returns the parameters an optimizer would own under `mode`."""
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "heads_only":
        for head in (model.camera_head, model.point_head, model.depth_head):
            if head is not None:
                for p in head.parameters():
                    p.requires_grad_(True)
    else:
        raise ValueError(mode)
    return [p for p in model.parameters() if p.requires_grad]


def _synthetic_targets(pred, device):
    """Random targets of the right shape.

    Values are meaningless -- the probe measures the cost of the graph, not
    convergence -- but the shapes must match so the backward pass is the real
    one and not a cheaper fiction.
    """
    t = {}
    for k in ("depth", "world_points", "pose_enc"):
        if k in pred:
            t[k] = torch.randn_like(pred[k])
    return t


def step(model, images, mode, opt, device):
    """One training step, timed in three parts.

    Returns (aggregator_fwd_s, heads_fwd_s, bwd_opt_s). The third is backward
    AND the optimizer step AND target construction -- for `full` that is an
    AdamW update over 1.19B parameters, which is not "backward" by any
    reasonable reading, so the name says so. `step_s` and `aggregator_share`
    are unaffected.

    The split matters because under `heads_only` the aggregator is a fixed
    function of the input, so its share is what caching its output would save.
    """
    torch.cuda.synchronize()
    opt.zero_grad(set_to_none=True)

    if mode == "heads_only":
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, patch_start_idx = model.aggregator(images)
        # Detach so autograd cannot walk back into the frozen trunk even if a
        # head holds a reference to these tensors.
        tokens = [t.detach() if t is not None else None for t in tokens]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred = _run_heads(model, tokens, patch_start_idx, images)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
    else:
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, patch_start_idx = model.aggregator(images)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred = _run_heads(model, tokens, patch_start_idx, images)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

    tgt = _synthetic_targets(pred, device)
    loss = sum((pred[k].float() - v).abs().mean() for k, v in tgt.items())
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2


def _run_heads(model, tokens, patch_start_idx, images):
    """Heads in fp32, matching upstream's `autocast(enabled=False)` block."""
    pred = {}
    with torch.autocast("cuda", enabled=False):
        if model.camera_head is not None:
            pred["pose_enc"] = model.camera_head(tokens)[-1]
        if model.depth_head is not None:
            d, _ = model.depth_head(tokens, images=images,
                                    patch_start_idx=patch_start_idx)
            pred["depth"] = d
        if model.point_head is not None:
            p, _ = model.point_head(tokens, images=images,
                                    patch_start_idx=patch_start_idx)
            pred["world_points"] = p
    return pred


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #

def probe_one(model, mode, views, res, device, warmup=1, iters=3):
    """Peak VRAM and step time for one configuration, or an OOM record."""
    params = set_trainable(model, mode)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=1e-5)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    row = {"mode": mode, "views": views, "resolution": res,
           "trainable_params_M": round(n_train / 1e6, 1)}
    try:
        images = torch.rand(1, views, 3, res, res, device=device)
        for _ in range(warmup):
            step(model, images, mode, opt, device)
        agg = heads = back = 0.0
        for _ in range(iters):
            a, h, b = step(model, images, mode, opt, device)
            agg += a; heads += h; back += b
        agg, heads, back = agg / iters, heads / iters, back / iters
        row.update({
            "ok": True,
            "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "aggregator_fwd_s": round(agg, 4),
            "heads_fwd_s": round(heads, 4),
            "bwd_opt_s": round(back, 4),
            "step_s": round(agg + heads + back, 4),
            "steps_per_s": round(1.0 / max(agg + heads + back, 1e-9), 3),
            "aggregator_share": round(agg / max(agg + heads + back, 1e-9), 3),
            "note": "",
        })
    except torch.cuda.OutOfMemoryError:
        row.update({"ok": False, "note": "OOM"})
    except RuntimeError as e:
        # Some cuDNN/cuBLAS paths surface OOM as a plain RuntimeError; if that
        # is recorded as a named exception instead, report()'s "full fine-tune
        # did not fit" verdict silently loses its evidence.
        msg = str(e).lower()
        oom = "out of memory" in msg or "cuda error" in msg and "memory" in msg
        row.update({"ok": False, "note": "OOM" if oom else f"RuntimeError: {e}"[:160]})
    except Exception as e:                       # keep the sweep going
        row.update({"ok": False, "note": f"{type(e).__name__}: {e}"[:160]})
    opt.zero_grad(set_to_none=True)
    del opt
    for prm in model.parameters():
        prm.grad = None
    return row


def budget(rows, steps_per_run=20000):
    """Wall-clock projection for the arm matrix, from measured throughput.

    `steps_per_run` is a planning figure, not a measurement: the ACCV study ran
    12 epochs of 8,000 two-view clips at an effective batch of 16, which is
    about 6,000 optimizer steps. 20,000 is deliberately generous because the
    frozen-trunk recipe trains far fewer parameters and may need longer.
    """
    out = []
    for r in rows:
        if not r.get("ok"):
            continue
        hrs = steps_per_run / r["steps_per_s"] / 3600.0
        out.append({**{k: r[k] for k in ("mode", "views", "resolution",
                                         "peak_alloc_GiB", "steps_per_s")},
                    "hours_per_run": round(hrs, 2),
                    # 6 arms x 5 seeds is the full pre-registered matrix:
                    # baseline, curvature, gradient control, unit-mean,
                    # binary-clamp control, area-normalized estimator.
                    "hours_30_runs": round(hrs * 30, 1),
                    "days_30_runs": round(hrs * 30 / 24.0, 1)})
    return out


def report(rows, budget_rows, total_vram, modes_attempted):
    L = []
    L.append("# VGGT trainability probe\n")
    L.append(f"GPU: {torch.cuda.get_device_name(0)}  ({total_vram:.1f} GiB)")
    L.append(f"torch {torch.__version__}, synthetic inputs, no dataset.\n")

    L.append("## Measured")
    hdr = ["mode", "views", "resolution", "trainable_params_M", "peak_alloc_GiB",
           "aggregator_fwd_s", "heads_fwd_s", "bwd_opt_s", "step_s",
           "steps_per_s", "aggregator_share", "note"]
    L.append("".join(f"{h:<20}" for h in hdr))
    for r in rows:
        L.append("".join(f"{str(r.get(h, '--')):<20}" for h in hdr))

    if budget_rows:
        L.append("\n## Projected cost (20,000 optimizer steps per run)")
        hdr2 = ["mode", "views", "resolution", "peak_alloc_GiB", "steps_per_s",
                "hours_per_run", "hours_30_runs", "days_30_runs"]
        L.append("".join(f"{h:<18}" for h in hdr2))
        for r in budget_rows:
            L.append("".join(f"{str(r.get(h, '--')):<18}" for h in hdr2))

    ok = [r for r in rows if r.get("ok")]
    L.append("\n## Verdict\n")
    if not ok:
        # Name the modes actually attempted. An earlier version said "both the
        # frozen-trunk and full fine-tune exceed this card" after a run that
        # probed only `full` -- the same overclaim this function already guards
        # against further down.
        L.append(f"No configuration completed a training step for "
                 f"{', '.join(sorted(modes_attempted))}. "
                 f"This report says nothing about any mode not listed there.")
        return "\n".join(L)

    heads = [r for r in ok if r["mode"] == "heads_only"]
    full = [r for r in ok if r["mode"] == "full"]
    if heads:
        best = max(heads, key=lambda r: (r["resolution"], r["views"]))
        L.append(
            f"Frozen-trunk training fits: at {best['views']} views and "
            f"{best['resolution']}px it peaks at {best['peak_alloc_GiB']} GiB of "
            f"{total_vram:.1f} GiB and runs at {best['steps_per_s']} steps/s, "
            f"training {best['trainable_params_M']}M parameters.")
        share = best["aggregator_share"]
        L.append(
            f"\nThe frozen aggregator accounts for {share:.0%} of each step. "
            f"Because it is a fixed function of the input under this recipe, "
            f"caching its token output would cut step time by about that much, "
            f"at the cost of disk and of fixing the augmentation pipeline.")
    if not full and "full" in modes_attempted:
        L.append(
            "\nA full fine-tune did not complete any configuration, which is the "
            "expected result: ~1.26B parameters at 16 bytes each for AdamW is "
            "~19 GiB before activations. The frozen-trunk recipe is not a "
            "convenience here, it is the only option on this hardware.")
    elif not full:
        # Never imply a result for a mode that was not run. An earlier version
        # printed the paragraph above after a heads-only smoke test, asserting
        # an OOM that had never been attempted.
        L.append("\nA full fine-tune was not probed in this run, so this report "
                 "says nothing about whether it fits.")
    else:
        b = max(full, key=lambda r: r["peak_alloc_GiB"])
        L.append(f"\nA full fine-tune also ran, peaking at {b['peak_alloc_GiB']} GiB "
                 f"at {b['views']} views / {b['resolution']}px.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--views", default=",".join(map(str, VIEWS)))
    ap.add_argument("--resolutions", default=",".join(map(str, RESOLUTIONS)))
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--steps-per-run", type=int, default=20000)
    ap.add_argument("--max-foreign-mib", type=int, default=2048,
                    help="refuse to start if others hold more VRAM than this")
    ap.add_argument("--list", action="store_true", help="print the sweep and exit")
    args = ap.parse_args()

    modes = args.modes.split(",")
    views = [int(v) for v in args.views.split(",")]
    res = [int(r) for r in args.resolutions.split(",")]

    if args.list:
        for m in modes:
            for v in views:
                for r in res:
                    print(f"{m:<12} views={v:<3} res={r}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30

    foreign, others = foreign_vram_mib()
    if foreign > args.max_foreign_mib:
        raise SystemExit(
            f"another process holds {foreign} MiB of VRAM ({others}).\n"
            f"This is a shared GPU; not starting. Re-run later, or raise "
            f"--max-foreign-mib deliberately if you know the job is yours.")
    print(f"GPU {torch.cuda.get_device_name(0)}  {total:.1f} GiB  "
          f"(others hold {foreign} MiB)")

    print("loading model...")
    model = load_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params / 1e9:.2f}B parameters")

    rows, stop = [], False
    for m in modes:
        if stop:
            break
        for v in views:
            if stop:
                break
            for r in res:
                f, _ = foreign_vram_mib()
                if f > args.max_foreign_mib:
                    # `break` alone only left the innermost loop, so the probe
                    # announced it was stopping and then kept allocating on a
                    # GPU somebody else had just claimed.
                    print(f"  stopping: another process now holds {f} MiB")
                    stop = True
                    break
                print(f"  probing {m:<12} views={v} res={r} ...", end="", flush=True)
                row = probe_one(model, m, v, r, device, iters=args.iters)
                rows.append(row)
                # Collect OUTSIDE probe_one: an OOM traceback holds references
                # to the activation graph via its frames, and those cycles need
                # a gc pass before empty_cache() can actually release anything.
                # Without this, every configuration after the first OOM inherits
                # the previous one's allocation and OOMs spuriously.
                gc.collect()
                torch.cuda.empty_cache()
                print(f" {row.get('peak_alloc_GiB', '--')} GiB  "
                      f"{row.get('steps_per_s', '--')} steps/s  {row.get('note', '')}")

    b = budget(rows, args.steps_per_run)
    text = report(rows, b, total, set(modes))
    print("\n" + text)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "probe.md"), "w") as f:
            f.write(text + "\n")
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(args.out, "probe.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(args.out, "probe.json"), "w") as f:
            json.dump({"rows": rows, "budget": b, "gpu": torch.cuda.get_device_name(0),
                       "total_vram_GiB": total}, f, indent=1, default=float)
        print(f"\nwrote {args.out}/probe.md, probe.csv, probe.json")


if __name__ == "__main__":
    main()
