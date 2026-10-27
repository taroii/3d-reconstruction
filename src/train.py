"""
Training driver: fine-tune VGGT's prediction heads with a geometry-weighted loss.

One arm of the study is one invocation. The arm is (signal, form, lam); the
baseline is `none:clamp:0`, which runs through exactly the same code path and
reproduces VGGT's published objective -- a property `weighted_loss.py` verifies
numerically against upstream rather than asserting.

What is trained
---------------
The DINOv2 patch embedder and the 24-layer alternating aggregator are FROZEN and
run under `no_grad`; only the camera head and the DPT depth head are trained.
This is not a preference. `probe_memory.py` measured a full fine-tune to OOM at
every one of six configurations on this 20 GB card, which the parameter
arithmetic predicts (1.19B params x 16 B/param for AdamW is ~19 GB before a
single activation). The frozen-trunk recipe is also what the ACCV study did with
D2USt3R's encoder, so it is continuous with the work being revised.

The point head is left out: upstream's own default config sets `point: null`, so
the published objective does not supervise it, and training a head the baseline
does not train would break the matched comparison.

Resolution
----------
518 px wide, VGGT's native training resolution. This is a scientific constraint,
not a performance one: the signal under study is a second difference over a
pixel grid whose physical spacing is Z/f, so its magnitude depends on the focal
length IN PIXELS, which changes when you resize. Training at anything other than
native would put a scale confound inside the quantity being measured.
TartanAir is 640x480, so frames are resized to height 392 (a multiple of the 14
px patch size) and centre-cropped to width 518, with the intrinsics carried
through the same transform.

Reproducibility
---------------
Every arm at every seed sets the same torch/numpy/python seeds, uses
deterministic algorithms where available, and draws its clips from the same
`sample_starts(seed=DATA_SEED)` ordering -- so arms differ by the loss and
nothing else. The run config, the git commit and the resolved clip list are
written next to the checkpoint.

  python src/train.py --arm none:clamp:0 --seed 0 --steps 100 --smoke
  python src/train.py --arm umbrella:clamp:1 --seed 0 --steps 6000
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

# Must be set before torch creates its cuBLAS handles, hence before the import.
# Without it `torch.use_deterministic_algorithms` warns and GEMMs are free to
# vary run to run, which would quietly break the reproducibility gate.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.environ.get("PROJ", os.path.abspath(os.path.join(_HERE, "..", "..")))
_VGGT = os.path.join(_PROJ, "third_party", "vggt")
for _p in (_VGGT, os.path.join(_VGGT, "training"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import datasets as DS                                    # noqa: E402
import weighted_loss as WL                               # noqa: E402

CKPT = os.path.join(_PROJ, "ckpt", "model.safetensors")
RUNS = os.path.join(_PROJ, "runs")

# Native width; height is the nearest multiple of the 14 px patch size that
# preserves TartanAir's 4:3 aspect without distortion.
IMG_W, IMG_H = 518, 392
N_VIEWS, STRIDE = 4, 3


# --------------------------------------------------------------------------- #
# batch construction
# --------------------------------------------------------------------------- #

def _resize_view(view, out_h=IMG_H, out_w=IMG_W):
    """Resize + centre-crop a View, carrying the intrinsics through.

    Depth is resampled NEAREST. Bilinear would average across depth
    discontinuities and manufacture exactly the intermediate surfaces this
    study is about -- a blended edge is indistinguishable from a model that
    predicts one, so the target would contain the artefact we are measuring.
    """
    import cv2
    H, W = view.depth.shape
    s = out_h / H
    rw = int(round(W * s))
    rgb = cv2.resize(view.rgb, (rw, out_h), interpolation=cv2.INTER_AREA)
    d = cv2.resize(np.nan_to_num(view.depth, nan=0.0), (rw, out_h),
                   interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(view.valid.astype(np.uint8), (rw, out_h),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    x0 = max(0, (rw - out_w) // 2)
    rgb, d, valid = rgb[:, x0:x0 + out_w], d[:, x0:x0 + out_w], valid[:, x0:x0 + out_w]

    K = view.K.copy()
    K[0, 0] *= s; K[1, 1] *= s
    K[0, 2] = K[0, 2] * s - x0
    K[1, 2] = K[1, 2] * s
    depth = np.where(valid, d, np.nan).astype(np.float32)
    return DS.View(rgb=rgb, depth=depth, K=K, c2w=view.c2w, meta=view.meta)


def build_batch(views, device="cpu"):
    """VGGT's batch contract from a list of Views (one clip).

    Keys and conventions are upstream's: `extrinsics` is (B,S,3,4) WORLD-TO-
    CAMERA in OpenCV convention, `intrinsics` is in pixels, `world_points` is
    (B,S,H,W,3), and the whole scene is normalized to the first camera with
    upstream's own helper so our targets match what the published objective
    was trained against.
    """
    from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

    vs = [_resize_view(v) for v in views]
    S = len(vs)
    H, W = vs[0].depth.shape

    images = torch.from_numpy(
        np.stack([v.rgb for v in vs]).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    depths = torch.from_numpy(np.stack([np.nan_to_num(v.depth, nan=0.0) for v in vs]))
    masks = torch.from_numpy(np.stack([v.valid for v in vs]))
    world = torch.from_numpy(
        np.stack([np.nan_to_num(v.world_points(), nan=0.0) for v in vs]).astype(np.float32))
    # Upstream's normalizer dereferences cam_points unconditionally despite
    # annotating it Optional, so it is computed and passed even though the loss
    # itself never reads it.
    cam = torch.from_numpy(
        np.stack([np.nan_to_num(v.camera_points(), nan=0.0) for v in vs]).astype(np.float32))
    K = torch.from_numpy(np.stack([v.K for v in vs]).astype(np.float32))

    # c2w -> w2c. Upstream wants camera-from-world.
    w2c = np.stack([np.linalg.inv(v.c2w) for v in vs]).astype(np.float32)
    extr = torch.from_numpy(w2c[:, :3, :])

    b = {k: v[None] for k, v in
         (("images", images), ("depths", depths), ("point_masks", masks),
          ("world_points", world), ("extrinsics", extr), ("intrinsics", K))}

    # Upstream's normalization: recentre on the first camera and scale to unit
    # mean point distance. Done with THEIR function so the targets are the ones
    # the published objective expects. It asserts a CPU device.
    e, _, wp, dp = normalize_camera_extrinsics_and_points_batch(
        b["extrinsics"], cam[None], b["world_points"], b["depths"],
        point_masks=b["point_masks"])
    b["extrinsics"], b["world_points"], b["depths"] = e, wp, dp
    return {k: v.to(device) for k, v in b.items()}


class _ClipDataset(torch.utils.data.Dataset):
    """Yields ready-made batches so workers absorb the CPU cost of building them.

    `batch_size=None` on the DataLoader: each item is already a full batch of
    N views, because a "sample" in VGGT's sense is a whole clip. Returning None
    for a broken clip rather than raising keeps one unreadable trajectory from
    killing a multi-hour run; the caller skips Nones and the reason is logged.
    """

    def __init__(self, ta, starts, steps):
        self.ta, self.starts = ta, starts
        # Walk the clip list as many times as the step budget needs. The order
        # is fixed, so every arm and seed sees the same clips in the same
        # sequence -- the data is part of the control, not part of the noise.
        self.n = max(steps, 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        traj, start, nv, st = self.starts[i % len(self.starts)]
        try:
            return build_batch(self.ta.clip(traj, start, nv, st))
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def load_model(device):
    from vggt.models.vggt import VGGT
    m = VGGT(enable_camera=True, enable_point=False, enable_depth=True,
             enable_track=False)
    from safetensors.torch import load_file
    sd = load_file(CKPT)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    dropped = [k for k in missing if not k.startswith(("track_head", "point_head"))]
    if dropped:
        raise RuntimeError(f"checkpoint is missing trained weights: {dropped[:5]}")
    return m.to(device)


def trainable_params(model):
    """Freeze the trunk; train the camera and depth heads."""
    for p in model.parameters():
        p.requires_grad_(False)
    heads = [h for h in (model.camera_head, model.depth_head) if h is not None]
    for h in heads:
        for p in h.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def forward(model, images):
    """Frozen aggregator under no_grad, heads with gradients."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, patch_start_idx = model.aggregator(images)
    tokens = [t.detach() if t is not None else None for t in tokens]
    pred = {}
    with torch.autocast("cuda", enabled=False):
        pose_list = model.camera_head(tokens)
        pred["pose_enc"], pred["pose_enc_list"] = pose_list[-1], pose_list
        d, dc = model.depth_head(tokens, images=images, patch_start_idx=patch_start_idx)
        pred["depth"], pred["depth_conf"] = d, dc
    return pred


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(_HERE),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_arm(a):
    """'signal:form:lam' -> (signal, form, lam), validated against the registry."""
    parts = a.split(":")
    if len(parts) != 3:
        raise ValueError(f"arm must be signal:form:lam, got {a!r}")
    sig, form, lam = parts[0], parts[1], float(parts[2])
    if sig != "none" and sig not in WL.SIGNALS:
        raise ValueError(f"unknown signal {sig!r}; have {sorted(WL.SIGNALS)} or 'none'")
    if form not in WL.WEIGHTS:
        raise ValueError(f"unknown form {form!r}; have {sorted(WL.WEIGHTS)}")
    if sig == "none" and lam != 0.0:
        raise ValueError("signal 'none' with lam != 0 is meaningless")
    return sig, form, lam


def train(args):
    sig, form, lam = parse_arm(args.arm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out = args.out or os.path.join(RUNS, f"{sig}_{form}_lam{lam:g}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)

    # --seed drives the CLIP ORDER, and that is the only stochastic element in
    # a run. This is deliberate and it took a measurement to get right.
    #
    # The heads initialise from the pretrained checkpoint, not randomly; there
    # is no dropout; and with a fixed clip order nothing else varies. Measured:
    # seed 0 and seed 1 produced BITWISE IDENTICAL losses. A five-seed study
    # like that reports zero variance and error bars that mean nothing.
    #
    # So the seed shuffles the data order. Every ARM at a given seed draws the
    # same order from the same seeded sampler, so arms stay perfectly matched
    # within a seed; across seeds the order changes, which is a real and
    # honest source of run-to-run variance. What the error bars then estimate
    # is sensitivity to the data ordering, and that is what they should be
    # described as -- not "training noise" in general.
    ta = DS.TartanAir()
    starts = ta.sample_starts(n_views=N_VIEWS, stride=STRIDE, seed=args.seed)
    if not starts:
        raise SystemExit("no clips available; is TartanAir staged?")

    cfg = dict(arm=args.arm, signal=sig, form=form, lam=lam, seed=args.seed,
               steps=args.steps, lr=args.lr, n_views=N_VIEWS, stride=STRIDE,
               img_h=IMG_H, img_w=IMG_W,
               n_clips=len(starts), envs=ta.envs, commit=git_commit(),
               tau=WL.DEFAULT_TAU, q=WL.DEFAULT_Q)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print(json.dumps(cfg, indent=1))

    model = load_model(device)
    params = trainable_params(model)
    print(f"trainable: {sum(p.numel() for p in params)/1e6:.1f}M of "
          f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    from loss import compute_camera_loss
    log = open(os.path.join(out, "train.log"), "a")

    # Building a batch costs ~0.49 s against ~0.56 s of GPU work, so running it
    # inline left the card idle for nearly half of every step. A worker pool
    # overlaps the two. `starts` order is fixed and the sampler is seeded, so
    # prefetching changes throughput and nothing else.
    loader = torch.utils.data.DataLoader(
        _ClipDataset(ta, starts, args.steps), batch_size=None,
        num_workers=args.workers, pin_memory=True,
        prefetch_factor=2 if args.workers else None,
        persistent_workers=bool(args.workers))

    t0, step = time.time(), 0
    for batch in loader:
        if step >= args.steps:
            break
        if batch is None or batch["point_masks"].sum() < 100:
            continue
        gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        pred = forward(model, gpu["images"])
        cam = compute_camera_loss(pred, gpu, loss_type="l1")
        dep = WL.make_depth_loss(pred, gpu, signal=sig, form=form, lam=lam)
        loss = (cam["loss_camera"] * 5.0
                + dep["loss_conf_depth"] + dep["loss_reg_depth"])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            msg = (f"step {step:>6}/{args.steps}  loss {loss.item():.4f}  "
                   f"cam {cam['loss_camera'].item():.4f}  "
                   f"conf {dep['loss_conf_depth'].item():.4f}  "
                   f"reg {dep['loss_reg_depth'].item():.4f}  "
                   f"lr {sched.get_last_lr()[0]:.2e}  "
                   f"{(time.time()-t0)/step:.2f}s/step")
            print(msg); print(msg, file=log, flush=True)
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({"step": step, "cfg": cfg,
                        "heads": {k: v for k, v in model.state_dict().items()
                                  if k.startswith(("camera_head", "depth_head"))}},
                       os.path.join(out, "checkpoint.pt"))

    torch.save({"step": step, "cfg": cfg,
                "heads": {k: v for k, v in model.state_dict().items()
                          if k.startswith(("camera_head", "depth_head"))}},
               os.path.join(out, "checkpoint.pt"))
    # Completion marker. `checkpoint.pt` is ALSO written periodically mid-run,
    # so its existence says nothing about whether training finished. Without a
    # separate marker a run that OOMs at step 500 leaves a checkpoint that the
    # matrix runner skips as "already complete" and that evaluate.py scores as
    # a finished arm -- a half-trained model silently entering the paper's
    # table. Written last, after the final save, so it can only mean "the
    # requested number of steps ran and the weights on disk are those weights".
    with open(os.path.join(out, "done.json"), "w") as f:
        json.dump({"step": step, "steps_requested": args.steps,
                   "arm": args.arm, "seed": args.seed}, f, indent=1)
    print(f"done: {step} steps in {(time.time()-t0)/60:.1f} min -> {out}")
    log.close()


# --------------------------------------------------------------------------- #

def _selftest():
    """Batch construction only -- no GPU, no checkpoint needed."""
    ta = DS.TartanAir()
    assert ta.trajs, "TartanAir not staged"
    views = ta.clip(ta.trajs[0], 0, N_VIEWS, STRIDE)
    assert len(views) == N_VIEWS

    v = _resize_view(views[0])
    assert v.depth.shape == (IMG_H, IMG_W), f"resize gave {v.depth.shape}"
    # intrinsics must scale with the image, or every pointmap is wrong
    s = IMG_H / views[0].depth.shape[0]
    assert abs(v.K[1, 1] - views[0].K[1, 1] * s) < 1e-3, "fy not scaled"
    assert abs(v.K[1, 2] - views[0].K[1, 2] * s) < 1e-3, "cy not scaled"
    # depth must stay NEAREST: no value may appear that was not in the original
    orig = set(np.unique(views[0].depth[views[0].valid]).tolist())
    new = set(np.unique(v.depth[v.valid]).tolist())
    assert new <= orig, "resampling invented depth values -- not nearest-neighbour"

    b = build_batch(views)
    for k, shp in (("images", (1, N_VIEWS, 3, IMG_H, IMG_W)),
                   ("depths", (1, N_VIEWS, IMG_H, IMG_W)),
                   ("point_masks", (1, N_VIEWS, IMG_H, IMG_W)),
                   ("world_points", (1, N_VIEWS, IMG_H, IMG_W, 3)),
                   ("extrinsics", (1, N_VIEWS, 3, 4)),
                   ("intrinsics", (1, N_VIEWS, 3, 3))):
        assert tuple(b[k].shape) == shp, f"{k}: {tuple(b[k].shape)} != {shp}"
    assert b["images"].min() >= 0 and b["images"].max() <= 1, "images not in [0,1]"
    assert torch.isfinite(b["world_points"]).all(), "non-finite world points"
    assert b["point_masks"].sum() > 1000, "almost nothing valid after resize"

    # normalization must put the first camera at the origin
    e0 = b["extrinsics"][0, 0]
    assert torch.allclose(e0[:, :3], torch.eye(3), atol=1e-4) and \
        torch.allclose(e0[:, 3], torch.zeros(3), atol=1e-4), \
        "scene is not normalized to the first camera"

    # the arm parser must reject nonsense rather than train something silently
    for bad in ("umbrella:clamp", "nosuchsignal:clamp:1", "umbrella:nosuchform:1",
                "none:clamp:1"):
        try:
            parse_arm(bad); raise AssertionError(f"accepted bad arm {bad!r}")
        except ValueError:
            pass
    assert parse_arm("none:clamp:0") == ("none", "clamp", 0.0)
    print(f"selftest: batch ok  {N_VIEWS} views  {IMG_H}x{IMG_W}  "
          f"{int(b['point_masks'].sum())} valid px")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="none:clamp:0", help="signal:form:lam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4,
                    help="batch-building workers; 0 runs inline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the loop executes end to end")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if a.smoke:
        a.steps, a.log_every, a.ckpt_every = min(a.steps, 5), 1, 0
    train(a)


if __name__ == "__main__":
    main()
