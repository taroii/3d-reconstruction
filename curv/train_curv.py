"""
Train the curvature head (brief Option 1, standalone prototype): a DPT head on
the frozen D2USt3R encoder, supervised by GT curvature from one or more datasets
(TartanAir + PointOdyssey for training; Sintel is eval-only). Head-only; encoder
frozen.

This is the learnability gate -- it proves the encoder features carry recoverable
curvature before we add the lambda*L_curv term to D2USt3R's real pair-training
loop (PLAN.md Phase 3, in the DDUSt3R training code).

  python train_curv.py --overfit                                   # learnability smoke
  python train_curv.py --datasets tartanair                        # single-dataset baseline
  python train_curv.py --datasets tartanair,pointodyssey \
        --max-per-dataset 6000 --epochs 30 --mode mean             # generalizable run

Run from inside curv/. Checkpoints -> curv/checkpoints/curv_<tag>.pth.
GT curvature is cached under ../cache/curv_targets/<dataset>/<scene>/<idx>.npz.
"""
import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import backbone as B
import datasets as DS
import curvature as CV
from curvature_head import CurvatureHead, curvature_conf_loss

CACHE = "../cache"


def cached_grid_curvature(fr, mode, compress):
    """GT curvature+validity on the backbone grid, cached per frame + mode."""
    tag = f"{mode}{'_c' if compress else ''}"
    p = os.path.join(CACHE, "curv_targets", tag, fr.key + ".npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["k"].astype(np.float32), d["valid"]
    depth, K = DS.load_depth_K(fr)
    cfg = DS.CFG[fr.dataset]
    Kmap, valid = CV.curvature_on_grid_from_depth_K(
        depth, K, mode=mode, rel_thresh=cfg["rel_thresh"],
        max_depth=cfg["max_depth"], compress=compress)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, k=Kmap.astype(np.float16), valid=valid)
    return Kmap.astype(np.float32), valid


class CurvatureDataset(Dataset):
    def __init__(self, frames, mode, compress, scale=1.0):
        from dust3r.utils.image import load_images
        self._load = load_images
        self.frames = frames
        self.mode = mode
        self.compress = compress
        self.scale = scale          # standardize target to ~unit variance

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        fr = self.frames[k]
        g = self._load([fr.rgb_path], size=512, verbose=False)[0]
        img = g["img"][0]
        ts = np.asarray(g["true_shape"][0])
        k_gt, valid = cached_grid_curvature(fr, self.mode, self.compress)
        k_gt = k_gt / self.scale
        return img, ts, torch.from_numpy(k_gt), torch.from_numpy(valid), fr.dataset


def estimate_target_scale(frames, mode, compress, k=200, seed=0):
    """Std of the (valid) curvature targets over a sample -- used to standardize
    the target to ~unit variance so the regression head gets O(1) gradients.
    Targets are ~1e-4 raw, which strands Huber in its vanishing-gradient regime."""
    fs = list(frames)
    random.Random(seed).shuffle(fs)
    vals = []
    for fr in fs[:k]:
        g, v = cached_grid_curvature(fr, mode, compress)
        if v.any():
            vals.append(g[v].astype(np.float32))
    if not vals:
        return 1.0
    s = float(np.concatenate(vals).std())
    return s if s > 1e-9 else 1.0


def collate(batch):
    imgs, ts, kgt, val, ds = zip(*batch)
    return (torch.stack(imgs), torch.tensor(np.stack(ts)),
            torch.stack(kgt), torch.stack(val), ds[0])


class DatasetBatchSampler(Sampler):
    """Batch within a single dataset only (datasets have different grid shapes,
    so cross-dataset frames cannot be stacked). Batches interleave across epoch."""
    def __init__(self, frames, bs, shuffle):
        self.bs, self.shuffle = bs, shuffle
        self.buckets = {}
        for i, fr in enumerate(frames):
            self.buckets.setdefault(fr.dataset, []).append(i)

    def __iter__(self):
        batches = []
        for idxs in self.buckets.values():
            idxs = list(idxs)
            if self.shuffle:
                random.shuffle(idxs)
            batches += [idxs[s:s + self.bs] for s in range(0, len(idxs), self.bs)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum((len(v) + self.bs - 1) // self.bs for v in self.buckets.values())


def loader_for(frames, bs, shuffle, mode, compress, scale=1.0):
    return DataLoader(CurvatureDataset(frames, mode, compress, scale), collate_fn=collate,
                      batch_sampler=DatasetBatchSampler(frames, bs, shuffle))


@torch.no_grad()
def evaluate(net, loader, dev):
    """Overall + per-dataset held-out curvature L1 MAE (target space)."""
    net.eval()
    agg = {}
    for img, ts, kgt, valid, ds in loader:
        img, ts, kgt, valid = img.to(dev), ts.to(dev), kgt.to(dev), valid.to(dev)
        curv, omega = net(img, ts)
        _, mae = curvature_conf_loss(curv, omega, kgt, valid)
        v = valid.sum().item()
        s, n = agg.get(ds, (0.0, 0.0))
        agg[ds] = (s + mae.item() * v, n + v)
    per = {d: s / max(n, 1) for d, (s, n) in agg.items()}
    tot = sum(s for s, _ in agg.values()) / max(sum(n for _, n in agg.values()), 1)
    return tot, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="d2ust3r")
    ap.add_argument("--datasets", default="tartanair",
                    help="comma list: tartanair,pointodyssey,spring (sintel is eval-only)")
    ap.add_argument("--mode", default="mean", choices=["mean", "gaussian"])
    ap.add_argument("--no-compress", action="store_true",
                    help="supervise raw curvature instead of signed-log target")
    ap.add_argument("--max-per-dataset", type=int, default=None,
                    help="cap frames/dataset (balances big synthetic sets)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=1.0, help="Huber transition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    compress = not args.no_compress
    random.seed(args.seed); torch.manual_seed(args.seed)

    bb = B.load_backbone(args.backbone)
    net = CurvatureHead(bb).to(dev)
    opt = torch.optim.AdamW(net.trainable_parameters(), lr=args.lr, weight_decay=1e-4)

    if args.overfit:
        frames = DS.build_frames(["sintel"], "train")[:args.bs]   # any data with GT
        scale = estimate_target_scale(frames, args.mode, compress, k=args.bs)
        print(f"target scale (std) = {scale:.5f}", flush=True)
        loader = loader_for(frames, args.bs, False, args.mode, compress, scale)
        img, ts, kgt, valid, _ = next(iter(loader))
        img, ts, kgt, valid = img.to(dev), ts.to(dev), kgt.to(dev), valid.to(dev)
        net.train()
        for step in range(150):
            opt.zero_grad()
            curv, omega = net(img, ts)
            loss, mae = curvature_conf_loss(curv, omega, kgt, valid, args.alpha, args.delta)
            loss.backward(); opt.step()
            if step % 25 == 0 or step == 149:
                print(f"  step {step:3d}  loss {loss.item():.4f}  MAE {mae.item():.4f}",
                      flush=True)
        return

    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tag = ("_".join(names) if len(names) > 1 else names[0]) + f"_{args.mode}"
    print(f"datasets: {names}  mode: {args.mode}  compress: {compress}", flush=True)
    print("building train frames...", flush=True)
    tr = DS.build_frames(names, "train", args.max_per_dataset, args.seed)
    print("building val frames...", flush=True)
    va = DS.build_frames(names, "val", args.max_per_dataset, args.seed)
    print(f"train {len(tr)}  val {len(va)}", flush=True)
    scale = estimate_target_scale(tr, args.mode, compress)
    print(f"target scale (std) = {scale:.5f}  -> standardizing target to ~unit var",
          flush=True)
    tr_ld = loader_for(tr, args.bs, True, args.mode, compress, scale)
    va_ld = loader_for(va, args.bs, False, args.mode, compress, scale)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/curv_{tag}.pth"
    best = 1e9
    for ep in range(args.epochs):
        net.train()
        run, npx = 0.0, 0.0
        for img, ts, kgt, valid, _ in tr_ld:
            img, ts, kgt, valid = img.to(dev), ts.to(dev), kgt.to(dev), valid.to(dev)
            opt.zero_grad()
            curv, omega = net(img, ts)
            loss, mae = curvature_conf_loss(curv, omega, kgt, valid, args.alpha, args.delta)
            loss.backward(); opt.step()
            v = valid.sum().item(); run += mae.item() * v; npx += v
        sched.step()
        val_mae, per = evaluate(net, va_ld, dev)
        flag = ""
        if val_mae < best:
            best = val_mae
            torch.save({"head": net.dpt.state_dict(), "hooks": net.hooks,
                        "backbone": args.backbone, "datasets": names,
                        "mode": args.mode, "compress": compress,
                        "target_scale": scale, "val_mae": val_mae}, ckpt)
            flag = "  *saved"
        perstr = " ".join(f"{d}={m:.4f}" for d, m in sorted(per.items()))
        print(f"epoch {ep:2d}  train {run/max(npx,1):.4f}  val {val_mae:.4f} "
              f"[{perstr}]{flag}", flush=True)
    print(f"best val MAE {best:.4f} -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
