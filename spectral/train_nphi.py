"""
Train N_phi (PLAN step 1a): DPT normal head on the frozen DUSt3R encoder,
supervised by GT normals from one or more datasets (Sintel + TartanAir +
PointOdyssey). Head-only; encoder frozen.

  python train_nphi.py --overfit                                  # learnability smoke
  python train_nphi.py --datasets sintel                          # Sintel-only baseline
  python train_nphi.py --datasets sintel,tartanair,pointodyssey \
        --max-per-dataset 6000 --epochs 30                        # generalizable run

Run from inside spectral/. Checkpoints -> spectral/checkpoints/nphi_<tag>.pth.
Normals are cached under ../cache/nphi_normals/<dataset>/<scene>/<idx>.npz.
"""
import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import backbone as B
import datasets as DS
import normals as NM
from normal_head import NormalHead, angular_conf_loss

CACHE = "../cache"


def cached_grid_normals(fr):
    """GT normals+validity on the backbone grid, cached per frame by dataset key."""
    p = os.path.join(CACHE, "nphi_normals", fr.key + ".npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["n"].astype(np.float32), d["valid"]
    depth, K = DS.load_depth_K(fr)
    cfg = DS.CFG[fr.dataset]
    n, valid = NM.normals_on_grid_from_depth_K(
        depth, K, rel_thresh=cfg["rel_thresh"], max_depth=cfg["max_depth"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, n=n.astype(np.float16), valid=valid)
    return n.astype(np.float32), valid


class NormalDataset(Dataset):
    def __init__(self, frames):
        from dust3r.utils.image import load_images
        self._load = load_images
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        fr = self.frames[k]
        g = self._load([fr.rgb_path], size=512, verbose=False)[0]
        img = g["img"][0]
        ts = np.asarray(g["true_shape"][0])
        n_gt, valid = cached_grid_normals(fr)
        return img, ts, torch.from_numpy(n_gt), torch.from_numpy(valid), fr.dataset


def collate(batch):
    imgs, ts, ngt, val, ds = zip(*batch)
    return (torch.stack(imgs), torch.tensor(np.stack(ts)),
            torch.stack(ngt), torch.stack(val), ds[0])


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


def loader_for(frames, bs, shuffle):
    return DataLoader(NormalDataset(frames), collate_fn=collate,
                      batch_sampler=DatasetBatchSampler(frames, bs, shuffle))


@torch.no_grad()
def evaluate(net, loader, dev):
    """Overall + per-dataset held-out angular MAE (degrees)."""
    net.eval()
    agg = {}
    for img, ts, ngt, valid, ds in loader:
        img, ts, ngt, valid = img.to(dev), ts.to(dev), ngt.to(dev), valid.to(dev)
        n_pred, omega = net(img, ts)
        _, mae = angular_conf_loss(n_pred, omega, ngt, valid)
        v = valid.sum().item()
        s, n = agg.get(ds, (0.0, 0.0))
        agg[ds] = (s + mae.item() * v, n + v)
    per = {d: s / max(n, 1) for d, (s, n) in agg.items()}
    tot = sum(s for s, _ in agg.values()) / max(sum(n for _, n in agg.values()), 1)
    return tot, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="d2ust3r")
    ap.add_argument("--datasets", default="sintel",
                    help="comma list: sintel,tartanair,pointodyssey")
    ap.add_argument("--max-per-dataset", type=int, default=None,
                    help="cap frames/dataset (balances big synthetic sets vs Sintel)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    random.seed(args.seed); torch.manual_seed(args.seed)

    bb = B.load_backbone(args.backbone)
    net = NormalHead(bb).to(dev)
    opt = torch.optim.AdamW(net.trainable_parameters(), lr=args.lr, weight_decay=1e-4)

    if args.overfit:
        frames = DS.build_frames(["sintel"], "train")[:args.bs]
        loader = loader_for(frames, args.bs, shuffle=False)
        img, ts, ngt, valid, _ = next(iter(loader))
        img, ts, ngt, valid = img.to(dev), ts.to(dev), ngt.to(dev), valid.to(dev)
        net.train()
        for step in range(150):
            opt.zero_grad()
            n_pred, omega = net(img, ts)
            loss, mae = angular_conf_loss(n_pred, omega, ngt, valid, args.alpha)
            loss.backward(); opt.step()
            if step % 25 == 0 or step == 149:
                print(f"  step {step:3d}  loss {loss.item():.4f}  MAE {mae.item():.1f} deg",
                      flush=True)
        return

    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tag = "_".join(names) if len(names) > 1 else names[0]
    print(f"datasets: {names}", flush=True)
    print("building train frames...", flush=True)
    tr = DS.build_frames(names, "train", args.max_per_dataset, args.seed)
    print("building val frames...", flush=True)
    va = DS.build_frames(names, "val", args.max_per_dataset, args.seed)
    print(f"train {len(tr)}  val {len(va)}", flush=True)
    tr_ld = loader_for(tr, args.bs, shuffle=True)
    va_ld = loader_for(va, args.bs, shuffle=False)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/nphi_{tag}.pth"
    best = 1e9
    for ep in range(args.epochs):
        net.train()
        run, npx = 0.0, 0.0
        for img, ts, ngt, valid, _ in tr_ld:
            img, ts, ngt, valid = img.to(dev), ts.to(dev), ngt.to(dev), valid.to(dev)
            opt.zero_grad()
            n_pred, omega = net(img, ts)
            loss, mae = angular_conf_loss(n_pred, omega, ngt, valid, args.alpha)
            loss.backward(); opt.step()
            v = valid.sum().item(); run += mae.item() * v; npx += v
        sched.step()
        val_mae, per = evaluate(net, va_ld, dev)
        flag = ""
        if val_mae < best:
            best = val_mae
            torch.save({"head": net.dpt.state_dict(), "hooks": net.hooks,
                        "backbone": args.backbone, "datasets": names,
                        "val_mae": val_mae}, ckpt)
            flag = "  *saved"
        perstr = " ".join(f"{d}={m:.1f}" for d, m in sorted(per.items()))
        print(f"epoch {ep:2d}  train {run/max(npx,1):.1f}  val {val_mae:.1f} "
              f"[{perstr}]{flag}", flush=True)
    print(f"best val MAE {best:.1f} deg -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
