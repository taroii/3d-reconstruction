"""
Train N_phi (PLAN step 1a): DPT normal head on the frozen DUSt3R encoder,
supervised by Sintel GT normals (step 0). Head-only; encoder frozen.

  .../sheaf/python.exe train_nphi.py --overfit            # wiring/learnability smoke
  .../sheaf/python.exe train_nphi.py --epochs 30          # full run

Checkpoints -> spectral/checkpoints/nphi_<backbone>.pth (gitignored: *.pth).
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import backbone as B
import sintel as SI
import normals as NM
from normal_head import NormalHead, angular_conf_loss

ROOT = "../data"
CACHE = "../cache"
# held-out scenes (diverse: indoor/outdoor, low/high motion); rest = train
VAL_SCENES = {"market_5", "cave_4", "ambush_6", "temple_3"}


def all_scenes():
    d = os.path.join(ROOT, "training", "clean")
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*")))


def cached_grid_normals(scene, idx0):
    """GT normals+valid on the backbone grid, cached to cache/<scene>/normals/."""
    p = os.path.join(CACHE, scene, "normals", f"{idx0:04d}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["n"].astype(np.float32), d["valid"]
    n, valid = NM.gt_normals_on_grid(ROOT, scene, idx0 + 1)   # GT files are 1-based
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, n=n.astype(np.float16), valid=valid)
    return n.astype(np.float32), valid


class NormalDataset(Dataset):
    def __init__(self, scenes, limit=None):
        from dust3r.utils.image import load_images
        self._load = load_images
        self.items = []
        for s in scenes:
            paths = SI.frame_paths(ROOT, s, "clean")
            if limit:
                paths = paths[:limit]
            self.items += [(s, i, p) for i, p in enumerate(paths)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        scene, idx0, path = self.items[k]
        g = self._load([path], size=512, verbose=False)[0]
        img = g["img"][0]                                  # (3,H,W)
        ts = np.asarray(g["true_shape"][0])                # (2,)
        n_gt, valid = cached_grid_normals(scene, idx0)
        return img, ts, torch.from_numpy(n_gt), torch.from_numpy(valid)


def collate(batch):
    imgs, ts, ngt, val = zip(*batch)
    return (torch.stack(imgs), torch.tensor(np.stack(ts)),
            torch.stack(ngt), torch.stack(val))


@torch.no_grad()
def evaluate(net, loader, dev):
    net.eval()
    tot, npx = 0.0, 0.0
    for img, ts, ngt, valid in loader:
        img, ts = img.to(dev), ts.to(dev)
        ngt, valid = ngt.to(dev), valid.to(dev)
        n_pred, omega = net(img, ts)
        _, mae = angular_conf_loss(n_pred, omega, ngt, valid)
        v = valid.sum().item()
        tot += mae.item() * v
        npx += v
    return tot / max(npx, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="d2ust3r")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None, help="frames/scene cap")
    ap.add_argument("--overfit", action="store_true",
                    help="overfit one batch to check learnability")
    args = ap.parse_args()
    dev = "cuda"

    bb = B.load_backbone(args.backbone)
    net = NormalHead(bb).to(dev)
    opt = torch.optim.AdamW(net.trainable_parameters(), lr=args.lr, weight_decay=1e-4)

    if args.overfit:
        ds = NormalDataset(["alley_1"], limit=args.bs)
        loader = DataLoader(ds, batch_size=args.bs, collate_fn=collate)
        img, ts, ngt, valid = next(iter(loader))
        img, ts, ngt, valid = img.to(dev), ts.to(dev), ngt.to(dev), valid.to(dev)
        net.train()
        for step in range(150):
            opt.zero_grad()
            n_pred, omega = net(img, ts)
            loss, mae = angular_conf_loss(n_pred, omega, ngt, valid, args.alpha)
            loss.backward(); opt.step()
            if step % 25 == 0 or step == 149:
                print(f"  step {step:3d}  loss {loss.item():.4f}  "
                      f"MAE {mae.item():.1f} deg  omega {omega.mean().item():.2f}", flush=True)
        return

    scenes = all_scenes()
    tr = [s for s in scenes if s not in VAL_SCENES]
    va = [s for s in scenes if s in VAL_SCENES]
    print(f"train scenes {len(tr)}  val scenes {len(va)}: {va}", flush=True)
    tr_ds = NormalDataset(tr, limit=args.limit)
    va_ds = NormalDataset(va, limit=args.limit)
    print(f"train frames {len(tr_ds)}  val frames {len(va_ds)}", flush=True)
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    va_ld = DataLoader(va_ds, batch_size=args.bs, collate_fn=collate)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/nphi_{args.backbone}.pth"
    best = 1e9
    for ep in range(args.epochs):
        net.train()
        run, npx = 0.0, 0.0
        for img, ts, ngt, valid in tr_ld:
            img, ts = img.to(dev), ts.to(dev)
            ngt, valid = ngt.to(dev), valid.to(dev)
            opt.zero_grad()
            n_pred, omega = net(img, ts)
            loss, mae = angular_conf_loss(n_pred, omega, ngt, valid, args.alpha)
            loss.backward(); opt.step()
            v = valid.sum().item(); run += mae.item() * v; npx += v
        sched.step()
        val_mae = evaluate(net, va_ld, dev)
        tag = ""
        if val_mae < best:
            best = val_mae
            torch.save({"head": net.dpt.state_dict(), "hooks": net.hooks,
                        "backbone": args.backbone, "val_mae": val_mae}, ckpt)
            tag = "  *saved"
        print(f"epoch {ep:2d}  train MAE {run/max(npx,1):.1f}  "
              f"val MAE {val_mae:.1f} deg{tag}", flush=True)
    print(f"best val MAE {best:.1f} deg -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
