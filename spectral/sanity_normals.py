"""
Per-dataset GT-normals smoke test. Validates that a dataset's loader is wired
correctly: depth+K load, normals derive, validity is sane, and the result lands
pixel-perfectly on the backbone grid. Run this for EACH dataset on the server
before a full training run.

  .../sheaf/python.exe sanity_normals.py --dataset sintel       --index 0
  .../sheaf/python.exe sanity_normals.py --dataset tartanair    --index 0
  .../sheaf/python.exe sanity_normals.py --dataset pointodyssey --index 0

Writes normal/validity visualizations to results/normals_sanity/ to eyeball.
"""
import os
import sys
import argparse
import numpy as np
import imageio.v2 as imageio

import datasets as DS
import normals as NM


def main(dataset, index, out="../results/normals_sanity"):
    os.makedirs(out, exist_ok=True)
    frames = DS.build_frames([dataset], "train")
    if not frames:
        sys.exit(f"no frames found for '{dataset}' -- check DS.CFG['{dataset}']['root'] "
                 f"= {DS.CFG[dataset]['root']} and the directory layout (see SETUP.md).")
    fr = frames[index % len(frames)]
    cfg = DS.CFG[dataset]

    depth, K = DS.load_depth_K(fr)
    print(f"[{dataset}] scene={fr.scene} idx={fr.idx}  depth {depth.shape} "
          f"{np.nanmin(depth):.2f}..{np.nanmax(depth):.2f}  f={K[0,0]:.1f} "
          f"cx,cy={K[0,2]:.1f},{K[1,2]:.1f}")

    n, valid = NM.normals_from_depth_K(depth, K, cfg["rel_thresh"], cfg["max_depth"])
    print(f"[full-res] valid {100*valid.mean():.1f}%")

    n_g, v_g = NM.normals_on_grid_from_depth_K(
        depth, K, rel_thresh=cfg["rel_thresh"], max_depth=cfg["max_depth"])
    mag = np.linalg.norm(n_g[v_g], axis=-1)
    print(f"[on-grid ] {n_g.shape}  valid {100*v_g.mean():.1f}%  "
          f"|n|={mag.mean():.4f}+/-{mag.std():.4f}")

    from dust3r.utils.image import load_images
    g = load_images([fr.rgb_path], size=512, verbose=False)[0]
    true_hw = tuple(int(x) for x in g["true_shape"][0])
    ok = true_hw == n_g.shape[:2]
    print(f"[grid-match] backbone {true_hw} vs ours {n_g.shape[:2]}  "
          f"{'OK' if ok else 'MISMATCH'}")

    tag = f"{dataset}_{fr.idx:04d}"
    imageio.imwrite(f"{out}/{tag}_normal_grid.png",
                    (NM.normal_to_rgb(n_g) * 255).astype(np.uint8))
    imageio.imwrite(f"{out}/{tag}_valid_grid.png", (v_g * 255).astype(np.uint8))
    print(f"wrote {tag}_*.png -> {out}")
    if not ok:
        sys.exit("GRID MISMATCH -- the loader's resolution differs from the "
                 "backbone grid; check the dataset's native image size.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sintel",
                    choices=["sintel", "tartanair", "pointodyssey"])
    ap.add_argument("--index", type=int, default=0, help="frame index into the list")
    args = ap.parse_args()
    main(args.dataset, args.index)
