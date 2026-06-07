"""Step-0 sanity check: GT normals full-res + on-grid, grid-match vs the real
backbone grid, and saved visualizations. Run with the sheaf env python.

  .../envs/sheaf/python.exe sanity_normals.py --scene alley_1 --idx 1
"""
import os
import argparse
import numpy as np
import imageio.v2 as imageio

import sintel as SI
import normals as NM


def main(scene, idx, root="../data", out="../results/normals_sanity"):
    os.makedirs(out, exist_ok=True)

    n, valid, depth, K = NM.gt_normals(root, scene, idx)
    print(f"[full-res] normals {n.shape}  valid {100*valid.mean():.1f}%  "
          f"depth {depth.min():.2f}..{depth.max():.2f}  f={K[0,0]:.1f}")

    n_g, v_g = NM.gt_normals_on_grid(root, scene, idx)
    print(f"[on-grid ] normals {n_g.shape}  valid {100*v_g.mean():.1f}%")

    # unit-norm check on valid grid pixels
    mag = np.linalg.norm(n_g[v_g], axis=-1)
    print(f"[on-grid ] |n| on valid: {mag.mean():.4f} +/- {mag.std():.4f}")

    # grid must match what the backbone actually produces for this frame
    from dust3r.utils.image import load_images
    fp = SI.frame_paths(root, scene, "clean")[idx - 1]
    gimg = load_images([fp], size=512, verbose=False)[0]
    true_hw = tuple(int(x) for x in gimg["true_shape"][0])
    print(f"[grid-match] backbone true_shape={true_hw}  ours={n_g.shape[:2]}  "
          f"{'OK' if true_hw == n_g.shape[:2] else 'MISMATCH'}")

    imageio.imwrite(f"{out}/{scene}_{idx:04d}_normal_fullres.png",
                    (NM.normal_to_rgb(n) * 255).astype(np.uint8))
    imageio.imwrite(f"{out}/{scene}_{idx:04d}_normal_grid.png",
                    (NM.normal_to_rgb(n_g) * 255).astype(np.uint8))
    imageio.imwrite(f"{out}/{scene}_{idx:04d}_valid_grid.png",
                    (v_g * 255).astype(np.uint8))
    dvis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    imageio.imwrite(f"{out}/{scene}_{idx:04d}_depth.png",
                    (dvis * 255).astype(np.uint8))
    print(f"wrote visualizations -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="alley_1")
    ap.add_argument("--idx", type=int, default=1)
    args = ap.parse_args()
    main(args.scene, args.idx)
