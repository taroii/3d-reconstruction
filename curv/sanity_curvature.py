"""
Sanity checks for the discrete curvature estimators (curvature.py), plus an
optional qualitative dump on a real frame.

  python sanity_curvature.py                       # synthetic-surface unit checks
  python sanity_curvature.py --dataset sintel --scene alley_1 --idx 1
                                                   # dump a real-frame curvature PNG

Synthetic checks build the pointmap P directly for surfaces with known curvature
and verify SIGNS and relative magnitudes (the discrete angle-deficit Gaussian is
the integrated form, so we check sign/uniformity, not an exact 1/r^2 value):
  - plane        -> K_gauss ~ 0,  H ~ 0
  - convex bump  -> H > 0
  - saddle       -> K_gauss < 0
"""
import argparse
import numpy as np

import curvature as CV


def _grid(n=64, span=2.0):
    t = np.linspace(-span, span, n)
    x, y = np.meshgrid(t, t)
    return x, y


def _interior_mean(field):
    return float(np.mean(field[2:-2, 2:-2]))


def synthetic_checks():
    x, y = _grid()
    ok = True

    # plane z=const: both curvatures ~ 0
    P = np.stack([x, y, np.full_like(x, 5.0)], axis=-1)
    Kg = CV.gaussian_curvature_from_points(P)
    H = CV.mean_curvature_from_points(P)
    print(f"plane:       <|K_gauss|>={_interior_mean(np.abs(Kg)):.2e}  "
          f"<|H|>={_interior_mean(np.abs(H)):.2e}  (expect ~0)")
    ok &= _interior_mean(np.abs(Kg)) < 1e-3 and _interior_mean(np.abs(H)) < 1e-3

    # convex paraboloid z = 5 - 0.3(x^2+y^2): mean curvature > 0 (with our
    # camera-facing normal orientation), Gaussian deficit > 0
    z = 5.0 - 0.3 * (x ** 2 + y ** 2)
    P = np.stack([x, y, z], axis=-1)
    Kg = CV.gaussian_curvature_from_points(P)
    H = CV.mean_curvature_from_points(P)
    print(f"convex bump: <K_gauss>={_interior_mean(Kg):+.3e}  "
          f"<H>={_interior_mean(H):+.3e}  (expect K>0, H>0)")
    ok &= _interior_mean(Kg) > 0 and _interior_mean(H) > 0

    # saddle z = 5 + 0.3(x^2 - y^2): Gaussian curvature < 0
    z = 5.0 + 0.3 * (x ** 2 - y ** 2)
    P = np.stack([x, y, z], axis=-1)
    Kg = CV.gaussian_curvature_from_points(P)
    print(f"saddle:      <K_gauss>={_interior_mean(Kg):+.3e}  (expect K<0)")
    ok &= _interior_mean(Kg) < 0

    print("SYNTHETIC CHECKS:", "PASS" if ok else "FAIL")
    return ok


def real_dump(dataset, scene, idx, mode):
    import imageio.v2 as imageio
    from datasets import CFG
    root = CFG[dataset]["root"]
    if dataset != "sintel":
        raise SystemExit("real dump wrapper currently wired for sintel only "
                         "(uses curvature.gt_curvature); other datasets via "
                         "curvature.curvature_from_depth_K + datasets.load_depth_K).")
    Kmap, valid, depth, K = CV.gt_curvature(root, scene, idx, mode=mode)
    rgb = (CV.curvature_to_rgb(np.where(valid, Kmap, 0.0)) * 255).astype(np.uint8)
    out = f"curv_{dataset}_{scene}_{idx:04d}_{mode}.png"
    imageio.imwrite(out, rgb)
    print(f"valid {valid.mean()*100:.1f}%  K range [{Kmap[valid].min():+.3f}, "
          f"{Kmap[valid].max():+.3f}]  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--scene", default="alley_1")
    ap.add_argument("--idx", type=int, default=1)
    ap.add_argument("--mode", default="mean", choices=["mean", "gaussian"])
    args = ap.parse_args()
    if args.dataset is None:
        synthetic_checks()
    else:
        real_dump(args.dataset, args.scene, args.idx, args.mode)


if __name__ == "__main__":
    main()
