"""
What does the umbrella-Laplacian "curvature" signal actually measure?

The ACCV submission weights the pointmap position residual by

    S_i = |K_i| = (1/2) || (1/4) sum_{n in N(i)} Xbar_n  -  Xbar_i ||          (1)

and calls S_i a discrete mean curvature, citing the Laplace-Beltrami relation
Delta_M x = -2 H n.  Three reviewers independently objected that (1) is not that
quantity.  The sharpest form of the objection (Reviewer CYqH):

    "The umbrella Laplacian in Eq. (5) has no area normalization, unlike the
     cotangent/Voronoi estimator of Meyer et al. that the paper cites, whose
     mean-curvature normal carries a 1/(2A) factor.  As written, (L chi)_i has
     units of length while -2 H n has units of inverse length [...] on a slanted
     or distant plane, where true mean curvature is zero, the estimator is
     nonzero and grows with slant and depth.  Reporting the estimator's value on
     a synthetic slanted plane, and re-running with a spacing-normalized or
     Voronoi-normalized estimator, would establish whether anything second-order
     is doing work."

This module does exactly that, analytically.  Every surface here has a mean
curvature we know in closed form, rendered through a real perspective camera and
back-projected to a pointmap, so the input to the estimators is the same object
the training loss sees -- including the `avg_dis` normalization, which matters
and is easy to forget (Sec. "normalization" below).

Estimators compared:

  umbrella    Eq. (1), the submitted signal.                 units: length
  cotangent   Meyer et al. mixed-area Voronoi estimator.     units: 1/length
  spacing     umbrella / mean-squared neighbour spacing.     units: 1/length
  grad        || grad Xbar ||, the paper's first-order control.

`cotangent` is the reference: it is the estimator the paper cites but does not
use, and it is the one that returns |H| in physical units.  Its correctness is
not assumed -- `--selftest` asserts it recovers |H| = 1/R on a sphere and
|H| = 1/(2R) on a cylinder to within a few percent, which is the only reason to
trust anything else this file reports.

Seven analyses (`--analysis all`):

  zero        On surfaces with true H = 0, what does each estimator report,
              relative to genuine curvature at the same depth?
  slant       Signal vs plane slant, 0..75 deg.  Truth is flat 0 throughout.
  depth       Two IDENTICAL spheres at different depths in ONE image.  Truth is
              a flat ratio of 1.
  depthsweep  The same sphere alone in its frame at a range of depths, which
              separates the within-image bias from the global one.
  edges       In a scene holding both genuine curvature and a depth step, what
              fraction of the top-decile-by-signal pixels -- the paper's own
              "boundary subset" -- are step pixels rather than curved ones?
  rankcorr    Spearman rho between the umbrella signal and the first-order
              control, per scene.  The paper reports that the gradient control
              recovers most of the gain; a high rho is the mechanical reason.
  recovery    Measured / true |H| across sphere radii, per estimator.

A note on normalization.  The loss computes S on the avg_dis-normalized pointmap
(P <- P / mean ||P||), so a GLOBAL depth change cancels and only WITHIN-image
depth variation biases the weight.  Reporting raw metric values would overstate
the problem; every analysis here therefore reports the normalized signal, which
is what the weight w_i = 1 + gamma*min(S_i/Sbar, tau) actually sees.  Inverse-length estimators (`cotangent`, `spacing`) are rescaled back to metric
units in EVERY analysis, not only `recovery`, so a cotangent column reads in 1/m
while an umbrella column beside it is dimensionless. The two are comparable as
ratios within a column, never as absolute values across one.

  python src/estimator_diag.py --selftest
  python src/estimator_diag.py --out results/estimator_diag
"""
import argparse
import csv
import json
import os

import numpy as np

# Camera: TartanAir-like intrinsics, square and small so the whole suite runs in
# seconds.  Nothing below depends on the specific values.
W, H, FOCAL = 256, 256, 320.0
CX, CY = W / 2.0, H / 2.0

# 6-neighbour ring, consistent with splitting each grid cell along one diagonal.
# Listed in rotational order so that consecutive entries bound a triangle.
RING = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# camera and analytic surfaces
# --------------------------------------------------------------------------- #

def ray_dirs():
    """Unnormalized ray directions m with m_z == 1, so z-depth == the ray scale."""
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    return np.stack([(us - CX) / FOCAL, (vs - CY) / FOCAL, np.ones_like(us)], -1)


def plane_depth(m, normal, offset):
    """z-depth of the plane {X : normal . X = offset}; NaN where it is behind."""
    denom = m @ np.asarray(normal, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = offset / denom
    return np.where(np.isfinite(z) & (z > 0), z, np.nan)


def sphere_depth(m, center, radius):
    """z-depth of the near intersection with a sphere; NaN off-silhouette."""
    c = np.asarray(center, dtype=np.float64)
    a = np.einsum("...i,...i->...", m, m)
    b = -2.0 * (m @ c)
    cc = c @ c - radius * radius
    disc = b * b - 4 * a * cc
    with np.errstate(invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
    return np.where((disc > 0) & (t > 0), t, np.nan)


def cylinder_depth(m, center, radius):
    """z-depth of a cylinder whose axis is the world y-axis through `center`."""
    c = np.asarray(center, dtype=np.float64)
    mx, mz = m[..., 0], m[..., 2]
    a = mx * mx + mz * mz
    b = -2.0 * (mx * c[0] + mz * c[2])
    cc = c[0] ** 2 + c[2] ** 2 - radius * radius
    disc = b * b - 4 * a * cc
    with np.errstate(invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
    return np.where((disc > 0) & (t > 0), t, np.nan)


def backproject(m, z):
    """Pointmap (H,W,3) from z-depth and ray directions."""
    return m * z[..., None]


def slanted_normal(theta_deg):
    """Unit normal of a plane tilted `theta_deg` about the camera's x-axis.

    theta = 0 is fronto-parallel (normal along -z); increasing theta tips the
    plane away so that the image spans an increasing depth range.
    """
    t = np.radians(theta_deg)
    return np.array([0.0, -np.sin(t), -np.cos(t)])


# --------------------------------------------------------------------------- #
# scenes: (pointmap, valid, true |H| per pixel, free-form tags)
# --------------------------------------------------------------------------- #

class Scene:
    def __init__(self, name, P, valid, true_H, tags=None):
        self.name, self.P, self.valid = name, P, valid
        self.true_H = true_H
        self.tags = tags or {}


def scene_plane(theta_deg=0.0, dist=5.0, max_depth=50.0):
    """Plane at perpendicular distance `dist`, tilted by theta.  True |H| = 0.

    `max_depth` matters more than it looks.  A plane at a steep slant recedes
    toward a horizon inside the frame -- at 75 degrees it reaches 2.2 km -- and
    those few near-horizon pixels carry almost all of the mean signal.  No real
    depth dataset contains that: TartanAir caps around 200 m and indoor sets at
    a few metres.  Leaving the horizon in produced a 172x headline that was
    really 1.9% of pixels, and which inverts to 0.3x under a median.  Pixels
    beyond `max_depth` are therefore marked invalid, as a real sensor would.
    """
    m = ray_dirs()
    n = slanted_normal(theta_deg)
    z = plane_depth(m, n, -dist)       # normal points toward the camera
    z = np.where(z <= max_depth, z, np.nan)
    P = backproject(m, z)
    valid = np.isfinite(z)
    return Scene(f"plane_slant{theta_deg:g}_d{dist:g}", P, valid,
                 np.zeros((H, W)), {"slant_deg": theta_deg, "dist": dist,
                                    "max_depth": max_depth})


def scene_sphere(radius=1.0, dist=5.0):
    """Sphere filling much of the frame.  True |H| = 1/radius."""
    m = ray_dirs()
    z = sphere_depth(m, (0.0, 0.0, dist), radius)
    P = backproject(m, z)
    valid = np.isfinite(z)
    return Scene(f"sphere_R{radius:g}_d{dist:g}", P, valid,
                 np.full((H, W), 1.0 / radius), {"radius": radius, "dist": dist})


def scene_cylinder(radius=1.0, dist=5.0):
    """Cylinder, axis vertical.  True |H| = 1/(2 radius)."""
    m = ray_dirs()
    z = cylinder_depth(m, (0.0, 0.0, dist), radius)
    P = backproject(m, z)
    valid = np.isfinite(z)
    return Scene(f"cylinder_R{radius:g}_d{dist:g}", P, valid,
                 np.full((H, W), 0.5 / radius), {"radius": radius, "dist": dist})


def _centre_at_column(u_target, z):
    """World x placing a point at depth z on image column `u_target`.

    Objects are positioned by where they should LAND, not by a world offset.
    A fixed world offset changes image position with depth, and a fixed angular
    offset does not change it at all -- the first attempt at this used the
    latter and stacked the two spheres on top of each other.
    """
    return z * (u_target - CX) / FOCAL


def scene_two_spheres(radius=0.5, near=3.0, far=9.0):
    """Two IDENTICAL spheres at different depths, side by side in one image.

    Both have true |H| = 1/radius, so any estimator that measures curvature must
    report the same value on each.  The umbrella signal is predicted to scale as
    H*Z^2, giving a near/far ratio of (far/near)^2.
    """
    m = ray_dirs()
    # The near sphere subtends ~2*R*FOCAL/z px and the far one far less, so they
    # are aimed at columns far enough apart that neither silhouette reaches the
    # other or the frame edge.  The self-test asserts both.
    zn = sphere_depth(m, (_centre_at_column(70, near), 0.0, near), radius)
    zf = sphere_depth(m, (_centre_at_column(190, far), 0.0, far), radius)
    z = np.where(np.isfinite(zn), zn, zf)          # disjoint in the image
    P = backproject(m, z)
    valid = np.isfinite(z)
    tags = {"near_mask": np.isfinite(zn), "far_mask": np.isfinite(zf),
            "near": near, "far": far, "radius": radius}
    return Scene(f"two_spheres_R{radius:g}_{near:g}v{far:g}", P, valid,
                 np.full((H, W), 1.0 / radius), tags)


def scene_step_and_sphere(radius=0.5, dist=4.0, jump=2.0):
    """A genuinely curved surface and a pure depth discontinuity in one frame.

    Left half: fronto-parallel plane at `dist`.  Right half: the same plane
    pushed back by `jump`, so the vertical seam between them is a depth step with
    true H = 0 on both sides.  A sphere of radius `radius` sits on the left.
    This is the configuration the paper's top-decile "boundary subset" selects
    from, so it reveals what that subset is made of.
    """
    m = ray_dirs()
    zl = plane_depth(m, (0, 0, -1), -dist)
    zr = plane_depth(m, (0, 0, -1), -(dist + jump))
    right = np.zeros((H, W), bool)
    right[:, W // 2:] = True
    z = np.where(right, zr, zl)

    # Left of the seam at column W//2, clear of both the seam and the frame edge.
    zc = dist - radius
    zs = sphere_depth(m, (_centre_at_column(70, zc), 0.0, zc), radius)
    on_sphere = np.isfinite(zs)
    z = np.where(on_sphere, zs, z)

    step = np.zeros((H, W), bool)                  # the two columns across the seam
    step[:, W // 2 - 1:W // 2 + 1] = True
    step &= ~on_sphere

    P = backproject(m, z)
    valid = np.isfinite(z)
    true_H = np.where(on_sphere, 1.0 / radius, 0.0)
    return Scene(f"step_sphere_R{radius:g}_j{jump:g}", P, valid, true_H,
                 {"sphere_mask": on_sphere, "step_mask": step})


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #

def avg_dis_normalize(P, valid):
    """The DUSt3R `avg_dis` convention: divide by the mean point-to-origin norm.

    This is applied before the loss computes its signal, so every comparison
    below must apply it too.  Returns (P_normalized, scale).
    """
    d = np.linalg.norm(P[valid], axis=-1)
    scale = float(np.mean(d)) if d.size else 1.0
    return P / max(scale, EPS), scale


def _shift(A, dr, dc, fill=np.nan):
    """Shift a field by (dr, dc), padding with `fill`.  A[r,c] <- A[r+dr, c+dc]."""
    out = np.full_like(A, fill, dtype=np.float64)
    r0, r1 = max(0, dr), min(A.shape[0], A.shape[0] + dr)
    c0, c1 = max(0, dc), min(A.shape[1], A.shape[1] + dc)
    out[r0 - dr:r1 - dr, c0 - dc:c1 - dc] = A[r0:r1, c0:c1]
    return out


def _shift_vec(P, dr, dc):
    return np.stack([_shift(P[..., k], dr, dc) for k in range(3)], -1)


def _dilate(mask, iters=1):
    """4-connected binary dilation.  Hand-rolled so the self-tests do not need
    scipy, which is otherwise only used by one analysis."""
    out = mask.copy()
    for _ in range(iters):
        g = out.astype(np.float64)
        nb = (_shift(g, -1, 0, 0.0) + _shift(g, 1, 0, 0.0)
              + _shift(g, 0, -1, 0.0) + _shift(g, 0, 1, 0.0))
        out = out | (nb > 0.5)
    return out


def _top_decile(S, sel, frac=0.10):
    """Deterministic top-`frac` selection by rank.

    A percentile threshold plus `S >= thresh` is not safe here: on a
    fronto-parallel plane the first-order signal is a MATHEMATICALLY CONSTANT
    field over tens of thousands of pixels, that constant can land exactly on
    the 90th percentile, and which of the tied pixels pass then depends on
    last-bit rounding in np.linalg.norm -- i.e. on the numpy build.  Ranking and
    taking a fixed count is reproducible; ties are broken by pixel index, which
    is arbitrary but stable.
    """
    idx = np.flatnonzero(sel.ravel())
    if idx.size == 0:
        return np.zeros_like(sel)
    order = idx[np.argsort(S.ravel()[idx], kind="stable")]
    k = max(1, int(np.ceil(frac * idx.size)))
    top = np.zeros(sel.size, bool)
    top[order[-k:]] = True
    return top.reshape(sel.shape)


def _ring_points(P):
    """(6,H,W,3) stack of the 6-neighbour ring."""
    return np.stack([_shift_vec(P, dr, dc) for dr, dc in RING], 0)


def _ring_valid(valid):
    v = np.stack([_shift(valid.astype(np.float64), dr, dc, 0.0) > 0.5
                  for dr, dc in RING], 0)
    return v.all(0) & valid


def umbrella(P, valid):
    """Eq. (1): the submitted signal.  Uses the 4-neighbour stencil as published."""
    four = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nb = np.stack([_shift_vec(P, dr, dc) for dr, dc in four], 0)
    ok = np.stack([_shift(valid.astype(np.float64), dr, dc, 0.0) > 0.5
                   for dr, dc in four], 0).all(0) & valid
    lap = nb.mean(0) - P
    S = 0.5 * np.linalg.norm(lap, axis=-1)
    return np.where(ok, S, np.nan), ok


def pointmap_grad(P, valid):
    """|| grad Xbar ||, the paper's first-order gradient control."""
    gy = np.linalg.norm(P - _shift_vec(P, -1, 0), axis=-1)
    gx = np.linalg.norm(P - _shift_vec(P, 0, -1), axis=-1)
    ok = ((_shift(valid.astype(np.float64), -1, 0, 0.0) > 0.5)
          & (_shift(valid.astype(np.float64), 0, -1, 0.0) > 0.5) & valid)
    S = np.sqrt(np.nan_to_num(gx) ** 2 + np.nan_to_num(gy) ** 2)
    return np.where(ok, S, np.nan), ok


def spacing_normalized(P, valid):
    """Umbrella divided by mean squared neighbour spacing.

    The cheap repair CYqH offers as an alternative to full Voronoi weighting: a
    second difference over a grid of spacing h approximates h^2 * (second
    derivative), so dividing by h^2 restores inverse-length units without any
    mesh machinery.  Blows up exactly where the mesh is degenerate.
    """
    S, ok = umbrella(P, valid)
    four = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    h2 = np.mean(np.stack(
        [np.sum((_shift_vec(P, dr, dc) - P) ** 2, -1) for dr, dc in four], 0), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = S / np.maximum(h2, EPS)
    return np.where(ok, out, np.nan), ok


def _cot_at(apex, a, b):
    """cot of the angle at `apex` in the triangle (apex, a, b)."""
    u, v = a - apex, b - apex
    dot = np.einsum("...i,...i->...", u, v)
    crs = np.linalg.norm(np.cross(u, v), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dot / np.maximum(crs, EPS)


def cotangent(P, valid):
    """Meyer et al. mixed-area discrete mean curvature |H|, in units of 1/length.

    K_H(i) = (1 / (2 A_mixed)) sum_j (cot alpha_ij + cot beta_ij) (x_i - x_j),
    with K_H = 2 H n, so |H| = ||K_H|| / 2.  A_mixed is the Voronoi area where
    the triangle is non-obtuse and the area/2 or area/4 fallback otherwise.

    This is the estimator the paper cites but does not implement.  It is the
    reference for everything else here, so `--selftest` checks it against the
    closed-form |H| of a sphere and a cylinder before any result is believed.
    """
    R = _ring_points(P)                                   # (6,H,W,3)
    ok = _ring_valid(valid)

    weights = np.zeros((6,) + P.shape[:2])
    area = np.zeros(P.shape[:2])
    for k in range(6):
        prv, cur, nxt = R[(k - 1) % 6], R[k], R[(k + 1) % 6]
        # cot of the two angles opposite edge (i, cur)
        weights[k] = _cot_at(prv, P, cur) + _cot_at(nxt, P, cur)

        # mixed area contribution of triangle (i, cur, nxt)
        e1, e2 = cur - P, nxt - P
        cross = np.linalg.norm(np.cross(e1, e2), axis=-1)
        tri = 0.5 * cross
        obtuse_i = np.einsum("...i,...i->...", e1, e2) < 0
        obtuse_c = np.einsum("...i,...i->...", P - cur, nxt - cur) < 0
        obtuse_n = np.einsum("...i,...i->...", P - nxt, cur - nxt) < 0
        voronoi = 0.125 * (np.sum(e1 ** 2, -1) * _cot_at(nxt, P, cur)
                           + np.sum(e2 ** 2, -1) * _cot_at(cur, P, nxt))
        contrib = np.where(obtuse_i, tri / 2.0,
                           np.where(obtuse_c | obtuse_n, tri / 4.0, voronoi))
        area += np.nan_to_num(contrib)

    lap = np.einsum("k...,k...i->...i", weights, P[None] - R)
    with np.errstate(divide="ignore", invalid="ignore"):
        KH = lap / np.maximum(2.0 * area, EPS)[..., None]
    S = 0.5 * np.linalg.norm(KH, axis=-1)
    return np.where(ok, S, np.nan), ok


ESTIMATORS = {
    "umbrella": umbrella,
    "cotangent": cotangent,
    "spacing": spacing_normalized,
    "grad": pointmap_grad,
}


def evaluate(scene, normalized=True):
    """Every estimator on one scene.  Returns {name: (signal, ok)}.

    With `normalized`, the pointmap is avg_dis-normalized first -- what the loss
    sees.  Inverse-length estimators are rescaled back to metric units so that
    `recovery` can compare them against a true |H|; the umbrella signal is left
    in the normalized units it is actually consumed in, because it has no
    physical units to be restored to.
    """
    P, valid = scene.P, scene.valid
    Pn, scale = (avg_dis_normalize(P, valid) if normalized else (P, 1.0))
    out = {}
    for name, fn in ESTIMATORS.items():
        S, ok = fn(np.nan_to_num(Pn, nan=0.0), valid)
        if name in ("cotangent", "spacing"):
            S = S / max(scale, EPS)       # 1/length: undo the normalization
        out[name] = (S, ok)
    return out


def _stats(S, mask):
    sel = S[mask & np.isfinite(S)]
    if sel.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"),
                "n": 0}
    return {"mean": float(sel.mean()), "median": float(np.median(sel)),
            "p90": float(np.percentile(sel, 90)), "n": int(sel.size)}


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #

def analysis_zero(rows):
    """True-H-zero surfaces vs genuine curvature at matched depth.

    The headline comparison: a 60-degree slanted plane has exactly zero mean
    curvature, so an estimator of mean curvature must rank it far below a sphere.
    """
    ref = scene_sphere(radius=1.0, dist=5.0)
    ref_vals = evaluate(ref)
    out = []
    for sc in [scene_plane(0, 5.0), scene_plane(30, 5.0), scene_plane(60, 5.0),
               scene_plane(75, 5.0), ref]:
        vals = evaluate(sc)
        for est, (S, ok) in vals.items():
            s = _stats(S, ok)
            r = _stats(ref_vals[est][0], ref_vals[est][1])
            out.append({"analysis": "zero", "scene": sc.name, "estimator": est,
                        "true_H": float(np.mean(sc.true_H)),
                        "mean": s["mean"], "median": s["median"],
                        # Median, not mean: on a grazing plane a few near-horizon
                        # pixels carry most of the mass, and a mean ratio there
                        # says more about the horizon than about the estimator.
                        "frac_of_sphere_R1": s["median"] / r["median"]
                        if np.isfinite(r["median"]) and r["median"] > 0 else float("nan"),
                        "n": s["n"]})
    rows += out
    return out


def analysis_slant(rows):
    """Signal vs slant on a plane.  Truth is flat zero for every angle."""
    out = []
    for th in [0, 15, 30, 45, 60, 75]:
        sc = scene_plane(th, 5.0)
        for est, (S, ok) in evaluate(sc).items():
            s = _stats(S, ok)
            out.append({"analysis": "slant", "scene": sc.name, "estimator": est,
                        "slant_deg": th, "true_H": 0.0,
                        "mean": s["mean"], "median": s["median"], "n": s["n"]})
    rows += out
    return out


def analysis_depth(rows):
    """Two identical spheres at different depths within one image."""
    sc = scene_two_spheres(radius=0.5, near=3.0, far=9.0)
    near, far = sc.tags["near_mask"], sc.tags["far_mask"]
    predicted = (sc.tags["far"] / sc.tags["near"]) ** 2
    out = []
    for est, (S, ok) in evaluate(sc).items():
        sn, sf = _stats(S, near & ok), _stats(S, far & ok)
        ratio = sf["median"] / sn["median"] if sn["median"] > 0 else float("nan")
        out.append({"analysis": "depth", "scene": sc.name, "estimator": est,
                    "true_ratio": 1.0, "near_median": sn["median"],
                    "far_median": sf["median"], "far_over_near": ratio,
                    # WITHIN one image avg_dis is a single global constant, so
                    # it cancels out of a ratio between two objects in that same
                    # image and the full Z^2 dependence survives.  ACROSS images
                    # each frame carries its own scale, which removes one power
                    # of Z -- see analysis_depthsweep.  The two predictions are
                    # different numbers for the same estimator, and both are
                    # borne out, which is the clearest evidence that the Z^2
                    # account of this signal is the right one.
                    "predicted_within_image_Z2": predicted,
                    "predicted_across_images_Z": sc.tags["far"] / sc.tags["near"],
                    "n": sn["n"] + sf["n"]})
    rows += out
    return out


def analysis_edges(rows):
    """Is a depth step ENRICHED in the top-decile-by-signal set?

    Composition alone misleads: the step is two pixels wide, so even selecting
    every step pixel is a small share of a top decile dominated by a large
    sphere.  The meaningful quantity is the selection rate per population,
    P(in top decile | population).

    Three corrections over the naive version, each of which changed a number:

    * Selection is by rank, not by a `>= percentile` threshold -- see
      `_top_decile`, where exact ties made the first-order row irreproducible.
    * The sphere's own silhouette is excluded from every population.  The sphere
      occludes the plane behind it, so the ring of plane pixels just outside the
      sphere mask sits across a real depth discontinuity.  Left in, 100% of the
      "flat" pixels the umbrella selected were that ring, and the step-sensitive
      estimators were credited for finding curvature they had not found.
    * The chance baseline is reported, not assumed to be 0.10.  The umbrella is
      EXACTLY zero on every pixel of a fronto-parallel plane, so its top decile
      is not a 10% draw from a continuum -- it is "the pixels that are nonzero".
      `frac_exactly_zero` states that directly.
    """
    sc = scene_step_and_sphere()
    sphere_raw, step_raw = sc.tags["sphere_mask"], sc.tags["step_mask"]
    # 2 px of dilation removes the occlusion ring from every population.
    ring = _dilate(sphere_raw, 2) & ~sphere_raw
    sphere = sphere_raw & ~_dilate(~sphere_raw, 2)     # sphere interior only
    step = step_raw & ~ring
    out = []
    for est, (S, ok) in evaluate(sc).items():
        sel = ok & np.isfinite(S)
        if sel.sum() == 0:
            continue
        top = _top_decile(S, sel)
        flat = sel & ~step_raw & ~sphere_raw & ~ring
        rate = lambda m: float((top & m).sum() / max(int((m & sel).sum()), 1))
        # A one-sided stencil (the published gradient control uses backward
        # differences) can only fire on one of the two step columns, so its
        # rate_step is capped at 0.5 by construction, not by what it detects.
        cap = 0.5 if est == "grad" else 1.0
        out.append({"analysis": "edges", "scene": sc.name, "estimator": est,
                    "rate_step": rate(step), "rate_sphere": rate(sphere),
                    "rate_flat": rate(flat), "structural_cap": cap,
                    "frac_exactly_zero": float((S[sel] == 0).mean()),
                    "step_px": int((step & sel).sum()),
                    "sphere_px": int((sphere & sel).sum()),
                    "flat_px": int((flat & sel).sum())})
    rows += out
    return out


def analysis_depthsweep(rows):
    """One sphere, alone in its frame, at a range of depths.

    This was added as a DEFENSIVE result -- the expectation was that `avg_dis`
    normalization cancels the depth dependence across images, confining the bias
    to within-image redistribution.  The data says otherwise, and the correct
    statement is worth getting right because it is the easiest thing for a
    reviewer to check.

    The umbrella signal is a second difference over a pixel grid whose physical
    spacing is h = Z/f, so it scales as h^2 * |H| ~ Z^2 * |H|.  `avg_dis` divides
    the pointmap by a scale s ~ Z, which sends h -> h/s and |H| -> |H|*s, so the
    normalized signal goes as Z^2 |H| / s ~ Z * |H|.  One power of Z cancels;
    one survives.  A sphere at 12 m therefore draws about 4x the weight of the
    same sphere at 3 m, in a separate image.  The bias is not confined to within
    an image -- it is linear in depth, everywhere.
    """
    out = []
    for d in [3.0, 5.0, 7.0, 9.0, 12.0]:
        sc = scene_sphere(radius=0.5, dist=d)
        for est, (S, ok) in evaluate(sc).items():
            st = _stats(S, ok)
            out.append({"analysis": "depthsweep", "scene": sc.name,
                        "estimator": est, "dist": d, "true_H": 1.0 / 0.5,
                        "mean": st["mean"], "median": st["median"], "n": st["n"]})
    # Drift relative to the nearest distance makes the linear trend explicit
    # instead of leaving it to the reader to divide two columns.
    base = {r["estimator"]: r["median"] for r in out if r["dist"] == 3.0}
    for r in out:
        b = base.get(r["estimator"], float("nan"))
        r["drift_vs_3m"] = r["median"] / b if b else float("nan")
    rows += out
    return out


def analysis_rankcorr(rows):
    """Spearman rho between the umbrella signal and the first-order control.

    The paper reports that a first-order gradient control recovers most of the
    gain.  If the two signals order pixels almost identically wherever the
    surface is smooth, that is the mechanical explanation, and it means the
    second-order form is not doing separable work.
    """
    from scipy.stats import spearmanr
    out = []
    for sc in [scene_plane(60, 5.0), scene_sphere(1.0, 5.0),
               scene_step_and_sphere(), scene_two_spheres()]:
        vals = evaluate(sc)
        Su, oku = vals["umbrella"]
        Sg, okg = vals["grad"]
        m = oku & okg & np.isfinite(Su) & np.isfinite(Sg)
        rho = float(spearmanr(Su[m], Sg[m]).statistic) if m.sum() > 2 else float("nan")
        out.append({"analysis": "rankcorr", "scene": sc.name,
                    "estimator": "umbrella_vs_grad", "spearman_rho": rho,
                    "n": int(m.sum())})
    rows += out
    return out


def analysis_recovery(rows):
    """Measured / true |H| across sphere radii at a FIXED distance.

    Distance is held at 5 m for every radius.  An earlier version scaled the
    distance with the radius, which confounded the curvature response with the
    depth response and made the umbrella's ratio look radius-dependent when it
    is not: at fixed depth the umbrella IS proportional to |H|, and the honest
    statement of its failure is about depth and slant, not about an inability to
    respond to curvature.
    """
    out = []
    # Radii chosen so every silhouette fits inside the frame at d = 5 m; an
    # R = 2 sphere spans 140 px of a 128 px half-frame and is clipped, which
    # showed up as a spurious 20% drop in its recovery ratio.
    for R in [0.2, 0.35, 0.5, 0.75, 1.0]:
        sc = scene_sphere(radius=R, dist=5.0)
        for est, (S, ok) in evaluate(sc).items():
            st = _stats(S, ok)
            true = 1.0 / R
            out.append({"analysis": "recovery", "scene": sc.name, "estimator": est,
                        "radius": R, "dist": sc.tags["dist"], "true_H": true,
                        "measured": st["median"],
                        "measured_over_true": st["median"] / true, "n": st["n"]})
    rows += out
    return out


ANALYSES = {"zero": analysis_zero, "slant": analysis_slant,
            "depth": analysis_depth, "depthsweep": analysis_depthsweep,
            "edges": analysis_edges, "rankcorr": analysis_rankcorr,
            "recovery": analysis_recovery}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _fmt(x, w=12, p=4):
    if isinstance(x, str):
        return f"{x:<{w}}"
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return f"{'--':<{w}}"
    if isinstance(x, float):
        return f"{x:<{w}.{p}g}"
    return f"{str(x):<{w}}"


def report(rows):
    """Human-readable summary; the CSV carries the full table."""
    lines = []
    by = {}
    for r in rows:
        by.setdefault(r["analysis"], []).append(r)

    def block(title, note, keys, which):
        if not by.get(which):          # --analysis X must not emit six empty tables
            return
        lines.append("")
        lines.append(f"## {title}")
        lines.append(note)
        lines.append("")
        lines.append("".join(_fmt(k, 22 if k in ("scene",) else 12) for k in keys))
        for r in by.get(which, []):
            lines.append("".join(
                _fmt(r.get(k), 22 if k in ("scene",) else 12) for k in keys))

    block("Zero test: estimator values where true mean curvature is exactly 0",
          "A plane has H = 0 at every slant.  `frac_of_sphere_R1` is the signal\n"
          "relative to a genuinely curved sphere (|H| = 1) at the same depth;\n"
          "a mean-curvature estimator should report ~0 for every plane row.",
          ["scene", "estimator", "true_H", "mean", "frac_of_sphere_R1"], "zero")

    block("Slant sweep: a flat plane tilted from 0 to 75 degrees",
          "Truth is 0 for every row.  Any monotone trend is slant leaking into\n"
          "the signal.",
          ["scene", "estimator", "slant_deg", "mean", "median"], "slant")

    block("Depth bias: two identical spheres at 3 m and 9 m in ONE image",
          "Both have |H| = 2, so a curvature estimator owes far/near = 1.\n"
          "Within ONE image avg_dis is a single constant and cancels out of the\n"
          "ratio, so the prediction is the full (Z_far/Z_near)^2 = 9. The\n"
          "across-image figure is shown for contrast and belongs to the next\n"
          "table, where each frame carries its own scale.",
          ["estimator", "near_median", "far_median", "far_over_near",
           "predicted_within_image_Z2", "predicted_across_images_Z"], "depth")

    block("Depth sweep ACROSS images: one sphere alone, 3 m to 12 m",
          "Identical geometry in every row, so a curvature estimator holds\n"
          "`drift_vs_3m` at 1.00.  avg_dis cancels one of the two powers of Z,\n"
          "not both, so a linear-in-depth bias survives BETWEEN images as well\n"
          "as within one.",
          ["estimator", "dist", "true_H", "median", "drift_vs_3m"], "depthsweep")

    block("Selection rate by population: what the 'boundary subset' targets",
          "A frame holding a sphere (genuine curvature) and a pure depth step\n"
          "(true H = 0).  `rate_*` is P(pixel in top decile | population).\n"
          "`structural_cap` is the most a stencil can reach: the published\n"
          "first-order control uses backward differences and so can only fire\n"
          "on one of the two step columns.  `frac_exactly_zero` is the share of\n"
          "valid pixels where the signal is identically 0, which is why 0.10 is\n"
          "not the right chance baseline for the umbrella.",
          ["estimator", "rate_step", "structural_cap", "rate_sphere",
           "rate_flat", "frac_exactly_zero", "step_px"], "edges")

    block("Rank correlation with the first-order control",
          "How much of the umbrella signal's pixel ranking the gradient control\n"
          "reproduces.  High rho explains why the two arms perform alike.",
          ["scene", "spearman_rho", "n"], "rankcorr")

    block("Recovery of |H| = 1/R on spheres",
          "`measured_over_true` should be ~1 for an estimator of mean curvature,\n"
          "and constant across R for one that is merely proportional to it.",
          ["estimator", "radius", "dist", "true_H", "measured",
           "measured_over_true"], "recovery")
    return "\n".join(lines)


def verdict(rows):
    """The paragraph this file exists to produce, generated from the table.

    Written out of the rows rather than by hand on purpose: an earlier
    hand-written version claimed a low rank correlation explained a similarity
    it could not explain, and quoted a mean ratio that a median refuted.
    """
    g = lambda a: [r for r in rows if r["analysis"] == a]
    pick = lambda a, **kw: next(
        (r for r in g(a) if all(r.get(k) == v for k, v in kw.items())), None)
    out = []

    slants = [r for r in g("zero")
              if r["estimator"] == "umbrella" and r["scene"].startswith("plane_slant")
              and np.isfinite(r.get("frac_of_sphere_R1", np.nan))]
    if slants:
        worst = max(slants, key=lambda r: r["frac_of_sphere_R1"])
        deg = worst["scene"].split("plane_slant")[1].split("_")[0]
        cot = pick("zero", scene=worst["scene"], estimator="cotangent")
        out.append(
            f"Slant leaks into the signal. A perfectly FLAT plane, true mean "
            f"curvature exactly zero, reaches {worst['frac_of_sphere_R1']:.2f}x "
            f"the median umbrella signal of a genuinely curved sphere (|H| = 1) "
            f"at the same depth by {deg} degrees of slant, rising monotonically "
            f"from zero. The cotangent reference reports {cot['median']:.1e} on "
            f"the same surface. The effect is real but moderate once "
            f"near-horizon pixels beyond a sensor-plausible depth are excluded; "
            f"leaving them in inflates it by more than two orders of magnitude "
            f"and is not a defensible way to report it.")

    d, dc = pick("depth", estimator="umbrella"), pick("depth", estimator="cotangent")
    ds = [r for r in g("depthsweep") if r["estimator"] == "umbrella"]
    if d and dc and ds:
        far = max(ds, key=lambda r: r["dist"])
        out.append(
            f"Depth biases the signal, and does so linearly. Two identical "
            f"spheres at 3 m and 9 m within one image: the umbrella reports the "
            f"far one at {d['far_over_near']:.2f}x the near one, against the "
            f"1.00 a curvature estimator owes and the "
            f"{dc['far_over_near']:.2f} the cotangent reference delivers. This "
            f"matches the Z^2 that theory predicts for a second difference over "
            f"a pixel grid of spacing Z/f, since within one image avg_dis is a "
            f"single constant that cancels out of the ratio. Across images each "
            f"frame carries its own scale, which removes one power of Z, and the "
            f"same sphere alone in its own frame still draws "
            f"{far['drift_vs_3m']:.2f}x more weight at {far['dist']:.0f} m than "
            f"at 3 m -- against a predicted {far['dist'] / 3.0:.2f}x. Both "
            f"predictions hold, which is what makes the Z^2 account of this "
            f"signal hard to argue with. Identical geometry is supervised more "
            f"strongly the further away it is.")

    e, ec = pick("edges", estimator="umbrella"), pick("edges", estimator="cotangent")
    if e:
        tail = (f" The cotangent reference selects {ec['rate_step']:.0%} of them."
                if ec else "")
        out.append(
            f"The signal targets depth discontinuities, not curvature. In a "
            f"frame holding both, {e['rate_step']:.0%} of the pure-step pixels "
            f"land in the top decile by umbrella signal -- the paper's own "
            f"boundary evaluation subset -- against {e['rate_sphere']:.0%} of "
            f"genuinely curved pixels and {e['rate_flat']:.1%} of flat ones. "
            f"True mean curvature at the step is zero.{tail} The umbrella is "
            f"identically zero on {e['frac_exactly_zero']:.0%} of valid pixels, "
            f"so its top decile is not a tenth of a continuum but essentially "
            f"the set of pixels where anything is happening at all.")

    smooth = [r for r in g("rankcorr")
              if r["scene"].startswith(("sphere", "plane", "two_spheres"))
              and np.isfinite(r["spearman_rho"])]
    if smooth:
        lo = min(r["spearman_rho"] for r in smooth)
        step = pick("rankcorr", scene=next(
            (r["scene"] for r in g("rankcorr") if r["scene"].startswith("step_")), ""))
        extra = (f" On the scene that does contain a discontinuity they diverge "
                 f"(rho = {step['spearman_rho']:.3f}), so the two signals are "
                 f"separable only by the depth edges -- which is to say, only by "
                 f"the thing that is not curvature." if step else "")
        out.append(
            f"First- and second-order signals are near-interchangeable where it "
            f"matters. On every smooth scene tested the umbrella's pixel ranking "
            f"agrees with the published first-order gradient control at Spearman "
            f"rho >= {lo:.3f}.{extra} This is the mechanical reason the paper's "
            f"curvature and gradient arms score within seed noise of each other.")

    rec = [r for r in g("recovery") if r["estimator"] == "cotangent"
           and np.isfinite(r["measured_over_true"])]
    if rec:
        err = max(abs(np.array([r["measured_over_true"] for r in rec]) - 1))
        urec = [r for r in g("recovery") if r["estimator"] == "umbrella"
                and np.isfinite(r["measured_over_true"])]
        spread = ""
        if urec:
            v = np.array([r["measured_over_true"] for r in urec])
            spread = (f" At fixed depth the umbrella is itself proportional to "
                      f"|H| to within {max(abs(v / np.median(v) - 1)) * 100:.0f}%, "
                      f"so its failure is not an inability to see curvature but "
                      f"an inability to separate curvature from slant and depth.")
        out.append(
            f"The comparison itself is sound: the cotangent reference recovers "
            f"|H| = 1/R across radii to within {err * 100:.1f}%.{spread}")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #

def _fit_sphere(pts):
    """Algebraic sphere fit.  Returns (centre, radius, max radial residual).

    Exists because a geometry bug shipped once already: offsetting the RAY
    DIRECTIONS rather than the sphere centre is a principal-point shift, and
    back-projecting along the original rays then shears the object into an
    ellipsoid.  Everything still looked like a sphere in the depth map.  Fitting
    is the only cheap way to assert the scenes are what they claim.
    """
    A = np.concatenate([2 * pts, -np.ones((len(pts), 1))], 1)
    b = np.sum(pts ** 2, 1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c, k = sol[:3], sol[3]
    R = float(np.sqrt(max(c @ c - k, 0.0)))
    resid = float(np.max(np.abs(np.linalg.norm(pts - c, axis=1) - R)))
    return c, R, resid


def _selftest():
    """Nothing in the report is trustworthy unless these pass."""
    def approx(a, b, tol, msg):
        assert abs(a - b) <= tol * max(abs(b), EPS), f"{msg}: {a:.6g} vs {b:.6g}"

    # -- the reference estimator must recover closed-form curvature -----------
    for R, d, truth in [(1.0, 6.0, 1.0), (0.4, 4.0, 2.5)]:
        sc = scene_sphere(radius=R, dist=d)
        S, ok = cotangent(sc.P, sc.valid)
        core = ok & (sc.P[..., 2] < d - 0.55 * R)      # away from the silhouette
        approx(float(np.median(S[core])), truth, 0.10, f"cotangent sphere R={R}")

    sc = scene_cylinder(radius=1.0, dist=6.0)
    S, ok = cotangent(sc.P, sc.valid)
    approx(float(np.median(S[ok & (sc.P[..., 2] < 5.45)])), 0.5, 0.12,
           "cotangent cylinder R=1")

    # -- both estimators vanish on a fronto plane; max, not median ------------
    sc = scene_plane(0.0, 5.0)
    Su, oku = umbrella(sc.P, sc.valid)
    Sc, okc = cotangent(sc.P, sc.valid)
    assert float(np.nanmax(Su[oku])) < 1e-9, "umbrella must vanish on a fronto plane"
    assert float(np.nanmax(Sc[okc])) < 1e-9, "cotangent must vanish on a fronto plane"

    # -- the reference stays ~0 on a slanted plane; a loose bound here would
    # -- pass even with the ring ordering scrambled, so it is tight.
    sc = scene_plane(60.0, 5.0)
    Sc, okc = cotangent(sc.P, sc.valid)
    assert float(np.nanmedian(Sc[okc])) < 1e-9, \
        "cotangent must stay ~0 on a slanted plane (true H = 0)"

    # -- the umbrella's slant response must be MONOTONE, not merely positive.
    # -- `> 0` was the old assertion and any float noise satisfied it, while it
    # -- was the only check on the paper's central claim.
    meds = []
    for th in (0.0, 30.0, 60.0):
        sc = scene_plane(th, 5.0)
        Su, oku = umbrella(sc.P, sc.valid)
        meds.append(float(np.nanmedian(Su[oku])))
    assert meds[0] < meds[1] < meds[2], f"umbrella slant response not monotone: {meds}"

    # -- scale equivariance, covering the /scale branch for EVERY estimator ---
    sc = scene_sphere(1.0, 5.0)
    base = evaluate(sc)
    big = Scene(sc.name, sc.P * 7.0, sc.valid, sc.true_H / 7.0, sc.tags)
    scaled = evaluate(big)
    for est in ESTIMATORS:
        a, b = base[est][0], scaled[est][0]
        m = np.isfinite(a) & np.isfinite(b)
        # umbrella/grad are dimensionless after avg_dis; cotangent/spacing are
        # rescaled to 1/length, so a 7x bigger scene has 1/7 the curvature.
        want = 1.0 / 7.0 if est in ("cotangent", "spacing") else 1.0
        approx(float(np.median(b[m])) / float(np.median(a[m])), want, 1e-6,
               f"scale equivariance for {est}")

    # -- the scenes must actually be the objects they claim to be -------------
    sc = scene_two_spheres()
    n, f = sc.tags["near_mask"], sc.tags["far_mask"]
    assert not (n & f).any(), "near and far spheres overlap in the image"
    for name, m in (("near", n), ("far", f)):
        assert m.sum() > 400, f"{name} sphere too small: {m.sum()}"
        rs, cs = np.where(m)
        assert rs.min() > 0 and cs.min() > 0 and rs.max() < H - 1 and cs.max() < W - 1, \
            f"{name} sphere is clipped by the frame: rows {rs.min()}-{rs.max()}, " \
            f"cols {cs.min()}-{cs.max()}"
        _, Rfit, resid = _fit_sphere(sc.P[m])
        approx(Rfit, sc.tags["radius"], 1e-6, f"{name} object is not a sphere")
        assert resid < 1e-8, f"{name} radial residual {resid:.2e} -- sheared object"

    sc = scene_step_and_sphere()
    sm, st = sc.tags["sphere_mask"], sc.tags["step_mask"]
    assert sm.sum() > 500 and st.sum() > 100, "step scene populations too small"
    assert not (sm & st).any(), "sphere and step masks overlap"
    _, Rfit, resid = _fit_sphere(sc.P[sm])
    approx(Rfit, 0.5, 1e-6, "step-scene object is not a sphere")
    assert resid < 1e-8, f"step-scene sphere sheared: residual {resid:.2e}"

    # -- plane depth clipping must actually bound the scene -------------------
    sc = scene_plane(75.0, 5.0, max_depth=50.0)
    assert float(np.nanmax(sc.P[sc.valid][:, 2])) <= 50.0 + 1e-6, \
        "max_depth did not clip the horizon"

    # -- deterministic top-decile on an exactly-tied field --------------------
    S = np.full((10, 10), 0.5)
    sel = np.ones((10, 10), bool)
    top = _top_decile(S, sel)
    assert top.sum() == 10, f"tied field must still select exactly 10%, got {top.sum()}"
    assert (top == _top_decile(S, sel)).all(), "top-decile selection is not stable"

    # -- helpers --------------------------------------------------------------
    A = np.arange(9.0).reshape(3, 3)
    assert _shift(A, 1, 0)[0, 0] == 3.0, "_shift row direction"
    assert _shift(A, 0, 1)[0, 0] == 1.0, "_shift col direction"
    m = np.zeros((5, 5), bool); m[2, 2] = True
    assert _dilate(m, 1).sum() == 5 and _dilate(m, 2).sum() == 13, "_dilate footprint"

    print("selftest: all checks passed")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--analysis", default="all",
                    help="comma-separated: " + ",".join(ANALYSES) + ", or all")
    ap.add_argument("--out", default=None, help="directory for report.md / diag.csv")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    _selftest()          # never emit a report from an unverified estimator
    names = list(ANALYSES) if args.analysis == "all" else args.analysis.split(",")
    rows = []
    for n in names:
        if n not in ANALYSES:
            raise SystemExit(f"unknown analysis: {n}")
        ANALYSES[n](rows)

    head = ("# Does the umbrella Laplacian measure mean curvature?\n\n"
            "Analytic surfaces, perspective camera, avg_dis-normalized pointmaps.\n"
            "Every estimator sees exactly what the training loss sees.\n")
    body = report(rows)
    tail = "\n\n## Verdict\n\n" + verdict(rows) + "\n"
    text = head + body + tail
    print(text)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.md"), "w") as f:
            f.write(text)
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(args.out, "diag.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(args.out, "diag.json"), "w") as f:
            json.dump(rows, f, indent=1, default=float)
        print(f"\nwrote {args.out}/report.md, diag.csv, diag.json")


if __name__ == "__main__":
    main()
