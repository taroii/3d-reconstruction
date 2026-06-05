"""
Sheaf Reconstruction proof-of-concept: does the harmonic H^1 representative
localize on a moving object?

This module implements the *linearized reconstruction sheaf* described in the
paper (pose model, Remark 1) and the numerical extraction of its harmonic
1-cochain. The load-bearing claim under test:

    "the obstruction mass concentrates on the edges and regions where motion
     and occlusion make local predictions globally irreconcilable"

We test the cleanest, most falsifiable instance of it: a static scene observed
by N moving cameras, plus ONE rigid object that moves between frames. Ground
truth tells us exactly which scene points are dynamic, so we can ask whether
harmonic energy concentrates on them.

Construction (pose model, Sim(3) or SE(3) tangent stalks)
---------------------------------------------------------
* Vertices = views. Stalk F(v) = tangent of view v's world placement:
  se(3) (dim 6) or sim(3) (dim 7). A 0-cochain is a per-view pose correction.
* Edges = pairs the "network" related. The edge stalk over e=(i,j) is
  decomposed PER CO-VISIBLE POINT, one R^3 block per point. This per-point
  decomposition is what lets the harmonic 1-cochain localize *spatially*
  within the scene rather than merely per-view.
* For point k co-visible on edge (i,j), each endpoint proposes a world
  position for k using its current pose estimate; the base residual is their
  disagreement
        r0_{ij,k} = w^i_k - w^j_k   in R^3.
  A pose correction xi_i perturbs w^i_k linearly via the left-perturbation
  Jacobian B_{i,k}, so the coboundary is
        (delta xi)_{ij,k} = B_{i,k} xi_i - B_{j,k} xi_j.

The sheaf Laplacian is L = delta^T delta (the BA/Gauss-Newton information
matrix). The harmonic representative of H^1 is the part of r0 that NO pose
correction can cancel:
        h = r0 + delta xi*,   xi* = argmin || r0 + delta xi ||^2,
i.e. the projection of r0 onto ker(delta^T). For a static rigid scene a global
section exists and h -> 0; the irreducible h is where consistency is genuinely
obstructed. Per-point harmonic energy E_k = sum over incident edges ||h_{ij,k}||^2.

The decisive comparison is E_k (harmonic) vs the raw residual energy
R_k = sum ||r0_{ij,k}||^2. Bundle adjustment minimizes R_k and discards the
leftover as noise; the sheaf view keeps it as cohomology. When camera poses are
perturbed (drift / network pose error), R_k is large on static points too and
fails to isolate the object -- while h, having quotiented out everything a pose
correction explains, should stay clean. That gap is the whole thesis.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


# --------------------------------------------------------------------------
# SO(3) / SE(3) helpers
# --------------------------------------------------------------------------
def skew(v):
    """3-vector -> 3x3 skew-symmetric matrix, so that skew(a) @ b = a x b."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def so3_exp(phi):
    """Rodrigues: axis-angle 3-vector -> rotation matrix."""
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3)
    k = phi / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def random_rotation(rng, max_angle):
    """Random rotation with angle <= max_angle (radians)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = rng.uniform(0.0, max_angle)
    return so3_exp(axis * angle)


# --------------------------------------------------------------------------
# Scene generation
# --------------------------------------------------------------------------
class Scene:
    """A synthetic multi-view scene: static background + one rigid moving object."""

    def __init__(self, static_pts, dyn_local, dyn_obj_R, dyn_obj_t,
                 cam_R, cam_t):
        self.static_pts = static_pts          # (Ms, 3) world positions, fixed
        self.dyn_local = dyn_local            # (Md, 3) object-local coords
        self.dyn_obj_R = dyn_obj_R            # (N, 3, 3) object rotation per view-time
        self.dyn_obj_t = dyn_obj_t            # (N, 3) object translation per view-time
        self.cam_R = cam_R                    # (N, 3, 3) world-from-camera
        self.cam_t = cam_t                    # (N, 3)
        self.N = cam_R.shape[0]
        self.Ms = static_pts.shape[0]
        self.Md = dyn_local.shape[0]
        self.P = self.Ms + self.Md
        # label: 0 static, 1 dynamic, indexed over the P scene points
        self.is_dynamic = np.concatenate([np.zeros(self.Ms, bool),
                                           np.ones(self.Md, bool)])

    def world_point(self, k, view):
        """Ground-truth world position of scene point k at the time of `view`."""
        if k < self.Ms:
            return self.static_pts[k]
        d = k - self.Ms
        return self.dyn_obj_R[view] @ self.dyn_local[d] + self.dyn_obj_t[view]

    def point_in_camera(self, k, view):
        """Ground-truth position of point k in camera `view`'s frame."""
        Xw = self.world_point(k, view)
        return self.cam_R[view].T @ (Xw - self.cam_t[view])


def make_scene(rng, n_views=12, n_static=80, n_dynamic=20,
               object_speed=0.18, object_spin=0.10, scene_radius=2.0):
    """Build a toy orbiting-camera scene with one rigid object drifting over time.

    object_speed : per-frame object translation magnitude (world units)
    object_spin  : per-frame object rotation magnitude (radians)
    """
    # Static background: points scattered in a shell.
    dirs = rng.normal(size=(n_static, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    radii = rng.uniform(0.6 * scene_radius, scene_radius, size=(n_static, 1))
    static_pts = dirs * radii

    # Moving object: a small compact blob of points, object-local coords.
    dyn_local = rng.normal(scale=0.18, size=(n_dynamic, 3))
    obj_center0 = np.array([0.0, 0.0, 0.0])

    # Object pose per frame: integrate a constant-ish velocity + spin.
    dyn_obj_R = np.zeros((n_views, 3, 3))
    dyn_obj_t = np.zeros((n_views, 3))
    vel = rng.normal(size=3)
    vel = object_speed * vel / (np.linalg.norm(vel) + 1e-12)
    spin_axis = rng.normal(size=3)
    spin_axis /= np.linalg.norm(spin_axis) + 1e-12
    R_acc = np.eye(3)
    for i in range(n_views):
        dyn_obj_R[i] = R_acc
        dyn_obj_t[i] = obj_center0 + vel * i
        R_acc = so3_exp(spin_axis * object_spin) @ R_acc

    # Cameras orbit the scene looking inward.
    cam_R = np.zeros((n_views, 3, 3))
    cam_t = np.zeros((n_views, 3))
    cam_radius = 2.5 * scene_radius
    for i in range(n_views):
        ang = 2.0 * np.pi * i / n_views * 0.6   # partial orbit (not full loop)
        pos = cam_radius * np.array([np.cos(ang), 0.3 * np.sin(2 * ang), np.sin(ang)])
        # look-at origin
        fwd = -pos / (np.linalg.norm(pos) + 1e-12)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, fwd); right /= np.linalg.norm(right) + 1e-12
        true_up = np.cross(fwd, right)
        # world-from-camera: columns are right, up, fwd
        R = np.stack([right, true_up, fwd], axis=1)
        cam_R[i] = R
        cam_t[i] = pos

    return Scene(static_pts, dyn_local, dyn_obj_R, dyn_obj_t, cam_R, cam_t)


def make_view_graph(n_views, loop_closures=True):
    """Edges: temporal chain + a few long-range loop-closure edges (cycles)."""
    edges = [(i, i + 1) for i in range(n_views - 1)]
    if loop_closures:
        # long-range edges create cycles -> a place for loop-inconsistency
        # modes to live, so we can check they don't swamp the localization.
        extra = [(0, n_views - 1), (0, n_views // 2), (n_views // 4, 3 * n_views // 4)]
        for e in extra:
            if e[0] != e[1] and e not in edges:
                edges.append(e)
    return edges


# --------------------------------------------------------------------------
# Linearized reconstruction sheaf
# --------------------------------------------------------------------------
def jacobian_block(w, use_scale):
    """Left-perturbation Jacobian B_{i,k} of a world point w w.r.t. view tangent.

    exp(xi^) acts on the world point: d w = rho - skew(w) phi (+ lambda w).
    Returns (3, dim) with dim = 7 (sim3) or 6 (se3).
    Tangent ordering: [translation(3), rotation(3), (log-scale)].
    """
    B = np.zeros((3, 7 if use_scale else 6))
    B[:, 0:3] = np.eye(3)
    B[:, 3:6] = -skew(w)
    if use_scale:
        B[:, 6] = w
    return B


def build_sheaf(scene, edges, pose_R_est, pose_t_est, use_scale=False,
                noise_std=0.0, rng=None, fix_scale_gauge=True,
                gauge_weight=1e3):
    """Assemble the coboundary delta (sparse) and base residual r0.

    pose_R_est, pose_t_est : the *estimated* world-from-camera poses used as the
        linearization base point. Pass the ground-truth poses for the clean
        case, or perturbed poses to simulate network pose error / drift.

    use_scale : include the Sim(3) log-scale generator per view. WARNING: the
        coordinate-model residual w_i - w_j is NOT scale-invariant, so the
        *uniform* scale direction (every view shrinking together) can collapse
        the whole reconstruction to a point and trivially cancel every residual
        -- including the obstruction we want to measure. This is the global
        scale gauge. `fix_scale_gauge` anchors it.
    fix_scale_gauge : when use_scale, append a heavy penalty row pinning the
        sum of per-view log-scales to zero, removing the uniform-collapse mode.
        Has no effect for SE(3).

    Returns
    -------
    delta : scipy.sparse (rows, dim*N) coboundary (+ gauge row if applicable)
    r0    : (rows,) base residual (gauge target = 0)
    incid : list of (edge_index, point_k, row_start) bookkeeping per incidence
    dim   : tangent dimension per view
    """
    N = scene.N
    dim = 7 if use_scale else 6
    rows_i, cols_i, vals = [], [], []
    r0_blocks = []
    incid = []
    row = 0

    for e_idx, (i, j) in enumerate(edges):
        for k in range(scene.P):
            # each endpoint's world estimate of point k, from its camera-frame
            # observation pushed through the *estimated* pose.
            p_i = scene.point_in_camera(k, i)
            p_j = scene.point_in_camera(k, j)
            if noise_std > 0 and rng is not None:
                p_i = p_i + rng.normal(scale=noise_std, size=3)
                p_j = p_j + rng.normal(scale=noise_std, size=3)
            w_i = pose_R_est[i] @ p_i + pose_t_est[i]
            w_j = pose_R_est[j] @ p_j + pose_t_est[j]
            r = w_i - w_j                      # base residual block (3,)

            B_i = jacobian_block(w_i, use_scale)
            B_j = jacobian_block(w_j, use_scale)

            for a in range(3):
                for b in range(dim):
                    if B_i[a, b] != 0.0:
                        rows_i.append(row + a); cols_i.append(i * dim + b)
                        vals.append(B_i[a, b])
                    if B_j[a, b] != 0.0:
                        rows_i.append(row + a); cols_i.append(j * dim + b)
                        vals.append(-B_j[a, b])
            r0_blocks.append(r)
            incid.append((e_idx, k, row))
            row += 3

    n_rows = row
    if use_scale and fix_scale_gauge:
        # one heavy constraint: sum_i (log-scale_i) = 0, anchoring the global
        # uniform-scale gauge that would otherwise collapse the reconstruction.
        for i in range(N):
            rows_i.append(n_rows); cols_i.append(i * dim + 6)
            vals.append(gauge_weight)
        n_rows += 1
        r0_blocks.append(np.zeros(1))

    delta = sparse.csr_matrix((vals, (rows_i, cols_i)), shape=(n_rows, dim * N))
    r0 = np.concatenate(r0_blocks)
    return delta, r0, incid, dim


def harmonic_projection(delta, r0, atol=1e-10, btol=1e-10, iter_lim=20000):
    """Project r0 onto ker(delta^T): the harmonic representative of H^1.

    Solve xi* = argmin ||r0 + delta xi||^2, return h = r0 + delta xi*.
    Uses lsqr, which returns the minimum-norm solution for the rank-deficient
    sheaf Laplacian (gauge freedom = global rigid+scale motion).
    """
    out = lsqr(delta, -r0, atol=atol, btol=btol, iter_lim=iter_lim)
    xi = out[0]
    h = r0 + delta.dot(xi)
    return h, xi


def per_point_energy(values, incid, n_points):
    """Aggregate a 1-cochain's blocks into per-point energy and per-point counts.

    Returns (energy[P], count[P]) where energy_k = sum over incident edges
    ||value_{e,k}||^2 and count_k = number of incidences (edges) for point k.
    """
    energy = np.zeros(n_points)
    count = np.zeros(n_points)
    for (e_idx, k, row) in incid:
        block = values[row:row + 3]
        energy[k] += float(block @ block)
        count[k] += 1
    return energy, count


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def auroc(scores, labels):
    """AUROC of `scores` predicting boolean `labels` (1 = positive), via rank sum."""
    labels = np.asarray(labels, bool)
    n_pos = labels.sum()
    n_neg = (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = ranks[labels].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def separation(energy, labels):
    """Median(dynamic) / median(static) energy ratio -- a scale-free contrast."""
    labels = np.asarray(labels, bool)
    md = np.median(energy[labels])
    ms = np.median(energy[~labels]) + 1e-12
    return md / ms
