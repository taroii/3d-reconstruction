"""
The reconstruction sheaf on real pointmap predictions (pure NumPy/SciPy).

Implements plan.md S4. Everything here sits *below the cache boundary*: it
consumes per-view pointmaps + per-edge correspondences + initial poses (all
produced once by the frozen backbone and cached) and produces the harmonic
H^1 energy that localizes moving regions. No torch, no network.

Conventions
-----------
* A view v has a pointmap X_v of shape (H, W, 3): the backbone's 3D prediction
  for each pixel, expressed in view v's own camera frame.
* Pixels are addressed (row, col) = (y, x), matching numpy indexing X_v[y, x].
* A pose is a similarity g_v = (s_v, R_v, t_v) in Sim(3); world point of a local
  point p is  w = s_v (R_v p) + t_v.
* Tangent (left perturbation) xi_v = (dt in R^3, dw in R^3, ds in R^1) in R^7,
  acting on a world point as  d w = dt - [w]_x dw + (ds) w, i.e. B(w) xi with
  B(w) = [ I3 | -[w]_x | w ]  in R^{3x7}.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


# --------------------------------------------------------------------------
# SO(3)/Sim(3) helpers
# --------------------------------------------------------------------------
def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3)
    k = phi / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclass
class Poses:
    """Per-view Sim(3): s (N,), R (N,3,3), t (N,3)."""
    s: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @staticmethod
    def identity(n):
        return Poses(np.ones(n), np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)))

    def copy(self):
        return Poses(self.s.copy(), self.R.copy(), self.t.copy())

    def world(self, v, local):
        """Map a local point (or (M,3) array) of view v to the world frame."""
        return self.s[v] * (local @ self.R[v].T) + self.t[v]


def retract(poses, xi, dim=7):
    """Apply per-view tangent step (left perturbation) onto the poses.
    w <- e^ds exp(dw) w + dt  =>  R<-exp(dw)R, t<-e^ds exp(dw) t + dt, s<-e^ds s."""
    p = poses.copy()
    for v in range(len(p.s)):
        xv = xi[v * dim:(v + 1) * dim]
        dt, dw = xv[0:3], xv[3:6]
        ds = xv[6] if dim == 7 else 0.0
        Rd, el = so3_exp(dw), np.exp(ds)
        p.R[v] = Rd @ p.R[v]
        p.t[v] = el * (Rd @ p.t[v]) + dt
        p.s[v] = el * p.s[v]
    return p


# --------------------------------------------------------------------------
# Coboundary assembly
# --------------------------------------------------------------------------
def _jac(w, use_scale):
    B = np.zeros((3, 7 if use_scale else 6))
    B[:, 0:3] = np.eye(3)
    B[:, 3:6] = -skew(w)
    if use_scale:
        B[:, 6] = w
    return B


def build_coboundary(pointmaps, edges, matches, poses, use_scale=True,
                     fix_scale_gauge=True, gauge_weight=1e3):
    """Assemble delta (sparse 3M+g x dim*N), r0 (3M+g,), W weights, and an index.

    pointmaps : list of (H,W,3) arrays, one per view (camera-frame).
    edges     : list of (i,j).
    matches   : dict edge_index -> Matches(pix_i (M,2), pix_j (M,2), conf (M,)).
    poses     : Poses.
    index     : list of (edge_idx, k, row, vi, yi, xi, vj, yj, xj, nu) per corr,
                mapping coboundary rows back to pixels for splatting.
    """
    N = len(pointmaps)
    dim = 7 if use_scale else 6
    rows_i, cols_i, vals = [], [], []
    r0_blocks, nu_list, index = [], [], []
    row = 0
    for e_idx, (i, j) in enumerate(edges):
        m = matches[e_idx]
        for k in range(len(m.conf)):
            yi, xi_ = int(m.pix_i[k, 0]), int(m.pix_i[k, 1])
            yj, xj_ = int(m.pix_j[k, 0]), int(m.pix_j[k, 1])
            w_i = poses.world(i, pointmaps[i][yi, xi_])
            w_j = poses.world(j, pointmaps[j][yj, xj_])
            r0_blocks.append(w_i - w_j)
            Bi, Bj = _jac(w_i, use_scale), _jac(w_j, use_scale)
            for a in range(3):
                for b in range(dim):
                    if Bi[a, b] != 0.0:
                        rows_i.append(row + a); cols_i.append(i * dim + b)
                        vals.append(Bi[a, b])
                    if Bj[a, b] != 0.0:
                        rows_i.append(row + a); cols_i.append(j * dim + b)
                        vals.append(-Bj[a, b])
            nu_list.append(float(m.conf[k]))
            index.append((e_idx, k, row, i, yi, xi_, j, yj, xj_, float(m.conf[k])))
            row += 3

    n_rows = row
    if use_scale and fix_scale_gauge:
        # anchor the global uniform-scale gauge: sum_v (log-scale) = 0.
        for v in range(N):
            rows_i.append(n_rows); cols_i.append(v * dim + 6)
            vals.append(gauge_weight)
        r0_blocks.append(np.zeros(1)); n_rows += 1

    delta = sparse.csr_matrix((vals, (rows_i, cols_i)), shape=(n_rows, dim * N))
    r0 = np.concatenate(r0_blocks)
    nu = np.asarray(nu_list)
    # per-row weights: gauge row gets weight 1
    w_row = np.ones(n_rows)
    w_row[:3 * len(index)] = np.repeat(np.maximum(nu, 1e-6), 3)
    return delta, r0, w_row, index, dim


# --------------------------------------------------------------------------
# Harmonic solve (gauge + IRLS + outer Gauss-Newton)
# --------------------------------------------------------------------------
@dataclass
class SolveCfg:
    use_scale: bool = True
    n_iters: int = 2          # outer Gauss-Newton (relinearization)
    n_irls: int = 3           # IRLS reweighting steps per solve
    huber_k: float = 1.345
    gauge_weight: float = 1e3
    use_conf: bool = True     # fold backbone confidence into W


def _huber(norms, k):
    s = 1.4826 * np.median(norms) + 1e-9
    c = k * s
    return np.where(norms <= c, 1.0, c / (norms + 1e-12))


def _weighted_solve(delta, r0, w_row):
    sw = np.sqrt(w_row)
    D = sparse.diags(sw) @ delta
    xi = lsqr(D, -(sw * r0), atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    return xi


@dataclass
class SheafResult:
    h: np.ndarray                  # harmonic residual (3M,)
    eps_k: np.ndarray              # per-correspondence harmonic energy (M,)
    raw_k: np.ndarray             # per-correspondence raw residual energy (M,)
    poses: Poses                   # converged poses
    index: list                    # row -> pixel bookkeeping
    irls_w: np.ndarray            # final per-correspondence IRLS weight (M,)


def solve_harmonic(pointmaps, edges, matches, poses0, cfg=SolveCfg()):
    """Outer GN with robust IRLS inner solve. Returns SheafResult.
    Correspondences are fixed within the solve (refresh between outer iters is
    the caller's job, per Remark-1 co-evolution)."""
    poses = poses0.copy()
    dim = 7 if cfg.use_scale else 6
    for _ in range(cfg.n_iters):
        delta, r0, w_base, index, dim = build_coboundary(
            pointmaps, edges, matches, poses, use_scale=cfg.use_scale,
            gauge_weight=cfg.gauge_weight)
        if not cfg.use_conf:
            w_base = np.ones_like(w_base)
        w_row = w_base.copy()
        M = len(index)
        for _ in range(cfg.n_irls):
            xi = _weighted_solve(delta, r0, w_row)
            h = r0 + delta.dot(xi)
            norms = np.array([np.linalg.norm(h[3 * t:3 * t + 3]) for t in range(M)])
            iw = _huber(norms, cfg.huber_k)
            w_row = w_base.copy()
            w_row[:3 * M] *= np.repeat(iw, 3)
        poses = retract(poses, xi, dim)

    # final readout at converged poses (one robust solve)
    delta, r0, w_base, index, dim = build_coboundary(
        pointmaps, edges, matches, poses, use_scale=cfg.use_scale,
        gauge_weight=cfg.gauge_weight)
    if not cfg.use_conf:
        w_base = np.ones_like(w_base)
    w_row = w_base.copy(); M = len(index)
    iw = np.ones(M)
    for _ in range(cfg.n_irls):
        xi = _weighted_solve(delta, r0, w_row)
        h = r0 + delta.dot(xi)
        norms = np.array([np.linalg.norm(h[3 * t:3 * t + 3]) for t in range(M)])
        iw = _huber(norms, cfg.huber_k)
        w_row = w_base.copy(); w_row[:3 * M] *= np.repeat(iw, 3)
    eps_k = np.array([float(h[3 * t:3 * t + 3] @ h[3 * t:3 * t + 3])
                      for t in range(M)])
    raw_k = np.array([float(r0[3 * t:3 * t + 3] @ r0[3 * t:3 * t + 3])
                      for t in range(M)])
    return SheafResult(h=h[:3 * M], eps_k=eps_k, raw_k=raw_k, poses=poses,
                       index=index, irls_w=iw)


# --------------------------------------------------------------------------
# Splat to pixels + dense mask
# --------------------------------------------------------------------------
def splat_to_pixels(energy_k, index, shapes, reduce="median"):
    """Accumulate per-correspondence energy onto per-view pixel maps.

    A pixel can be touched by several edges; `reduce` ('median'|'mean'|'sum')
    aggregates them. Median exploits that genuine motion is inconsistent on every
    edge while a correspondence outlier is sporadic (plan A1). Returns
    dict view -> (H,W) energy map and dict view -> (H,W) hit-count map."""
    buckets = {v: {} for v in range(len(shapes))}
    for (e_idx, k, row, vi, yi, xi_, vj, yj, xj_, nu) in index:
        t = row // 3
        e = energy_k[t]
        buckets[vi].setdefault((yi, xi_), []).append(e)
        buckets[vj].setdefault((yj, xj_), []).append(e)
    out, cnt = {}, {}
    red = (np.median if reduce == "median" else
           np.mean if reduce == "mean" else np.sum)
    for v, (H, W) in enumerate(shapes):
        em = np.zeros((H, W)); cm = np.zeros((H, W))
        for (y, x), vals in buckets[v].items():
            em[y, x] = red(vals); cm[y, x] = len(vals)
        out[v] = em; cnt[v] = cm
    return out, cnt


def render_mask(energy_map, valid, method="percentile", q=80.0):
    """Threshold a per-view energy map into a binary dynamic mask over `valid`
    pixels. 'percentile' uses a fixed q across clips; 'otsu' is data-driven."""
    vals = energy_map[valid]
    if vals.size == 0:
        return np.zeros_like(valid)
    score = np.sqrt(energy_map + 1e-15)
    sv = np.sqrt(vals + 1e-15)
    if method == "otsu":
        thr = _otsu(sv)
    else:
        thr = np.percentile(sv, q)
    return (score >= thr) & valid


def _otsu(x, bins=128):
    hist, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist); wb = w / max(w[-1], 1)
    mu = np.cumsum(hist * centers)
    mt = mu[-1] if mu[-1] > 0 else 1.0
    denom = wb * (1 - wb) + 1e-12
    sigma_b = (mt * wb - mu / max(w[-1], 1)) ** 2 / denom
    return centers[np.nanargmax(sigma_b)]
