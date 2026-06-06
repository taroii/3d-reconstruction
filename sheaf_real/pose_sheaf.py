"""
Pose / connection sheaf: H^1 as loop-closure inconsistency.

The view graph G=(V,E) carries, per edge e=(i,j), a measured relative SE(3)
transform M_ij (from the backbone's pairwise prediction). We instantiate the
cellular sheaf with stalks F(v)=F(e)=se(3) and restriction maps given by the
connection (parallel transport):

    rho_{i<|e} = Ad_{M_ij},   rho_{j<|e} = I,
    (delta x)_e = x_j - Ad_{M_ij} x_i.

delta^T delta is the pose-graph information matrix. The harmonic representative
of H^1 (= ker delta^T) is the part of the edge residual that NO choice of frame
corrections can remove -- the irreducible loop-closure obstruction. Its
projection onto a cycle is the linearized log-holonomy of that cycle; its support
on an edge is the bad-edge leverage.

Pure numpy (downstream of the network cache). SE(3) only for now; Sim(3) (per-pair
scale) is the real-data extension and is flagged where it bites.
"""
import numpy as np
from dataclasses import dataclass


# --------------------------------------------------------------------------
# SE(3) Lie group / algebra. Tangent ordering xi = [rho (3, transl), phi (3, rot)].
# --------------------------------------------------------------------------
def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.eye(3) + skew(phi)
    k = phi / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def so3_log(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-7:
        # small angle: skew part is the generator
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    return th / (2.0 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _V(phi):
    """Left SE(3) jacobian: t = V(phi) rho."""
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-7:
        return np.eye(3) + 0.5 * K
    a = (1.0 - np.cos(th)) / th**2
    b = (th - np.sin(th)) / th**3
    return np.eye(3) + a * K + b * (K @ K)


def se3_exp(xi):
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = _V(phi) @ rho
    return T


def se3_log(T):
    R, t = T[:3, :3], T[:3, 3]
    phi = so3_log(R)
    rho = np.linalg.solve(_V(phi), t)
    return np.concatenate([rho, phi])


def se3_Ad(T):
    """6x6 adjoint, ordering [rho, phi]:  [[R, [t]x R],[0, R]]."""
    R, t = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# --------------------------------------------------------------------------
# Pose-sheaf assembly
# --------------------------------------------------------------------------
DIM = 6


def spanning_tree_init(n, edges, meas):
    """Propagate measured relatives along a spanning tree to get absolute frames
    g_i (cam-to-world, 4x4). Tree edges then carry ~zero residual by construction;
    only loop-closure edges carry the obstruction. meas[k] = M_ij with the
    convention M_ij = g_i^{-1} g_j (frame j expressed in frame i)."""
    adj = {v: [] for v in range(n)}
    for k, (i, j) in enumerate(edges):
        adj[i].append((j, k, +1))
        adj[j].append((i, k, -1))
    g = [None] * n
    g[0] = np.eye(4)
    stack = [0]
    seen = {0}
    while stack:
        u = stack.pop()
        for v, k, s in adj[u]:
            if v in seen:
                continue
            M = meas[k] if s > 0 else inv(meas[k])   # g_u^{-1} g_v
            g[v] = g[u] @ M
            seen.add(v)
            stack.append(v)
    return g


def build(n, edges, meas, g=None, weights=None):
    """Assemble the connection-sheaf coboundary delta (|E|*6, |V|*6) and the
    edge residual cochain r0 (|E|*6). Convention: right perturbation
    g_k <- g_k exp(x_k), residual r_e = log( M_ij^{-1} g_i^{-1} g_j ). Its
    first-order linearization is
        (delta x)_e = x_j - Ad_{M_ij^{-1}} x_i
    so the restriction map onto i is Ad_{M_ij^{-1}} (= Ad_{M_ij}^{-1}). r0 is the
    current edge mismatch at the linearization point g (->0 on tree edges).
    Returns (delta, r0, g).  weights: per-edge sqrt scaling folded into both."""
    if g is None:
        g = spanning_tree_init(n, edges, meas)
    E = len(edges)
    delta = np.zeros((E * DIM, n * DIM))
    r0 = np.zeros(E * DIM)
    for k, (i, j) in enumerate(edges):
        w = 1.0 if weights is None else float(np.sqrt(weights[k]))
        Mi = inv(meas[k])
        Ad = se3_Ad(Mi)
        delta[k * DIM:(k + 1) * DIM, j * DIM:(j + 1) * DIM] = w * np.eye(DIM)
        delta[k * DIM:(k + 1) * DIM, i * DIM:(i + 1) * DIM] = -w * Ad
        resid = se3_log(Mi @ inv(g[i]) @ g[j])
        r0[k * DIM:(k + 1) * DIM] = w * resid
    return delta, r0, g


def retract(g, x):
    """Right perturbation g_k <- g_k exp(x_k) (matches the build() linearization)."""
    return [g[k] @ se3_exp(x[k * DIM:(k + 1) * DIM]) for k in range(len(g))]


def solve_iter(n, edges, meas, g0=None, n_iters=20, weights=None):
    """Sheaf-valued Gauss-Newton: re-linearize until the residual is orthogonal to
    im(delta). At convergence the leftover residual IS the harmonic obstruction and
    is independent of the starting frames g0 -- this is the exact (all-orders)
    drift invariance. Returns (h, g, delta, r0)."""
    g = spanning_tree_init(n, edges, meas) if g0 is None else [gi.copy() for gi in g0]
    cap = 0.5   # trust region: keep per-node steps inside the log linearization
    for _ in range(n_iters):
        delta, r0, g = build(n, edges, meas, g=g, weights=weights)
        x = -np.linalg.pinv(delta.T @ delta) @ (delta.T @ r0)   # min ||delta x + r0||
        step = np.abs(x).max()
        if step < 1e-12:
            break
        if step > cap:
            x *= cap / step
        g = retract(g, x)
    delta, r0, g = build(n, edges, meas, g=g, weights=weights)
    return harmonic(delta, r0), g, delta, r0


def harmonic(delta, r0):
    """Project r0 onto ker(delta^T) = harmonic space of H^1.
        h = (I - delta (delta^T delta)^+ delta^T) r0
    The removed part delta(...)delta^T r0 is the exact (im delta) component that
    frame corrections CAN absorb; h is the irreducible loop obstruction."""
    L = delta.T @ delta
    # x* = argmin ||delta x + r0|| ; im-delta part = delta x* ... we want
    # projection of r0 onto im(delta): P r0 = delta (L^+) delta^T r0
    P_im = delta @ np.linalg.pinv(L) @ delta.T
    return r0 - P_im @ r0


def edge_energy(h):
    """Per-edge harmonic mass ||h_e||^2 (bad-edge leverage)."""
    H = h.reshape(-1, DIM)
    return np.sum(H**2, axis=1)


def cycle_basis_incidence(n, edges):
    """Signed cycle-edge incidence B (C, E) from the fundamental cycle basis of a
    spanning tree. Each non-tree edge -> one cycle; +1/-1 along the tree path."""
    adj = {v: [] for v in range(n)}
    for k, (i, j) in enumerate(edges):
        adj[i].append((j, k))
        adj[j].append((i, k))
    parent = {0: (None, None)}
    order = [0]
    stack = [0]
    tree_edges = set()
    while stack:
        u = stack.pop()
        for v, k in adj[u]:
            if v not in parent:
                parent[v] = (u, k)
                tree_edges.add(k)
                stack.append(u if False else v)
                order.append(v)
    rows = []
    for k, (i, j) in enumerate(edges):
        if k in tree_edges:
            continue
        row = np.zeros(len(edges))
        row[k] = 1.0
        # path j -> i through tree gives the cycle closed by edge (i,j)
        def path_up(x):
            p = []
            while parent[x][0] is not None:
                u, ek = parent[x]
                p.append((x, u, ek))
                x = u
            return p
        pj, pi = path_up(j), path_up(i)
        seen = {}
        for (a, b, ek) in pj:
            seen[ek] = +1.0
        for (a, b, ek) in pi:
            seen[ek] = seen.get(ek, 0.0) - 1.0
        for ek, s in seen.items():
            row[ek] += s
        rows.append(row)
    return np.array(rows) if rows else np.zeros((0, len(edges)))


def cycle_energy(h, B):
    """Per-cycle harmonic energy ||(B h)_c||^2: how badly cycle c fails to close."""
    H = h.reshape(-1, DIM)              # (E, 6)
    Bh = B @ H                          # (C, 6)
    return np.sum(Bh**2, axis=1)


# --------------------------------------------------------------------------
# SO(3)-only connection sheaf (rotation synchronization). Scale-free, so it is
# the clean operator for DUSt3R relative poses (per-pair translation scale is
# arbitrary). Stalks so(3)=R^3; for SO(3) the adjoint is R itself, so the
# restriction map onto i is R_ij^{-1}=R_ij^T (right perturbation convention).
# --------------------------------------------------------------------------
def so3_tree_init(n, edges, Rmeas):
    adj = {v: [] for v in range(n)}
    for k, (i, j) in enumerate(edges):
        adj[i].append((j, k, +1))
        adj[j].append((i, k, -1))
    R = [None] * n
    R[0] = np.eye(3)
    stack, seen = [0], {0}
    while stack:
        u = stack.pop()
        for v, k, s in adj[u]:
            if v in seen:
                continue
            Re = Rmeas[k] if s > 0 else Rmeas[k].T   # R_uv = R_u^T R_v
            R[v] = R[u] @ Re
            seen.add(v)
            stack.append(v)
    return R


def so3_build(n, edges, Rmeas, Rabs, weights=None):
    E = len(edges)
    delta = np.zeros((E * 3, n * 3))
    r0 = np.zeros(E * 3)
    for k, (i, j) in enumerate(edges):
        w = 1.0 if weights is None else float(np.sqrt(weights[k]))
        delta[k*3:k*3+3, j*3:j*3+3] = w * np.eye(3)
        delta[k*3:k*3+3, i*3:i*3+3] = -w * Rmeas[k].T        # Ad_{R^{-1}} = R^T
        r0[k*3:k*3+3] = w * so3_log(Rmeas[k].T @ Rabs[i].T @ Rabs[j])
    return delta, r0


def so3_solve(n, edges, Rmeas, n_iters=30, weights=None):
    """Rotation synchronization by iterated GN; returns (h, Rabs, edge_leverage)."""
    R = so3_tree_init(n, edges, Rmeas)
    cap = 0.5
    for _ in range(n_iters):
        delta, r0 = so3_build(n, edges, Rmeas, R, weights=weights)
        x = -np.linalg.pinv(delta.T @ delta) @ (delta.T @ r0)
        step = np.abs(x).max()
        if step < 1e-12:
            break
        if step > cap:
            x *= cap / step
        R = [R[k] @ so3_exp(x[k*3:k*3+3]) for k in range(n)]
    delta, r0 = so3_build(n, edges, Rmeas, R, weights=weights)
    h = harmonic(delta, r0)
    lev = np.sum(h.reshape(-1, 3)**2, axis=1)
    return h, R, lev


@dataclass
class PoseSheafResult:
    h: np.ndarray            # harmonic cochain (E*6,)
    r0: np.ndarray           # raw residual cochain (E*6,)
    edge_e: np.ndarray       # per-edge leverage (E,)
    cycle_e: np.ndarray      # per-cycle energy (C,)
    g: list                  # initialized absolute frames


def analyze(n, edges, meas, weights=None, n_iters=8, g0=None):
    h, g, delta, r0 = solve_iter(n, edges, meas, g0=g0, n_iters=n_iters,
                                 weights=weights)
    B = cycle_basis_incidence(n, edges)
    return PoseSheafResult(h=h, r0=r0, edge_e=edge_energy(h),
                           cycle_e=cycle_energy(h, B), g=g)
