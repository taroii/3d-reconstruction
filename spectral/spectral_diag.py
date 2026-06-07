"""
Spectral band diagnostic (PLAN step 1c): the graph Laplacian on the reconstructed
point set and a Chebyshev low/high-pass split, used to measure per-band residual
energy. Sparse matvecs only -- never an eigendecomposition.

  L = D - W,  W_ij = exp(-||x_i - x_j||^2 / eps)  on a kNN graph.
  Rescale  L_hat = (2/lmax) L - I  in [-1,1]; apply h(L) by Chebyshev recursion.
  h_lo(lambda) = cos(pi * lambda / (2 lmax))      (Hann^{1/2} low-pass)
  h_hi = sqrt(1 - h_lo^2)  => h_lo^2 + h_hi^2 = I  (energy-partitioning split).

Everything is torch + GPU; points are subsampled to ~O(10k) per the plan
("full-res dense L is infeasible; you only ever need sparse L x").
"""
import torch


def knn_graph(pts, k=12, chunk=2048):
    """kNN indices + squared distances via chunked cdist. pts (M,3) -> (M,k)."""
    M = pts.shape[0]
    idx = torch.empty(M, k, dtype=torch.long, device=pts.device)
    d2 = torch.empty(M, k, device=pts.device)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        dist = torch.cdist(pts[s:e], pts)              # (c, M)
        dd, ii = dist.topk(k + 1, largest=False)       # incl self
        idx[s:e], d2[s:e] = ii[:, 1:], dd[:, 1:] ** 2
    return idx, d2


def build_laplacian(pts, k=12, eps=None):
    """Sparse symmetric graph Laplacian L = D - W (torch sparse COO)."""
    M = pts.shape[0]
    idx, d2 = knn_graph(pts, k)
    if eps is None:
        eps = d2.median().clamp(min=1e-12)             # self-tuning heuristic
    w = torch.exp(-d2 / eps)                            # (M,k)
    rows = torch.arange(M, device=pts.device).repeat_interleave(k)
    cols = idx.reshape(-1)
    vals = w.reshape(-1)
    # symmetrize: W <- max(W, W^T) by stacking both orientations
    r = torch.cat([rows, cols]); c = torch.cat([cols, rows]); v = torch.cat([vals, vals])
    W = torch.sparse_coo_tensor(torch.stack([r, c]), v, (M, M)).coalesce()
    deg = torch.sparse.sum(W, dim=1).to_dense()
    Wi, Wv = W.indices(), W.values()
    L_idx = torch.cat([Wi, torch.stack([torch.arange(M, device=pts.device)] * 2)], dim=1)
    L_val = torch.cat([-Wv, deg])
    return torch.sparse_coo_tensor(L_idx, L_val, (M, M)).coalesce(), eps


def lambda_max(L, iters=30):
    """Largest eigenvalue of (symmetric PSD) L via power iteration."""
    M = L.shape[0]
    v = torch.randn(M, device=L.device)
    v = v / v.norm()
    lm = torch.tensor(1.0, device=L.device)
    for _ in range(iters):
        Lv = torch.sparse.mm(L, v.unsqueeze(1)).squeeze(1)
        lm = Lv.norm()
        v = Lv / lm.clamp(min=1e-12)
    return float(lm) * 1.01                              # margin so L_hat in [-1,1]


def _cheb_coeffs(hfun, order=24, n=200):
    """Chebyshev coefficients of h(lambda) on [0, 1] (lambda already /lmax)."""
    j = torch.arange(n, dtype=torch.float64)
    theta = (j + 0.5) * torch.pi / n
    x = torch.cos(theta)                                 # nodes in [-1,1]
    lam = (x + 1) / 2                                    # map to [0,1]
    h = hfun(lam)
    c = torch.empty(order + 1, dtype=torch.float64)
    for m in range(order + 1):
        c[m] = (2.0 / n) * (h * torch.cos(m * theta)).sum()
    c[0] *= 0.5
    return c


def cheb_filter(L, x, lmax, hfun, order=24):
    """Apply h(L) x by Chebyshev recursion on L_hat = (2/lmax)L - I."""
    c = _cheb_coeffs(hfun, order).to(x.device, x.dtype)
    def Lhat(v):
        return (2.0 / lmax) * torch.sparse.mm(L, v.unsqueeze(1)).squeeze(1) - v
    T0 = x
    T1 = Lhat(x)
    out = c[0] * T0 + c[1] * T1
    for m in range(2, order + 1):
        T2 = 2 * Lhat(T1) - T0
        out = out + c[m] * T2
        T0, T1 = T1, T2
    return out


# Hann^{1/2} low-pass on lambda in [0,1] (already divided by lmax)
def h_lo(lam):
    return torch.cos(torch.pi * lam.clamp(0, 1) / 2)

def h_hi(lam):
    return torch.sqrt((1 - h_lo(lam) ** 2).clamp(min=0))


def band_energies(L, r, lmax, order=24):
    """Low/high band energy of residual field r (M,) or (M,C): returns (lo, hi)."""
    if r.ndim == 1:
        r = r.unsqueeze(1)
    lo = sum(cheb_filter(L, r[:, c], lmax, h_lo, order).pow(2).sum() for c in range(r.shape[1]))
    hi = sum(cheb_filter(L, r[:, c], lmax, h_hi, order).pow(2).sum() for c in range(r.shape[1]))
    return float(lo), float(hi)
