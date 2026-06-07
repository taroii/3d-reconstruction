"""
TUM RGB-D loader for the loop-closure / pose-sheaf experiments.

Parses rgb.txt + groundtruth.txt, associates each RGB frame to the nearest GT
pose by timestamp, and subsamples keyframes spanning the trajectory (fr1/room is
a handheld loop, so a uniform subsample revisits earlier viewpoints -> real
loop-closure edges in the complete view graph).

GT convention (TUM): each line is `t tx ty tz qx qy qz qw`, the pose of the color
optical frame in the world frame (camera-to-world). Optical frame is OpenCV-style
(z forward, x right, y down) -- same as DUSt3R, so relative poses compare directly.
"""
import os
import numpy as np


def _read_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx**2 + qy**2)],
    ])


def load(root, n_key=15, max_dt=0.02, t0_frac=0.0, t1_frac=1.0):
    """Return (paths, poses, stamps): keyframe RGB paths, GT cam-to-world 4x4
    (N,4,4), and timestamps. Keyframes uniformly subsampled over [t0,t1] fraction
    of the (associated) sequence."""
    rgb = _read_list(os.path.join(root, "rgb.txt"))          # [ts, file]
    gt = _read_list(os.path.join(root, "groundtruth.txt"))   # [ts, tx..qw]
    gt_ts = np.array([float(r[0]) for r in gt])
    gt_pose = {}
    rgb_ts = np.array([float(r[0]) for r in rgb])
    # associate each rgb to nearest gt within max_dt
    assoc = []   # (rgb_index, gt_index)
    for i, t in enumerate(rgb_ts):
        k = int(np.argmin(np.abs(gt_ts - t)))
        if abs(gt_ts[k] - t) <= max_dt:
            assoc.append((i, k))
    lo = int(t0_frac * len(assoc))
    hi = int(t1_frac * len(assoc))
    assoc = assoc[lo:hi]
    sel = np.linspace(0, len(assoc) - 1, n_key).round().astype(int)
    sel = sorted(set(sel.tolist()))
    paths, poses, stamps = [], [], []
    for s in sel:
        ri, gi = assoc[s]
        paths.append(os.path.join(root, rgb[ri][1]))
        g = gt[gi]
        T = np.eye(4)
        T[:3, :3] = quat_to_R(*(float(x) for x in g[4:8]))
        T[:3, 3] = [float(g[1]), float(g[2]), float(g[3])]
        poses.append(T)
        stamps.append(rgb_ts[ri])
    return paths, np.array(poses), np.array(stamps)


def rel_pose(Twi, Twj):
    """Relative transform frame j -> frame i: T_ij = Twi^{-1} Twj (so a point in
    frame j maps to frame i). Matches the M_ij = g_i^{-1} g_j sheaf convention."""
    Ri, ti = Twi[:3, :3], Twi[:3, 3]
    Tij = np.eye(4)
    Tij[:3, :3] = Ri.T @ Twj[:3, :3]
    Tij[:3, 3] = Ri.T @ (Twj[:3, 3] - ti)
    return Tij


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "../data/tum/rgbd_dataset_freiburg1_room"
    paths, poses, stamps = load(root, n_key=15)
    print(f"{len(paths)} keyframes over {stamps[-1]-stamps[0]:.1f}s")
    # trajectory extent + revisit check: min pairwise camera-center distance
    C = poses[:, :3, 3]
    span = C.max(0) - C.min(0)
    print(f"trajectory bbox span (m): {span.round(2)}")
    D = np.linalg.norm(C[:, None] - C[None, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    # how close does a non-adjacent frame get to an earlier one (loop revisit)?
    far_idx = [(i, j) for i in range(len(C)) for j in range(i + 2, len(C))]
    if far_idx:
        dmin = min(D[i, j] for i, j in far_idx)
        amin = min(far_idx, key=lambda ij: D[ij[0], ij[1]])
        print(f"closest non-adjacent revisit: frames {amin} at {dmin:.2f} m")
    print("first/last cam centers:", C[0].round(2), C[-1].round(2))
