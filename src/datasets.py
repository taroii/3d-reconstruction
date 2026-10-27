"""
Datasets: one training source, four evaluation sets, one contract.

Every dataset here reduces to the same `View` record -- RGB, metric depth,
intrinsics, an optional camera-to-world transform, and a validity mask -- so the
evaluation code is written once and the loss never learns which dataset it is
looking at.

    TRAIN   TartanAir      synthetic, 5 fixed environments (see data_tartanair.sh);
                           poses VERIFIED geometrically, 0.098% cross-view error
    EVAL    iBims-1        real, 100 images, purpose-built for DEPTH BOUNDARY
                           quality and shipping ground-truth edge maps
            NYUv2          real, 1449 indoor images, the standard depth benchmark
            Bonn RGB-D     real, 26 sequences. DEPTH ONLY -- its poses could
                           not be validated; see the note on the class.
            ETH3D          STUB. Archives extracted, no loader implemented.

Three of the four evaluation sets are REAL data. That is deliberate: the ACCV
reviewers' most repeated complaint was that the evidence rested on a single
synthetic zero-shot benchmark, and Reviewer Xcc4 specifically objected that the
method "appears to be mainly applicable to training with synthetic datasets".
Training on synthetic TartanAir and evaluating on real iBims-1/NYUv2/Bonn
answers that directly, and iBims-1's boundary metrics are the ones a paper about
boundary-aware supervision should live or die by.

------------------------------------------------------------------------------
Formats, all verified empirically rather than taken from documentation
------------------------------------------------------------------------------
TartanAir   depth  float32 .npy, metres, 480x640
            K      fx = fy = 320, cx = 320, cy = 240 (fixed for the dataset)
            pose   'tx ty tz qx qy qz qw', optical centre in a NED world frame
                   where x is FORWARD, y is RIGHT, z is DOWN -- not the computer
                   vision convention. See `CV_TO_NED` and `_selftest`.
iBims-1     depth  uint16 PNG; metres = raw * 50 / 65535. The readme's wording
                   ("depth_map*65535/50") is ambiguous and the literal reading
                   gives depths up to 1310 m indoors; the scale here was fixed
                   by checking that a corridor comes out at ~3 m, not ~90 m.
            K      per-image 'fx,fy,cx,cy' text file
            masks  mask_invalid, mask_transp -- True means VALID
            edges  binary PNG, ground-truth depth discontinuities
Bonn        depth  uint16 PNG, metres = raw / 5000 (TUM convention; checked
                   against a sequence that comes out at 1.5-3.7 m)
            K      fx 542.822841, fy 542.576870, cx 315.593520, cy 237.756098,
                   from the IPB dataset page; depth is pre-registered to RGB so
                   the colour intrinsics apply to both
            pose   groundtruth.txt, TUM format, associated to frames by timestamp
NYUv2       .mat   HDF5 v7.3; images (N,3,W,H) uint8 and depths (N,W,H) float32
                   ALREADY IN METRES. MATLAB is column-major, so both need a
                   transpose -- an untransposed 640x480 depth map still "looks"
                   fine and is wrong.
            K      the standard NYUv2 colour intrinsics

  python src/datasets.py --selftest
  python src/datasets.py --summary
"""
import argparse
import glob
import os
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.environ.get("PROJ", os.path.abspath(os.path.join(_HERE, "..", "..")))
DATA = os.environ.get("DATA_ROOT", os.path.join(_PROJ, "data"))

# Maps CV camera axes (x right, y down, z forward) -> NED body axes (x forward,
# y right, z down):  M @ [1,0,0]_cv = [0,1,0] = NED y = right, and so on.
#
# The direction in the NAME matters because this matrix has ORDER 3, not 2:
# M @ M != I, and M.T == M^-1 is a DIFFERENT matrix. An earlier version called
# it CV_TO_NED, which is backwards; anyone trusting that name to move a normal
# or a translation the other way picks up a silent 120-degree rotation about
# (1,1,1)/sqrt(3).
#
# Used as R_c2w = R_quat @ CV_TO_NED, i.e. RIGHT-multiplied, because permuting
# the CAMERA axes is a change of basis on the source space of R_c2w. Verified
# by cross-view reprojection in `_selftest`, not by reading the docs: the
# correct form gives a median cross-view depth error of 0.098%, against 16.8%
# for no permutation and 7.5% for the transpose, both of which still project
# most points INTO the image and so fail silently.
CV_TO_NED = np.array([[0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])

# TartanAir encodes sky as the fp16 maximum, exactly. Masking the sentinel by
# value is unambiguous; masking by a distance threshold is a judgement call, so
# the two are kept separate.
TARTANAIR_SKY = 65504.0
# Rendered backdrops sit at ~14-16 km in `westerndesert` and `office` and are
# not useful supervision -- they are a skybox, they carry no parallax, and they
# would dominate any scale normalization. Genuine long-range geometry in these
# environments lives below 1000 m, so that is the cut. Measured over ALL
# trajectories (not a 2-trajectory sample, which understated westerndesert's
# far fraction by 3x): a 1000 m cut discards 17.31% of westerndesert and 0.41%
# of office as backdrop, while retaining the 0.66% of westerndesert that is
# real geometry between 200 m and 1000 m.
TARTANAIR_MAX_DEPTH = 1000.0

TARTANAIR_K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])
BONN_K = np.array([[542.822841, 0.0, 315.593520],
                   [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]])
NYUV2_K = np.array([[5.1885790117450188e+02, 0.0, 3.2558244941119034e+02],
                    [0.0, 5.1946961112127485e+02, 2.5373616633400465e+02],
                    [0.0, 0.0, 1.0]])

IBIMS_SCALE = 50.0 / 65535.0        # uint16 -> metres
BONN_SCALE = 1.0 / 5000.0           # uint16 -> metres (TUM)


# --------------------------------------------------------------------------- #
# the common record
# --------------------------------------------------------------------------- #

@dataclass
class View:
    """One image with its geometry. `depth` is metres; invalid pixels are NaN."""
    rgb: np.ndarray                      # (H, W, 3) uint8
    depth: np.ndarray                    # (H, W) float32, NaN where invalid
    K: np.ndarray                        # (3, 3)
    c2w: np.ndarray = None               # (4, 4) or None for single-image sets
    meta: dict = field(default_factory=dict)

    @property
    def valid(self):
        return np.isfinite(self.depth) & (self.depth > 0)

    def camera_points(self):
        """(H, W, 3) points in THIS camera's frame. No pose needed."""
        H, W = self.depth.shape
        vs, us = np.mgrid[0:H, 0:W].astype(np.float64)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], -1)
        return rays * self.depth[..., None]

    def world_points(self):
        """(H, W, 3) points in the world frame. Requires `c2w`.

        This is what VGGT's point loss supervises and what the geometric signal
        is read from. Note the signal itself is rigid-invariant, so it could be
        computed from `camera_points` instead -- poses are needed for the
        pointmap target and the camera loss, not for the weighting.
        """
        if self.c2w is None:
            raise ValueError("this view has no pose; world points are undefined")
        P = self.camera_points()
        return P @ self.c2w[:3, :3].T + self.c2w[:3, 3]


def _quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def _rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _png(path):
    from PIL import Image
    return np.array(Image.open(path))


# --------------------------------------------------------------------------- #
# TartanAir -- training
# --------------------------------------------------------------------------- #

class TartanAir:
    """Training source. Trajectories of sequential frames with poses.

    The five environments are a stated design choice targeting the two confounds
    the estimator diagnostic found (large within-image depth range, planes at
    many slants), not a convenience sample -- see decision_log.md D28/D42.
    """
    NAME = "tartanair"

    def __init__(self, root=None, envs=None, split="all", val_envs=None,
                 max_depth=TARTANAIR_MAX_DEPTH):
        self.root = root or os.path.join(DATA, "tartanair")
        self.max_depth = max_depth
        found = sorted(d for d in os.listdir(self.root)
                       if os.path.isdir(os.path.join(self.root, d))
                       and not d.startswith("."))
        if envs is not None:
            missing = [e for e in envs if e not in found]
            if missing:
                raise ValueError(
                    f"environments not found under {self.root}: {missing}. "
                    f"Available: {found}")
        self.envs = list(envs) if envs is not None else list(found)
        self.trajs = []
        for e in self.envs:
            for diff in ("Easy", "Hard"):
                for p in sorted(glob.glob(os.path.join(self.root, e, diff, "P*"))):
                    if os.path.isdir(os.path.join(p, "depth_left")):
                        self.trajs.append(p)
        # Validation holds out whole ENVIRONMENTS -- held-out trajectories would
        # leak, because frames within a trajectory overlap heavily.
        #
        # There is deliberately NO default val split. An earlier version used
        # `self.envs[::4]`, which on the five training environments held out
        # `carwelding` AND `westerndesert`: a 40% holdout that removed the one
        # environment chosen to exercise the Z^2 depth confound (see
        # data_tartanair.sh) from training entirely. All five were picked to be
        # trained on. The plan of record (decision_log D42) is to stage extra
        # environments from the recovered drive for validation, which keeps the
        # training mix intact and gives a val set with no overlap at all.
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be all|train|val, got {split!r}")
        self.val = list(val_envs) if val_envs else []
        if split in ("train", "val") and not self.val:
            raise ValueError(
                "split= requires explicit val_envs. There is no safe default: "
                "carving validation out of the five training environments "
                "removes one that was chosen for a reason. Stage held-out "
                "environments from ~/server_backup/tartanair instead.")
        unknown = [e for e in self.val if e not in found]
        if unknown:
            raise ValueError(f"val_envs not present on disk: {unknown}")
        if split == "train":
            self.trajs = [t for t in self.trajs if self._env(t) not in self.val]
            self.envs = [e for e in self.envs if e not in self.val]
        elif split == "val":
            self.trajs = [t for t in self.trajs if self._env(t) in self.val]
            self.envs = [e for e in self.envs if e in self.val]

    @staticmethod
    def _env(traj):
        return traj.split(os.sep)[-3]

    def frames(self, traj):
        return sorted(glob.glob(os.path.join(traj, "depth_left", "*_left_depth.npy")))

    def poses(self, traj):
        return np.loadtxt(os.path.join(traj, "pose_left.txt"))

    def view(self, traj, idx, poses=None):
        dpath = self.frames(traj)[idx]
        stem = os.path.basename(dpath).replace("_left_depth.npy", "")
        ipath = os.path.join(traj, "image_left", f"{stem}_left.png")
        depth = np.load(dpath).astype(np.float32)
        depth[~np.isfinite(depth)] = np.nan
        depth[depth == TARTANAIR_SKY] = np.nan          # sky, unambiguous
        depth[depth > self.max_depth] = np.nan          # rendered backdrop
        p = (poses if poses is not None else self.poses(traj))[idx]
        c2w = _rt_to_T(_quat_to_R(*p[3:7]) @ CV_TO_NED, p[:3])
        if not os.path.exists(ipath):
            # Silently substituting a black image would feed the model
            # zero-filled RGB against valid depth targets -- the worst kind of
            # corruption for a training set, and invisible in every log.
            raise FileNotFoundError(
                f"depth exists but RGB is missing: {ipath}. The extraction for "
                f"this trajectory is incomplete; re-run data_tartanair.sh.")
        rgb = _png(ipath)
        return View(rgb=rgb, depth=depth, K=TARTANAIR_K.copy(), c2w=c2w,
                    meta={"dataset": self.NAME, "traj": traj, "idx": idx})

    def clip(self, traj, start, n_views=4, stride=3):
        """n_views frames at a fixed temporal stride -- VGGT consumes a SET."""
        poses = self.poses(traj)
        n = len(self.frames(traj))
        last = start + (n_views - 1) * stride
        if start < 0 or last >= n:
            raise IndexError(
                f"clip({n_views} views, stride {stride}) from {start} needs "
                f"frame {last} but {os.path.basename(traj)} has {n}")
        return [self.view(traj, start + k * stride, poses) for k in range(n_views)]

    def sample_starts(self, n_views=4, stride=3, seed=0, limit=None,
                      per_traj=None):
        """(traj, start, n_views, stride) tuples spanning the whole corpus.

        Renamed from `sample_clips`, which was a lie twice over: it returned
        (traj, start) tuples rather than clips, and it drew exactly ONE start
        per trajectory. With 41 trajectories that was 41 clips -- 164 frames of
        the 30,906 available, or 0.5% of the data, which is not a training set.

        `per_traj` defaults to as many non-overlapping clips as each trajectory
        holds, so long trajectories contribute proportionally more. The
        (n_views, stride) used for the span check travel WITH the start, so a
        caller cannot pair a start computed for 4 views with a `clip()` call
        asking for 8 and index off the end.
        """
        rng = np.random.default_rng(seed)
        span = (n_views - 1) * stride
        out = []
        for t in self.trajs:
            n = len(self.frames(t))
            if n <= span:
                continue
            k = per_traj if per_traj is not None else max(1, (n - span) // (span + 1))
            starts = rng.choice(n - span, size=min(k, n - span), replace=False)
            out.extend((t, int(st), n_views, stride) for st in starts)
        rng.shuffle(out)
        return out[:limit] if limit is not None else out

# --------------------------------------------------------------------------- #
# evaluation sets
# --------------------------------------------------------------------------- #

class IBims1:
    """100 real indoor images built to measure DEPTH BOUNDARY quality.

    Ships ground-truth edge maps and planarity masks, which is why it is the
    most relevant evaluation set we have: the paper's claim is about supervision
    at boundaries, and this is the benchmark that measures boundaries directly
    rather than averaging them away over a whole image.
    """
    NAME = "ibims1"

    def __init__(self, root=None):
        self.root = root or os.path.join(DATA, "ibims1")
        core = os.path.join(self.root, "ibims1_core_raw")
        self.core = core if os.path.isdir(core) else self.root
        self.names = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self.core, "rgb", "*.png"))
            if not os.path.basename(p).startswith("Thumbs"))

    def __len__(self):
        return len(self.names)

    def names_with_edges(self):
        """The 85 of 100 images that ship a ground-truth edge map.

        Coverage is partial by design of the dataset, not an error. Boundary
        metrics must be reported over THIS subset and said to be over 85
        images; computing them over all 100 and silently scoring 15 against a
        missing map is how a boundary claim becomes untrue.
        """
        have = {os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(self.core, "edges", "*.png"))
                if not os.path.basename(p).startswith("Thumbs")}
        return [n for n in self.names if n in have]

    def view(self, i):
        n = self.names[i]
        raw = _png(os.path.join(self.core, "depth", f"{n}.png")).astype(np.float64)
        depth = (raw * IBIMS_SCALE).astype(np.float32)
        depth[raw == 0] = np.nan            # no return
        # raw == 65535 is uint16 SATURATION, which reads as exactly 50.00 m.
        # 299 such pixels exist across the 100 images, in one image, and
        # mask_invalid does NOT exclude them. Negligible in count, but a 50 m
        # reading in a room is a huge outlier that would sit at the top of
        # every residual distribution and inside the quantile filter's tail.
        depth[raw == 65535] = np.nan
        for m in ("mask_invalid", "mask_transp"):
            p = os.path.join(self.core, m, f"{n}.png")
            if os.path.exists(p):
                depth[_png(p) == 0] = np.nan        # these masks are True = VALID
        fx, fy, cx, cy = [float(x) for x in open(
            os.path.join(self.core, "calib", f"{n}.txt")).read().replace(",", " ").split()]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        edges = None
        ep = os.path.join(self.core, "edges", f"{n}.png")
        if os.path.exists(ep):
            edges = _png(ep) > 0
        return View(rgb=_png(os.path.join(self.core, "rgb", f"{n}.png")),
                    depth=depth, K=K, c2w=None,
                    meta={"dataset": self.NAME, "name": n, "edges": edges})


class NYUv2:
    """1449 real indoor images from the labelled .mat (HDF5 v7.3).

    MATLAB stores column-major, so `images` is (N,3,W,H) and `depths` is
    (N,W,H); both need a transpose. An untransposed 640x480 depth map is still
    a plausible-looking array and is wrong, so `_selftest` checks the aspect
    ratio against the RGB rather than trusting the shape.
    """
    NAME = "nyuv2"

    def __init__(self, root=None):
        self.root = root or os.path.join(DATA, "nyuv2")
        self.path = os.path.join(self.root, "nyu_depth_v2_labeled.mat")
        self._h = None

    def _file(self):
        if self._h is None:
            import h5py
            self._h = h5py.File(self.path, "r")
        return self._h

    def __len__(self):
        return self._file()["depths"].shape[0] if os.path.exists(self.path) else 0

    def view(self, i):
        h = self._file()
        depth = np.array(h["depths"][i]).T.astype(np.float32)     # (W,H) -> (H,W)
        rgb = np.array(h["images"][i]).transpose(2, 1, 0)         # (3,W,H) -> (H,W,3)
        depth[depth <= 0] = np.nan
        return View(rgb=rgb.astype(np.uint8), depth=depth, K=NYUV2_K.copy(),
                    c2w=None, meta={"dataset": self.NAME, "idx": i})


class Bonn:
    """Real dynamic RGB-D sequences, TUM format, with ground-truth poses.

    RGB and depth are captured on separate clocks, so frames are associated by
    nearest timestamp within a tolerance -- the standard TUM procedure. Pairing
    by sorted index instead would drift and silently mismatch colour and depth.
    """
    NAME = "bonn"
    # Bonn's poses are NOT verified, and this flag says so rather than letting
    # silence imply they are. The reprojection test that pinned TartanAir's
    # convention to 0.098% cannot settle Bonn, for two independent reasons:
    #
    #   * at a LARGE frame gap the metric is dominated by OCCLUSION -- it does a
    #     nearest-pixel depth lookup with no visibility handling -- and static
    #     sequences degrade from 1.4% at gap 2 to 14% at gap 30 exactly as the
    #     dynamic ones do, so the growth is the metric, not the scene;
    #   * at a SMALL frame gap the baseline is ~1 cm and the rotation ~0.5 deg,
    #     too small to separate candidates: identity, CV_TO_NED, its transpose
    #     and two axis flips all land between 0.6% and 2.1%.
    #
    # So Bonn is used for DEPTH evaluation only, which needs no poses at all
    # (depth is scale-shift aligned per frame). `c2w` is still exposed, but do
    # not report a pose metric from it without validating the convention some
    # other way.
    POSES_VERIFIED = False
    MAX_DT = 0.02                        # rgb<->depth association, seconds
    MAX_POSE_DT = 0.05                   # frame<->groundtruth pose, seconds

    _cache = {}

    def __init__(self, root=None, seqs=None):
        self.root = root or os.path.join(DATA, "bonn")
        base = os.path.join(self.root, "rgbd_bonn_dataset")
        self.base = base if os.path.isdir(base) else self.root
        found = sorted(d for d in os.listdir(self.base)
                       if os.path.isdir(os.path.join(self.base, d)))
        self.seqs = [s for s in (seqs or found) if s in found]

    @staticmethod
    def _index(path):
        out = []
        if not os.path.exists(path):
            return out
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            out.append((float(parts[0]), parts[1]))
        return out

    def pairs(self, seq):
        """(rgb_path, depth_path, timestamp), associated one-to-one by timestamp.

        Cached: this was re-reading and re-associating both index files on every
        single `view()` call, making a sequence sweep quadratic in file parsing.

        The index files are NOT trusted. Our copy of this dataset is a partial
        backup: every sequence lists more depth frames in depth.txt than exist
        on disk (2-9 for most, 906 of 10906 for rgbd_bonn_static), and the
        staged tree matches the source byte for byte, so the gap is in the
        archive rather than in the copy. Entries whose file is absent are
        dropped here, and `integrity()` reports how many -- the alternative is
        a FileNotFoundError thousands of frames into an evaluation run, or
        worse, a silently shortened eval set.
        """
        key = ("pairs", self.base, seq)
        if key in Bonn._cache:
            return Bonn._cache[key]
        d = os.path.join(self.base, seq)
        rgb = [(t, p) for t, p in self._index(os.path.join(d, "rgb.txt"))
               if os.path.exists(os.path.join(d, p))]
        dep = [(t, p) for t, p in self._index(os.path.join(d, "depth.txt"))
               if os.path.exists(os.path.join(d, p))]
        if not rgb or not dep:
            return []
        dts = np.array([t for t, _ in dep])
        # One-to-one. Plain nearest-neighbour lets one depth frame serve several
        # RGB frames, which duplicates frames in the eval set and biases it
        # toward whatever the duplicated moments contain. Candidates are
        # consumed in order of match quality, as TUM's own associate.py does.
        cand = []
        for ri, (t, rp) in enumerate(rgb):
            j = int(np.argmin(np.abs(dts - t)))
            dt = abs(dts[j] - t)
            if dt <= self.MAX_DT:
                cand.append((dt, ri, j))
        cand.sort()
        used_r, used_d, out = set(), set(), []
        for dt, ri, j in cand:
            if ri in used_r or j in used_d:
                continue
            used_r.add(ri); used_d.add(j)
            out.append((os.path.join(d, rgb[ri][1]),
                        os.path.join(d, dep[j][1]), rgb[ri][0]))
        out.sort(key=lambda x: x[2])
        Bonn._cache[key] = out
        return out

    def integrity(self):
        """Per-sequence (listed, present, usable_pairs). Report, never hide."""
        rows = []
        for sq in self.seqs:
            d = os.path.join(self.base, sq)
            listed = len(self._index(os.path.join(d, "depth.txt")))
            present = len([1 for _, p in self._index(os.path.join(d, "depth.txt"))
                           if os.path.exists(os.path.join(d, p))])
            rows.append((sq, listed, present, len(self.pairs(sq))))
        return rows

    def poses(self, seq):
        key = ("poses", self.base, seq)
        if key in Bonn._cache:
            return Bonn._cache[key]
        p = os.path.join(self.base, seq, "groundtruth.txt")
        rows = []
        if os.path.exists(p):
            with open(p) as fh:
                rows = [l.split() for l in fh if not l.startswith("#") and l.strip()]
        out = np.array([[float(x) for x in r] for r in rows]) if rows else None
        Bonn._cache[key] = out
        return out

    def view(self, seq, i):
        pr = self.pairs(seq)
        rp, dp, t = pr[i]
        raw = _png(dp).astype(np.float64)
        depth = (raw * BONN_SCALE).astype(np.float32)
        depth[raw == 0] = np.nan
        # Record WHY a pose is absent instead of handing back a silently
        # pose-less View: downstream code that filters on `c2w is not None`
        # would otherwise report a metric over an unknown subset of frames.
        c2w, pose_status = None, "ok"
        gt = self.poses(seq)
        if gt is None or not len(gt):
            pose_status = "missing_groundtruth_file"
        else:
            j = int(np.argmin(np.abs(gt[:, 0] - t)))
            dt = abs(gt[j, 0] - t)
            if dt <= self.MAX_POSE_DT:
                c2w = _rt_to_T(_quat_to_R(*gt[j, 4:8]), gt[j, 1:4])
            else:
                pose_status = f"no_pose_within_{self.MAX_POSE_DT}s(dt={dt:.3f})"
        return View(rgb=_png(rp), depth=depth, K=BONN_K.copy(), c2w=c2w,
                    meta={"dataset": self.NAME, "seq": seq, "t": t,
                          "pose": pose_status})


class ETH3D:
    """STUB. The archives are extracted but no loader is implemented yet.

    Deliberately does NOT provide `view()`. The module docstring's promise that
    every dataset reduces to a `View` does not hold for this one, and
    `_summary` reports it as `stub` rather than `ok` so nothing reads as usable
    that is not. ETH3D ships its own undistortion and multi-camera rig layout,
    which is more than a format adapter.
    """
    NAME = "eth3d"
    IS_STUB = True

    def __init__(self, root=None):
        self.root = root or os.path.join(DATA, "eth3d")
        self.scenes = sorted(
            d for d in (os.listdir(self.root) if os.path.isdir(self.root) else [])
            if os.path.isdir(os.path.join(self.root, d))
            and d not in ("archives",) and not d.startswith("."))

    def __len__(self):
        return len(self.scenes)


EVAL_SETS = {"ibims1": IBims1, "nyuv2": NYUv2, "bonn": Bonn, "eth3d": ETH3D}


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #

def _relative_rotation_deg(va, vb):
    """Angle between two views' orientations, in degrees."""
    R = va.c2w[:3, :3].T @ vb.c2w[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _reprojection_error(va, vb):
    """Median relative depth error lifting view a into view b.

    The definitive test of a pose convention: a wrong one still projects most
    points into the image, so only comparing the predicted depth against the
    other camera's MEASURED depth can tell them apart.

    PRECONDITION: the two views must differ in ORIENTATION, not only position.
    If they share an orientation, every convention of the form
    `S_true @ rot(baseline_axis, theta)` yields EXACTLY zero error for any
    theta -- a continuous family of wrong conventions that the test cannot see.
    `_selftest` asserts the precondition rather than assuming a trajectory
    happens to turn.
    """
    Pw = va.world_points()
    R, t = vb.c2w[:3, :3], vb.c2w[:3, 3]
    Pc = (Pw - t) @ R
    z = Pc[..., 2]
    fx, fy, cx, cy = vb.K[0, 0], vb.K[1, 1], vb.K[0, 2], vb.K[1, 2]
    u = fx * Pc[..., 0] / z + cx
    v = fy * Pc[..., 1] / z + cy
    H, W = vb.depth.shape
    m = (va.valid & (z > 0.1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1))
    if m.sum() < 200:
        return float("nan"), int(m.sum())
    zb = vb.depth[np.round(v[m]).astype(int), np.round(u[m]).astype(int)]
    good = np.isfinite(zb) & (zb > 0)
    if good.sum() < 200:
        return float("nan"), int(good.sum())
    err = np.abs(z[m][good] - zb[good]) / zb[good]
    return float(np.median(err)), int(good.sum())


def _selftest(verbose=True):
    def say(*a):
        if verbose:
            print(" ", *a)
    checked = 0

    # -- TartanAir: the pose convention, verified geometrically ---------------
    troot = os.path.join(DATA, "tartanair")
    if os.path.isdir(troot) and os.listdir(troot):
        ta = TartanAir()
        assert ta.trajs, "TartanAir present but no trajectories found"
        traj = ta.trajs[0]
        poses = ta.poses(traj)
        assert len(poses) == len(ta.frames(traj)), \
            "pose count does not match frame count"
        # the sky sentinel must never survive into a View
        raw = np.load(ta.frames(traj)[0])
        v0 = ta.view(traj, 0, poses)
        assert not (v0.valid & (raw == TARTANAIR_SKY)).any(), \
            "the fp16 sky sentinel survived masking"
        assert np.nanmax(v0.depth) <= ta.max_depth + 1e-3, \
            f"depth {np.nanmax(v0.depth)} exceeds max_depth {ta.max_depth}"
        a, b = v0, ta.view(traj, 6, poses)
        # Without a rotation between the frames this test cannot distinguish a
        # one-parameter family of wrong conventions, so the precondition is
        # checked rather than hoped for.
        rot = _relative_rotation_deg(a, b)
        assert rot > 1.0, (
            f"frames {0} and {6} differ by only {rot:.3f} deg of rotation; the "
            f"reprojection test is degenerate on a purely translating pair and "
            f"would pass a whole family of wrong conventions. Pick frames that "
            f"turn.")
        err, n = _reprojection_error(a, b)
        assert np.isfinite(err) and err < 0.02, (
            f"TartanAir pose convention looks wrong: median cross-view depth "
            f"error {err:.3%} over {n} points. A correct convention gives "
            f"~0.1%; the plausible WRONG ones give 7-17% while still placing "
            f"most points in view, so this is the only check that separates "
            f"them.")
        say(f"TartanAir: pose convention OK ({err:.3%} over {n} px), "
            f"{len(ta.envs)} envs, {len(ta.trajs)} trajectories")
        # a deliberately wrong convention must FAIL this test
        bad = View(a.rgb, a.depth, a.K, _rt_to_T(a.c2w[:3, :3] @ CV_TO_NED, a.c2w[:3, 3]))
        bad_err, _ = _reprojection_error(bad, b)
        assert not (np.isfinite(bad_err) and bad_err < 0.02), \
            "the reprojection test cannot distinguish a wrong pose convention"
        checked += 1
    else:
        say("TartanAir: not present, skipped")

    # -- iBims-1: the depth scale, pinned by physical plausibility ------------
    if os.path.isdir(os.path.join(DATA, "ibims1")):
        ib = IBims1()
        assert len(ib) >= 90, f"expected ~100 iBims images, found {len(ib)}"
        v = ib.view(0)
        d = v.depth[v.valid]
        assert 0.5 < np.median(d) < 15.0, (
            f"iBims depth median {np.median(d):.2f} m is not an indoor scene -- "
            f"the uint16 scale is wrong (the readme's literal reading gives "
            f"~90 m)")
        assert v.rgb.shape[:2] == v.depth.shape, "iBims rgb/depth shape mismatch"
        assert v.K[0, 0] > 100, "iBims intrinsics not loaded"
        with_e = ib.names_with_edges()
        assert len(with_e) >= 80, (
            f"only {len(with_e)}/{len(ib)} iBims images have edge maps; the "
            f"boundary metrics this dataset was chosen for need them")
        ve = ib.view(ib.names.index(with_e[0]))
        assert ve.meta["edges"] is not None and ve.meta["edges"].any(), \
            "edge map loaded but empty"
        assert ve.meta["edges"].shape == ve.depth.shape, "edge/depth shape mismatch"
        say(f"iBims-1: {len(ib)} images, median depth {np.median(d):.2f} m, "
            f"{len(with_e)} with ground-truth edge maps "
            f"({ve.meta['edges'].mean():.2%} of pixels on the first)")
        checked += 1
    else:
        say("iBims-1: not present, skipped")

    # -- NYUv2: the MATLAB transpose -----------------------------------------
    if os.path.exists(os.path.join(DATA, "nyuv2", "nyu_depth_v2_labeled.mat")):
        ny = NYUv2()
        v = ny.view(0)
        assert v.depth.shape == v.rgb.shape[:2], (
            f"NYUv2 depth {v.depth.shape} and rgb {v.rgb.shape[:2]} disagree -- "
            f"the MATLAB column-major transpose is wrong")
        assert v.depth.shape[0] < v.depth.shape[1], (
            f"NYUv2 frame is portrait {v.depth.shape}; it should be landscape, "
            f"which means the transpose was skipped")
        d = v.depth[v.valid]
        assert 0.5 < np.median(d) < 10.0, f"NYUv2 median depth {np.median(d):.2f} m"
        say(f"NYUv2: {len(ny)} images, {v.depth.shape}, "
            f"median depth {np.median(d):.2f} m")
        checked += 1
    else:
        say("NYUv2: not present, skipped")

    # -- Bonn: timestamp association and the TUM depth scale ------------------
    if os.path.isdir(os.path.join(DATA, "bonn")):
        bn = Bonn()
        assert bn.seqs, "Bonn present but no sequences"
        pr = bn.pairs(bn.seqs[0])
        assert len(pr) > 50, f"Bonn association produced only {len(pr)} pairs"
        # every associated path must exist -- the index over-reports
        for rp, dp, _ in pr[:50]:
            assert os.path.exists(rp) and os.path.exists(dp), \
                f"pairs() returned a path that does not exist: {dp}"
        # The association is the only thing pairs() does, so test IT: the
        # DEPTH timestamp must be within tolerance of the rgb timestamp. The
        # previous assertion compared the rgb filename against the rgb
        # timestamp -- two fields of the same line -- and would have passed a
        # completely broken association.
        for rp, dp, t in pr[:50]:
            dt = abs(float(os.path.splitext(os.path.basename(dp))[0]) - t)
            assert dt <= Bonn.MAX_DT, f"rgb/depth association off by {dt:.4f}s"
        assert len({dp for _, dp, _ in pr}) == len(pr), \
            "a depth frame was paired with more than one RGB frame"
        v = bn.view(bn.seqs[0], 0)
        d = v.depth[v.valid]
        assert 0.3 < np.median(d) < 12.0, (
            f"Bonn median depth {np.median(d):.2f} m -- the /5000 TUM scale "
            f"is probably wrong")
        assert v.c2w is not None, "Bonn view has no pose; association failed"
        assert v.meta["pose"] == "ok", f"unexpected pose status {v.meta['pose']}"
        # Guard the honesty of the flag: flipping POSES_VERIFIED to True must
        # require adding a test that actually verifies them.
        assert not Bonn.POSES_VERIFIED, (
            "Bonn.POSES_VERIFIED is True but nothing here verifies Bonn poses. "
            "Add a validation that discriminates between pose conventions "
            "before claiming they are checked.")
        say(f"Bonn: {len(bn.seqs)} sequences, {len(pr)} associated frames in "
            f"{bn.seqs[0]}, median depth {np.median(d):.2f} m "
            f"(DEPTH ONLY -- poses unverified)")
        checked += 1
    else:
        say("Bonn: not present, skipped")

    # Every non-stub dataset registered for evaluation must actually load.
    for nm, cls in EVAL_SETS.items():
        if getattr(cls, "IS_STUB", False):
            continue
        assert hasattr(cls, "view"), f"{nm} is registered for eval but has no view()"
    assert checked >= 1, "no dataset was available to test"
    print(f"selftest: {checked} dataset(s) verified")


def _audit(verbose=True):
    """Validate depth scales across EVERY image, not a sample.

    Added because a spot-check on one image is not evidence about a dataset: a
    global scale error makes every image implausible at once, so the honest
    test sweeps the whole set. For these sizes that is cheap, and it is the
    only way to say "this scale is right" rather than "this scale looked right
    on the image I happened to open".

    For iBims-1 both candidate scales are reported side by side, because the
    dataset's own readme is self-inconsistent -- no literal reading of
    "depth_map*65535/50" produces an indoor scene.
    """
    def line(name, meds, lo, hi, maxes=None):
        m = np.asarray(meds, float)
        if m.size == 0:
            print(f"{name:<34} no data"); return 0
        bad = int(((m < lo) | (m > hi)).sum())
        mx = f"  max/img p50 {np.median(maxes):7.2f}" if maxes is not None else ""
        print(f"{name:<34} n={m.size:<6} median {np.median(m):7.2f} m  "
              f"range [{m.min():6.2f}, {m.max():7.2f}]{mx}  "
              f"outside[{lo},{hi}]: {bad}")
        return bad

    bad_total = 0
    if os.path.isdir(os.path.join(DATA, "ibims1")):
        ib = IBims1()
        for label, sc in (("iBims-1 raw*50/65535 (used)", IBIMS_SCALE),
                          ("iBims-1 raw/50 (readme literal)", 1.0 / 50.0)):
            meds, mxs = [], []
            for n in ib.names:
                raw = _png(os.path.join(ib.core, "depth", f"{n}.png")).astype(np.float64)
                d = (raw * sc)[(raw > 0) & (raw < 65535)]
                if d.size:
                    meds.append(np.median(d)); mxs.append(d.max())
            b = line(label, meds, 0.5, 15.0, mxs)
            if "used" in label:
                bad_total += b

    if os.path.exists(os.path.join(DATA, "nyuv2", "nyu_depth_v2_labeled.mat")):
        ny = NYUv2(); D = ny._file()["depths"]
        meds, mxs = [], []
        for i in range(D.shape[0]):
            d = np.array(D[i]); d = d[d > 0]
            if d.size:
                meds.append(np.median(d)); mxs.append(d.max())
        bad_total += line("NYUv2 metres as stored", meds, 0.5, 10.0, mxs)

    if os.path.isdir(os.path.join(DATA, "bonn")):
        bn = Bonn(); meds, mxs = [], []
        for sq in bn.seqs:
            pr = bn.pairs(sq)
            if not pr:
                continue
            for k in np.linspace(0, len(pr) - 1, min(20, len(pr))).astype(int):
                raw = _png(pr[k][1]).astype(np.float64)
                d = (raw * BONN_SCALE)[raw > 0]
                if d.size:
                    meds.append(np.median(d)); mxs.append(d.max())
        bad_total += line(f"Bonn raw/5000 ({len(bn.seqs)} seqs)", meds, 0.3, 12.0, mxs)
        rows = bn.integrity()
        listed = sum(r[1] for r in rows); present = sum(r[2] for r in rows)
        usable = sum(r[3] for r in rows)
        print(f"  Bonn integrity: {present}/{listed} depth frames present "
              f"({listed - present} missing from this partial backup), "
              f"{usable} usable RGB-D pairs after association")

    if os.path.isdir(os.path.join(DATA, "tartanair")):
        ta = TartanAir()
        print()
        for e in ta.envs:
            # EVERY trajectory, not the first two. Sampling two trajectories
            # per environment understated westerndesert's far-depth fraction by
            # 3x (6.1% vs the true 17.97%), which is exactly the kind of error
            # a convenience sample produces: it is not noisy, it is BIASED,
            # because trajectories within an environment differ systematically
            # in where the camera looks.
            tr = [t for t in ta.trajs if TartanAir._env(t) == e]
            vals = [np.load(f).astype(np.float32).ravel()
                    for t in tr for f in ta.frames(t)[::40]]
            if not vals:
                continue
            v = np.concatenate(vals); v = v[np.isfinite(v)]
            sky = (v == TARTANAIR_SKY).mean()
            drop = ((v > ta.max_depth) & (v != TARTANAIR_SKY)).mean()
            kept = v[(v != TARTANAIR_SKY) & (v <= ta.max_depth)]
            print(f"  TartanAir {e:<15} kept p50 {np.median(kept):7.2f}  "
                  f"p99 {np.percentile(kept, 99):8.2f}  "
                  f"sky {sky:7.3%}  backdrop>{ta.max_depth:.0f}m {drop:7.3%}  "
                  f"valid {1 - sky - drop:6.2%}")
    print(f"\nimages outside a physically plausible range: {bad_total}")
    return bad_total


def _summary():
    rows = []
    for name, cls in [("tartanair", TartanAir)] + list(EVAL_SETS.items()):
        root = os.path.join(DATA, name)
        if not os.path.exists(root):
            rows.append((name, "MISSING", "", ""))
            continue
        try:
            d = cls()
            if name == "tartanair":
                n = sum(len(d.frames(t)) for t in d.trajs)
                rows.append((name, "ok", f"{len(d.envs)} envs / {len(d.trajs)} trajs",
                             f"{n} frames"))
            elif name == "bonn":
                rows.append((name, "ok", f"{len(d.seqs)} seqs",
                             f"{sum(len(d.pairs(s)) for s in d.seqs)} frames"))
            elif name == "eth3d":
                rows.append((name, "STUB (no loader)", f"{len(d)} scenes extracted", ""))
            else:
                rows.append((name, "ok", f"{len(d)} images", ""))
        except Exception as e:
            rows.append((name, f"ERROR {type(e).__name__}", str(e)[:50], ""))
    print(f"{'dataset':<12}{'status':<22}{'size':<28}{'frames':<16}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:<22}{r[2]:<28}{r[3]:<16}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="validate depth scales across every image")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.audit:
        _audit()
    elif a.summary:
        _summary()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
