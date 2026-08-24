r"""
Per-view data access for the premise check, plus the Sec. 4 measurability gate.

One contract for every dataset:

    scenes(ds)              -> [scene_id, ...]          (the unit of analysis)
    views(ds, scene, n)     -> [View, ...]              (deterministic subsample)
    load(view)              -> ViewData(rgb, depth, valid, depth_raw, K)

`depth` is metric z-depth, float32, with invalid pixels set to NaN. `valid` is the
dataset's own validity mask AND finite AND in (0, max_depth]. `depth_raw` is the
same map *before* the validity mask is applied -- it is the `D_bar_raw` the GT
control regresses (Sec. 5). Every dataset here is renderer-produced, so raw and
clean differ only by the explicit mask and FP_GT ~= 0 is the expected control
result, not a bug.

Verified format facts (carried over from the retired curvature loaders, which had
been smoke-tested against the real files on the server):

  tartanair    depth_left/<base>_left_depth.npy, float32 metres, fixed
               K = [[320,0,320],[0,320,240]]. Sky is a very large finite value ->
               cut by max_depth. Scene = <env>/<Easy|Hard>/P0xx.
  pointodyssey depths/depth_<idx:05d>.png 16-bit, metres = png/65535*1000;
               K per frame from anno.npz['intrinsics'][idx] (the file is named
               anno.npz, NOT annot.npz). Scene = sequence dir.
  spring       disp1_left/disp1_left_<idx:04d>.dsp5, HDF5 key 'disparity', stored
               at 2x -> subsample [::2,::2] with NO value rescale; Z = fx*B/d with
               B = 0.065 m; d <= 0 is sky. K from cam_data/intrinsics.txt rows
               'fx fy cx cy'. Scene = sequence dir.
  sintel       depth/<scene>/frame_XXXX.dpt (MPI binary), K from
               camdata_left/*.cam, invalid/*.png != 0 marks invalid.

Tiering (Sec. 4): all four are synthetic with exact rendered boundaries, so all
are Tier A. No Tier B real-captured set with dense GT is present on the server --
that absence is itself a reportable finding, not a gap to paper over.

    python src/data.py --list
    python src/data.py --gate --out results/premise_check
"""
from __future__ import annotations

import os
import re
import csv
import glob
import json
from dataclasses import dataclass, field

import numpy as np

import fpmetrics as FM

ROOT = os.environ.get("PREMISE_DATA_ROOT", "data")

# tier: A = renderer-exact boundaries (primary), B = real capture (secondary),
#       C = negative control (expected to misbehave; validates the pipeline).
# contam: Tier 1 models known/suspected to have trained on it. Per Sec. 4 of
#       notes/server_requirements.md, "unknown" counts as "yes" for safety, and
#       the headline result must come from a dataset that is clean for ALL of
#       them. Sources are each model's training-data table; entries marked "?"
#       still need confirming against the paper before they are relied on.
SPECS = {
    "tartanair":    dict(tier="A", max_depth=200.0, sub="tartanair",
                         contam=["vggt?"], note="synthetic, exact depth"),
    "pointodyssey": dict(tier="A", max_depth=200.0, sub="pointodyssey",
                         contam=[], note="synthetic dynamic"),
    "spring":       dict(tier="A", max_depth=100.0, sub="spring",
                         contam=[], note="synthetic stereo"),
    "sintel":       dict(tier="A", max_depth=100.0, sub="training",
                         contam=[], note="synthetic, eval-only by convention"),
    "hypersim":     dict(tier="A", max_depth=100.0, sub="premise/hypersim",
                         contam=["vggt", "pi3?"],
                         note="RAY DISTANCE not planar z -- converted on load"),
    "middlebury":   dict(tier="A", max_depth=20.0, sub="premise/middlebury2014",
                         contam=[], note="structured light, very sharp boundaries"),
    "infinigen":    dict(tier="A", max_depth=200.0, sub="premise/infinigen",
                         contam=[], note="procedural, post-dates most training mixes"),
    "ibims":        dict(tier="B", max_depth=49.0, sub="premise/ibims1/ibims1_core_raw",
                         contam=[],
                         note="purpose-built for depth-boundary eval; laser GT, 100 images"),
    "eth3d":        dict(tier="B", max_depth=100.0, sub="premise/eth3d",
                         contam=["vggt?"], note="laser GT, real"),
    "nyuv2":        dict(tier="C", max_depth=10.0, sub="premise/nyuv2",
                         contam=["vggt?", "pi3?"],
                         note="control: GT itself is known to contain flying points"),
    "bonn":         dict(tier="C", max_depth=10.0, sub="premise/bonn/rgbd_bonn_dataset",
                         contam=[], note="control: GT masked at boundaries, should FAIL the gate"),
}
TIER1_MODELS = ("vggt", "pi3")

TARTANAIR_K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
SPRING_BASELINE = 0.065
_TAG_FLOAT = 202021.25
# Hypersim renders a fixed 1024x768 pinhole; this focal is the value the dataset
# authors give for converting their distance maps (ml-hypersim issue #9).
HYPERSIM_WH, HYPERSIM_FOCAL = (1024, 768), 886.81
BONN_DEPTH_SCALE = 5000.0        # TUM RGB-D convention: metres = png / 5000
# iBims-1: metres = png / 65535 * 50. NOTE the shipped readme states the inverse
# ("depth_map*65535/50"), which is dimensionally impossible -- it yields millions
# of metres. Verified empirically against three scenes: this direction gives the
# expected indoor ranges (1.0-7.3 m), the readme's does not. 65535 is the
# saturation ceiling and marks no-return, not a 50 m surface.
IBIMS_DEPTH_SCALE, IBIMS_SATURATED = 50.0 / 65535.0, 65535
# ETH3D DSLR native frame (Nikon D3X). GT depth is always this size.
ETH3D_NATIVE = (4032, 6048)


@dataclass
class View:
    dataset: str
    scene: str
    idx: int
    rgb: str
    depth: str
    cam: str = ""

    @property
    def key(self):
        return f"{self.dataset}/{self.scene.replace('/', '_')}/{self.idx:06d}"


@dataclass
class ViewData:
    rgb: np.ndarray | None
    depth: np.ndarray          # metric z-depth, invalid -> NaN
    valid: np.ndarray          # dataset mask AND finite AND in (0, max_depth]
    depth_raw: np.ndarray      # pre-mask depth; the GT control's D_bar_raw
    K: np.ndarray = field(default_factory=lambda: np.eye(3))
    mask: np.ndarray | None = None   # dataset-DECLARED validity alone, or None
    range_valid: np.ndarray | None = None  # finite & >0 & <= max_depth, no mask


def root_of(ds):
    return os.path.join(ROOT, SPECS[ds]["sub"])


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def _read_dpt(path):
    with open(path, "rb") as f:
        assert abs(np.fromfile(f, np.float32, 1)[0] - _TAG_FLOAT) < 1e-2, path
        w, h = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, w * h).reshape(h, w)


def _read_cam(path):
    with open(path, "rb") as f:
        assert abs(np.fromfile(f, np.float32, 1)[0] - _TAG_FLOAT) < 1e-2, path
        return np.fromfile(f, np.float64, 9).reshape(3, 3)


def _read_png(path):
    import imageio.v2 as imageio
    return np.asarray(imageio.imread(path))


def _read_pfm(path):
    """Middlebury .pfm -> (H,W) float32, top-down. inf marks unknown disparity."""
    with open(path, "rb") as f:
        if f.readline().strip() not in (b"Pf", b"PF"):
            raise IOError(f"{path}: not a PFM")
        colour = False
        w, h = (int(x) for x in f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4", w * h)
    return np.flipud(data.reshape(h, w)).astype(np.float32)


def _hypersim_ray2plane(h, w):
    """Hypersim stores DISTANCE TO THE CAMERA CENTRE, not planar z. Multiplying
    by this factor converts to the z-depth every boundary definition assumes.
    Skipping it makes every number computed from Hypersim wrong."""
    fx = HYPERSIM_FOCAL * w / HYPERSIM_WH[0]
    fy = HYPERSIM_FOCAL * h / HYPERSIM_WH[1]
    u = np.arange(w) - (w - 1) / 2.0
    v = np.arange(h) - (h - 1) / 2.0
    uu, vv = np.meshgrid(u, v)
    f = 0.5 * (fx + fy)
    return (f / np.sqrt(uu ** 2 + vv ** 2 + f ** 2)).astype(np.float32)


def _hypersim_K(h, w):
    fx = HYPERSIM_FOCAL * w / HYPERSIM_WH[0]
    fy = HYPERSIM_FOCAL * h / HYPERSIM_WH[1]
    return np.array([[fx, 0, (w - 1) / 2.0], [0, fy, (h - 1) / 2.0], [0, 0, 1.0]])


def _read_h5(path, key=None):
    import h5py
    with h5py.File(path, "r") as f:
        k = key or list(f.keys())[0]
        return np.asarray(f[k][()])


def _middlebury_calib(path):
    """calib.txt -> (K, baseline_m, doffs). Middlebury quotes baseline in mm."""
    kv = {}
    for line in open(path):
        if "=" in line:
            a, b = line.split("=", 1)
            kv[a.strip()] = b.strip()
    m = re.findall(r"[-+0-9.eE]+", kv.get("cam0", ""))
    fx, cx, fy, cy = float(m[0]), float(m[2]), float(m[4]), float(m[5])
    return (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
            float(kv.get("baseline", 0)) / 1000.0, float(kv.get("doffs", 0)))


_NYU_CACHE: dict[str, object] = {}


def _nyu(root):
    """The labeled NYUv2 .mat is HDF5 v7.3 with axes stored (N,3,W,H)/(N,W,H).

    It ships BOTH `depths` (in-painted, the usual GT) and `rawDepths` (the sensor
    output, with holes). That pairing is what makes NYUv2 a real control here:
    Sec. 5's GT control regresses the raw map against support computed from the
    cleaned one, so FP_GT is expected to be genuinely non-zero.
    """
    if root not in _NYU_CACHE:
        import h5py
        _NYU_CACHE[root] = h5py.File(os.path.join(root, "nyu_depth_v2_labeled.mat"), "r")
    return _NYU_CACHE[root]


def _nyu_scene_names(f):
    out = []
    for ref in f["sceneTypes"][0] if "sceneTypes" in f else f["scenes"][0]:
        out.append("".join(chr(c) for c in f[ref][:].ravel()))
    return out


_PO_K_CACHE: dict[str, np.ndarray] = {}


def _po_K(anno, idx):
    """anno.npz is ~200MB; read it once per sequence, not once per frame."""
    arr = _PO_K_CACHE.get(anno)
    if arr is None:
        arr = np.load(anno)["intrinsics"]
        _PO_K_CACHE[anno] = arr
    return arr[min(idx, len(arr) - 1)].astype(np.float64)


# --------------------------------------------------------------------------- #
# scene / view enumeration
# --------------------------------------------------------------------------- #
def scenes(ds):
    r = root_of(ds)
    if ds == "tartanair":
        return sorted(os.path.relpath(os.path.dirname(p), r).replace(os.sep, "/")
                      for p in glob.glob(os.path.join(r, "*", "*", "P*", "image_left")))
    if ds == "pointodyssey":
        # id includes the split: a sequence name can occur under both train/ and
        # val/, and collapsing them would merge two scenes into one unit of
        # analysis and make the resolved path filesystem-order dependent.
        return sorted(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
                      for p in glob.glob(os.path.join(r, "*", "*"))
                      if os.path.isdir(p) and os.path.exists(os.path.join(p, "anno.npz")))
    if ds == "spring":
        return sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(r, "train", "*")) if os.path.isdir(p))
    if ds == "sintel":
        return sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(r, "clean", "*")) if os.path.isdir(p))
    if ds == "hypersim":
        # one scene per (volume, camera trajectory): different cameras of the same
        # volume see different geometry and must not be pooled as one unit.
        return sorted(f"{os.path.basename(os.path.dirname(os.path.dirname(p)))}/"
                      f"{os.path.basename(p).replace('_geometry_hdf5', '')}"
                      for p in glob.glob(os.path.join(r, "*", "images", "*_geometry_hdf5")))
    if ds == "middlebury":
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(r, "*"))
                      if os.path.isfile(os.path.join(p, "calib.txt")))
    if ds == "infinigen":
        # tarballs extract to <seed>/<seed>/frames/{Depth,Image}/<camera>/
        return sorted(f"{p.split(os.sep)[-4]}/{os.path.basename(p)}"
                      for p in glob.glob(os.path.join(r, "*", "*", "frames", "Depth", "camera_*")))
    if ds == "ibims":
        # 100 independent single images from different rooms: each IS a scene,
        # which is exactly the unit of analysis Sec. 6.7 wants.
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(r, "depth", "*.png")))
    if ds == "eth3d":
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(r, "*"))
                      if os.path.isdir(os.path.join(p, "ground_truth_depth")))
    if ds == "bonn":
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(r, "*"))
                      if os.path.isdir(os.path.join(p, "depth")))
    if ds == "nyuv2":
        try:
            return sorted(set(_nyu_scene_names(_nyu(r))))
        except Exception:
            return []
    raise KeyError(ds)


def _scene_dir(ds, scene):
    r = root_of(ds)
    if ds == "tartanair":
        return os.path.join(r, *scene.split("/"))
    if ds == "pointodyssey":
        return os.path.join(r, *scene.split("/"))      # scene is "<split>/<seq>"
    if ds == "spring":
        return os.path.join(r, "train", scene)
    if ds in ("hypersim", "infinigen"):
        return os.path.join(r, *scene.split("/"))
    if ds in ("middlebury", "eth3d", "bonn"):
        return os.path.join(r, scene)
    return r


def views(ds, scene, n=None):
    """Deterministic evenly-spaced subsample of a scene's views (all if n=None)."""
    d = _scene_dir(ds, scene)
    out = []
    if ds == "tartanair":
        for p in sorted(glob.glob(os.path.join(d, "image_left", "*_left.png"))):
            b = os.path.basename(p).replace("_left.png", "")
            dp = os.path.join(d, "depth_left", b + "_left_depth.npy")
            if os.path.exists(dp):
                out.append(View(ds, scene, int(b), p, dp))
    elif ds == "pointodyssey":
        anno = os.path.join(d, "anno.npz")
        for p in sorted(glob.glob(os.path.join(d, "rgbs", "*.jpg")) +
                        glob.glob(os.path.join(d, "rgbs", "*.png"))):
            i = int("".join(filter(str.isdigit, os.path.basename(p))))
            dp = os.path.join(d, "depths", "depth_%05d.png" % i)
            if os.path.exists(dp):
                out.append(View(ds, scene, i, p, dp, anno))
    elif ds == "spring":
        cam = os.path.join(d, "cam_data", "intrinsics.txt")
        for p in sorted(glob.glob(os.path.join(d, "frame_left", "*.png"))):
            i = int("".join(filter(str.isdigit, os.path.basename(p)))[-4:])
            dp = os.path.join(d, "disp1_left", f"disp1_left_{i:04d}.dsp5")
            if os.path.exists(dp):
                out.append(View(ds, scene, i, p, dp, cam))
    elif ds == "sintel":
        for p in sorted(glob.glob(os.path.join(d, "clean", scene, "frame_*.png"))):
            i = int("".join(filter(str.isdigit, os.path.basename(p))))
            dp = os.path.join(d, "depth", scene, f"frame_{i:04d}.dpt")
            cm = os.path.join(d, "camdata_left", scene, f"frame_{i:04d}.cam")
            if os.path.exists(dp):
                out.append(View(ds, scene, i, p, dp, cm))
    elif ds == "hypersim":
        vol, cam = scene.split("/")
        gd = os.path.join(root_of(ds), vol, "images", f"{cam}_geometry_hdf5")
        pv = os.path.join(root_of(ds), vol, "images", f"{cam}_final_preview")
        for p in sorted(glob.glob(os.path.join(gd, "frame.*.depth_meters.hdf5"))):
            i = int(os.path.basename(p).split(".")[1])
            rgb = os.path.join(pv, f"frame.{i:04d}.tonemap.jpg")
            out.append(View(ds, scene, i, rgb, p))
    elif ds == "middlebury":
        d0 = os.path.join(root_of(ds), scene)
        if os.path.exists(os.path.join(d0, "disp0.pfm")):
            out.append(View(ds, scene, 0, os.path.join(d0, "im0.png"),
                            os.path.join(d0, "disp0.pfm"),
                            os.path.join(d0, "calib.txt")))
    elif ds == "infinigen":
        seed, cam = scene.split("/")
        fr = os.path.join(root_of(ds), seed, seed, "frames")
        for p in sorted(glob.glob(os.path.join(fr, "Depth", cam, "*.npy"))):
            # Depth_0_0_0120_0.npy pairs with Image_0_0_0120_0.png; the numeric
            # field is the frame index.
            base = os.path.basename(p)
            parts = base.replace(".npy", "").split("_")
            i = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else len(out)
            rgb = os.path.join(fr, "Image", cam, base.replace("Depth", "Image")
                               .replace(".npy", ".png"))
            if not os.path.exists(rgb):
                cand = sorted(glob.glob(os.path.join(fr, "Image", cam, f"*{parts[3]}*.png"))) \
                       if len(parts) > 3 else []
                rgb = cand[0] if cand else ""
            out.append(View(ds, scene, i, rgb, p))
    elif ds == "ibims":
        r0 = root_of(ds)
        out.append(View(ds, scene, 0, os.path.join(r0, "rgb", scene + ".png"),
                        os.path.join(r0, "depth", scene + ".png"),
                        os.path.join(r0, "calib", scene + ".txt")))
    elif ds == "eth3d":
        b = os.path.join(root_of(ds), scene)
        for p in sorted(glob.glob(os.path.join(b, "ground_truth_depth", "**", "*"),
                                  recursive=True)):
            if os.path.isdir(p):
                continue
            rel = os.path.relpath(p, os.path.join(b, "ground_truth_depth"))
            rgb = os.path.join(b, "images", rel + ".JPG")
            if not os.path.exists(rgb):
                rgb = os.path.join(b, "images", rel)
            out.append(View(ds, scene, len(out), rgb, p,
                            os.path.join(b, "cameras.txt")))
    elif ds == "bonn":
        b = os.path.join(root_of(ds), scene)
        rgbs = sorted(glob.glob(os.path.join(b, "rgb", "*.png")))
        for p in sorted(glob.glob(os.path.join(b, "depth", "*.png"))):
            i = len(out)
            out.append(View(ds, scene, i, rgbs[min(i, len(rgbs) - 1)] if rgbs else "", p))
    elif ds == "nyuv2":
        f = _nyu(root_of(ds))
        names = _nyu_scene_names(f)
        for i, nm in enumerate(names):
            if nm == scene:
                out.append(View(ds, scene, i, "", ""))
    if n and len(out) > n:
        out = [out[i] for i in np.linspace(0, len(out) - 1, n).round().astype(int)]
    return out


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(v, with_rgb=False):
    ds, mx = v.dataset, SPECS[v.dataset]["max_depth"]
    K = np.eye(3)
    mask = None                                   # dataset-declared invalid pixels

    if ds == "tartanair":
        raw = np.load(v.depth).astype(np.float32)
        K = TARTANAIR_K.copy()
    elif ds == "pointodyssey":
        raw = (_read_png(v.depth).astype(np.float32) / 65535.0) * 1000.0
        K = _po_K(v.cam, v.idx)
    elif ds == "spring":
        import h5py
        with h5py.File(v.depth, "r") as f:
            disp = np.asarray(f["disparity"][()])[::2, ::2].astype(np.float32)
        row = np.loadtxt(v.cam)
        row = row if row.ndim == 1 else row[min(v.idx - 1, len(row) - 1)]
        fx, fy, cx, cy = row[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (fx * SPRING_BASELINE / disp).astype(np.float32)
        mask = disp > 0                            # d <= 0 is sky
    elif ds == "sintel":
        raw = _read_dpt(v.depth).astype(np.float32)
        K = _read_cam(v.cam) if os.path.exists(v.cam) else np.eye(3)
        inv = v.depth.replace(f"{os.sep}depth{os.sep}", f"{os.sep}invalid{os.sep}") \
                     .replace(".dpt", ".png")
        if os.path.exists(inv):
            mask = _read_png(inv) == 0
    elif ds == "hypersim":
        dist = _read_h5(v.depth, "dataset").astype(np.float32)
        h, w = dist.shape
        raw = dist * _hypersim_ray2plane(h, w)     # distance -> planar z, MANDATORY
        K = _hypersim_K(h, w)
        mask = np.isfinite(dist)                   # sky/windows are NaN by design
    elif ds == "middlebury":
        disp = _read_pfm(v.depth)
        K, B, doffs = _middlebury_calib(v.cam)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = (K[0, 0] * B / (disp + doffs)).astype(np.float32)
        mask = np.isfinite(disp) & (disp > 0)      # inf = unknown disparity
    elif ds == "infinigen":
        raw = np.load(v.depth).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        mask = np.isfinite(raw)
        K = np.eye(3)                              # not needed for the measurement
    elif ds == "ibims":
        png = _read_png(v.depth).astype(np.float32)
        raw = png * IBIMS_DEPTH_SCALE
        fx, fy, cx, cy = (float(x) for x in open(v.cam).read().strip().split(","))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        r0 = root_of(ds)
        # Despite the name, mask_invalid is TRUE where the pixel is VALID
        # (verified: its False regions carry 0% depth). mask_transp marks
        # non-transparent surfaces, where the laser return is trustworthy.
        mask = (png > 0) & (png < IBIMS_SATURATED)
        for m in ("mask_invalid", "mask_transp"):
            f = os.path.join(r0, m, v.scene + ".png")
            if os.path.exists(f):
                mask &= _read_png(f).astype(bool)
    elif ds == "eth3d":
        buf = np.fromfile(v.depth, dtype="<f4")
        # ETH3D GT depth is a raw float32 grid in the ORIGINAL (distorted) DSLR
        # frame -- constant 4032x6048 across every scene we checked. The
        # `dslr_images_undistorted` JPGs are a DIFFERENT, per-scene size
        # (6205x4135, 6220x4141, ...) and do NOT pair with it pixel-for-pixel, so
        # they must not be used as the RGB input for these depth maps.
        hw = None
        if v.rgb and os.path.exists(v.rgb):
            from PIL import Image
            w0, h0 = Image.open(v.rgb).size
            if w0 * h0 == buf.size:
                hw = (h0, w0)
        if hw is None:
            for h0 in (ETH3D_NATIVE[0], ETH3D_NATIVE[1]):
                if buf.size % h0 == 0:
                    hw = (h0, buf.size // h0)
                    break
        if hw is None:
            raise IOError(f"{v.depth}: cannot infer depth shape from {buf.size} floats")
        raw = buf.reshape(hw).astype(np.float32)
        mask = np.isfinite(raw) & (raw > 0)
        K = np.eye(3)
    elif ds == "bonn":
        raw = (_read_png(v.depth).astype(np.float32) / BONN_DEPTH_SCALE)
        mask = raw > 0                             # 0 = no return, the masked case
        K = np.eye(3)
    elif ds == "nyuv2":
        f = _nyu(root_of(ds))
        raw = np.asarray(f["rawDepths"][v.idx]).T.astype(np.float32)
        clean = np.asarray(f["depths"][v.idx]).T.astype(np.float32)
        # `raw` is the sensor map WITH holes; `clean` is the in-painted GT the
        # support is built from. Sec. 5's control wants exactly this pairing.
        mask = clean > 0
        K = np.eye(3)
        range_valid = np.isfinite(clean) & (clean > 0) & (clean <= mx)
        valid = range_valid & mask
        depth = np.where(valid, clean, np.nan).astype(np.float32)
        rgb = None
        if with_rgb:
            rgb = np.asarray(f["images"][v.idx]).transpose(2, 1, 0)[..., :3]
        return ViewData(rgb, depth, valid, raw, K, mask, range_valid)
    else:
        raise KeyError(ds)

    # Kept apart on purpose: the Sec. 4 gate has to distinguish pixels the DATASET
    # deleted from pixels we drop for being sky or out of range. Folding them
    # together would charge our own range cut to the dataset's mask.
    range_valid = np.isfinite(raw) & (raw > 0) & (raw <= mx)
    valid = range_valid & mask if mask is not None else range_valid.copy()
    depth = np.where(valid, raw, np.nan).astype(np.float32)

    rgb = None
    if with_rgb and v.rgb and os.path.exists(v.rgb):
        img = _read_png(v.rgb)
        if img.ndim == 2:                       # greyscale: do not slice the width axis
            img = np.repeat(img[..., None], 3, 2)
        rgb = img[..., :3]
    return ViewData(rgb, depth, valid, raw.astype(np.float32), K, mask, range_valid)


# --------------------------------------------------------------------------- #
# Sec. 4 measurability gate -- run FIRST, it is cheap and can end the study
# --------------------------------------------------------------------------- #
def gate(ds, n_scenes=8, n_views=4, eta=FM.ETA0, w=3):
    """Sec. 4 measurability gate.

    frac_boundary_valid = share of DETECTED boundary pixels that still have valid
    GT on both sides. The detection must happen BEFORE the dataset's validity
    mask is applied: a dataset that deletes its boundaries would otherwise erase
    those pixels from numerator and denominator alike and score ~1.0, and the
    gate would pass exactly the datasets it exists to reject.

    frac_boundary_masked = share of those detected boundary pixels that the
    dataset's OWN mask removes -- not counting our sky/max-range cut, which is
    our choice and not evidence about the data.

    Both are pooled over boundary pixels, not averaged over views: Sec. 4
    thresholds this at 0.5, and a view with a handful of boundary pixels must not
    weigh as much as one with tens of thousands.
    """
    spec = SPECS[ds]
    sc = scenes(ds)
    if not sc:
        return dict(dataset=ds, tier=spec["tier"], status="absent", n_scenes=0,
                    n_views=0, n_boundary=0, frac_boundary_valid=float("nan"),
                    frac_boundary_masked=float("nan"), usable=False,
                    note=f"no scenes found under {root_of(ds)}")
    sc = [sc[i] for i in np.linspace(0, len(sc) - 1, min(n_scenes, len(sc))).round().astype(int)]

    n_det = n_both = n_masked = 0
    nv = 0
    for s_ in sc:
        for v in views(ds, s_, n_views):
            try:
                d = load(v)
            except Exception as e:
                print(f"    ! {v.key}: {type(e).__name__}: {e}", flush=True)
                continue
            nv += 1
            # The pre-mask validity must ALSO exclude non-positive raw values.
            # NYUv2's raw sensor map has 0-valued holes where the cleaned map was
            # in-painted, so range_valid (computed from the cleaned map) can be
            # True at a raw 0. Those zeros reach rel_jump's min(D(p),D(q))
            # denominator and produce inf/NaN jumps -- which is what the
            # "divide by zero" warning was, and it moves a gate verdict that sits
            # right on the 0.5 threshold.
            rv = d.range_valid if d.range_valid is not None else np.isfinite(d.depth_raw)
            rv = rv & np.isfinite(d.depth_raw) & (d.depth_raw > 0)
            pre = np.where(rv, d.depth_raw, np.nan).astype(np.float64)
            B = FM.boundary_set(pre, rv, eta)          # detected BEFORE the mask
            pix = np.argwhere(B)
            if not len(pix):
                continue
            n_det += len(pix)
            if d.mask is not None:
                n_masked += int((~d.mask[pix[:, 0], pix[:, 1]]).sum())
            # both sides still resolvable in the data we would actually measure
            K = FM.depth_support(d.depth.astype(np.float64), d.valid, pix, w=w, eta=eta)["K"]
            n_both += int((K >= 2).sum())

    if not n_det:
        return dict(dataset=ds, tier=spec["tier"], status="absent", n_scenes=len(sc),
                    n_views=nv, n_boundary=0, frac_boundary_valid=float("nan"),
                    frac_boundary_masked=float("nan"), usable=False,
                    note="no boundary pixels detected in the sampled views")
    fv, fm = n_both / n_det, n_masked / n_det
    return dict(dataset=ds, tier=spec["tier"], status="ok" if fv >= 0.5 else "fail",
                n_scenes=len(sc), n_views=nv, n_boundary=n_det,
                frac_boundary_valid=fv, frac_boundary_masked=fm,
                usable=bool(fv >= 0.5),
                note="" if fv >= 0.5 else "GATE FAIL (frac_boundary_valid < 0.5)")


def matrix(out=None):
    """Sec. 4 contamination matrix: rows = datasets, cols = Tier 1 models.

    "unknown" counts as "yes". A dataset is CLEAN only when no Tier 1 model is
    known or suspected to have trained on it -- the headline result must come
    from one of those.
    """
    rows = []
    for ds, sp in SPECS.items():
        c = {m: ("yes" if m in sp["contam"] else
                 "unknown" if f"{m}?" in sp["contam"] else "no")
             for m in TIER1_MODELS}
        clean = all(v == "no" for v in c.values())
        rows.append(dict(dataset=ds, tier=sp["tier"],
                         **{f"in_{m}_training": c[m] for m in TIER1_MODELS},
                         clean_for_all_tier1=clean, note=sp["note"]))
    w = max(len(r["dataset"]) for r in rows)
    print(f"{'dataset':<{w}}  tier  " + "  ".join(f"{m:>8}" for m in TIER1_MODELS)
          + "   clean?")
    for r in rows:
        print(f"{r['dataset']:<{w}}   {r['tier']}    "
              + "  ".join(f"{r[f'in_{m}_training']:>8}" for m in TIER1_MODELS)
              + f"   {'YES' if r['clean_for_all_tier1'] else 'no'}   {r['note']}")
    ok = [r["dataset"] for r in rows if r["clean_for_all_tier1"] and r["tier"] == "A"]
    print(f"\nTier A and clean for every Tier 1 model: {ok or 'NONE'}")
    if not ok:
        print("  -> No clean exact-boundary set. Sec. 4 says the fallback is to render "
              "a small held-out set ourselves before trusting any headline number.")
    if out:
        os.makedirs(out, exist_ok=True)
        pth = os.path.join(out, "contamination.csv")
        with open(pth, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
        print("wrote", pth)
    return rows


def verify(ds, n=2):
    """Load a couple of views and report whether the decoded depth is sane.

    Most of these loaders are written to documented formats and have never seen
    the real files. This is the smoke test to run FIRST on the server: a loader
    that silently returns disparity, millimetres, or ray distance instead of
    metric z-depth would corrupt every downstream number while looking fine.
    """
    try:
        sc = scenes(ds)
    except Exception as e:
        return dict(dataset=ds, status="ERROR", detail=f"scenes(): {type(e).__name__}: {e}")
    if not sc:
        return dict(dataset=ds, status="absent", detail=f"no scenes under {root_of(ds)}")
    rep = []
    for s_ in sc[:n]:
        vs = views(ds, s_, 1)
        if not vs:
            rep.append(f"{s_}: no views")
            continue
        try:
            d = load(vs[0], with_rgb=True)
        except Exception as e:
            rep.append(f"{s_}: load failed {type(e).__name__}: {e}")
            continue
        v = d.depth[np.isfinite(d.depth)]
        if not v.size:
            rep.append(f"{s_}: all pixels invalid")
            continue
        rep.append(f"{s_}: {d.depth.shape} valid={d.valid.mean():.2f} "
                   f"depth[min={v.min():.3f} med={np.median(v):.3f} max={v.max():.3f}]m "
                   f"rgb={'ok' if d.rgb is not None else 'MISSING'}")
    bad = [r for r in rep if "failed" in r or "no views" in r or "all pixels" in r]
    return dict(dataset=ds, status="FAIL" if bad else "ok", detail=" | ".join(rep))


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="premise-check data layer + Sec.4 gate")
    ap.add_argument("--list", action="store_true", help="scene/view inventory")
    ap.add_argument("--matrix", action="store_true", help="Sec.4 contamination matrix")
    ap.add_argument("--verify", action="store_true",
                    help="decode a few views per dataset and sanity-check the depths")
    ap.add_argument("--gate", action="store_true", help="run the measurability gate")
    ap.add_argument("--datasets", default=",".join(SPECS))
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--out", default="results/premise_check")
    a = ap.parse_args()
    dss = [d for d in a.datasets.split(",") if d]

    if a.matrix:
        matrix(a.out)
    if a.verify:
        bad = 0
        for ds in dss:
            r = verify(ds)
            print(f"{r['status']:>6}  {r['dataset']:<13} {r['detail']}")
            bad += r["status"] in ("FAIL", "ERROR")
        if bad:
            print(f"\n{bad} dataset(s) failed to decode. Fix before measuring.")
            return 4
    if a.list:
        for ds in dss:
            sc = scenes(ds)
            nv = len(views(ds, sc[0])) if sc else 0
            print(f"{ds:14s} tier {SPECS[ds]['tier']}  scenes={len(sc):4d}  "
                  f"views/scene~{nv:5d}  root={root_of(ds)}")
            if sc:
                print(f"{'':14s} e.g. {sc[:3]}")
    if a.gate:
        os.makedirs(a.out, exist_ok=True)
        rows = []
        for ds in dss:
            print(f"gate: {ds} ...", flush=True)
            r = gate(ds, a.scenes, a.views)
            rows.append(r)
            print(f"  status={r['status']:6s} valid={r['frac_boundary_valid']:.3f} "
                  f"masked={r['frac_boundary_masked']:.3f} boundary_px={r['n_boundary']} "
                  f"{r['note']}", flush=True)
        p = os.path.join(a.out, "gate.csv")
        with open(p, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
        print("wrote", p)
        absent = [r for r in rows if r["status"] == "absent"]
        failed = [r for r in rows if r["status"] == "fail"]
        if absent:
            print(f"\n{len(absent)} dataset(s) have NO DATA on disk: "
                  f"{', '.join(r['dataset'] for r in absent)}. That is a setup problem, "
                  f"NOT a measurability finding -- check the paths under {ROOT}/ before "
                  f"reading anything into it.")
        if not any(r["usable"] for r in rows):
            if failed:
                print("\nNO-GO on measurability: every dataset that HAD data failed the "
                      "gate (Sec. 2). The boundaries have been deleted from the data.")
                return 2
            print("\nNothing could be gated because nothing was on disk. Not a verdict.")
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
