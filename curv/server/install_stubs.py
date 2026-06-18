"""
Install import-satisfying stubs into a DDUSt3R clone (training-only path).

Three transitive imports break a fresh clone but are never exercised by the
finetune:
  - the optimizer hard-imports `load_RAFT` / `build_sam2_video_predictor`
    (flow/seg, only used when flow_loss_weight>0 -- we keep it 0);
  - `dust3r.training` imports `pose_eval -> demo -> viz_demo ->
    datasets_preprocess.sintel_get_dynamics.compute_optical_flow`, a
    preprocessing/demo-viz module not shipped in the clone.
These stubs satisfy the imports and raise loudly if actually called. Files that
already exist are left untouched (so a real datasets_preprocess is never
clobbered).

Usage (after cloning DDUSt3R):
    python curv/server/install_stubs.py /path/to/DDUSt3R
    # or, from repo root with the default layout:
    python curv/server/install_stubs.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "..", "..", "DDUSt3R")

RAFT = '''"""Flow-free stub: this project keeps flow_loss_weight=0, so RAFT is never
loaded. Satisfies the import; raises if flow is switched on by accident."""


def load_RAFT(model_path=None):
    raise RuntimeError(
        "RAFT is stubbed out: this project is flow-free. A call to load_RAFT "
        "means flow_loss_weight > 0 somewhere -- keep it at 0.")
'''

SAM2 = '''"""Flow-free / segmentation-free stub (see third_party/raft.py). SAM2 is only
used by the optimizer's sam2_mask_refine path, inside the flow_loss_weight>0
block we never enter. Satisfies the import; raises if called."""


def build_sam2_video_predictor(*args, **kwargs):
    raise RuntimeError("SAM2 is stubbed out: this project does not use "
                       "segmentation-based mask refinement.")
'''

SINTEL_DYN = '''"""Stub: training does not use sintel dynamic-flow preprocessing / demo viz.
Satisfies the transitive import dust3r.training -> pose_eval -> demo -> viz_demo.
Raises if actually called."""


def compute_optical_flow(*args, **kwargs):
    raise RuntimeError("datasets_preprocess.sintel_get_dynamics is stubbed "
                       "(demo / pose-eval viz only; not used in training).")
'''

FILES = {
    "third_party/__init__.py": "",
    "third_party/raft.py": RAFT,
    "sam2/__init__.py": "",
    "sam2/build_sam.py": SAM2,
    "datasets_preprocess/__init__.py": "",
    "datasets_preprocess/sintel_get_dynamics.py": SINTEL_DYN,
}

if not os.path.isdir(DD):
    sys.exit(f"DDUSt3R dir not found: {DD}")

for rel, content in FILES.items():
    p = os.path.join(DD, rel)
    if os.path.exists(p):                       # never clobber a real module
        print("exists, skip", os.path.normpath(p))
        continue
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        f.write(content)
    print("wrote", os.path.normpath(p))
print("stubs installed.")
