"""
Install the flow-free RAFT/SAM2 import stubs into a DDUSt3R clone.

The bundled optimizer (DDUSt3R/dust3r/cloud_opt/optimizer.py) hard-imports
`load_RAFT` and `build_sam2_video_predictor` at module load, but never CALLS
them as long as flow_loss_weight=0 (which is our default everywhere). A fresh
DDUSt3R clone does not ship these modules, so the import fails. These stubs
satisfy the import and raise loudly if flow/segmentation is ever switched on.

Usage (after cloning DDUSt3R):
    python mfc/server/install_stubs.py /path/to/DDUSt3R
    # or, from repo root with the default layout:
    python mfc/server/install_stubs.py
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

FILES = {
    "third_party/__init__.py": "",
    "third_party/raft.py": RAFT,
    "sam2/__init__.py": "",
    "sam2/build_sam.py": SAM2,
}

if not os.path.isdir(DD):
    sys.exit(f"DDUSt3R dir not found: {DD}")

for rel, content in FILES.items():
    p = os.path.join(DD, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(content)
    print("wrote", os.path.normpath(p))
print("stubs installed.")
