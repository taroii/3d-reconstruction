"""
Install the curvature-weighted loss (Option 2) into a DDUSt3R clone.

Copies curv/curv_loss.py -> DDUSt3R/dust3r/curv_loss.py and appends
`from dust3r.curv_loss import *` to DDUSt3R/dust3r/losses.py so the criterion
string `CurvWeightedConfLoss(...)` resolves in training.py's eval() context.
Idempotent.

  python curv/server/install_curv_loss.py             # default ../../DDUSt3R
  python curv/server/install_curv_loss.py /path/DDUSt3R
"""
import os
import sys
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "..", "curv_loss.py")
DD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "..", "..", "DDUSt3R")

if not os.path.isdir(DD):
    sys.exit(f"DDUSt3R dir not found: {DD}")

dst = os.path.join(DD, "dust3r", "curv_loss.py")
shutil.copyfile(SRC, dst)
print("wrote", os.path.normpath(dst))

losses = os.path.join(DD, "dust3r", "losses.py")
marker = "from dust3r.curv_loss import *"
with open(losses) as f:
    content = f.read()
if marker not in content:
    with open(losses, "a") as f:
        f.write(f"\n\n# curvature-weighted loss (Option 2)\n{marker}\n")
    print("patched", os.path.normpath(losses))
else:
    print("already patched", os.path.normpath(losses))

print("done. Arm B criterion string:")
print("  CurvWeightedConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, gamma=1.0)")
