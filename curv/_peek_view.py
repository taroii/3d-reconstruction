"""Throwaway: dump what fields PointOdysseyDUSt3R puts on each view, so we can
locate the matching masks/mask_<frame>.png for the dynamic-premise test."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "DDUSt3R"))
from dust3r.datasets import get_data_loader

ds = ("4 @ PointOdysseyDUSt3R(dset='train', dataset_location='../data/pointodyssey',"
      " S=2, strides=[4], resolution=(512,288))")
loader = get_data_loader(ds, batch_size=2, num_workers=0, shuffle=False, drop_last=False)
d = getattr(loader, "dataset", None)
if d is not None and hasattr(d, "set_epoch"):
    d.set_epoch(0)

for batch in loader:
    v1 = batch[0] if isinstance(batch, (list, tuple)) else batch["view1"]
    print("VIEW KEYS:", sorted(v1.keys()))
    for k in ("dataset", "label", "instance", "idx"):
        if k in v1:
            val = v1[k]
            try:
                val = [val[i] for i in range(min(2, len(val)))]
            except Exception:
                pass
            print(f"  {k}: {val}")
    for k in ("img", "depthmap", "true_shape"):
        if k in v1:
            print(f"  {k}.shape: {tuple(getattr(v1[k], 'shape', ()))}")
    break
