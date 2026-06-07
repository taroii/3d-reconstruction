"""
Pluggable correspondence sources (plan S1b, S5). NETWORK SIDE (mostly).

The sheaf edge needs to know which pixel in view i and which in view j are the
SAME physical point. Source matters for the "no flow" claim:
  - CrossAttnCorr  : PRIMARY. argmax of the backbone's decoder cross-attention
                     (appearance, rigid-agnostic, NOT a flow estimator).
  - MASt3RCorr     : fallback. MASt3R feature reciprocal matching.
  - Synthetic3DNN  : CONTAMINATED (3D nearest-neighbour on pointmaps); ablation
                     ONLY — never for the headline (manufactures the signal).

`inject_outliers` supports realism experiment A1.
"""

from typing import Protocol
import numpy as np
from datatypes import Matches


class Correspondence(Protocol):
    def match(self, pair) -> Matches: ...


class CrossAttnCorr:
    """PRIMARY. Decoder cross-attention argmax → reciprocal-consistent matches.
    TODO(env): read attention from PairPred decoder features; mutual-argmax with
    a confidence floor; cap at max_matches_per_edge (plan S4 defaults)."""
    def __init__(self, conf_floor=0.0, max_matches=3000):
        self.conf_floor, self.max_matches = conf_floor, max_matches

    def match(self, pair) -> Matches:
        raise NotImplementedError("fill once backbone features are available")


class MASt3RCorr:
    """Fallback: MASt3R feature reciprocal matching."""
    def match(self, pair) -> Matches:
        raise NotImplementedError


class Synthetic3DNN:
    """CONTAMINATED baseline (3D-NN on pointmaps) — ablation only (plan S1b)."""
    def match(self, pair) -> Matches:
        raise NotImplementedError


def inject_outliers(m: Matches, rate: float, kind="scattered", seed=0) -> Matches:
    """Corrupt a fraction of matches by remapping the j-side pixel (experiment
    A1). 'scattered' picks random correspondences and random wrong targets."""
    rng = np.random.default_rng(seed)
    pix_i, pix_j, conf = m.pix_i.copy(), m.pix_j.copy(), m.conf.copy()
    n = len(conf); n_out = int(round(rate * n))
    if n_out > 0:
        idx = rng.choice(n, size=n_out, replace=False)
        pix_j[idx] = pix_j[rng.integers(0, n, size=n_out)]
    return Matches(pix_i, pix_j, conf)
