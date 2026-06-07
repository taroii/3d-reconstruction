"""Lightweight data types crossing the cache boundary (plan S5/S6)."""

from dataclasses import dataclass
import numpy as np


@dataclass
class PairPred:
    """Frozen-backbone output for one ordered pair (i, j), pointmaps in frame i."""
    X_ii: np.ndarray   # (H,W,3) view i points in frame i
    X_ji: np.ndarray   # (H,W,3) view j points in frame i
    conf_i: np.ndarray  # (H,W)
    conf_j: np.ndarray  # (H,W)

    def save(self, path):
        np.savez_compressed(path, X_ii=self.X_ii, X_ji=self.X_ji,
                            conf_i=self.conf_i, conf_j=self.conf_j)

    @staticmethod
    def load(path):
        d = np.load(path)
        return PairPred(d["X_ii"], d["X_ji"], d["conf_i"], d["conf_j"])


@dataclass
class Matches:
    """Correspondences on one edge: pix_i, pix_j as (M,2) int (row,col); conf (M,)."""
    pix_i: np.ndarray
    pix_j: np.ndarray
    conf: np.ndarray

    def __len__(self):
        return len(self.conf)

    def save(self, path):
        np.savez_compressed(path, pix_i=self.pix_i, pix_j=self.pix_j, conf=self.conf)

    @staticmethod
    def load(path):
        d = np.load(path)
        return Matches(d["pix_i"], d["pix_j"], d["conf"])
