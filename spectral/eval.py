"""Evaluation metrics (plan S8). Pure NumPy."""

import numpy as np


def auroc(scores, labels):
    """Threshold-free AUROC of `scores` predicting boolean `labels` (rank-sum,
    tie-corrected)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2.0
        i = j + 1
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mask_iou_f1(pred, gt, valid=None):
    """IoU and F1 of boolean masks over `valid` pixels."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if valid is not None:
        v = np.asarray(valid, bool); pred, gt = pred & v, gt & v
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    tp, fp, fn = inter, (pred & ~gt).sum(), (~pred & gt).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return float(iou), float(f1)


def map_auroc(energy_maps, gt_masks, valid_masks):
    """AUROC over all valid pixels pooled across a clip's views."""
    sc, lab = [], []
    for v in energy_maps:
        vm = valid_masks[v]
        sc.append(np.sqrt(energy_maps[v][vm] + 1e-15))
        lab.append(gt_masks[v][vm])
    return auroc(np.concatenate(sc), np.concatenate(lab))
