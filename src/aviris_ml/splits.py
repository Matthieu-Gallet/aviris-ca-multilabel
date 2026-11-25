"""Spatial folds by 10 km blocks (leakage-safe: a block never spans two folds).

Blocks (17 x 17 windows of 600 m = 10.2 km) are assigned greedily, largest first, to the fold
that keeps the per-class window-label counts (and the window count) most balanced. The
assignment is deterministic (ties broken by a seeded permutation).
"""
from __future__ import annotations

import numpy as np

from . import config


def assign_folds(window_ids: np.ndarray, labels: np.ndarray, n_folds: int = config.N_FOLDS, seed: int = 0) -> np.ndarray:
    """window_ids (N,), labels (N, K) in {0, 1} -> fold (N,) in [0, n_folds)."""
    blocks = np.array([config.block_id(w) for w in window_ids])
    ub, inv = np.unique(blocks, return_inverse=True)
    K = labels.shape[1]
    feat = np.zeros((len(ub), K + 1))
    np.add.at(feat[:, :K], inv, np.clip(labels, 0, 1))
    np.add.at(feat[:, K], inv, 1)
    tot = feat.sum(0)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(len(ub)), -feat[:, K]))
    load = np.zeros((n_folds, K + 1))
    bf = np.zeros(len(ub), int)
    for b in order:
        cand = (load + feat[b]) / np.maximum(tot, 1) * n_folds  # 1.0 = exactly balanced
        score = cand.max(1) + 1e-3 * cand[:, K]
        f = int(np.argmin(score))
        load[f] += feat[b]
        bf[b] = f
    return bf[inv]


def check_folds(window_ids: np.ndarray, folds: np.ndarray, xy: np.ndarray) -> dict:
    """Leakage checks: blocks in more than one fold, duplicated windows, test-train distances."""
    from scipy.spatial import cKDTree

    blocks = np.array([config.block_id(w) for w in window_ids])
    multi = sum(1 for b in np.unique(blocks) if len(np.unique(folds[blocks == b])) > 1)
    dists = []
    for f in np.unique(folds):
        d, _ = cKDTree(xy[folds != f]).query(xy[folds == f])
        dists.append(d)
    d = np.concatenate(dists)
    return dict(n_blocks=int(len(np.unique(blocks))), blocks_in_several_folds=int(multi),
                duplicated_windows=int(len(window_ids) - len(np.unique(window_ids))),
                windows_per_fold={int(f): int((folds == f).sum()) for f in np.unique(folds)},
                nearest_other_fold_m=dict(min=float(d.min()), p5=float(np.percentile(d, 5)), median=float(np.median(d)),
                                          frac_le_600m=float((d <= 600.5).mean()), frac_le_1km=float((d <= 1000).mean())))
