"""Multilabels from CDL proportions.

* Proportions: class counts / mapped CDL pixels of the window (codes outside the 40 groups are
  excluded from the denominator), for one CDL year.
* Per-date label: proportions of the CDL year of the acquisition date; a class is present if its
  proportion is >= 10 %. Padded dates (no acquisition) get label -1 and NaN proportions.
  Dates that fail QC still get their CDL label (QC only concerns the image).
* Window label (decision to validate): proportions averaged over the *distinct* CDL years of
  the window's acquisition dates (each year weighted once), then thresholded at 10 %.
"""
from __future__ import annotations

import numpy as np

from . import config
from .groundtruth import class_counts


def year_props(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, Y, 40) counts -> (N, Y, 13) proportions and (N, Y) number of mapped pixels."""
    cc = class_counts(counts)
    tot = cc.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = cc / tot[..., None]
    p[tot == 0] = np.nan
    return p.astype("float32"), tot


def date_props(props_year: np.ndarray, dates: np.ndarray, years: list[int]) -> np.ndarray:
    """(N, Y, 13), (N, T) ISO dates ('' = padding) -> (N, T, 13), NaN for padded dates."""
    N, T = dates.shape
    out = np.full((N, T, props_year.shape[-1]), np.nan, "float32")
    yi = {y: i for i, y in enumerate(years)}
    for n in range(N):
        for t in range(T):
            d = str(dates[n, t])
            if d:
                out[n, t] = props_year[n, yi[int(d[:4])]]
    return out


def window_props(props_year: np.ndarray, dates: np.ndarray, years: list[int]) -> np.ndarray:
    """Mean of the yearly proportions over the distinct years of each window's dates."""
    N = dates.shape[0]
    out = np.full((N, props_year.shape[-1]), np.nan, "float32")
    yi = {y: i for i, y in enumerate(years)}
    for n in range(N):
        ys = sorted({int(str(d)[:4]) for d in dates[n] if str(d)})
        if ys:
            out[n] = np.nanmean(props_year[n, [yi[y] for y in ys]], 0)
    return out


def multilabel(props: np.ndarray, thr: float = config.PRESENCE_THRESHOLD) -> np.ndarray:
    """Presence (int8 0/1) where proportion >= thr, -1 where proportions are missing (NaN)."""
    lab = (props >= thr - 1e-9).astype(np.int8)
    lab[np.isnan(props)] = -1
    return lab
