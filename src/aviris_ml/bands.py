"""Band table (wavelength, FWHM), base absorption mask and residual-spike statistic.

Spike statistic. For the window-mean spectrum s of a (window, date) and a band b inside a
contiguous run of base-mask bands (never across a removed gap):
    e_b = sign(a) * min(|a|, |c|)  if a = s_b - s_{b-1} and c = s_b - s_{b+1} have the same sign
                                   (an isolated peak or dip), else 0; at the first / last band of
                                   a run e_b = s_b - (its only neighbour in the run).
The anomaly is A_b = e_b - median over the window's QC-ok dates of e_b, which removes the
features of the surface itself (red edge, absorption shoulders) and keeps date-specific
artefacts. A (window, date) has a spike at b when |A_b| > SPIKE_REL * max(s_b, SPIKE_FLOOR).
The spike rate of a band is the fraction of QC-ok (window, date) pairs with a spike at b.
Rule: drop the base-mask bands whose rate exceeds SPIKE_RATE_FACTOR x the median rate.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config
from .envi import parse_header


def header_band_table(hdr_paths: list[str]) -> pd.DataFrame:
    """Median wavelength / FWHM over flight-line headers, with the min-max spread across flights."""
    wl, fw = [], []
    for p in hdr_paths:
        with open(p) as f:
            h = parse_header(f.read())
        if h.wavelength is not None and len(h.wavelength) == config.N_BANDS_RAW:
            wl.append(h.wavelength)
            if h.fwhm is not None and len(h.fwhm) == config.N_BANDS_RAW:
                fw.append(h.fwhm)
    wl, fw = np.array(wl), np.array(fw)
    return pd.DataFrame(dict(
        band=np.arange(config.N_BANDS_RAW), wavelength_nm=np.round(np.median(wl, 0), 2), fwhm_nm=np.round(np.median(fw, 0), 3),
        wavelength_min_nm=np.round(wl.min(0), 2), wavelength_max_nm=np.round(wl.max(0), 2), n_headers=len(wl)))


def headers_in(dir_: str) -> list[str]:
    return sorted(glob.glob(os.path.join(dir_, "*.hdr")))


def base_mask(reference_wavelength: np.ndarray) -> np.ndarray:
    """True = kept. Removes the BASE_REMOVED_RANGES_NM (water vapour, detector edges, > 2440 nm)."""
    w = np.asarray(reference_wavelength)
    keep = np.ones(len(w), bool)
    for lo, hi in config.BASE_REMOVED_RANGES_NM:
        keep &= ~((w >= lo) & (w <= hi))
    return keep


def runs(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    cut = np.nonzero(np.diff(idx) > 1)[0] + 1
    return np.split(idx, cut)


def peak_deviation(spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Isolated-band deviation e (..., B) on the kept bands, NaN elsewhere."""
    s = np.asarray(spec, "float64")
    e = np.full(s.shape, np.nan)
    for r in runs(mask):
        if len(r) < 2:
            e[..., r] = 0.0
            continue
        v = s[..., r]
        o = np.zeros_like(v)
        if len(r) > 2:
            a = v[..., 1:-1] - v[..., :-2]
            c = v[..., 1:-1] - v[..., 2:]
            o[..., 1:-1] = np.where(np.sign(a) == np.sign(c), np.sign(a) * np.minimum(np.abs(a), np.abs(c)), 0.0)
        o[..., 0] = v[..., 0] - v[..., 1]
        o[..., -1] = v[..., -1] - v[..., -2]
        e[..., r] = o
    return e


def spike_flags(spec: np.ndarray, ok: np.ndarray, mask: np.ndarray, rel: float = config.SPIKE_REL,
                floor: float = config.SPIKE_FLOOR) -> np.ndarray:
    """spec (N, T, B) window-mean spectra, ok (N, T) QC-ok dates -> (N, T, B) bool spikes
    (False outside the mask and on non-ok dates)."""
    import warnings

    e = peak_deviation(spec, mask)
    e = np.where(ok[..., None], e, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        A = e - np.nanmedian(e, axis=1, keepdims=True)
    level = np.maximum(np.abs(np.asarray(spec, "float64")), floor)
    with np.errstate(invalid="ignore"):
        f = np.abs(A) > rel * level
    return f & ok[..., None] & mask[None, None, :]


def spike_rates(spec: np.ndarray, ok: np.ndarray, mask: np.ndarray, rel: float = config.SPIKE_REL) -> np.ndarray:
    """Per-band spike rate over the QC-ok (window, date) pairs; NaN outside the mask."""
    f = spike_flags(spec, ok, mask, rel)
    r = f[ok].mean(0).astype("float64")
    r[~mask] = np.nan
    return r


def select_bands(spec: np.ndarray, ok: np.ndarray, mask0: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Single pass: drop base-mask bands with rate > max(SPIKE_RATE_FACTOR x median rate, SPIKE_RATE_MIN).
    Returns (final mask, rate under the base mask, dropped bands sorted by decreasing rate)."""
    rate0 = spike_rates(spec, ok, mask0)
    thr = max(config.SPIKE_RATE_FACTOR * np.nanmedian(rate0), config.SPIKE_RATE_MIN)
    drop = [int(b) for b in np.argsort(-np.nan_to_num(rate0, nan=-1.0)) if mask0[b] and rate0[b] > thr]
    mask = mask0.copy()
    mask[drop] = False
    return mask, rate0, drop
