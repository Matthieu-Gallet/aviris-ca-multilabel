"""Per-date quality control and per-date spectral statistics (one pass over the cubes).

QC heuristics (not validated against a reference, documented as such):
  * no acquisition (padding)                         -> invalid
  * no-data pixels > 20 % (first band)               -> not ok
  * snow: NDSI(560, 1650) > 0.4 and blue(480) > 0.15  -> not ok
  * very bright scene: blue > 0.30 and NIR(850) > 0.40 -> not ok
  * > 50 % of the base-mask bands with mean <= 5e-4 ("zero" spectra) -> not ok
Medians are spatial medians of the window at the band closest to the given wavelength.
An extra column, ``outlier_frac``, gives the fraction of pixels with at least one base-mask
band outside OUTLIER_RANGE (reported, not used by the QC decision).
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config
from .assemble import load_cube

log = logging.getLogger(__name__)
HIST_EDGES = np.round(np.arange(-0.20, 2.0001, 0.01), 4)  # 220 bins + under/overflow


def qc_bands(wavelength: np.ndarray) -> dict[str, int]:
    return {k: int(np.argmin(np.abs(wavelength - v))) for k, v in config.QC_BANDS_NM.items()}


def qc_date(x: np.ndarray, nan_frac: float, base_mask: np.ndarray, qb: dict[str, int]) -> dict:
    """QC flags of one (224, 32, 32) date. Reproduces the source dataset rules exactly."""
    x = x.astype("float32")
    p = int(base_mask.sum())
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.nanmean(x[base_mask].reshape(p, -1), axis=1) if np.isfinite(x).any() else np.full(p, np.nan)
            med = {k: float(np.nanmedian(x[i])) if np.isfinite(x[i]).any() else np.nan for k, i in qb.items()}
    ndsi = (med["green"] - med["swir"]) / (med["green"] + med["swir"] + 1e-9)
    snow = int(ndsi > config.QC_SNOW_NDSI and med["blue"] > config.QC_SNOW_BLUE)
    bright = int(med["blue"] > config.QC_BRIGHT_BLUE and med["nir"] > config.QC_BRIGHT_NIR)
    zero = int(np.isfinite(s).any() and (s <= config.QC_ZERO_LEVEL).mean() > config.QC_ZERO_FRAC)
    nanhigh = int(float(nan_frac) > config.QC_MAX_NAN_FRAC)
    ok = int(not nanhigh and not snow and not bright and not zero)
    return dict(valid=1, nan_frac=round(float(nan_frac), 4), nan_high=nanhigh, snow=snow, bright=bright, zero=zero, ok=ok,
                blue=med["blue"], green=med["green"], nir=med["nir"], swir=med["swir"])


def _one(args):
    path, base_mask, qb = args
    import warnings

    warnings.simplefilter("ignore", RuntimeWarning)
    w = load_cube(path)
    cube = w["cube"]
    T, B = cube.shape[:2]
    mean_spec = np.full((T, B), np.nan, "float32")
    nanfrac_b = np.ones((T, B), "float32")
    vmin = np.full((T, B), np.nan, "float32")
    vmax = np.full((T, B), np.nan, "float32")
    hist_all = np.zeros((B, len(HIST_EDGES) + 1), np.int64)
    hist_ok = np.zeros_like(hist_all)
    rows = []
    for t in range(T):
        d = w["dates"][t]
        if not d:
            rows.append(dict(valid=0, nan_frac=1.0, nan_high=1, snow=0, bright=0, zero=0, ok=0,
                             blue=np.nan, green=np.nan, nir=np.nan, swir=np.nan, outlier_frac=np.nan))
            continue
        x = cube[t].astype("float32").reshape(B, -1)
        r = qc_date(cube[t], w["nan_frac"][t], base_mask, qb)
        xb = x[base_mask]
        lo, hi = config.OUTLIER_RANGE
        with np.errstate(invalid="ignore"):
            out_px = ((xb < lo) | (xb > hi)).any(0)
        has = np.isfinite(xb).any(0)
        r["outlier_frac"] = float(out_px[has].mean()) if has.any() else np.nan
        rows.append(r)
        fin = np.isfinite(x)
        nanfrac_b[t] = 1 - fin.mean(1)
        if fin.any():
            mean_spec[t] = np.nanmean(x, 1)
            vmin[t] = np.nanmin(np.where(fin, x, np.inf), 1)
            vmax[t] = np.nanmax(np.where(fin, x, -np.inf), 1)
        nb = len(HIST_EDGES) + 1
        idx = np.searchsorted(HIST_EDGES, np.where(fin, x, 0.0), side="right") + np.arange(B)[:, None] * nb
        h = np.bincount(idx[fin], minlength=B * nb).reshape(B, nb)
        hist_all += h
        if r["ok"]:
            hist_ok += h
    return dict(window_id=w["window_id"], dates=w["dates"], flights=w["flights"], rows=rows, mean_spec=mean_spec,
                nanfrac_b=nanfrac_b, vmin=vmin, vmax=vmax, hist_all=hist_all, hist_ok=hist_ok)


def run_qc(paths: dict[int, str], wavelength: np.ndarray, base_mask: np.ndarray, out_npz: str, workers: int = 3) -> str:
    """One pass over all cubes; writes QC flags and spectral statistics to ``out_npz``."""
    if os.path.exists(out_npz):
        return out_npz
    qb = qc_bands(wavelength)
    ids = sorted(paths)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_one, [(paths[i], base_mask, qb) for i in ids], chunksize=4)):
            res.append(r)
            if (k + 1) % 200 == 0:
                log.info("qc: %d/%d windows, %.0f s", k + 1, len(ids), time.time() - t0)
    keys = ["valid", "nan_frac", "nan_high", "snow", "bright", "zero", "ok", "blue", "green", "nir", "swir", "outlier_frac"]
    flags = {k: np.array([[row[k] for row in r["rows"]] for r in res], "float32") for k in keys}
    tmp = out_npz + ".tmp.npz"
    np.savez_compressed(
        tmp, window_id=np.array([r["window_id"] for r in res]), dates=np.array([r["dates"] for r in res]),
        flights=np.array([r["flights"] for r in res]), mean_spec=np.stack([r["mean_spec"] for r in res]),
        nanfrac_band=np.stack([r["nanfrac_b"] for r in res]).astype("float16"), vmin=np.stack([r["vmin"] for r in res]),
        vmax=np.stack([r["vmax"] for r in res]), hist_all=sum(r["hist_all"] for r in res), hist_ok=sum(r["hist_ok"] for r in res),
        hist_edges=HIST_EDGES, **{"qc_" + k: v for k, v in flags.items()})
    os.replace(tmp, out_npz)
    log.info("qc: done in %.0f s", time.time() - t0)
    return out_npz
