"""Window assembly: native pixel block -> 32 x 32 grid in EPSG:5070, reflectance 0-1, float16.

The source flight lines are north-up or rotated (20-44 degrees); the rotation is carried by
the affine geotransform derived by GDAL from the ENVI 'map info', so a plain
``rasterio.warp.reproject`` (area average) onto the window grid handles it.
"""
from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np

from . import config

log = logging.getLogger(__name__)


def nodata_mask(arr: np.ndarray) -> np.ndarray:
    """(rows, cols) True where the spectrum is no-data (any -9999, all 0, all 65535 or non-finite)."""
    return (arr == -9999).any(-1) | (arr == 0).all(-1) | (arr == 65535).all(-1) | ~np.isfinite(arr).all(-1)


def to_window(block: np.ndarray, scale: float, src_transform, src_crs, window_id: int) -> tuple[np.ndarray, float]:
    """Reproject a (rows, cols, bands) native block to the (bands, 32, 32) window grid.

    Returns the float16 cube (NaN = no data) and the no-data fraction of the first band."""
    from rasterio.crs import CRS
    from rasterio.transform import Affine, from_origin
    from rasterio.warp import Resampling, reproject

    arr = block.astype("float32")
    nod = nodata_mask(arr)
    arr *= scale
    arr[nod] = np.nan
    x0, y0, x1, y1 = config.cell_bounds(window_id)
    dst = np.full((arr.shape[-1], config.NPIX, config.NPIX), np.nan, "float32")
    reproject(np.moveaxis(arr, -1, 0), dst, src_transform=Affine(*src_transform) if not isinstance(src_transform, Affine) else src_transform,
              src_crs=src_crs, dst_transform=from_origin(x0, y1, config.CELL_M / config.NPIX, config.CELL_M / config.NPIX),
              dst_crs=CRS.from_epsg(5070), resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    dst = np.clip(dst, -config.CLIP_ABS, config.CLIP_ABS)
    return dst.astype("float16"), float(np.isnan(dst[0]).mean())


def assemble_window(raw_dir: str, window_id: int, out_path: str, keep_raw: bool = True) -> str | None:
    """Assemble all downloaded dates of a window into ``out_path`` (npz: cube (14,224,32,32),
    dates, flights, nan_frac). Missing dates are NaN-padded with an empty date string."""
    if os.path.exists(out_path):
        return out_path
    wdir = os.path.join(raw_dir, str(window_id))
    sides = sorted(glob.glob(os.path.join(wdir, "????-??-??.json")))
    if not sides:
        return None
    cubes, dates, flights, nanf = [], [], [], []
    for s in sides[: config.N_DATES]:
        with open(s) as f:
            m = json.load(f)
        block = np.load(s[:-5] + ".npy")
        c, nf = to_window(block, m["scale"], m["transform"], m["crs"], window_id)
        cubes.append(c)
        dates.append(m["date"])
        flights.append(m["flight_id"])
        nanf.append(nf)
    nb = cubes[0].shape[0]
    while len(cubes) < config.N_DATES:
        cubes.append(np.full((nb, config.NPIX, config.NPIX), np.nan, "float16"))
        dates.append("")
        flights.append("")
        nanf.append(1.0)
    failures = []
    sp = os.path.join(wdir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            failures = json.load(f).get("failures", [])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, cube=np.stack(cubes), dates=np.array(dates), flights=np.array(flights),
                        nan_frac=np.array(nanf, "float32"), window_id=int(window_id), failures=np.array(failures))
    os.replace(tmp, out_path)
    if not keep_raw:
        for p in glob.glob(os.path.join(wdir, "*.npy")):
            os.remove(p)
    return out_path


def load_cube(path: str) -> dict:
    """Load a window npz written by :func:`assemble_window` or by the original MIX build."""
    z = np.load(path, allow_pickle=False)
    return dict(cube=z["cube"], dates=[str(d) for d in z["dates"]], flights=[str(x) for x in z["flights"]],
                nan_frac=z["nan_frac"].astype("float32"), window_id=int(z["window_id"]))


def index_existing(mix_dir: str) -> dict[int, str]:
    """window_id -> npz path for an existing MIX folder (files are named <id>_<dominant>.npz)."""
    out = {}
    for p in glob.glob(os.path.join(mix_dir, "windows", "*.npz")):
        b = os.path.basename(p)
        if b.endswith(".tmp.npz"):
            continue
        out[int(b.split("_")[0])] = p
    return out
