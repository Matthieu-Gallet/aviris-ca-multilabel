"""Ground truth: USDA NASS Cropland Data Layer (CDL) counts per window and per year.

The window grid is aligned on the native 30 m CDL grid (EPSG:5070), so each window covers
exactly 20 x 20 CDL pixels: counts are exact block counts, no resampling. CDL tiles are read
from Microsoft Planetary Computer (collection ``usda-cdl``, anonymous SAS token). The signed
URLs contain a token and are never logged.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict

import numpy as np
import requests

from . import config

log = logging.getLogger(__name__)


class Token:
    def __init__(self):
        self._t, self._ts, self._lock = None, 0.0, threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            if force or self._t is None or time.time() - self._ts > 1200:
                r = requests.get(config.PC_SAS, timeout=60)
                r.raise_for_status()
                self._t, self._ts = r.json()["token"], time.time()
            return self._t


def _item_year(f: dict) -> int | None:
    fid = f.get("id", "")
    parts = fid.split("_")
    for p in parts:
        if p.isdigit() and len(p) == 4:
            return int(p)
    dt_ = f.get("properties", {}).get("start_datetime") or f.get("properties", {}).get("datetime")
    return int(dt_[:4]) if dt_ else None


def search_items(lon: float, lat: float, years=config.CDL_YEARS) -> dict[int, str]:
    """year -> href of the CDL 'cropland' COG covering (lon, lat)."""
    body = {"collections": [config.CDL_COLLECTION], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{min(years)}-01-01T00:00:00Z/{max(years)}-12-31T23:59:59Z", "limit": 200}
    for k in range(4):
        try:
            r = requests.post(config.PC_STAC + "/search", json=body, timeout=120)
            r.raise_for_status()
            feats = r.json()["features"]
            break
        except requests.RequestException:  # pragma: no cover - network
            time.sleep(4 * (k + 1))
    else:
        raise RuntimeError("STAC search failed")
    out = {}
    for f in feats:
        if "cropland" in f.get("assets", {}) and f["id"].startswith("cropland"):
            y = _item_year(f)
            if y in years:
                out[y] = f["assets"]["cropland"]["href"]
    return out


def counts_from_codes(codes: np.ndarray, lut: np.ndarray | None = None) -> np.ndarray:
    """(20, 20) CDL codes -> (40,) group counts (codes outside the groups are ignored)."""
    lut = config.cdl_lut() if lut is None else lut
    g = lut[codes.astype(np.uint8).ravel()]
    return np.bincount(g[g < 255], minlength=len(config.GROUP_NAMES)).astype(np.uint16)


def read_window_codes(ds, window_id: int) -> np.ndarray:
    """Exact 20 x 20 CDL block of a window from an open rasterio dataset on the CDL grid."""
    from rasterio.windows import Window

    x0, y0, x1, y1 = config.cell_bounds(window_id)
    inv = ~ds.transform
    c, r = inv * (x0 + 1e-3, y1 - 1e-3)
    c, r = int(np.floor(c)), int(np.floor(r))
    n = config.CDL_PIX_PER_CELL
    col_x = ds.transform * (c, r)
    if abs(col_x[0] - x0) > 1e-3 or abs(col_x[1] - y1) > 1e-3:
        raise ValueError("window not aligned on the CDL grid")
    return ds.read(1, window=Window(c, r, n, n), boundless=True, fill_value=0)


def fetch_counts(windows: list[dict], years=config.CDL_YEARS, workers: int = 3) -> dict[int, np.ndarray]:
    """window_id -> (len(years), 40) uint16 group counts, fetched from Planetary Computer."""
    import rasterio

    tok = Token()
    lut = config.cdl_lut()
    hrefs: dict[tuple[int, str], list[int]] = defaultdict(list)
    for w in windows:
        items = search_items(float(w["lon"]), float(w["lat"]), years)
        for y in years:
            if y in items:
                hrefs[(y, items[y])].append(int(w["window_id"]))
    out = {int(w["window_id"]): np.zeros((len(years), len(config.GROUP_NAMES)), np.uint16) for w in windows}
    yidx = {y: i for i, y in enumerate(years)}

    def job(key):
        y, href = key
        for k in range(4):
            try:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                    with rasterio.open("/vsicurl/" + href + "?" + tok.get(force=k > 0)) as ds:
                        return key, {wid: counts_from_codes(read_window_codes(ds, wid), lut) for wid in hrefs[key]}
            except Exception as e:  # pragma: no cover - network
                log.warning("CDL read failed for year %d (%s), retry %d", y, type(e).__name__, k + 1)
                time.sleep(5 * (k + 1))
        return key, {}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as ex:
        for (y, _), res in ex.map(job, list(hrefs)):
            for wid, c in res.items():
                out[wid][yidx[y]] = c
    return out


def save_counts(path: str, counts: dict[int, np.ndarray], years=config.CDL_YEARS) -> None:
    ids = sorted(counts)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, window_id=np.array(ids), counts=np.stack([counts[i] for i in ids]),
                        groups=np.array(config.GROUP_NAMES), years=np.array(years))
    os.replace(tmp, path)


def load_counts(path: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    z = np.load(path)
    if list(z["groups"]) != config.GROUP_NAMES:
        raise ValueError("CDL group list in the cache differs from config.CDL_GROUPS")
    return z["window_id"], z["counts"], [int(y) for y in z["years"]]


def class_counts(counts: np.ndarray) -> np.ndarray:
    """(..., 40) group counts -> (..., 13) class counts."""
    m = np.zeros((len(config.GROUP_NAMES), config.N_CLASSES), np.float64)
    for g, c in enumerate(config.group_to_class_index()):
        m[g, c] = 1
    return counts.astype(np.float64) @ m


def write_legend_json(path: str) -> None:
    with open(path, "w") as f:
        json.dump({"groups": config.CDL_GROUPS, "classes": config.CLASS_NAMES}, f, indent=1)
