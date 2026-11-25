"""Window-level download of AVIRIS-C L2 reflectance by HTTP Range reads (ORNL DAAC).

For each (window, date) the flight line covering the window is chosen from the CMR
footprints, its ENVI header is fetched, the pixel rows intersecting the window are
read with a single ``Range`` request (BIL/BIP files are row-contiguous) and only the
columns around the window are kept on disk (native dtype, a few MB per date).

Credentials: Earthdata Login credentials are read by ``requests`` from ``~/.netrc``
(``machine urs.earthdata.nasa.gov``). They are never printed, logged or written
elsewhere; session cookies stay in memory.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import numpy as np
import requests

from . import cmr, config, envi

log = logging.getLogger(__name__)


class EarthdataSession(requests.Session):
    """requests' default redirect handling already does what Earthdata needs: the
    Authorization header is dropped when the host changes and the ~/.netrc entry of the new
    host (urs.earthdata.nasa.gov) is applied. Cookies set by URS stay in memory."""


_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = EarthdataSession()
        s.trust_env = True  # lets requests pick credentials from ~/.netrc
        _local.s = s
    return _local.s


class FatalHTTP(IOError):
    """Status codes that retrying will not fix (auth, missing file)."""


def fetch(url: str, out: str, byte_range: tuple[int, int] | None = None, expected: int | None = None,
          retries: int = 4, timeout: int = 600) -> int:
    """Download ``url`` (optionally bytes [a, b] inclusive) to ``out`` atomically.

    Resumable at file level: an existing ``out`` of the expected size is kept. Returns bytes on disk."""
    if os.path.exists(out) and (expected is None or os.path.getsize(out) == expected):
        return os.path.getsize(out)
    part = out + ".part"
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    want_status = 206 if byte_range else 200
    last = "?"
    for k in range(retries):
        try:
            with session().get(url, headers=headers, stream=True, timeout=timeout) as r:
                last = str(r.status_code)
                if r.status_code in (401, 403, 404):
                    raise FatalHTTP(f"HTTP {r.status_code}")
                if r.status_code != want_status:
                    raise IOError(f"HTTP {r.status_code}")
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            size = os.path.getsize(part)
            if expected is not None and size != expected:
                raise IOError(f"size {size} != {expected}")
            os.replace(part, out)
            return size
        except FatalHTTP:
            if os.path.exists(part):
                os.remove(part)
            raise
        except (requests.RequestException, IOError) as e:
            log.warning("download failed (%s) for %s, retry %d", e, os.path.basename(urlparse(url).path), k + 1)
            time.sleep(3 * (k + 1))
    if os.path.exists(part):
        os.remove(part)
    raise IOError(f"download failed after {retries} tries (last status {last})")


class Downloader:
    def __init__(self, work: str, flights: dict, workers: int = 2):
        self.work = work
        self.flights = flights
        self.polys = cmr.flight_polygons_5070(flights)
        self.by_date: dict[str, list[str]] = {}
        for fid, rec in flights.items():
            self.by_date.setdefault(rec["date"], []).append(fid)
        self.workers = workers
        self.hdr_dir = os.path.join(work, "hdr")
        self.stub_dir = os.path.join(work, "stubs")
        self.raw_dir = os.path.join(work, "raw")
        self._hlock = threading.Lock()
        self._hlocks: dict[str, threading.Lock] = {}
        self._headers: dict[str, tuple] = {}
        self.bytes_read = 0

    # -------------------------------------------------------------- headers
    def header(self, fid: str):
        with self._hlock:
            if fid in self._headers:
                return self._headers[fid]
            lk = self._hlocks.setdefault(fid, threading.Lock())
        with lk:
            if fid in self._headers:
                return self._headers[fid]
            os.makedirs(self.hdr_dir, exist_ok=True)
            p = os.path.join(self.hdr_dir, fid + ".hdr")
            fetch(self.flights[fid]["url"][:-4] + ".hdr", p)
            with open(p) as f:
                h = envi.parse_header(f.read())
            tr, crs = envi.georef_from_stub(h, self.stub_dir, fid)
            with self._hlock:
                self._headers[fid] = (h, tr, crs)
            return self._headers[fid]

    # -------------------------------------------------------------- one date
    def read_date(self, window_id: int, date: str) -> tuple[dict | None, str | None]:
        """Download the pixel block of one window at one date. Returns (sidecar, error)."""
        from pyproj import Transformer

        wdir = os.path.join(self.raw_dir, str(window_id))
        side = os.path.join(wdir, f"{date}.json")
        if os.path.exists(side) and os.path.exists(os.path.join(wdir, f"{date}.npy")):
            with open(side) as f:
                return json.load(f), None
        bounds = config.cell_bounds(window_id)
        fid, cov = cmr.best_flight(bounds, date, self.flights, self.polys, self.by_date)
        if fid is None:
            return None, "no flight"
        if cov < config.MIN_FLIGHT_COVERAGE:
            return None, f"coverage {cov:.2f}"
        h, tr, crs = self.header(fid)
        if not h.supported:
            return None, f"format {h.interleave}/{h.dtype}"
        x0, y0, x1, y1 = bounds
        t = Transformer.from_crs(5070, crs, always_xy=True)
        xs, ys = t.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        inv = ~tr
        pc = [inv * (a, b) for a, b in zip(xs, ys)]
        c0 = int(np.floor(min(p[0] for p in pc))) - 2
        c1 = int(np.ceil(max(p[0] for p in pc))) + 2
        r0 = int(np.floor(min(p[1] for p in pc))) - 2
        r1 = int(np.ceil(max(p[1] for p in pc))) + 2
        if c0 < 0 or r0 < 0 or c1 > h.samples or r1 > h.lines:
            return None, "outside image"
        os.makedirs(wdir, exist_ok=True)
        tmp = os.path.join(wdir, f"{date}.rows")
        n = (r1 - r0) * h.row_bytes
        fetch(self.flights[fid]["url"], tmp, (r0 * h.row_bytes, r1 * h.row_bytes - 1), expected=n)
        with open(tmp, "rb") as f:
            rows = envi.decode_rows(f.read(), h, r1 - r0)
        block = np.ascontiguousarray(rows[:, c0:c1, :])
        np.save(os.path.join(wdir, f"{date}.npy"), block)
        os.remove(tmp)
        from rasterio.transform import Affine

        sub = tr * Affine.translation(c0, r0)
        meta = dict(window_id=int(window_id), date=date, flight_id=fid, coverage=round(cov, 4), r0=r0, r1=r1, c0=c0, c1=c1,
                    transform=list(sub)[:6], crs=crs.to_wkt(), dtype=h.dtype, scale=h.scale, interleave=h.interleave,
                    bytes_read=n, wavelength=None if h.wavelength is None else h.wavelength.tolist())
        with open(side + ".part", "w") as f:
            json.dump(meta, f)
        os.replace(side + ".part", side)
        with self._hlock:
            self.bytes_read += n
        return meta, None

    # -------------------------------------------------------------- one window
    def window(self, row: dict, max_dates: int = config.N_DATES) -> dict:
        """Download up to ``max_dates`` dates of a window: the preferred dates first, then
        replacement dates (spread in time) taken from the candidate list if some fail."""
        wid = int(row["window_id"])
        want = [d for d in row["dates"].split() if d][:max_dates]
        pool = [d for d in row.get("dates_all", "").split() if d]
        got, why, tried = {}, [], set()
        for _ in range(3):
            for d in want:
                if d in got or d in tried:
                    continue
                tried.add(d)
                try:
                    meta, err = self.read_date(wid, d)
                except Exception as e:  # network or decoding failure: recorded, never fatal
                    meta, err = None, f"exception {type(e).__name__}"
                if meta is None:
                    why.append(f"{d}:{err}")
                else:
                    got[d] = meta
            if len(got) >= max_dates:
                break
            want = spread([d for d in pool if d not in tried], list(got), max_dates)[len(got):]
            if not want:
                break
        summary = dict(window_id=wid, dates=sorted(got)[:max_dates], failures=why)
        wdir = os.path.join(self.raw_dir, str(wid))
        os.makedirs(wdir, exist_ok=True)
        with open(os.path.join(wdir, "summary.json"), "w") as f:
            json.dump(summary, f)
        return summary

    def run(self, rows: list[dict], max_dates: int = config.N_DATES) -> list[dict]:
        t0 = time.time()
        out = []
        with ThreadPoolExecutor(self.workers) as ex:
            for k, s in enumerate(ex.map(lambda r: self.window(r, max_dates), rows)):
                out.append(s)
                if (k + 1) % 10 == 0 or k + 1 == len(rows):
                    log.info("download: %d/%d windows, %.2f GB read, %.0f s", k + 1, len(rows), self.bytes_read / 1e9, time.time() - t0)
        return out


def spread(pool: list[str], chosen: list[str], k: int) -> list[str]:
    """Greedy farthest-point completion of ``chosen`` with dates from ``pool`` up to ``k`` dates."""
    pool = [d for d in pool if d not in chosen]
    ords = {d: dt.date.fromisoformat(d).toordinal() for d in pool + chosen}
    chosen = list(chosen)
    while len(chosen) < k and pool:
        best = max(pool, key=lambda d: min((abs(ords[d] - ords[c]) for c in chosen), default=1e9))
        chosen.append(best)
        pool.remove(best)
    return chosen
